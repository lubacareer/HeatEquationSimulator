"""Held-out field errors and independent accepted-snapshot conservation gates."""
from __future__ import annotations

import math
import platform
import threading
from typing import TYPE_CHECKING

import numpy as np

from .. import __version__
from ..models import Progress
from .fields import STEFAN_BOLTZMANN, _cancelled, _scalar_field

if TYPE_CHECKING:
    from .training import TrainingRun


def _field_metrics(run: TrainingRun, predicted: dict[str, np.ndarray], initial_validation: float) -> dict[str, float]:
    field = run.field
    data = field.data
    scaling = data.scaling
    result = data.result
    conductor_mask = np.asarray(result.metadata.get('conductor_mask', np.ones(data.point_count)), dtype=bool)
    metrics: dict[str, float] = {"initial_validation_loss": initial_validation,
                                "validation_loss": run.history["validation_loss"][-1],
                                "training_loss": run.history["loss"][-1],
                                "physics_loss": run.history["physics_loss"][-1],
                                "boundary_loss": run.history["boundary_loss"][-1],
                                "epochs": float(data.config.pinn_epochs)}
    vi, vp = np.divmod(data.validation_ids, data.point_count)
    zero_reference_fields = []
    normalization = {}
    scale_start = 0
    for name, field_width in zip(scaling.names, scaling.widths):
        normalization[name] = float(scaling.scales[scale_start])
        scale_start += field_width
    normalization["temperature"] = max(float(np.ptp(np.asarray(result.fields["temperature"]))), 1.)
    normalization["liquid_fraction"] = 1.
    run.metadata["validation_field_scales"] = normalization
    for name, predicted_field in predicted.items():
        if name not in result.fields:
            continue
        expected = np.asarray(result.fields[name])
        actual = predicted_field
        if expected.ndim == 3 and expected.shape[2] == 1 and actual.ndim == 2:
            actual = actual[..., None]
        valid_samples = np.ones(len(vp), dtype=bool) if name in {"magnetic_field", "vector_potential"} else conductor_mask[vp]
        if not data.thermal and name == "pressure":
            valid_samples &= vi > 0
        if not valid_samples.any():
            # Tiny datasets can place no conductor labels in the split. Do
            # not fabricate a zero error metric for an untested field.
            continue
        difference = (actual - expected)[vi[valid_samples], vp[valid_samples]]
        rmse = float(np.sqrt(np.mean(difference ** 2)))
        amplitude = float(np.sqrt(np.mean(expected[vi[valid_samples], vp[valid_samples]] ** 2)))
        metrics[f"{name}_rmse"] = rmse
        metrics[f"{name}_normalized_rmse"] = rmse / normalization[name]
        if amplitude > max(1e-15, normalization[name] * 1e-9):
            metrics[f"{name}_relative_l2"] = rmse / amplitude
        else:
            zero_reference_fields.append(name)
    run.metadata["relative_l2_omitted_for_zero_references"] = zero_reference_fields
    return metrics


def _thermal_conservation(run: TrainingRun, predicted: dict[str, np.ndarray], metrics: dict[str, float],
                          cancel: threading.Event) -> None:
    field = run.field
    data = field.data
    config = data.config
    operator = data.operator
    result = data.result
    assert operator is not None
    h = _scalar_field(predicted["enthalpy"], "enthalpy")
    # Integrated global energy defect uses accepted-time differences and
    # midpoint boundary loss, separately from continuous training residual.
    total = config.density * np.einsum("tn,n->t", h, operator.volumes)
    energy_defects = []
    residual_square_sum = 0.
    residual_element_count = 0
    checked = set(np.unique(np.linspace(1, data.snapshot_count - 1, min(10, data.snapshot_count - 1), dtype=np.int64)).tolist())
    for k, dt in enumerate(np.diff(result.times), 1):
        _cancelled(cancel)
        time = (result.times[k - 1] + result.times[k]) / 2
        power = config.heating_power if time < config.charge_fraction * config.duration else -config.discharge_power
        temperature = predicted["temperature"][k]
        boundary_node_w = config.emissivity * STEFAN_BOLTZMANN * operator.boundary_area * (
            temperature ** 4 - config.ambient_temperature ** 4)
        boundary_w = float(boundary_node_w.sum())
        energy_defects.append(float((total[k] - total[k - 1]) - dt * (power - boundary_w)))
        if k in checked:
            transfer = operator.conductance * (temperature[operator.edge_i] - temperature[operator.edge_j])
            conduction = np.zeros(data.point_count)
            np.add.at(conduction, operator.edge_i, transfer)
            np.add.at(conduction, operator.edge_j, -transfer)
            residual = (config.density * operator.volumes * (h[k] - h[k - 1]) / dt
                        + conduction + boundary_node_w - power * operator.source_weights)
            residual /= config.density * operator.volumes * data.scaling.scales[0] / field.time_scale
            residual_square_sum += float(np.sum(residual ** 2))
            residual_element_count += data.point_count
    metrics["energy_defect_max_j"] = max(abs(v) for v in energy_defects)
    scale_j = max(abs(float(total[-1] - total[0])),
                  config.heating_power * config.duration, config.discharge_power * config.duration, 1.0)
    metrics["energy_defect_relative"] = metrics["energy_defect_max_j"] / scale_j
    metrics["enthalpy_residual_rms"] = math.sqrt(residual_square_sum / residual_element_count)
    run.metadata["conservation_snapshot_indices"] = sorted(checked)
    metrics["phase_fraction_rmse"] = metrics.get("liquid_fraction_rmse", 0.0)


def _coupled_conservation(run: TrainingRun, predicted: dict[str, np.ndarray], metrics: dict[str, float],
                          progress: Progress, cancel: threading.Event) -> None:
    field = run.field
    data = field.data
    torch = field.torch
    assert data.weak is not None
    # Evaluate final accepted-snapshot fields independently of the random
    # last training batch, including both ends of the physical timeline.
    residual_sums: dict[str, float] = {}
    residual_counts: dict[str, int] = {}
    selected = np.unique(np.linspace(1, data.snapshot_count - 1, min(10, data.snapshot_count - 1), dtype=np.int64))
    with torch.no_grad():
        for position, ti in enumerate(selected):
            _cancelled(cancel)
            fields = {key: torch.as_tensor(value[ti - 1:ti + 1], **field.kw) for key, value in predicted.items()}
            components = data.weak(data.config, data.result, fields, np.asarray([ti - 1, ti], dtype=np.int64))
            for key, value in components.items():
                if not isinstance(value, torch.Tensor) or not bool(torch.isfinite(value).all()) or not value.numel():
                    raise ValueError(f"Final FE PINN residual {key} is empty or non-finite.")
                residual_sums[key] = residual_sums.get(key, 0.) + float(value.square().sum().item())
                residual_counts[key] = residual_counts.get(key, 0) + value.numel()
            progress(.98 + .015 * (position + 1) / len(selected), "Checking accepted-snapshot conservation and boundary defects")
    for key, square_sum in residual_sums.items():
        metrics[f"{key}_residual_rms"] = math.sqrt(square_sum / residual_counts[key])
    run.metadata["conservation_snapshot_indices"] = selected.tolist()


def validate_prediction(run: TrainingRun, predicted: dict[str, np.ndarray], initial_validation: float,
                        progress: Progress, cancel: threading.Event) -> dict[str, float]:
    field = run.field
    data = field.data
    metrics = _field_metrics(run, predicted, initial_validation)
    if data.thermal:
        _thermal_conservation(run, predicted, metrics, cancel)
    else:
        _coupled_conservation(run, predicted, metrics, progress, cancel)
    if not all(math.isfinite(value) for value in metrics.values()):
        raise FloatingPointError("PINN validation produced non-finite errors or conservation defects; no result was published.")
    conservation_ok = (metrics.get("energy_defect_relative", 0.) <= .05
                       and metrics["enthalpy_residual_rms"] <= .1 if data.thermal else
                       all(value <= .1 for key, value in metrics.items() if key.endswith("_residual_rms")))
    fields_ok = all(value <= .1 for key, value in metrics.items() if key.endswith("_normalized_rmse"))
    missing_validation = [name for name in data.scaling.names if name + "_normalized_rmse" not in metrics]
    run.metadata["fields_without_heldout_labels"] = missing_validation
    run.metadata["validation_status"] = ("passed" if metrics["validation_loss"] <= 1e-2
                                      and metrics["physics_loss"] <= 1e-2
                                      and metrics["boundary_loss"] <= 1e-2 and conservation_ok
                                      and fields_ok and not missing_validation else "failed")
    run.metadata["validation_criteria"] = (
        "held-out normalized MSE ≤ 0.01; each field normalized RMSE ≤ 0.1; training physics/boundary MSE ≤ 0.01; "
        + ("maximum interval energy defect / reference energy ≤ 0.05 and independent enthalpy residual RMS ≤ 0.1" if data.thermal else
           "each independently checked FE normalized residual RMS ≤ 0.1"))
    run.metadata["validation_notes"] = ("Agreement with this source run is assessed; this does not establish "
                                    "physical validity or mesh convergence of the source simulation.")
    run.metadata["software_versions"] = {"torch": field.torch.__version__, "numpy": np.__version__,
                                     "python": platform.python_version(), "simulator": __version__}
    return metrics
