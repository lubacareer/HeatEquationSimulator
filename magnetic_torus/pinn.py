"""Scenario-bound PyTorch PINNs using the numerical worker's discrete equations.

The thermal benchmark uses its conservative finite-volume spatial operator and
accepted backward-Euler intervals. Coupled jobs use the *same differentiable
weak residual* as the FE worker; a strong-form surrogate is never substituted for
the mixed VMS turbulence closure.  This module imports PyTorch lazily so the
dashboard can inspect simulations without starting a CUDA runtime.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from importlib import import_module
import json
import math
import os
from pathlib import Path
import platform
import threading
from typing import Any, Callable, Mapping
from uuid import uuid4

import numpy as np

from .models import PINNResult, Progress, RunResult, SimulationCancelled
from . import __version__

STEFAN_BOLTZMANN = 5.670374419e-8
FORMAT_VERSION = 1


class UnsupportedPINNConfiguration(ValueError):
    """A source result cannot supply the matching, validated physics operator."""


def validate_pinn_source(pinn: PINNResult, source: RunResult) -> None:
    """Reject stale/misattributed saved or worker results before displaying them."""
    if pinn.source_run_id != source.run_id or pinn.config_hash != source.config.config_hash:
        raise ValueError("This PINN belongs to a different numerical run or scenario configuration. Select its source run or train a new PINN.")
    if (pinn.metadata.get("source_physics_hash") is not None
            and pinn.metadata["source_physics_hash"] != source.config.physics_hash):
        raise ValueError("PINN physical configuration differs from this source; observer-only model reuse cannot change its physics.")
    if not pinn.metadata.get("explicit_prior_model_reuse", False):
        if (pinn.metadata.get("source_run_id", source.run_id) != source.run_id
                or pinn.metadata.get("source_config_hash", source.config.config_hash) != source.config.config_hash):
            raise ValueError("PINN provenance disagrees with the claimed source run/configuration.")
    expected_shape = (len(source.times), len(source.points))
    for name, field in pinn.fields.items():
        if field.shape[:2] != expected_shape or not np.isfinite(field).all():
            raise ValueError(f"PINN field {name} is incompatible with its source snapshots.")
    if "source_snapshot_times_s" in pinn.metadata and not np.array_equal(
            np.asarray(pinn.metadata["source_snapshot_times_s"]), source.times):
        raise ValueError("PINN accepted snapshot times differ from the source run.")


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PINN training needs PyTorch in the torch-gpu environment.") from exc
    return torch


def _cancelled(cancel: threading.Event) -> None:
    if cancel.is_set():
        raise SimulationCancelled("PINN training cancelled; the numerical result is unchanged.")


def _scalar_field(array: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(array, dtype=np.float64)
    if value.ndim == 3 and value.shape[2] == 1:
        value = value[..., 0]
    if value.ndim != 2 or not np.isfinite(value).all():
        raise ValueError(f"PINN {name} references must have finite shape (time, point).")
    return value


def enthalpy_to_state_torch(enthalpy: Any, config: Any) -> tuple[Any, Any]:
    """Differentiable sharp enthalpy law, SI units and T_m as zero-energy solid.

    The material plateau is exact; its derivative is zero inside (0, LH).
    At the two nonsmooth endpoints PyTorch uses the selected one-sided branch.
    """
    torch = _torch()
    solid = config.melting_temperature + enthalpy / config.heat_capacity_solid
    liquid = config.melting_temperature + (enthalpy - config.latent_heat) / config.heat_capacity_liquid
    plateau = torch.full_like(enthalpy, config.melting_temperature)
    temperature = torch.where(enthalpy < 0, solid,
                              torch.where(enthalpy > config.latent_heat, liquid, plateau))
    if config.latent_heat > 0:
        fraction = torch.clamp(enthalpy / config.latent_heat, 0.0, 1.0)
    else:
        fraction = (enthalpy > 0).to(enthalpy.dtype)
    return temperature, fraction


def regularized_enthalpy_to_state_torch(enthalpy: Any, config: Any) -> tuple[Any, Any]:
    """Exact inverse of the FE worker's regularized apparent-capacity law."""
    torch = _torch()
    width = config.transition_width
    lower = -.5 * config.heat_capacity_solid * width
    upper = config.latent_heat + .5 * config.heat_capacity_liquid * width
    middle_h = torch.clamp(enthalpy, min=lower, max=upper) - lower
    linear = config.heat_capacity_solid * width + config.latent_heat
    quadratic = .5 * (config.heat_capacity_liquid - config.heat_capacity_solid) * width
    # Rationalized quadratic root stays well-conditioned when the two heat
    # capacities coincide, and avoids subtracting nearly equal numbers.
    root = torch.sqrt(torch.clamp(linear ** 2 + 4 * quadratic * middle_h, min=1e-24))
    middle_fraction = 2 * middle_h / (linear + root)
    middle_temperature = config.melting_temperature + width * (middle_fraction - .5)
    temperature = torch.where(enthalpy < lower,
                              config.melting_temperature + enthalpy / config.heat_capacity_solid,
                              torch.where(enthalpy > upper, config.melting_temperature
                                          + (enthalpy - config.latent_heat) / config.heat_capacity_liquid,
                                          middle_temperature))
    fraction = torch.where(enthalpy < lower, torch.zeros_like(enthalpy),
                           torch.where(enthalpy > upper, torch.ones_like(enthalpy), middle_fraction))
    return temperature, fraction


@dataclass(frozen=True)
class ThermalOperator:
    """Conservative thermal spatial operator exported by the benchmark worker.

    Each undirected edge contributes g_ij (T_i-T_j) in watts.  P1 stiffness
    entries on non-Delaunay meshes can have either sign.  Source weights sum to
    one, node volumes are m^3, and exposed lumped boundary areas are m^2.
    """
    edge_i: np.ndarray
    edge_j: np.ndarray
    conductance: np.ndarray
    volumes: np.ndarray
    boundary_area: np.ndarray
    source_weights: np.ndarray

    @classmethod
    def from_result(cls, result: RunResult) -> "ThermalOperator":
        raw = result.metadata.get("pinn_physics", {})
        if not isinstance(raw, Mapping) or raw.get("kind") not in {"thermal_p1_lumped_v1", "thermal_fv_lumped_v1"}:
            raise UnsupportedPINNConfiguration(
                "This result lacks the thermal physics operator. Recalculate the Thermal "
                "benchmark with the current numerical worker before training.")
        n = len(result.points)
        # Arrays can be inline, or an artifact path keeps run-history JSON small.
        if "operator_path" in raw:
            path = Path(str(raw["operator_path"]))
            if not path.is_file():
                raise UnsupportedPINNConfiguration(f"Thermal PINN operator artifact is missing: {path}")
            with np.load(path, allow_pickle=False) as artifact:
                values = {key: artifact[key] for key in artifact.files}
        else:
            values = raw
        try:
            edge_i = np.asarray(values["edge_i"], dtype=np.int64)
            edge_j = np.asarray(values["edge_j"], dtype=np.int64)
            conductance = np.asarray(values["conductance_w_k"], dtype=np.float64)
            volumes = np.asarray(values["node_volume"], dtype=np.float64)
            area = np.asarray(values["boundary_area"], dtype=np.float64)
            weights = np.asarray(values["source_weights"], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as exc:
            raise UnsupportedPINNConfiguration("Thermal PINN operator artifact is incomplete or invalid.") from exc
        if (edge_i.ndim != 1 or edge_i.shape != edge_j.shape or edge_i.shape != conductance.shape
                or np.any(edge_i < 0) or np.any(edge_j < 0)
                or np.any(edge_i >= n) or np.any(edge_j >= n) or np.any(edge_i == edge_j)
                or volumes.shape != (n,) or area.shape != (n,) or weights.shape != (n,)
                or not all(np.isfinite(v).all() for v in (conductance, volumes, area, weights))
                or np.any(volumes <= 0) or np.any(area < 0) or np.any(weights < 0)
                or not np.isclose(weights.sum(), 1.0, atol=1e-7)):
            raise UnsupportedPINNConfiguration("Thermal PINN operator has invalid edges, volumes, areas, or source weights.")
        return cls(edge_i, edge_j, conductance, volumes, area, weights)

    def residual(self, enthalpy_rate: Any, temperature: Any, time: Any, config: Any) -> Any:
        """Full-node physical residual in watts (used by validation and tests)."""
        torch = _torch()
        kw = {"device": temperature.device, "dtype": temperature.dtype}
        i = torch.as_tensor(self.edge_i, device=temperature.device)
        j = torch.as_tensor(self.edge_j, device=temperature.device)
        g = torch.as_tensor(self.conductance, **kw)
        conduction = torch.zeros_like(temperature)
        transfer = g * (temperature[i] - temperature[j])
        conduction = conduction.index_add(0, i, transfer).index_add(0, j, -transfer)
        volume = torch.as_tensor(self.volumes, **kw)
        area = torch.as_tensor(self.boundary_area, **kw)
        weights = torch.as_tensor(self.source_weights, **kw)
        t = torch.as_tensor(time, **kw)
        power = torch.where(t <= config.charge_fraction * config.duration,
                            torch.as_tensor(config.heating_power, **kw),
                            torch.as_tensor(-config.discharge_power, **kw))
        radiation = config.emissivity * STEFAN_BOLTZMANN * area * (
            temperature ** 4 - config.ambient_temperature ** 4)
        return config.density * volume * enthalpy_rate + conduction + radiation - power * weights

    def neighborhood(self, centers: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Local edge stencil; no full N by N matrix is constructed on the GPU."""
        centers = np.asarray(centers, dtype=np.int64)
        lookup = np.full(len(self.volumes), -1, dtype=np.int64)
        lookup[centers] = np.arange(len(centers))
        left = lookup[self.edge_i] >= 0
        right = lookup[self.edge_j] >= 0
        rows = np.concatenate((lookup[self.edge_i[left]], lookup[self.edge_j[right]]))
        first = np.concatenate((self.edge_i[left], self.edge_j[right]))
        second = np.concatenate((self.edge_j[left], self.edge_i[right]))
        g = np.concatenate((self.conductance[left], self.conductance[right]))
        support, inverse = np.unique(np.concatenate((centers, first, second)), return_inverse=True)
        c = len(centers)
        e = len(first)
        # Each selected row has center and neighbor contribution.
        sparse_rows = np.concatenate((rows, rows))
        sparse_cols = np.concatenate((inverse[c:c + e], inverse[c + e:]))
        sparse_values = np.concatenate((g, -g))
        return support, inverse[:c], np.stack((sparse_rows, sparse_cols)), sparse_values


@dataclass(frozen=True)
class FieldScaling:
    names: tuple[str, ...]
    widths: tuple[int, ...]
    offsets: np.ndarray
    scales: np.ndarray

    @classmethod
    def from_result(cls, result: RunResult, coupled: bool,
                    training_ids: np.ndarray | None = None) -> "FieldScaling":
        names = ("enthalpy", "velocity", "magnetic_field", "vector_potential", "pressure",
                 "electric_potential") if coupled else ("enthalpy",)
        widths = (1, 3, 3, 3, 1, 1) if coupled else (1,)
        offsets: list[np.ndarray] = []
        scales: list[np.ndarray] = []
        for name, width in zip(names, widths):
            if name not in result.fields:
                raise UnsupportedPINNConfiguration(f"The source run has no {name.replace('_', ' ')} reference field.")
            array = np.asarray(result.fields[name], dtype=np.float64)
            if width == 1:
                array = _scalar_field(array, name)[..., None]
            if array.shape != (len(result.times), len(result.points), width) or not np.isfinite(array).all():
                raise ValueError(f"PINN {name} references have an invalid shape or non-finite values.")
            conductor_mask = np.asarray(result.metadata.get("conductor_mask", np.ones(len(result.points))), dtype=bool)
            if conductor_mask.shape != (len(result.points),) or not conductor_mask.any():
                raise ValueError("PINN conductor mask must select at least one source mesh point.")
            if training_ids is None:
                flattened = (array[:, conductor_mask] if name not in {"magnetic_field", "vector_potential"}
                             else array).reshape(-1, width)
            else:
                ti, pi = np.divmod(training_ids, len(result.points))
                valid = (conductor_mask[pi] if name not in {"magnetic_field", "vector_potential"}
                         else np.ones(len(pi), dtype=bool))
                if name == "pressure":
                    valid &= ti > 0
                if not valid.any():
                    raise UnsupportedPINNConfiguration(f"The reference split has no training labels for {name}; use a larger benchmark.")
                flattened = array[ti[valid], pi[valid]]
            offset = flattened.mean(axis=0)
            # Per-field scales preserve vector-component comparability.  A floor
            # prevents constant reference fields amplifying harmless roundoff.
            floor = {"enthalpy": result.config.heat_capacity_solid,
                     "velocity": 1e-3, "magnetic_field": 1e-6,
                     "vector_potential": 1e-6 * result.config.major_radius,
                     "pressure": 1e-3, "electric_potential": 1e-6}[name]
            amplitude = float(np.sqrt(np.mean(flattened ** 2)))
            if name != "enthalpy" and amplitude > 1e-15:
                # Small but physically nonzero flow/potential must not vanish
                # behind a large absolute floor. Exact-zero references retain
                # a characteristic floor and report absolute errors.
                floor = amplitude * .01
            scale = max(float(flattened.std()), float(np.ptp(flattened, axis=0).max()) / 4, floor)
            offsets.append(offset)
            scales.append(np.full(width, scale))
        return cls(names, widths, np.concatenate(offsets), np.concatenate(scales))

    def references(self, result: RunResult) -> np.ndarray:
        arrays = []
        for name, width in zip(self.names, self.widths):
            value = np.asarray(result.fields[name], dtype=np.float64)
            arrays.append(value[..., None] if width == 1 and value.ndim == 2 else value)
        return np.concatenate(arrays, axis=2)

    def split(self, tensor: Any) -> dict[str, Any]:
        fields: dict[str, Any] = {}
        start = 0
        for name, width in zip(self.names, self.widths):
            field = tensor[..., start:start + width]
            fields[name] = field[..., 0] if width == 1 else field
            start += width
        return fields


def _make_network(torch: Any, output_width: int, input_width: int = 4) -> Any:
    return torch.nn.Sequential(torch.nn.Linear(input_width, 64), torch.nn.Tanh(),
                               torch.nn.Linear(64, 64), torch.nn.Tanh(),
                               torch.nn.Linear(64, 64), torch.nn.Tanh(),
                               torch.nn.Linear(64, output_width))


def _weak_adapter(result: RunResult) -> Callable[..., Mapping[str, Any]]:
    """Only accept the trusted in-package operator, never a module from metadata."""
    if result.metadata.get("pinn_physics", {}).get("kind") != "fenics_a_phi_vms_v1":
        raise UnsupportedPINNConfiguration("This coupled run has no matching A–phi/VMS weak-operator version. Recalculate it with the current FEniCS worker.")
    try:
        module = import_module("magnetic_torus.physics")
        adapter = getattr(module, "evaluate_weak_residual_torch")
    except (ImportError, AttributeError) as exc:
        raise UnsupportedPINNConfiguration(
            "Coupled PINN training needs the numerical worker's differentiable FE weak operator. "
            "Recalculate with the current FEniCS worker, or choose the Thermal benchmark. "
            "A strong-form PINN cannot replace the mixed VMS turbulence equations.") from exc
    path = result.metadata.get("weak_operator_path")
    if not path or not Path(str(path)).is_file():
        raise UnsupportedPINNConfiguration(
            "This coupled result lacks its FE weak-operator artifact. Recalculate the scenario "
            "and retain the run directory before training its PINN.")
    if len(result.times) < 2:
        raise UnsupportedPINNConfiguration("Coupled PINNs require at least two accepted FE snapshots.")
    return adapter


def _operator_fingerprint(operator: ThermalOperator | None, result: RunResult) -> str:
    digest = sha256()
    if operator is not None:
        for array in (operator.edge_i, operator.edge_j, operator.conductance,
                      operator.volumes, operator.boundary_area, operator.source_weights):
            digest.update(np.ascontiguousarray(array).tobytes())
    else:
        with Path(str(result.metadata["weak_operator_path"])).open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _coupled_trial_fields(result: RunResult, scaling: FieldScaling) -> np.ndarray:
    """Known particular fields, never copied from unknown numerical solutions.

    Imposed A comes from the *coil/seed specification* in the FE artifact. B is
    its matching FE curl projection, not a sampled reference solution. The
    initial enthalpy and integral of prescribed volumetric input are also
    known. Learning the induced/circulating correction avoids rediscovering a
    sharply varying external coil field from sparse labels.
    """
    with np.load(str(result.metadata["weak_operator_path"]), allow_pickle=False) as artifact:
        required = ("imposed_vector_potential", "cells", "gradients", "volumes", "node_volumes")
        try:
            op = {key: artifact[key] for key in required}
        except KeyError as exc:
            raise UnsupportedPINNConfiguration("Coupled PINN trial fields require the matching imposed-field FE artifact.") from exc
    nt, nn = len(result.times), len(result.points)
    imposed = np.asarray(op["imposed_vector_potential"], dtype=np.float64)
    if imposed.shape != (nt, nn, 3) or not np.isfinite(imposed).all():
        raise UnsupportedPINNConfiguration("The FE imposed-field artifact does not match the accepted snapshots.")
    cells, gradients, volumes = op["cells"], op["gradients"], op["volumes"]
    magnetic = np.empty_like(imposed)
    for k in range(nt):
        gradient = np.einsum("cai,caj->cij", imposed[k, cells], gradients)
        curl = np.column_stack((gradient[:, 2, 1] - gradient[:, 1, 2],
                                gradient[:, 0, 2] - gradient[:, 2, 0],
                                gradient[:, 1, 0] - gradient[:, 0, 1]))
        projected = np.zeros((nn, 3))
        weight = np.zeros(nn)
        for local_node in range(4):
            np.add.at(projected, cells[:, local_node], volumes[:, None] * curl)
            np.add.at(weight, cells[:, local_node], volumes)
        if np.any(weight <= 0):
            raise UnsupportedPINNConfiguration("The coupled FE artifact has isolated mesh nodes.")
        magnetic[k] = projected / weight[:, None]
    config = result.config
    delta = config.initial_temperature - config.melting_temperature
    fraction = float(np.clip(delta / config.transition_width + .5, 0., 1.))
    h0 = (config.heat_capacity_solid * delta if delta < -config.transition_width / 2 else
          config.latent_heat + config.heat_capacity_liquid * delta if delta > config.transition_width / 2 else
          config.heat_capacity_solid * delta + config.latent_heat * fraction
          + .5 * (config.heat_capacity_liquid - config.heat_capacity_solid) * config.transition_width * fraction ** 2)
    charge_end = config.charge_fraction * config.duration
    supplied = (config.heating_power * np.minimum(result.times, charge_end)
                - config.discharge_power * np.maximum(result.times - charge_end, 0.))
    conductor = np.asarray(op["node_volumes"]) > 0
    total_volume = float(np.sum(op["node_volumes"]))
    if total_volume <= 0:
        raise UnsupportedPINNConfiguration("The coupled FE artifact has no conductor volume.")
    h = h0 + supplied[:, None] / (config.density * total_volume) * conductor[None]
    fixed_temperature = np.asarray(result.metadata.get("pinn_physics", {}).get("bc_nodes", {}).get("4", ()), dtype=int)
    h[:, fixed_temperature] = h0
    fields = {"enthalpy": np.broadcast_to(h, (nt, nn)), "vector_potential": imposed,
              "magnetic_field": magnetic}
    return np.concatenate([fields.get(name, np.zeros((nt, nn, width)))
                           if width > 1 else fields.get(name, np.zeros((nt, nn)))[..., None]
                           for name, width in zip(scaling.names, scaling.widths)], axis=2)


def _checkpoint(path: Path, model: Any, optimizer: Any, epoch: int, result: RunResult,
                scaling: FieldScaling, history: Mapping[str, list[float]], metadata: Mapping[str, Any]) -> None:
    torch = _torch()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial")
    torch.save({"format_version": FORMAT_VERSION, "epoch": epoch,
                "source_run_id": result.run_id, "config_hash": result.config.config_hash,
                "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                "field_names": scaling.names, "field_widths": scaling.widths,
                "field_offsets": scaling.offsets.tolist(), "field_scales": scaling.scales.tolist(),
                "scenario": result.config.to_dict(), "history": dict(history),
                "metadata": dict(metadata)}, temporary)
    temporary.replace(path)


def train_pinn(result: RunResult, progress: Progress, cancel: threading.Event,
               output_dir: Path | None = None) -> PINNResult:
    """Train and validate one scenario without modifying the numerical result.

    Runtime errors and cancellation never publish a partial PINNResult.  Finite
    but inaccurate training is returned with ``validation_status='failed'`` so
    the dashboard can display errors without implying numerical agreement.
    """
    _cancelled(cancel)
    config = result.config.validate()
    thermal = config.solver == "thermal_benchmark"
    if thermal and config.turbulence:
        raise UnsupportedPINNConfiguration("The Thermal benchmark PINN cannot model turbulence. Disable turbulence or use FEniCS.")
    operator = ThermalOperator.from_result(result) if thermal else None
    weak = None if thermal else _weak_adapter(result)
    if len(result.times) < 2 or len(result.points) < 1:
        raise UnsupportedPINNConfiguration("PINN training requires a nonempty mesh and at least two accepted times.")
    rng = np.random.default_rng(config.seed)
    all_ids = np.arange(len(result.times) * len(result.points), dtype=np.int64)
    rng.shuffle(all_ids)
    holdout_count = min(len(all_ids) - 1, max(1, len(all_ids) // 5))
    validation_ids, train_ids = all_ids[:holdout_count], all_ids[holdout_count:]
    scaling = FieldScaling.from_result(result, coupled=not thermal, training_ids=train_ids)
    for name in ("temperature", "liquid_fraction"):
        if name not in result.fields:
            raise UnsupportedPINNConfiguration(f"The source run has no {name.replace('_', ' ')} reference field.")
        _scalar_field(result.fields[name], name)
    reference = scaling.references(result)
    nt, nn, width = reference.shape
    if thermal and not np.allclose(reference[0, :, 0], reference[0, 0, 0], rtol=1e-12, atol=1e-9):
        raise UnsupportedPINNConfiguration("The thermal PINN trial function requires the benchmark's uniform initial enthalpy.")
    torch = _torch()
    requested_device = os.environ.get("MAGNETIC_TORUS_PINN_DEVICE", "auto")
    if requested_device not in {"auto", "cpu", "cuda"}:
        raise ValueError("MAGNETIC_TORUS_PINN_DEVICE must be auto, cpu, or cuda.")
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was explicitly requested for PINN training but is unavailable.")
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if requested_device == "auto" else requested_device)
    # fork_rng avoids changing the caller's global CPU/CUDA random sequence.
    devices = [torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(config.seed)
        model = _make_network(torch, width, 4 if thermal else 21).to(device=device, dtype=torch.float64)
        # Begin with the prescribed particular solution; learn only the
        # correction. No unknown reference fields initialize neural weights.
        torch.nn.init.zeros_(model[-1].weight)
        torch.nn.init.zeros_(model[-1].bias)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, config.pinn_epochs), eta_min=1e-5)
    maximum_batch = min(config.pinn_batch_size, 4096)
    # A full FE pair retains its autograd graph. Bound this explicit, rather
    # than implying that split forward calls alone limit activation memory.
    if not thermal and nn * width * 2 > 2_000_000:
        raise UnsupportedPINNConfiguration(
            "The coupled PINN mesh exceeds the bounded training budget. Recalculate a smaller "
            "mesh (at most 2,000,000 scalar outputs per snapshot pair) before training.")
    dtype = torch.float64
    kw = {"device": device, "dtype": dtype}
    offsets = torch.as_tensor(scaling.offsets, **kw)
    scales = torch.as_tensor(scaling.scales, **kw)
    temporal_frequencies = torch.as_tensor(2 * math.pi * 2. ** np.arange(6), **kw)
    conductor_mask = np.asarray(result.metadata.get("conductor_mask", np.ones(nn)), dtype=bool)
    field_validity = np.ones((nn, width), dtype=bool)
    field_start = 0
    for name, field_width in zip(scaling.names, scaling.widths):
        if name not in {"magnetic_field", "vector_potential"}:
            field_validity[:, field_start:field_start + field_width] = conductor_mask[:, None]
        field_start += field_width
    length_scale = config.major_radius + config.minor_radius
    start_time = float(result.times[0])
    time_scale = float(result.times[-1] - result.times[0])
    if not len(train_ids):
        raise ValueError("Too few reference samples for a disjoint held-out validation split.")
    run_id = "pinn-" + uuid4().hex[:12]
    checkpoint_path = Path(output_dir) / run_id / "checkpoint.pt" if output_dir is not None else None
    toroidal_thermal_inputs = thermal and "structured_shape" in result.metadata
    coupled_trial = None if thermal else _coupled_trial_fields(result, scaling)
    coupled_free = np.ones((nn, width), dtype=np.float64)
    if not thermal:
        conditions = result.metadata.get("pinn_physics", {}).get("bc_nodes", {})
        slices = {"enthalpy": (0, 4), "velocity": (1, 0), "magnetic_field": (4, None),
                  "vector_potential": (7, 2), "pressure": (10, 1), "electric_potential": (11, 3)}
        for name, field_width in zip(scaling.names, scaling.widths):
            start, condition_index = slices[name]
            if condition_index is not None:
                nodes = np.asarray(conditions.get(str(condition_index), ()), dtype=np.int64)
                coupled_free[nodes, start:start + field_width] = 0.
    metadata: dict[str, Any] = {
        "format_version": FORMAT_VERSION, "source_run_id": result.run_id,
        "source_config_hash": config.config_hash, "source_snapshot_times_s": result.times.tolist(),
        "source_physics_hash": config.physics_hash, "training_source_run_id": result.run_id,
        "training_config_hash": config.config_hash, "training_epochs": config.pinn_epochs,
        "physics": result.metadata["pinn_physics"]["kind"] if thermal else "coupled_fe_weak_operator",
        "turbulence": bool(config.turbulence), "device": str(device), "torch_version": torch.__version__,
        "seed": config.seed, "training_samples": int(len(train_ids)),
        "heldout_samples": int(len(validation_ids)), "max_forward_batch": maximum_batch,
        "heldout_split": "disjoint seeded point-time labels; physics collocation may use the same locations",
        "normalization_fitted_on": "training labels only",
        "data_loss_aggregation": "mean of per-field valid-component MSE; conductor fields masked; initial pressure is not a dynamical initial condition",
        "excluded_initial_labels": [] if thermal else ["pressure"],
        "field_names": list(scaling.names), "field_offsets": scaling.offsets.tolist(),
        "field_scales": scaling.scales.tolist(), "length_scale_m": length_scale,
        "input_transform": ("axisymmetric toroidal coordinates and time" if toroidal_thermal_inputs else
                            "Cartesian and toroidal geometry features plus time" if not thermal else "Cartesian coordinates and time"),
        "architecture": {"widths": [4 if thermal else 21, 64, 64, 64, width], "activation": "tanh", "dtype": "float64"},
        "temporal_features": [] if thermal else ["sin/cos(2π·2^k·normalized_time), k=0,…,5"],
        "optimizer": {"name": "Adam", "initial_learning_rate": .001,
                      "schedule": "cosine", "final_learning_rate": 1e-5, "gradient_norm_limit": 10.},
        "hard_constraints": "specified dynamical initial state and FE Dirichlet node masks; pressure is elliptic" if not thermal else "specified initial state",
        "particular_solution": ("local prescribed heating/discharge with initial-state linearized radiation; neural correction enforces exact nonlinear radiation and conduction" if thermal else
                                "prescribed initial enthalpy, integrated input and imposed coil/seed FE field; neural correction learns coupled response"),
        "time_scale_s": time_scale,
        "time_discretization": "source worker's accepted backward-Euler intervals and source schedule",
        "enthalpy_law": "sharp T(E) plateau" if thermal else "worker apparent-capacity regularization",
        "source_validation_status": result.metadata.get("validation_status", "unvalidated"),
        "source_backend": result.metadata.get("backend", "unspecified"),
        "source_model": result.metadata.get("model", "unspecified"),
        "source_material_provenance": config.provenance,
        "source_operator_sha256": _operator_fingerprint(operator, result),
        "source_closure": result.metadata.get("closure", "not applicable"),
        "source_closure_reference": result.metadata.get("closure_reference", "not applicable"),
        "source_software_versions": {key: str(result.metadata[key]) for key in
            ("dolfinx_version", "petsc_version", "gmsh_version") if key in result.metadata},
        "validation_status": "unvalidated", "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "weak_operator_path": result.metadata.get("weak_operator_path"),
        "thermal_physics_only": thermal,
    }
    history: dict[str, list[float]] = {key: [] for key in (
        "epoch", "loss", "data_loss", "physics_loss", "boundary_loss", "validation_loss")}
    progress(0.0, f"Preparing {'thermal' if thermal else 'coupled weak-form'} PINN on {device}")

    def input_at(time_ids: np.ndarray, point_ids: np.ndarray, *, grad: bool = False) -> Any:
        coordinate = np.column_stack((result.points[point_ids] / length_scale,
                                       (result.times[time_ids] - start_time) / time_scale))
        return torch.tensor(coordinate, requires_grad=grad, **kw)

    def forward(inputs: Any, point_ids: np.ndarray | None = None, time_ids: np.ndarray | None = None) -> Any:
        network_inputs = inputs
        if toroidal_thermal_inputs:
            # The native thermal boundary/source and geometry are invariant
            # around the major circle. Enforce that exact symmetry instead of
            # asking labels to rediscover it in Cartesian coordinates.
            major_r = torch.sqrt(inputs[:, 0] ** 2 + inputs[:, 1] ** 2) * length_scale
            horizontal = major_r - config.major_radius
            vertical = inputs[:, 2] * length_scale
            tube_r = torch.sqrt(horizontal ** 2 + vertical ** 2 + 1e-24)
            network_inputs = torch.stack((tube_r / config.minor_radius,
                                          horizontal / tube_r, vertical / tube_r, inputs[:, 3]), dim=1)
        elif not thermal:
            major_r = torch.sqrt(inputs[:, 0] ** 2 + inputs[:, 1] ** 2 + 1e-24)
            horizontal = major_r * length_scale - config.major_radius
            vertical = inputs[:, 2] * length_scale
            tube_r = torch.sqrt(horizontal ** 2 + vertical ** 2 + 1e-24)
            # Cartesian coordinates keep all 3D degrees of freedom; geometry
            # features resolve the thin tube without imposing axisymmetry.
            network_inputs = torch.cat((inputs[:, :3], torch.stack((tube_r / config.minor_radius,
                horizontal / tube_r, vertical / tube_r, inputs[:, 1] / major_r,
                inputs[:, 0] / major_r, inputs[:, 3]), dim=1)), dim=1)
            phase = inputs[:, 3:4] * temporal_frequencies[None]
            # Accepted adaptive timesteps can resolve rapid changes in the
            # elliptic fields. A time-only Fourier basis avoids asking a
            # low-frequency tanh network to smooth those changes away.
            network_inputs = torch.cat((network_inputs, torch.sin(phase), torch.cos(phase)), dim=1)
        correction = torch.cat([model(chunk) for chunk in network_inputs.split(maximum_batch)], dim=0)
        if not thermal:
            assert coupled_trial is not None
            if point_ids is None or time_ids is None:
                raise ValueError("Coupled weak-form PINN predictions require their accepted node/time indices.")
            base = torch.as_tensor(coupled_trial[time_ids, point_ids], **kw)
            free = torch.as_tensor(coupled_free[point_ids], **kw)
            time_factor = inputs[:, 3:4].expand_as(correction).clone()
            # Incompressible pressure is an elliptic Lagrange multiplier, not
            # a dynamical variable with a prescribed zero initial condition.
            time_factor[:, 10] = 1.
            return (base - offsets) / scales + time_factor * free * correction
        assert operator is not None
        if point_ids is None:
            raise ValueError("Thermal graph PINN predictions require their source mesh node indices.")
        t = start_time + inputs[:, 3:4] * time_scale
        charge_end = config.charge_fraction * config.duration
        initial_h = float(reference[0, 0, 0])
        mass = config.density * torch.as_tensor(operator.volumes[point_ids, None], **kw)
        source = torch.as_tensor(operator.source_weights[point_ids, None], **kw) / mass
        area = torch.as_tensor(operator.boundary_area[point_ids, None], **kw)
        initial_radiation = config.emissivity * STEFAN_BOLTZMANN * area * (
            config.initial_temperature ** 4 - config.ambient_temperature ** 4) / mass
        initial_cp = (config.heat_capacity_solid if config.initial_temperature <= config.melting_temperature else
                      config.heat_capacity_liquid)
        decay = 4 * config.emissivity * STEFAN_BOLTZMANN * area * config.initial_temperature ** 3 / (mass * initial_cp)
        charging_time = torch.clamp(t, min=0., max=charge_end)
        discharge_time = torch.clamp(t - charge_end, min=0.)
        def response(duration: Any) -> Any:
            return torch.where(decay > 1e-14, -torch.expm1(-decay * duration) / decay.clamp(min=1e-14), duration)
        charging_h = (config.heating_power * source - initial_radiation) * response(charging_time)
        discharge_h = (-config.discharge_power * source - initial_radiation) * response(discharge_time)
        base = initial_h + charging_h * torch.exp(-decay * discharge_time) + discharge_h
        return (base - offsets) / scales + inputs[:, 3:4] * correction

    def predict(inputs: Any, point_ids: np.ndarray | None = None, time_ids: np.ndarray | None = None) -> dict[str, Any]:
        fields = scaling.split(forward(inputs, point_ids, time_ids) * scales + offsets)
        temperature, fraction = (enthalpy_to_state_torch(fields["enthalpy"], config) if thermal else
                                 regularized_enthalpy_to_state_torch(fields["enthalpy"], config))
        fields["temperature"], fields["liquid_fraction"] = temperature, fraction
        return fields

    def errors_at(ids: np.ndarray) -> tuple[Any, np.ndarray]:
        ti, pi = np.divmod(ids, nn)
        target = torch.as_tensor((reference[ti, pi] - scaling.offsets) / scaling.scales, **kw)
        validity = field_validity[pi].copy()
        if not thermal:
            validity[ti == 0, 10] = False
        return (forward(input_at(ti, pi), pi, ti) - target).square(), validity

    def sampled_loss(ids: np.ndarray) -> Any:
        errors, validity = errors_at(ids)
        valid_tensor = torch.as_tensor(validity, **kw)
        losses = []
        column = 0
        for field_width in scaling.widths:
            count = int(validity[:, column:column + field_width].sum())
            if count:
                losses.append((errors[:, column:column + field_width] *
                               valid_tensor[:, column:column + field_width]).sum() / count)
            column += field_width
        return torch.stack(losses).mean()

    def validation_loss() -> float:
        totals = np.zeros(len(scaling.names))
        counts = np.zeros(len(scaling.names), dtype=np.int64)
        with torch.no_grad():
            for offset in range(0, len(validation_ids), maximum_batch):
                _cancelled(cancel)
                ids = validation_ids[offset:offset + maximum_batch]
                errors, validity = errors_at(ids)
                valid_tensor = torch.as_tensor(validity, **kw)
                column = 0
                for field_index, field_width in enumerate(scaling.widths):
                    counts[field_index] += int(validity[:, column:column + field_width].sum())
                    totals[field_index] += float((errors[:, column:column + field_width] *
                        valid_tensor[:, column:column + field_width]).sum().item())
                    column += field_width
        valid_fields = counts > 0
        return float(np.mean(totals[valid_fields] / counts[valid_fields]))

    def thermal_physics() -> tuple[Any, Any]:
        assert operator is not None
        # Match the reference's accepted backward-Euler interval exactly. A
        # continuous-time strong residual would conflict with coarse reference
        # snapshots merely because of numerical temporal truncation error.
        ti = int(rng.integers(1, nt))
        physical_time = float((result.times[ti - 1] + result.times[ti]) / 2)
        centers = rng.choice(nn, min(maximum_batch, nn), replace=False)
        support, positions, edge_indices, edge_values = operator.neighborhood(centers)
        pair_points = np.tile(support, 2)
        pair_times = np.repeat(np.asarray([ti - 1, ti]), len(support))
        fields = predict(input_at(pair_times, pair_points), pair_points, pair_times)
        enthalpy = fields["enthalpy"].reshape(2, len(support))
        dt = float(result.times[ti] - result.times[ti - 1])
        rate = (enthalpy[1, positions] - enthalpy[0, positions]) / dt
        temperature = fields["temperature"].reshape(2, len(support))[1]
        if edge_indices.shape[1]:
            row = torch.as_tensor(edge_indices[0], device=device)
            column = torch.as_tensor(edge_indices[1], device=device)
            contribution = torch.as_tensor(edge_values, **kw) * temperature[column]
            conduction = torch.zeros(len(centers), **kw).index_add(0, row, contribution)
        else:
            conduction = torch.zeros(len(centers), **kw)
        volume = torch.as_tensor(operator.volumes[centers], **kw)
        area = torch.as_tensor(operator.boundary_area[centers], **kw)
        weights = torch.as_tensor(operator.source_weights[centers], **kw)
        local_temperature = temperature[positions]
        power = config.heating_power if physical_time <= config.charge_fraction * config.duration else -config.discharge_power
        radiation = config.emissivity * STEFAN_BOLTZMANN * area * (
            local_temperature ** 4 - config.ambient_temperature ** 4)
        scale = config.density * volume * float(scaling.scales[0]) / time_scale
        residual = (config.density * volume * rate + conduction + radiation - power * weights) / scale
        # Uniform source initial state is enforced separately at sampled nodes.
        initial = predict(input_at(np.zeros(len(centers), dtype=np.int64), centers), centers,
                           np.zeros(len(centers), dtype=np.int64))["enthalpy"]
        boundary = ((initial - torch.as_tensor(reference[0, centers, 0], **kw)) / scaling.scales[0]).square().mean()
        return residual.square().mean(), boundary

    def coupled_physics() -> tuple[Any, Any]:
        assert weak is not None
        ti = int(rng.integers(1, nt))
        tids = np.repeat(np.array([ti - 1, ti], dtype=np.int64), nn)
        pids = np.tile(np.arange(nn, dtype=np.int64), 2)
        values = predict(input_at(tids, pids), pids, tids)
        fields = {key: value.reshape((2, nn) + value.shape[1:]) for key, value in values.items()}
        components = weak(config, result, fields, np.asarray([ti - 1, ti], dtype=np.int64))
        if not isinstance(components, Mapping) or not components:
            raise UnsupportedPINNConfiguration("The coupled FE adapter returned no weak residuals.")
        required = {"enthalpy", "momentum", "continuity", "induction", "charge", "gauge"}
        if not required.issubset(components):
            raise UnsupportedPINNConfiguration("The coupled FE weak adapter omits required conservation/gauge equations: "
                                               + ", ".join(sorted(required - set(components))))
        # Adapter residuals are nondimensionalized by the exact worker scales.
        physics_terms, boundary_terms = [], []
        for key, value in components.items():
            if not isinstance(value, torch.Tensor) or not value.requires_grad or not bool(torch.isfinite(value).all()):
                raise ValueError(f"The FE PINN residual {key} is non-finite or detached from the neural fields.")
            (boundary_terms if key.startswith("boundary") else physics_terms).append(value.square().mean())
        return torch.stack(physics_terms).mean(), (torch.stack(boundary_terms).mean() if boundary_terms else
                                                   torch.zeros((), **kw))

    initial_validation = validation_loss()
    for epoch in range(config.pinn_epochs):
        _cancelled(cancel)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        chosen = rng.choice(train_ids, min(maximum_batch, len(train_ids)), replace=False)
        data_loss = sampled_loss(chosen)
        physics_loss, boundary_loss = thermal_physics() if thermal else coupled_physics()
        loss = data_loss + physics_loss + boundary_loss
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("PINN loss became non-finite; previous numerical results are unchanged. Reduce the learning scale or rerun a smaller benchmark.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0, error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        values = {"epoch": float(epoch + 1), "loss": float(loss.item()),
                  "data_loss": float(data_loss.item()), "physics_loss": float(physics_loss.item()),
                  "boundary_loss": float(boundary_loss.item())}
        # Evaluate held-out labels regularly, and retain a finite displayed
        # value between evaluations rather than NaN history placeholders.
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == config.pinn_epochs:
            model.eval()
            current_validation = validation_loss()
        else:
            current_validation = history["validation_loss"][-1]
        values["validation_loss"] = current_validation
        for key, value in values.items():
            history[key].append(value)
        progress(.9 * (epoch + 1) / config.pinn_epochs,
                 f"Epoch {epoch + 1}/{config.pinn_epochs} · loss {values['loss']:.3g} · held-out {current_validation:.3g}")
        if checkpoint_path is not None and ((epoch + 1) % 50 == 0 or epoch + 1 == config.pinn_epochs):
            _checkpoint(checkpoint_path, model, optimizer, epoch + 1, result, scaling, history, metadata)

    _cancelled(cancel)
    model.eval()
    predicted = {name: np.empty_like(np.asarray(result.fields[name]), dtype=np.float64)
                 for name in scaling.names}
    predicted["temperature"] = np.empty((nt, nn), dtype=np.float64)
    predicted["liquid_fraction"] = np.empty((nt, nn), dtype=np.float64)
    # Scalar source fields may carry a trailing singleton; retain the same
    # shape as the source for direct numerical/PINN view switching.
    with torch.no_grad():
        next_progress = 0
        for start in range(0, nt * nn, maximum_batch):
            _cancelled(cancel)
            ids = np.arange(start, min(start + maximum_batch, nt * nn))
            ti, pi = np.divmod(ids, nn)
            fields = predict(input_at(ti, pi), pi, ti)
            for name, values in fields.items():
                array = values.detach().cpu().numpy()
                output = predicted[name]
                if output.ndim == 3 and output.shape[2] == 1 and array.ndim == 1:
                    output[ti, pi, 0] = array
                else:
                    output[ti, pi] = array
            if start >= next_progress or start + len(ids) == nt * nn:
                progress(.9 + .08 * (start + len(ids)) / (nt * nn), "Validating held-out fields and conservation")
                next_progress = start + max(maximum_batch, nt * nn // 20)
    if not all(np.isfinite(value).all() for value in predicted.values()):
        raise FloatingPointError("PINN produced non-finite fields; no result was published.")
    metrics: dict[str, float] = {"initial_validation_loss": initial_validation,
                                "validation_loss": history["validation_loss"][-1],
                                "training_loss": history["loss"][-1],
                                "physics_loss": history["physics_loss"][-1],
                                "boundary_loss": history["boundary_loss"][-1],
                                "epochs": float(config.pinn_epochs)}
    vi, vp = np.divmod(validation_ids, nn)
    zero_reference_fields = []
    normalization = {}
    scale_start = 0
    for name, field_width in zip(scaling.names, scaling.widths):
        normalization[name] = float(scaling.scales[scale_start])
        scale_start += field_width
    normalization["temperature"] = max(float(np.ptp(np.asarray(result.fields["temperature"]))), 1.)
    normalization["liquid_fraction"] = 1.
    metadata["validation_field_scales"] = normalization
    for name, predicted_field in predicted.items():
        if name not in result.fields:
            continue
        expected = np.asarray(result.fields[name])
        actual = predicted_field
        if expected.ndim == 3 and expected.shape[2] == 1 and actual.ndim == 2:
            actual = actual[..., None]
        valid_samples = np.ones(len(vp), dtype=bool) if name in {"magnetic_field", "vector_potential"} else conductor_mask[vp]
        if not thermal and name == "pressure":
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
    metadata["relative_l2_omitted_for_zero_references"] = zero_reference_fields
    if thermal:
        assert operator is not None
        h = _scalar_field(predicted["enthalpy"], "enthalpy")
        # Integrated global energy defect uses accepted-time differences and
        # midpoint boundary loss, separately from continuous training residual.
        total = config.density * np.einsum("tn,n->t", h, operator.volumes)
        energy_defects = []
        residual_square_sum = 0.
        residual_element_count = 0
        checked = set(np.unique(np.linspace(1, nt - 1, min(10, nt - 1), dtype=np.int64)).tolist())
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
                conduction = np.zeros(nn)
                np.add.at(conduction, operator.edge_i, transfer)
                np.add.at(conduction, operator.edge_j, -transfer)
                residual = (config.density * operator.volumes * (h[k] - h[k - 1]) / dt
                            + conduction + boundary_node_w - power * operator.source_weights)
                residual /= config.density * operator.volumes * scaling.scales[0] / time_scale
                residual_square_sum += float(np.sum(residual ** 2))
                residual_element_count += nn
        metrics["energy_defect_max_j"] = max(abs(v) for v in energy_defects)
        scale_j = max(abs(float(total[-1] - total[0])),
                      config.heating_power * config.duration, config.discharge_power * config.duration, 1.0)
        metrics["energy_defect_relative"] = metrics["energy_defect_max_j"] / scale_j
        metrics["enthalpy_residual_rms"] = math.sqrt(residual_square_sum / residual_element_count)
        metadata["conservation_snapshot_indices"] = sorted(checked)
        metrics["phase_fraction_rmse"] = metrics.get("liquid_fraction_rmse", 0.0)
    else:
        assert weak is not None
        # Evaluate final accepted-snapshot fields independently of the random
        # last training batch, including both ends of the physical timeline.
        residual_sums: dict[str, float] = {}
        residual_counts: dict[str, int] = {}
        selected = np.unique(np.linspace(1, nt - 1, min(10, nt - 1), dtype=np.int64))
        with torch.no_grad():
            for position, ti in enumerate(selected):
                _cancelled(cancel)
                fields = {key: torch.as_tensor(value[ti - 1:ti + 1], **kw) for key, value in predicted.items()}
                components = weak(config, result, fields, np.asarray([ti - 1, ti], dtype=np.int64))
                for key, value in components.items():
                    if not isinstance(value, torch.Tensor) or not bool(torch.isfinite(value).all()) or not value.numel():
                        raise ValueError(f"Final FE PINN residual {key} is empty or non-finite.")
                    residual_sums[key] = residual_sums.get(key, 0.) + float(value.square().sum().item())
                    residual_counts[key] = residual_counts.get(key, 0) + value.numel()
                progress(.98 + .015 * (position + 1) / len(selected), "Checking accepted-snapshot conservation and boundary defects")
        for key, square_sum in residual_sums.items():
            metrics[f"{key}_residual_rms"] = math.sqrt(square_sum / residual_counts[key])
        metadata["conservation_snapshot_indices"] = selected.tolist()
    if not all(math.isfinite(value) for value in metrics.values()):
        raise FloatingPointError("PINN validation produced non-finite errors or conservation defects; no result was published.")
    conservation_ok = (metrics.get("energy_defect_relative", 0.) <= .05
                       and metrics["enthalpy_residual_rms"] <= .1 if thermal else
                       all(value <= .1 for key, value in metrics.items() if key.endswith("_residual_rms")))
    fields_ok = all(value <= .1 for key, value in metrics.items() if key.endswith("_normalized_rmse"))
    missing_validation = [name for name in scaling.names if name + "_normalized_rmse" not in metrics]
    metadata["fields_without_heldout_labels"] = missing_validation
    metadata["validation_status"] = ("passed" if metrics["validation_loss"] <= 1e-2
                                      and metrics["physics_loss"] <= 1e-2
                                      and metrics["boundary_loss"] <= 1e-2 and conservation_ok
                                      and fields_ok and not missing_validation else "failed")
    metadata["validation_criteria"] = (
        "held-out normalized MSE ≤ 0.01; each field normalized RMSE ≤ 0.1; training physics/boundary MSE ≤ 0.01; "
        + ("maximum interval energy defect / reference energy ≤ 0.05 and independent enthalpy residual RMS ≤ 0.1" if thermal else
           "each independently checked FE normalized residual RMS ≤ 0.1"))
    metadata["validation_notes"] = ("Agreement with this source run is assessed; this does not establish "
                                    "physical validity or mesh convergence of the source simulation.")
    metadata["software_versions"] = {"torch": torch.__version__, "numpy": np.__version__,
                                     "python": platform.python_version(), "simulator": __version__}
    if checkpoint_path is not None:
        _checkpoint(checkpoint_path, model, optimizer, config.pinn_epochs, result, scaling, history, metadata)
        (checkpoint_path.parent / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    _cancelled(cancel)
    output = PINNResult(run_id=run_id, source_run_id=result.run_id, config_hash=config.config_hash,
                        fields=predicted, history={key: np.asarray(value) for key, value in history.items()},
                        metrics=metrics, metadata=metadata)
    validate_pinn_source(output, result)
    progress(1.0, f"PINN complete · held-out validation {metadata['validation_status']}")
    return output
