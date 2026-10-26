"""Constitutive laws, numerical operator contracts, and neural field scaling.

PyTorch is imported only when a tensor operation or training explicitly needs it.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from importlib import import_module
import threading
from typing import Any, Callable, Mapping

import numpy as np

from ..models import PINNResult, RunResult, SimulationCancelled

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
        # Arrays can be inline or explicitly captured; JSON paths grant no I/O.
        if "operator_path" in raw:
            if "thermal_operator_path" not in result.artifacts:
                raise UnsupportedPINNConfiguration("Thermal PINN operator is not an embedded/captured artifact; recalculate the thermal benchmark.")
            with np.load(BytesIO(result.artifact_bytes("thermal_operator_path")), allow_pickle=False) as artifact:
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
    if "weak_operator_path" not in result.artifacts:
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
        digest.update(result.artifact_bytes("weak_operator_path"))
    return digest.hexdigest()


def _coupled_trial_fields(result: RunResult, scaling: FieldScaling) -> np.ndarray:
    """Known particular fields, never copied from unknown numerical solutions.

    Imposed A comes from the *coil/seed specification* in the FE artifact. B is
    its matching FE curl projection, not a sampled reference solution. The
    initial enthalpy and integral of prescribed volumetric input are also
    known. Learning the induced/circulating correction avoids rediscovering a
    sharply varying external coil field from sparse labels.
    """
    with np.load(BytesIO(result.artifact_bytes("weak_operator_path")), allow_pickle=False) as artifact:
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
