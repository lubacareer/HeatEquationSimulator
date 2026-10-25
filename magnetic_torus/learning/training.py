"""Explicit preparation, loss evaluation, optimization, and prediction stages."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import threading
from typing import Any, Callable, Mapping
from uuid import uuid4

import numpy as np

from ..models import PINNResult, Progress, RunResult, ScenarioConfig
from .fields import (FORMAT_VERSION, STEFAN_BOLTZMANN, FieldScaling, ThermalOperator,
                     UnsupportedPINNConfiguration, _cancelled, _coupled_trial_fields,
                     _make_network, _operator_fingerprint, _scalar_field, _torch,
                     _weak_adapter, enthalpy_to_state_torch,
                     regularized_enthalpy_to_state_torch, validate_pinn_source)
from .validation import validate_prediction


@dataclass(frozen=True)
class TrainingData:
    """Source identity, disjoint labels, and the matching discrete equations."""
    result: RunResult
    thermal: bool
    operator: ThermalOperator | None
    weak: Callable[..., Mapping[str, Any]] | None
    rng: np.random.Generator
    train_ids: np.ndarray
    validation_ids: np.ndarray
    scaling: FieldScaling
    reference: np.ndarray

    @property
    def config(self) -> ScenarioConfig:
        return self.result.config

    @property
    def snapshot_count(self) -> int:
        return self.reference.shape[0]

    @property
    def point_count(self) -> int:
        return self.reference.shape[1]

    @property
    def output_width(self) -> int:
        return self.reference.shape[2]


@dataclass
class NeuralField:
    """Network prediction with the source's coordinates and hard constraints."""
    data: TrainingData
    torch: Any
    device: Any
    model: Any
    maximum_batch: int
    kw: dict[str, Any]
    offsets: Any
    scales: Any
    temporal_frequencies: Any
    length_scale: float
    start_time: float
    time_scale: float
    toroidal_thermal_inputs: bool
    coupled_trial: np.ndarray | None
    coupled_free: np.ndarray

    def input_at(self, time_ids: np.ndarray, point_ids: np.ndarray, *, grad: bool = False) -> Any:
        data = self.data
        result = data.result
        coordinate = np.column_stack((result.points[point_ids] / self.length_scale,
                                       (result.times[time_ids] - self.start_time) / self.time_scale))
        return self.torch.tensor(coordinate, requires_grad=grad, **self.kw)

    def forward(self, inputs: Any, point_ids: np.ndarray | None = None, time_ids: np.ndarray | None = None) -> Any:
        data = self.data
        config = data.config
        operator = data.operator
        torch = self.torch
        network_inputs = inputs
        if self.toroidal_thermal_inputs:
            # The native thermal boundary/source and geometry are invariant
            # around the major circle. Enforce that exact symmetry instead of
            # asking labels to rediscover it in Cartesian coordinates.
            major_r = torch.sqrt(inputs[:, 0] ** 2 + inputs[:, 1] ** 2) * self.length_scale
            horizontal = major_r - config.major_radius
            vertical = inputs[:, 2] * self.length_scale
            tube_r = torch.sqrt(horizontal ** 2 + vertical ** 2 + 1e-24)
            network_inputs = torch.stack((tube_r / config.minor_radius,
                                          horizontal / tube_r, vertical / tube_r, inputs[:, 3]), dim=1)
        elif not data.thermal:
            major_r = torch.sqrt(inputs[:, 0] ** 2 + inputs[:, 1] ** 2 + 1e-24)
            horizontal = major_r * self.length_scale - config.major_radius
            vertical = inputs[:, 2] * self.length_scale
            tube_r = torch.sqrt(horizontal ** 2 + vertical ** 2 + 1e-24)
            # Cartesian coordinates keep all 3D degrees of freedom; geometry
            # features resolve the thin tube without imposing axisymmetry.
            network_inputs = torch.cat((inputs[:, :3], torch.stack((tube_r / config.minor_radius,
                horizontal / tube_r, vertical / tube_r, inputs[:, 1] / major_r,
                inputs[:, 0] / major_r, inputs[:, 3]), dim=1)), dim=1)
            phase = inputs[:, 3:4] * self.temporal_frequencies[None]
            # Accepted adaptive timesteps can resolve rapid changes in the
            # elliptic fields. A time-only Fourier basis avoids asking a
            # low-frequency tanh network to smooth those changes away.
            network_inputs = torch.cat((network_inputs, torch.sin(phase), torch.cos(phase)), dim=1)
        correction = torch.cat([self.model(chunk) for chunk in network_inputs.split(self.maximum_batch)], dim=0)
        if not data.thermal:
            assert self.coupled_trial is not None
            if point_ids is None or time_ids is None:
                raise ValueError("Coupled weak-form PINN predictions require their accepted node/time indices.")
            base = torch.as_tensor(self.coupled_trial[time_ids, point_ids], **self.kw)
            free = torch.as_tensor(self.coupled_free[point_ids], **self.kw)
            time_factor = inputs[:, 3:4].expand_as(correction).clone()
            # Incompressible pressure is an elliptic Lagrange multiplier, not
            # a dynamical variable with a prescribed zero initial condition.
            time_factor[:, 10] = 1.
            return (base - self.offsets) / self.scales + time_factor * free * correction
        assert operator is not None
        if point_ids is None:
            raise ValueError("Thermal graph PINN predictions require their source mesh node indices.")
        t = self.start_time + inputs[:, 3:4] * self.time_scale
        charge_end = config.charge_fraction * config.duration
        initial_h = float(data.reference[0, 0, 0])
        mass = config.density * torch.as_tensor(operator.volumes[point_ids, None], **self.kw)
        source = torch.as_tensor(operator.source_weights[point_ids, None], **self.kw) / mass
        area = torch.as_tensor(operator.boundary_area[point_ids, None], **self.kw)
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
        return (base - self.offsets) / self.scales + inputs[:, 3:4] * correction

    def predict(self, inputs: Any, point_ids: np.ndarray | None = None, time_ids: np.ndarray | None = None) -> dict[str, Any]:
        data = self.data
        config = data.config
        fields = data.scaling.split(self.forward(inputs, point_ids, time_ids) * self.scales + self.offsets)
        temperature, fraction = (enthalpy_to_state_torch(fields["enthalpy"], config) if data.thermal else
                                 regularized_enthalpy_to_state_torch(fields["enthalpy"], config))
        fields["temperature"], fields["liquid_fraction"] = temperature, fraction
        return fields


@dataclass
class TrainingLosses:
    """Label, boundary, and matching thermal/coupled conservation losses."""
    field: NeuralField
    field_validity: np.ndarray
    cancel: threading.Event

    def errors_at(self, ids: np.ndarray) -> tuple[Any, np.ndarray]:
        field = self.field
        data = field.data
        scaling = data.scaling
        ti, pi = np.divmod(ids, data.point_count)
        target = field.torch.as_tensor((data.reference[ti, pi] - scaling.offsets) / scaling.scales, **field.kw)
        validity = self.field_validity[pi].copy()
        if not data.thermal:
            validity[ti == 0, 10] = False
        return (field.forward(field.input_at(ti, pi), pi, ti) - target).square(), validity

    def sampled_loss(self, ids: np.ndarray) -> Any:
        field = self.field
        torch = field.torch
        errors, validity = self.errors_at(ids)
        valid_tensor = torch.as_tensor(validity, **field.kw)
        losses = []
        column = 0
        for field_width in field.data.scaling.widths:
            count = int(validity[:, column:column + field_width].sum())
            if count:
                losses.append((errors[:, column:column + field_width] *
                               valid_tensor[:, column:column + field_width]).sum() / count)
            column += field_width
        return torch.stack(losses).mean()

    def validation_loss(self) -> float:
        field = self.field
        data = field.data
        scaling = data.scaling
        torch = field.torch
        totals = np.zeros(len(scaling.names))
        counts = np.zeros(len(scaling.names), dtype=np.int64)
        with torch.no_grad():
            for offset in range(0, len(data.validation_ids), field.maximum_batch):
                _cancelled(self.cancel)
                ids = data.validation_ids[offset:offset + field.maximum_batch]
                errors, validity = self.errors_at(ids)
                valid_tensor = torch.as_tensor(validity, **field.kw)
                column = 0
                for field_index, field_width in enumerate(scaling.widths):
                    counts[field_index] += int(validity[:, column:column + field_width].sum())
                    totals[field_index] += float((errors[:, column:column + field_width] *
                        valid_tensor[:, column:column + field_width]).sum().item())
                    column += field_width
        valid_fields = counts > 0
        return float(np.mean(totals[valid_fields] / counts[valid_fields]))

    def thermal_physics(self) -> tuple[Any, Any]:
        field = self.field
        data = field.data
        config = data.config
        operator = data.operator
        scaling = data.scaling
        result = data.result
        torch = field.torch
        kw = field.kw
        assert operator is not None
        # Match the reference's accepted backward-Euler interval exactly. A
        # continuous-time strong residual would conflict with coarse reference
        # snapshots merely because of numerical temporal truncation error.
        ti = int(data.rng.integers(1, data.snapshot_count))
        physical_time = float((result.times[ti - 1] + result.times[ti]) / 2)
        centers = data.rng.choice(data.point_count, min(field.maximum_batch, data.point_count), replace=False)
        support, positions, edge_indices, edge_values = operator.neighborhood(centers)
        pair_points = np.tile(support, 2)
        pair_times = np.repeat(np.asarray([ti - 1, ti]), len(support))
        fields = field.predict(field.input_at(pair_times, pair_points), pair_points, pair_times)
        enthalpy = fields["enthalpy"].reshape(2, len(support))
        dt = float(result.times[ti] - result.times[ti - 1])
        rate = (enthalpy[1, positions] - enthalpy[0, positions]) / dt
        temperature = fields["temperature"].reshape(2, len(support))[1]
        if edge_indices.shape[1]:
            row = torch.as_tensor(edge_indices[0], device=field.device)
            column = torch.as_tensor(edge_indices[1], device=field.device)
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
        scale = config.density * volume * float(scaling.scales[0]) / field.time_scale
        residual = (config.density * volume * rate + conduction + radiation - power * weights) / scale
        # Uniform source initial state is enforced separately at sampled nodes.
        initial = field.predict(field.input_at(np.zeros(len(centers), dtype=np.int64), centers), centers,
                           np.zeros(len(centers), dtype=np.int64))["enthalpy"]
        boundary = ((initial - torch.as_tensor(data.reference[0, centers, 0], **kw)) / scaling.scales[0]).square().mean()
        return residual.square().mean(), boundary

    def coupled_physics(self) -> tuple[Any, Any]:
        field = self.field
        data = field.data
        torch = field.torch
        assert data.weak is not None
        ti = int(data.rng.integers(1, data.snapshot_count))
        tids = np.repeat(np.array([ti - 1, ti], dtype=np.int64), data.point_count)
        pids = np.tile(np.arange(data.point_count, dtype=np.int64), 2)
        values = field.predict(field.input_at(tids, pids), pids, tids)
        fields = {key: value.reshape((2, data.point_count) + value.shape[1:]) for key, value in values.items()}
        components = data.weak(data.config, data.result, fields, np.asarray([ti - 1, ti], dtype=np.int64))
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
                                                   torch.zeros((), **field.kw))


@dataclass
class TrainingRun:
    """Optimization state and publication identity for one prepared scenario."""
    field: NeuralField
    losses: TrainingLosses
    optimizer: Any
    scheduler: Any
    run_id: str
    checkpoint_path: Path | None
    metadata: dict[str, Any]
    history: dict[str, list[float]]


def _prepare_data(result: RunResult, cancel: threading.Event) -> TrainingData:
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
    if thermal and not np.allclose(reference[0, :, 0], reference[0, 0, 0], rtol=1e-12, atol=1e-9):
        raise UnsupportedPINNConfiguration("The thermal PINN trial function requires the benchmark's uniform initial enthalpy.")
    return TrainingData(result, thermal, operator, weak, rng, train_ids, validation_ids, scaling, reference)


def _prepare_training(data: TrainingData, progress: Progress, cancel: threading.Event,
                      output_dir: Path | None) -> TrainingRun:
    result, config = data.result, data.config
    thermal = data.thermal
    train_ids, scaling = data.train_ids, data.scaling
    nn, width = data.point_count, data.output_width
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
    field = NeuralField(data, torch, device, model, maximum_batch, kw, offsets, scales,
                        temporal_frequencies, length_scale, start_time, time_scale,
                        toroidal_thermal_inputs, coupled_trial, coupled_free)
    metadata = _training_metadata(data, field, checkpoint_path)
    history: dict[str, list[float]] = {key: [] for key in (
        "epoch", "loss", "data_loss", "physics_loss", "boundary_loss", "validation_loss")}
    progress(0.0, f"Preparing {'thermal' if thermal else 'coupled weak-form'} PINN on {device}")
    return TrainingRun(field, TrainingLosses(field, field_validity, cancel), optimizer,
                       scheduler, run_id, checkpoint_path, metadata, history)


def _training_metadata(data: TrainingData, field: NeuralField, checkpoint_path: Path | None) -> dict[str, Any]:
    config = data.config
    scaling = data.scaling
    result = data.result
    metadata: dict[str, Any] = {
        "format_version": FORMAT_VERSION, "source_run_id": result.run_id,
        "source_config_hash": config.config_hash, "source_snapshot_times_s": result.times.tolist(),
        "source_physics_hash": config.physics_hash, "training_source_run_id": result.run_id,
        "training_config_hash": config.config_hash, "training_epochs": config.pinn_epochs,
        "physics": result.metadata["pinn_physics"]["kind"] if data.thermal else "coupled_fe_weak_operator",
        "turbulence": bool(config.turbulence), "device": str(field.device), "torch_version": field.torch.__version__,
        "seed": config.seed, "training_samples": int(len(data.train_ids)),
        "heldout_samples": int(len(data.validation_ids)), "max_forward_batch": field.maximum_batch,
        "heldout_split": "disjoint seeded point-time labels; physics collocation may use the same locations",
        "normalization_fitted_on": "training labels only",
        "data_loss_aggregation": "mean of per-field valid-component MSE; conductor fields masked; initial pressure is not a dynamical initial condition",
        "excluded_initial_labels": [] if data.thermal else ["pressure"],
        "field_names": list(scaling.names), "field_offsets": scaling.offsets.tolist(),
        "field_scales": scaling.scales.tolist(), "length_scale_m": field.length_scale,
        "input_transform": ("axisymmetric toroidal coordinates and time" if field.toroidal_thermal_inputs else
                            "Cartesian and toroidal geometry features plus time" if not data.thermal else "Cartesian coordinates and time"),
        "architecture": {"widths": [4 if data.thermal else 21, 64, 64, 64, data.output_width], "activation": "tanh", "dtype": "float64"},
        "temporal_features": [] if data.thermal else ["sin/cos(2π·2^k·normalized_time), k=0,…,5"],
        "optimizer": {"name": "Adam", "initial_learning_rate": .001,
                      "schedule": "cosine", "final_learning_rate": 1e-5, "gradient_norm_limit": 10.},
        "hard_constraints": "specified dynamical initial state and FE Dirichlet node masks; pressure is elliptic" if not data.thermal else "specified initial state",
        "particular_solution": ("local prescribed heating/discharge with initial-state linearized radiation; neural correction enforces exact nonlinear radiation and conduction" if data.thermal else
                                "prescribed initial enthalpy, integrated input and imposed coil/seed FE field; neural correction learns coupled response"),
        "time_scale_s": field.time_scale,
        "time_discretization": "source worker's accepted backward-Euler intervals and source schedule",
        "enthalpy_law": "sharp T(E) plateau" if data.thermal else "worker apparent-capacity regularization",
        "source_validation_status": result.metadata.get("validation_status", "unvalidated"),
        "source_backend": result.metadata.get("backend", "unspecified"),
        "source_model": result.metadata.get("model", "unspecified"),
        "source_material_provenance": config.provenance,
        "source_operator_sha256": _operator_fingerprint(data.operator, result),
        "source_closure": result.metadata.get("closure", "not applicable"),
        "source_closure_reference": result.metadata.get("closure_reference", "not applicable"),
        "source_software_versions": {key: str(result.metadata[key]) for key in
            ("dolfinx_version", "petsc_version", "gmsh_version") if key in result.metadata},
        "validation_status": "unvalidated", "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "weak_operator_path": result.metadata.get("weak_operator_path"),
        "thermal_physics_only": data.thermal,
    }
    return metadata


def _train_model(run: TrainingRun, progress: Progress, cancel: threading.Event) -> float:
    field = run.field
    data = field.data
    config = data.config
    torch = field.torch
    initial_validation = run.losses.validation_loss()
    for epoch in range(config.pinn_epochs):
        _cancelled(cancel)
        field.model.train()
        run.optimizer.zero_grad(set_to_none=True)
        chosen = data.rng.choice(data.train_ids, min(field.maximum_batch, len(data.train_ids)), replace=False)
        data_loss = run.losses.sampled_loss(chosen)
        physics_loss, boundary_loss = run.losses.thermal_physics() if data.thermal else run.losses.coupled_physics()
        loss = data_loss + physics_loss + boundary_loss
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("PINN loss became non-finite; previous numerical results are unchanged. Reduce the learning scale or rerun a smaller benchmark.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(field.model.parameters(), 10.0, error_if_nonfinite=True)
        run.optimizer.step()
        run.scheduler.step()
        values = {"epoch": float(epoch + 1), "loss": float(loss.item()),
                  "data_loss": float(data_loss.item()), "physics_loss": float(physics_loss.item()),
                  "boundary_loss": float(boundary_loss.item())}
        # Evaluate held-out labels regularly, and retain a finite displayed
        # value between evaluations rather than NaN history placeholders.
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == config.pinn_epochs:
            field.model.eval()
            current_validation = run.losses.validation_loss()
        else:
            current_validation = run.history["validation_loss"][-1]
        values["validation_loss"] = current_validation
        for key, value in values.items():
            run.history[key].append(value)
        progress(.9 * (epoch + 1) / config.pinn_epochs,
                 f"Epoch {epoch + 1}/{config.pinn_epochs} · loss {values['loss']:.3g} · held-out {current_validation:.3g}")
        if run.checkpoint_path is not None and ((epoch + 1) % 50 == 0 or epoch + 1 == config.pinn_epochs):
            _checkpoint(run.checkpoint_path, field.model, run.optimizer, epoch + 1, data.result, data.scaling, run.history, run.metadata)

    return initial_validation


def _predict_fields(run: TrainingRun, progress: Progress, cancel: threading.Event) -> dict[str, np.ndarray]:
    field = run.field
    data = field.data
    _cancelled(cancel)
    field.model.eval()
    predicted = {name: np.empty_like(np.asarray(data.result.fields[name]), dtype=np.float64)
                 for name in data.scaling.names}
    predicted["temperature"] = np.empty((data.snapshot_count, data.point_count), dtype=np.float64)
    predicted["liquid_fraction"] = np.empty((data.snapshot_count, data.point_count), dtype=np.float64)
    # Scalar source fields may carry a trailing singleton; retain the same
    # shape as the source for direct numerical/PINN view switching.
    with field.torch.no_grad():
        next_progress = 0
        for start in range(0, data.snapshot_count * data.point_count, field.maximum_batch):
            _cancelled(cancel)
            ids = np.arange(start, min(start + field.maximum_batch, data.snapshot_count * data.point_count))
            ti, pi = np.divmod(ids, data.point_count)
            fields = field.predict(field.input_at(ti, pi), pi, ti)
            for name, values in fields.items():
                array = values.detach().cpu().numpy()
                output = predicted[name]
                if output.ndim == 3 and output.shape[2] == 1 and array.ndim == 1:
                    output[ti, pi, 0] = array
                else:
                    output[ti, pi] = array
            if start >= next_progress or start + len(ids) == data.snapshot_count * data.point_count:
                progress(.9 + .08 * (start + len(ids)) / (data.snapshot_count * data.point_count), "Validating held-out fields and conservation")
                next_progress = start + max(field.maximum_batch, data.snapshot_count * data.point_count // 20)
    if not all(np.isfinite(value).all() for value in predicted.values()):
        raise FloatingPointError("PINN produced non-finite fields; no result was published.")
    return predicted


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



def _assemble_result(run: TrainingRun, predicted: dict[str, np.ndarray], metrics: dict[str, float],
                     progress: Progress, cancel: threading.Event) -> PINNResult:
    field = run.field
    data = field.data
    config = data.config
    result = data.result
    if run.checkpoint_path is not None:
        _checkpoint(run.checkpoint_path, field.model, run.optimizer, config.pinn_epochs, result, data.scaling, run.history, run.metadata)
        (run.checkpoint_path.parent / "metadata.json").write_text(json.dumps(run.metadata, indent=2), encoding="utf-8")
    _cancelled(cancel)
    output = PINNResult(run_id=run.run_id, source_run_id=result.run_id, config_hash=config.config_hash,
                        fields=predicted, history={key: np.asarray(value) for key, value in run.history.items()},
                        metrics=metrics, metadata=run.metadata)
    validate_pinn_source(output, result)
    progress(1.0, f"PINN complete · held-out validation {run.metadata['validation_status']}")
    return output


def train_pinn(result: RunResult, progress: Progress, cancel: threading.Event,
               output_dir: Path | None = None) -> PINNResult:
    """Train, validate, and assemble one scenario without mutating its source.

    Each stage has explicit ownership. Failed accuracy gates remain a completed
    result marked failed; cancellation/errors never publish a partial result.
    """
    data = _prepare_data(result, cancel)
    run = _prepare_training(data, progress, cancel, output_dir)
    initial_validation = _train_model(run, progress, cancel)
    predicted = _predict_fields(run, progress, cancel)
    metrics = validate_prediction(run, predicted, initial_validation, progress, cancel)
    return _assemble_result(run, predicted, metrics, progress, cancel)
