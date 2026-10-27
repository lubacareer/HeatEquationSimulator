"""Explicit preparation, loss evaluation, optimization, and prediction stages."""
from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from copy import deepcopy
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

_HOST_TRAINING_CACHE_BYTES = 64 * 1024 ** 2
_FULL_LABEL_BATCH_SIZE = 4096


@dataclass(frozen=True)
class _FixedTrainingCache:
    targets: np.ndarray
    training_rows: np.ndarray
    points: np.ndarray
    times: np.ndarray

    @property
    def nbytes(self) -> int:
        return sum(array.nbytes for array in (self.targets, self.training_rows, self.points, self.times))


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
    cache: _FixedTrainingCache | None = None

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


def _fixed_training_cache(result: RunResult, scaling: FieldScaling, reference: np.ndarray,
                          train_ids: np.ndarray, byte_limit: int,
                          cancel: threading.Event | None = None) -> _FixedTrainingCache | None:
    nt, nn, width = reference.shape
    required = (len(train_ids) * width + nt * nn + nn * 3 + nt) * 8
    if required > byte_limit:
        return None
    targets = np.empty((len(train_ids), width), dtype=np.float64)
    for start in range(0, len(train_ids), _FULL_LABEL_BATCH_SIZE):
        if cancel is not None:
            _cancelled(cancel)
        ids = train_ids[start:start + _FULL_LABEL_BATCH_SIZE]
        ti, pi = np.divmod(ids, nn)
        np.divide(reference[ti, pi] - scaling.offsets, scaling.scales,
                  out=targets[start:start + len(ids)])
    if cancel is not None:
        _cancelled(cancel)
    rows = np.full(nt * nn, -1, dtype=np.int64)
    rows[train_ids] = np.arange(len(train_ids), dtype=np.int64)
    points = np.asarray(result.points / (result.config.major_radius + result.config.minor_radius), dtype=np.float64)
    times = np.asarray((result.times - float(result.times[0])) / float(result.times[-1] - result.times[0]), dtype=np.float64)
    for array in (targets, rows, points, times):
        array.setflags(write=False)
    return _FixedTrainingCache(targets, rows, points, times)


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
    full_label_batch_size: int = _FULL_LABEL_BATCH_SIZE

    def input_at(self, time_ids: np.ndarray, point_ids: np.ndarray, *, grad: bool = False) -> Any:
        data = self.data
        result = data.result
        if data.cache is None:
            coordinate = np.column_stack((result.points[point_ids] / self.length_scale,
                                           (result.times[time_ids] - self.start_time) / self.time_scale))
        else:
            coordinate = np.column_stack((data.cache.points[point_ids], data.cache.times[time_ids]))
        return self.torch.tensor(coordinate, requires_grad=grad, **self.kw)

    def forward(self, inputs: Any, point_ids: np.ndarray | None = None, time_ids: np.ndarray | None = None,
                *, forward_batch_size: int | None = None) -> Any:
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
        batch_size = self.maximum_batch if forward_batch_size is None else forward_batch_size
        if not 1 <= batch_size <= _FULL_LABEL_BATCH_SIZE:
            raise ValueError("Neural forward batches must contain between 1 and 4096 samples.")
        correction = torch.cat([self.model(chunk) for chunk in network_inputs.split(batch_size)], dim=0)
        if not data.thermal:
            assert self.coupled_trial is not None
            if point_ids is None or time_ids is None:
                raise ValueError("Coupled weak-form PINN predictions require their accepted node/time indices.")
            base = torch.as_tensor(self.coupled_trial[time_ids, point_ids], **self.kw)
            free = torch.as_tensor(self.coupled_free[point_ids], **self.kw)
            first_step = float(data.result.times[1] - self.start_time)
            time_factor = torch.clamp(inputs[:, 3:4] * self.time_scale / first_step,
                                      min=0., max=1.).expand_as(correction).clone()
            # Incompressible pressure is an elliptic Lagrange multiplier, not
            # a dynamical variable with a prescribed zero initial condition.
            time_factor[:, 10] = 1.
            return (base - self.offsets) / self.scales + time_factor * free * correction
        assert operator is not None
        if point_ids is None:
            raise ValueError("Thermal graph PINN predictions require their source mesh node indices.")
        t = self.start_time + inputs[:, 3:4] * self.time_scale
        charge_end = config.charge_fraction * config.duration
        delta = config.initial_temperature - config.melting_temperature
        initial_h = (config.heat_capacity_solid * delta if delta <= 0 else
                     config.latent_heat + config.heat_capacity_liquid * delta)
        # A local, conduction-free source trial overheats surface nodes and
        # creates enormous diffusion gradients. The bulk energy particular
        # solution uses only the prescribed source and geometric measures.
        mass = config.density * float(operator.volumes.sum())
        source = 1. / mass
        area = float(operator.boundary_area.sum())
        initial_radiation = config.emissivity * STEFAN_BOLTZMANN * area * (
            config.initial_temperature ** 4 - config.ambient_temperature ** 4) / mass
        initial_cp = (config.heat_capacity_solid if config.initial_temperature <= config.melting_temperature else
                      config.heat_capacity_liquid)
        decay = torch.as_tensor(4 * config.emissivity * STEFAN_BOLTZMANN * area *
                                config.initial_temperature ** 3 / (mass * initial_cp), **self.kw)
        charging_time = torch.clamp(t - self.start_time, min=0., max=max(charge_end - self.start_time, 0.))
        discharge_time = torch.clamp(t - max(charge_end, self.start_time), min=0.)
        def response(duration: Any) -> Any:
            return torch.where(decay > 1e-14, -torch.expm1(-decay * duration) / decay.clamp(min=1e-14), duration)
        charging_h = (config.heating_power * source - initial_radiation) * response(charging_time)
        discharge_h = (-config.discharge_power * source - initial_radiation) * response(discharge_time)
        base = initial_h + charging_h * torch.exp(-decay * discharge_time) + discharge_h
        return (base - self.offsets) / self.scales + inputs[:, 3:4] * correction

    def predict(self, inputs: Any, point_ids: np.ndarray | None = None, time_ids: np.ndarray | None = None,
                *, forward_batch_size: int | None = None) -> dict[str, Any]:
        data = self.data
        config = data.config
        fields = data.scaling.split(self.forward(inputs, point_ids, time_ids,
            forward_batch_size=forward_batch_size) * self.scales + self.offsets)
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
    training_pools: tuple[np.ndarray, ...] = dataclass_field(init=False, repr=False)
    training_counts: np.ndarray = dataclass_field(init=False, repr=False)

    def __post_init__(self):
        ids = self.field.data.train_ids
        validity = self.validity_at(ids, training=True)
        pools, counts = [], []
        column = 0
        for width in self.field.data.scaling.widths:
            valid = validity[:, column:column + width]
            pools.append(ids[valid.any(axis=1)])
            counts.append(int(valid.sum()))
            column += width
        self.training_pools = tuple(pools)
        self.training_counts = np.asarray(counts, dtype=np.int64)

    def validity_at(self, ids: np.ndarray, *, training: bool = False) -> np.ndarray:
        ti, pi = np.divmod(ids, self.field.data.point_count)
        validity = self.field_validity[pi].copy()
        if not self.field.data.thermal:
            validity[ti == 0, 10] = False
        if training:
            validity &= self.field.coupled_free[pi] != 0
            validity[ti == 0] = False
        return validity

    def errors_at(self, ids: np.ndarray, *, training: bool = False,
                  forward_batch_size: int | None = None) -> tuple[Any, np.ndarray]:
        field = self.field
        data = field.data
        scaling = data.scaling
        ti, pi = np.divmod(ids, data.point_count)
        if training and data.cache is not None:
            rows = data.cache.training_rows[ids]
            if np.any(rows < 0):
                raise ValueError("A held-out label cannot enter the training-target cache.")
            normalized = data.cache.targets[rows]
        else:
            normalized = (data.reference[ti, pi] - scaling.offsets) / scaling.scales
        target = field.torch.as_tensor(normalized, **field.kw)
        validity = self.validity_at(ids, training=training)
        return (field.forward(field.input_at(ti, pi), pi, ti,
                              forward_batch_size=forward_batch_size) - target).square(), validity

    def _label_loss(self, errors: Any, validity: np.ndarray, counts: np.ndarray | None = None) -> Any:
        field = self.field
        torch = field.torch
        valid_tensor = torch.as_tensor(validity, **field.kw)
        losses = []
        column = 0
        for index, field_width in enumerate(field.data.scaling.widths):
            count = int(validity[:, column:column + field_width].sum()) if counts is None else int(counts[index])
            if count:
                losses.append((errors[:, column:column + field_width] *
                               valid_tensor[:, column:column + field_width]).sum() / count)
            column += field_width
        return torch.stack(losses).mean() if losses else errors.sum() * 0.

    def sampled_loss(self, ids: np.ndarray) -> Any:
        return self._label_loss(*self.errors_at(ids, training=True))

    def sampled_training_loss(self) -> Any:
        """Sample each unknown field's labels without diluting rare free DOFs."""
        field, data = self.field, self.field.data
        eligible = [(index, pool) for index, pool in enumerate(self.training_pools) if len(pool)]
        if not eligible:
            return next(parameter for parameter in field.model.parameters() if parameter.requires_grad).sum() * 0.
        order = data.rng.permutation(len(eligible))
        quota, remainder = divmod(field.maximum_batch, len(eligible))
        samples, selections = [], []
        for position, pool_index in enumerate(order):
            index, pool = eligible[pool_index]
            count = min(len(pool), quota + (position < remainder))
            if count:
                samples.append(data.rng.choice(pool, count, replace=False))
                selections.append(np.full(count, index, dtype=np.int64))
        ids, selected_fields = np.concatenate(samples), np.concatenate(selections)
        errors, validity = self.errors_at(ids, training=True)
        column = 0
        for index, width in enumerate(data.scaling.widths):
            validity[:, column:column + width] &= (selected_fields == index)[:, None]
            column += width
        return self._label_loss(errors, validity)

    def stream_training_loss(self, *, backward: bool) -> Any:
        """Accumulate a full training-label mean while releasing each batch graph."""
        field, data = self.field, self.field.data
        total = field.torch.zeros((), **field.kw)
        for offset in range(0, len(data.train_ids), field.full_label_batch_size):
            _cancelled(self.cancel)
            ids = data.train_ids[offset:offset + field.full_label_batch_size]
            loss = self._label_loss(*self.errors_at(ids, training=True,
                forward_batch_size=field.full_label_batch_size), counts=self.training_counts)
            if backward:
                loss.backward()
            total += loss.detach()
            _cancelled(self.cancel)
        if not bool(field.torch.isfinite(total)):
            raise FloatingPointError("PINN training-label objective became non-finite.")
        return total

    def validation_loss(self) -> float:
        field = self.field
        data = field.data
        scaling = data.scaling
        torch = field.torch
        totals = torch.zeros(len(scaling.names), **field.kw)
        counts = np.zeros(len(scaling.names), dtype=np.int64)
        with torch.no_grad():
            for offset in range(0, len(data.validation_ids), field.maximum_batch):
                _cancelled(self.cancel)
                ids = data.validation_ids[offset:offset + field.maximum_batch]
                errors, validity = self.errors_at(ids)
                valid_tensor = torch.as_tensor(validity, **field.kw)
                column = 0
                sums = []
                for field_index, field_width in enumerate(scaling.widths):
                    counts[field_index] += int(validity[:, column:column + field_width].sum())
                    sums.append((errors[:, column:column + field_width] *
                                 valid_tensor[:, column:column + field_width]).sum())
                    column += field_width
                totals += torch.stack(sums)
        host_totals = totals.cpu().numpy()
        valid_fields = counts > 0
        return float(np.mean(host_totals[valid_fields] / counts[valid_fields]))

    def thermal_physics(self, ti: int | None = None, centers: np.ndarray | None = None) -> tuple[Any, Any]:
        data = self.field.data
        ti = int(data.rng.integers(1, data.snapshot_count)) if ti is None else ti
        centers = (data.rng.choice(data.point_count, min(self.field.maximum_batch, data.point_count), replace=False)
                   if centers is None else centers)
        terms = self.thermal_components(ti, centers)
        boundary = terms["initial_error"].square().mean()
        return terms["residual"].square().mean(), boundary

    def thermal_components(self, ti: int, centers: np.ndarray) -> dict[str, Any]:
        """Return local accepted-interval residuals at unique source nodes.

        The residual is the same conservative backward-Euler equation used by
        the reference. Research samplers may inspect detached values without
        reading numerical labels or replacing this discrete operator.
        """
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
        physical_time = float((result.times[ti - 1] + result.times[ti]) / 2)
        centers = np.asarray(centers, dtype=np.int64)
        if (centers.ndim != 1 or not len(centers) or len(np.unique(centers)) != len(centers)
                or np.any(centers < 0) or np.any(centers >= data.point_count)):
            raise ValueError("Thermal residual centers must be nonempty unique source-node indices.")
        support, positions, edge_indices, edge_values = operator.neighborhood(centers)
        pair_points = np.tile(support, 2)
        pair_times = np.repeat(np.asarray([ti - 1, ti]), len(support))
        fields = field.predict(field.input_at(pair_times, pair_points), pair_points, pair_times,
                               forward_batch_size=_FULL_LABEL_BATCH_SIZE)
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
                           np.zeros(len(centers), dtype=np.int64),
                           forward_batch_size=_FULL_LABEL_BATCH_SIZE)["enthalpy"]
        initial_error = (initial - torch.as_tensor(data.reference[0, centers, 0], **kw)) / scaling.scales[0]
        fraction = fields["liquid_fraction"].reshape(2, len(support))[1, positions]
        return {"residual": residual, "initial_error": initial_error, "liquid_fraction": fraction}

    def _coupled_components_for_fields(self, ti: int, fields: Mapping[str, Any]) -> Mapping[str, Any]:
        field = self.field
        data = field.data
        torch = field.torch
        assert data.weak is not None
        components = data.weak(data.config, data.result, fields, np.asarray([ti - 1, ti], dtype=np.int64))
        if not isinstance(components, Mapping) or not components:
            raise UnsupportedPINNConfiguration("The coupled FE adapter returned no weak residuals.")
        required = {"enthalpy", "momentum", "continuity", "induction", "charge", "gauge"}
        if not required.issubset(components):
            raise UnsupportedPINNConfiguration("The coupled FE weak adapter omits required conservation/gauge equations: "
                                               + ", ".join(sorted(required - set(components))))
        for key, value in components.items():
            if not isinstance(value, torch.Tensor) or not value.requires_grad or not bool(torch.isfinite(value).all()):
                raise ValueError(f"The FE PINN residual {key} is non-finite or detached from the neural fields.")
        return components

    def coupled_components(self, ti: int) -> Mapping[str, Any]:
        field = self.field
        data = field.data
        tids = np.repeat(np.array([ti - 1, ti], dtype=np.int64), data.point_count)
        pids = np.tile(np.arange(data.point_count, dtype=np.int64), 2)
        values = field.predict(field.input_at(tids, pids), pids, tids,
                               forward_batch_size=_FULL_LABEL_BATCH_SIZE)
        fields = {key: value.reshape((2, data.point_count) + value.shape[1:]) for key, value in values.items()}
        return self._coupled_components_for_fields(ti, fields)

    def _coupled_component_losses(self, components: Mapping[str, Any]) -> tuple[Any, Any]:
        torch = self.field.torch
        physics_terms, boundary_terms = [], []
        for key, value in components.items():
            (boundary_terms if key.startswith("boundary") else physics_terms).append(value.square().mean())
        return torch.stack(physics_terms).mean(), (torch.stack(boundary_terms).mean() if boundary_terms else
                                                   torch.zeros((), **self.field.kw))

    def coupled_physics(self, ti: int | None = None) -> tuple[Any, Any]:
        data = self.field.data
        ti = int(data.rng.integers(1, data.snapshot_count)) if ti is None else ti
        # Adapter residuals are nondimensionalized by the exact worker scales.
        return self._coupled_component_losses(self.coupled_components(ti))

    def coupled_physics_backprop(self, ti: int, *, weight: float = 1., batch_size: int = 1024,
                                 offload_saved_tensors: bool = False) -> tuple[Any, Any]:
        """Backpropagate the unchanged global FE loss through bounded neural chunks.

        The FE assembly and its reduction still cover the entire snapshot pair.
        Independent leaf fields first yield the global loss cotangents. Recomputing
        all neural outputs per chunk then applies their chain rule, including the
        enthalpy-derived temperature and fraction and unused field cotangents.
        """
        from contextlib import nullcontext

        field, data, torch = self.field, self.field.data, self.field.torch
        if data.thermal or data.weak is None:
            raise ValueError("Global FE cotangents require a coupled weak operator.")
        if not 1 <= ti < data.snapshot_count:
            raise ValueError("Global FE cotangents require a consecutive accepted snapshot pair.")
        if not 1 <= batch_size <= _FULL_LABEL_BATCH_SIZE or not math.isfinite(weight):
            raise ValueError("Invalid FE cotangent batch size or loss weight.")
        _cancelled(self.cancel)
        tids = np.repeat(np.array([ti - 1, ti], dtype=np.int64), data.point_count)
        pids = np.tile(np.arange(data.point_count, dtype=np.int64), 2)
        with torch.no_grad():
            values = field.predict(field.input_at(tids, pids), pids, tids,
                                   forward_batch_size=batch_size)
        leaves = {name: value.reshape((2, data.point_count) + value.shape[1:]).detach().requires_grad_(True)
                  for name, value in values.items()}
        del values
        context = (torch.autograd.graph.save_on_cpu(pin_memory=False)
                   if offload_saved_tensors and field.device.type == "cuda" else nullcontext())
        with context:
            components = self._coupled_components_for_fields(ti, leaves)
            physics, boundary = self._coupled_component_losses(components)
            objective = weight * (physics + boundary)
            if not bool(torch.isfinite(objective)):
                raise FloatingPointError("PINN physics objective became non-finite.")
            cotangents = torch.autograd.grad(objective, tuple(leaves.values()), allow_unused=True)
        physics_value, boundary_value = physics.detach(), boundary.detach()
        cotangents = {name: None if gradient is None else gradient.detach().reshape((len(tids),) + gradient.shape[2:])
                      for name, gradient in zip(leaves, cotangents)}
        del components, objective, physics, boundary, leaves
        _cancelled(self.cancel)
        for start in range(0, len(tids), batch_size):
            _cancelled(self.cancel)
            stop = min(start + batch_size, len(tids))
            chunk_tids, chunk_pids = tids[start:stop], pids[start:stop]
            predicted = field.predict(field.input_at(chunk_tids, chunk_pids), chunk_pids, chunk_tids,
                                      forward_batch_size=batch_size)
            outputs, gradients = [], []
            for name, value in predicted.items():
                gradient = cotangents[name]
                if gradient is not None and value.requires_grad:
                    outputs.append(value)
                    gradients.append(gradient[start:stop])
            if outputs:
                torch.autograd.backward(outputs, gradients)
            _cancelled(self.cancel)
        return physics_value, boundary_value


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


def _optimizer_steps(epochs: int) -> tuple[int, int]:
    adam = epochs if epochs < 50 else 4 * epochs // 5
    return adam, epochs - adam


def _physics_weight(epoch: int, epochs: int, *, thermal: bool) -> float:
    """Fixed native data warmup, followed by a fixed residual ramp."""
    adam, _ = _optimizer_steps(epochs)
    warmup = min(100, adam // 8) if thermal and epochs >= 50 else 0
    if epoch < warmup:
        return 0.
    if warmup and epoch < 2 * warmup:
        return (epoch - warmup + 1) / warmup
    return 1.


def _audit_intervals(data: TrainingData) -> np.ndarray:
    indices = np.unique(np.linspace(1, data.snapshot_count - 1,
        min(10, data.snapshot_count - 1), dtype=np.int64))
    if data.thermal:
        transition = data.config.charge_fraction * data.config.duration
        nearest = int(np.searchsorted(data.result.times, transition))
        indices = np.union1d(indices, [index for index in (nearest - 1, nearest, nearest + 1)
                                     if 1 <= index < data.snapshot_count])
    return indices.astype(np.int64)


def _training_diagnostics(run: TrainingRun, indices: np.ndarray) -> dict[str, dict[str, float]]:
    """Training labels and equation-only panels; held-out labels are never read."""
    field, losses = run.field, run.losses
    data, torch = field.data, field.torch
    sums = torch.zeros(len(data.scaling.names), **field.kw)
    with torch.no_grad():
        for offset in range(0, len(data.train_ids), field.full_label_batch_size):
            _cancelled(losses.cancel)
            errors, valid = losses.errors_at(data.train_ids[offset:offset + field.full_label_batch_size],
                training=True, forward_batch_size=field.full_label_batch_size)
            column = 0
            for index, width in enumerate(data.scaling.widths):
                sums[index] += (errors[:, column:column + width] *
                    torch.as_tensor(valid[:, column:column + width], **field.kw)).sum()
                column += width
    host = sums.cpu().numpy()
    fields = {name: math.sqrt(float(total) / int(count))
              for name, total, count in zip(data.scaling.names, host, losses.training_counts) if count}
    residual_sums, residual_counts = {}, {}
    for ti in indices:
        _cancelled(losses.cancel)
        if data.thermal:
            for offset in range(0, data.point_count, field.maximum_batch):
                centers = np.arange(offset, min(offset + field.maximum_batch, data.point_count))
                value, _ = losses.thermal_physics(int(ti), centers)
                residual_sums["enthalpy"] = residual_sums.get("enthalpy", 0.) + float(value.detach()) * len(centers)
                residual_counts["enthalpy"] = residual_counts.get("enthalpy", 0) + len(centers)
        else:
            for key, value in losses.coupled_components(int(ti)).items():
                residual_sums[key] = residual_sums.get(key, 0.) + float(value.detach().square().sum())
                residual_counts[key] = residual_counts.get(key, 0) + value.numel()
    residuals = {key: math.sqrt(value / residual_counts[key]) for key, value in residual_sums.items()}
    if not all(math.isfinite(value) for value in (*fields.values(), *residuals.values())):
        raise FloatingPointError("PINN training activation diagnostics became non-finite.")
    return {"fields": fields, "residuals": residuals}


def _correction_groups(data: TrainingData, diagnostics: Mapping[str, Mapping[str, float]]) -> tuple[str, ...]:
    fields, residuals = diagnostics["fields"], diagnostics["residuals"]
    def above(field_names: tuple[str, ...], equations: tuple[str, ...]) -> bool:
        return any(fields.get(name, 0.) > .05 for name in field_names) or any(
            residuals.get(name, 0.) > .05 for name in equations)
    if data.thermal:
        return ("thermal",) if above(("enthalpy",), ("enthalpy",)) else ()
    groups = []
    if above(("pressure", "electric_potential"), ("momentum", "charge")):
        groups.append("elliptic")
    if above(("enthalpy", "velocity"), ("continuity", "enthalpy")):
        groups.append("thermofluid")
    return tuple(groups)


def _validate_correction_memory(data: TrainingData, groups: tuple[str, ...]) -> None:
    if 2 * data.point_count * data.output_width * (1 + len(groups)) > 2_000_000:
        raise UnsupportedPINNConfiguration("The active correction architecture exceeds the bounded FE-pair training memory budget; use a smaller mesh.")


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
    cache = _fixed_training_cache(result, scaling, reference, train_ids, _HOST_TRAINING_CACHE_BYTES, cancel)
    return TrainingData(result, thermal, operator, weak, rng, train_ids, validation_ids, scaling, reference, cache)


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
    adam_steps, _ = _optimizer_steps(config.pinn_epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, adam_steps), eta_min=1e-5)
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
    losses = TrainingLosses(field, field_validity, cancel)
    metadata["training_unknown_components"] = dict(zip(scaling.names, losses.training_counts.tolist()))
    history: dict[str, list[float]] = {key: [] for key in (
        "epoch", "loss", "data_loss", "physics_loss", "boundary_loss", "validation_loss")}
    progress(0.0, f"Preparing {'thermal' if thermal else 'coupled weak-form'} PINN on {device}")
    return TrainingRun(field, losses, optimizer,
                       scheduler, run_id, checkpoint_path, metadata, history)


def _training_metadata(data: TrainingData, field: NeuralField, checkpoint_path: Path | None) -> dict[str, Any]:
    config = data.config
    scaling = data.scaling
    result = data.result
    adam_steps, lbfgs_steps = _optimizer_steps(config.pinn_epochs)
    metadata: dict[str, Any] = {
        "format_version": FORMAT_VERSION, "source_run_id": result.run_id,
        "parameterization_version": 2, "training_method_version": 5,
        "source_config_hash": config.config_hash, "source_snapshot_times_s": result.times.tolist(),
        "source_physics_hash": config.physics_hash, "training_source_run_id": result.run_id,
        "training_config_hash": config.config_hash, "training_epochs": config.pinn_epochs,
        "physics": result.metadata["pinn_physics"]["kind"] if data.thermal else "coupled_fe_weak_operator",
        "turbulence": bool(config.turbulence), "device": str(field.device), "torch_version": str(field.torch.__version__),
        "seed": config.seed, "training_samples": int(len(data.train_ids)),
        "heldout_samples": int(len(data.validation_ids)),
        "max_forward_batch": max(field.maximum_batch, field.full_label_batch_size),
        "stochastic_label_batch_size": field.maximum_batch,
        "full_label_batch_size": field.full_label_batch_size,
        "coupled_refinement_gradient": "full graph",
        "host_training_cache_bytes": data.cache.nbytes if data.cache is not None else 0,
        "host_training_cache_limit_bytes": _HOST_TRAINING_CACHE_BYTES,
        "host_training_cache_scope": "training targets and fixed coordinates only; streaming fallback above limit",
        "heldout_split": "disjoint seeded point-time labels; physics collocation may use the same locations",
        "normalization_fitted_on": "training labels only",
        "data_loss_aggregation": "mean of per-field free-DOF MSE; stratified training labels; prescribed initial/Dirichlet labels excluded from training only",
        "heldout_loss_aggregation": "unchanged mean of per-field valid-component MSE; conductor fields masked; initial pressure excluded",
        "excluded_initial_labels": [] if data.thermal else ["pressure"],
        "field_names": list(scaling.names), "field_offsets": scaling.offsets.tolist(),
        "field_scales": scaling.scales.tolist(), "length_scale_m": field.length_scale,
        "input_transform": ("axisymmetric toroidal coordinates and time" if field.toroidal_thermal_inputs else
                            "Cartesian and toroidal geometry features plus time" if not data.thermal else "Cartesian coordinates and time"),
        "architecture": {"widths": [4 if data.thermal else 21, 64, 64, 64, data.output_width], "activation": "tanh", "dtype": "float64"},
        "temporal_features": [] if data.thermal else ["sin/cos(2π·2^k·normalized_time), k=0,…,5"],
        "optimizer": {"name": "Adam + L-BFGS" if lbfgs_steps else "Adam",
                      "adam": {"steps": adam_steps, "initial_learning_rate": .001,
                               "schedule": "cosine", "final_learning_rate": 1e-5, "gradient_norm_limit": 10.},
                      "lbfgs": {"steps": lbfgs_steps, "max_iter": 1, "max_eval": 8,
                                "history_size": 20, "line_search": "strong_wolfe",
                                "closure_evaluation_cap": 8,
                                "budget_policy": "full rollback, then recover an evaluated Armijo-decreasing training trial; restart curvature/LR; rejected-step backoff has a 1e-3 floor"}},
        "optimizer_steps_completed": {"adam": 0, "lbfgs": 0},
        "requested_training_steps": config.pinn_epochs, "actual_training_steps": 0,
        "additional_optimizer_steps_completed": {"adam": 0, "lbfgs": 0},
        "conditioning_policy": {"activation_target": .05, "activation_source": "training free-DOF normalized RMSE and equation-only raw residual RMS",
            "additional_adam_limit": min(2000, 2 * config.pinn_epochs),
            "additional_lbfgs_limit": min(50, config.pinn_epochs // 20),
            "native_warmup_steps": min(100, adam_steps // 8) if data.thermal and config.pinn_epochs >= 50 else 0,
            "native_ramp_steps": min(100, adam_steps // 8) if data.thermal and config.pinn_epochs >= 50 else 0},
        "training_stages": [],
        "refinement_budget_limited_steps": 0,
        "refinement_step_outcomes": {"wolfe": 0, "armijo_recovered": 0, "rolled_back": 0,
                                     "restarted_nochange": 0, "converged": 0},
        "refinement_accepted_updates": 0, "refinement_nochange_steps": 0,
        "refinement_restarts": 0,
        "function_evaluations": {"adam": 0, "lbfgs_closures": 0,
                                 "refinement_objective": 0, "final_objective": 0},
        "hard_constraints": "specified dynamical initial state and FE Dirichlet node masks; pressure is elliptic" if not data.thermal else "specified initial state",
        "correction_time_envelope": "normalized physical time" if data.thermal else "accepted-first-step ramp; pressure remains elliptic",
        "particular_solution": ("uniform bulk prescribed heating/discharge with total-mass/area initial-state linearized radiation; neural correction enforces exact nonlinear radiation and conduction" if data.thermal else
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


def _deterministic_objective(run: TrainingRun, indices: np.ndarray, centers: np.ndarray,
                             *, backward: bool, coupled_vjp: bool | None = None,
                             offload_saved_tensors: bool = False,
                             vjp_batch_size: int = 1024) -> dict[str, Any]:
    """A fixed training objective, streamed without retaining all batch graphs."""
    losses, field = run.losses, run.field
    data_loss = losses.stream_training_loss(backward=backward)
    physics_loss = field.torch.zeros((), **field.kw)
    boundary_loss = field.torch.zeros((), **field.kw)
    for index in indices:
        _cancelled(losses.cancel)
        # The global FE VJP is available for bounded probes. Measurements on
        # the reference GPU selected the faster full graph for the normal path.
        use_vjp = backward and not field.data.thermal and bool(coupled_vjp)
        if use_vjp:
            physics, boundary = losses.coupled_physics_backprop(
                int(index), weight=1. / len(indices), batch_size=vjp_batch_size,
                offload_saved_tensors=offload_saved_tensors)
        else:
            physics, boundary = (losses.thermal_physics(int(index), centers) if field.data.thermal
                                 else losses.coupled_physics(int(index)))
        loss = (physics + boundary) / len(indices)
        if not bool(field.torch.isfinite(loss)):
            raise FloatingPointError("PINN physics objective became non-finite.")
        if backward and not use_vjp:
            loss.backward()
        physics_loss += physics.detach() / len(indices)
        boundary_loss += boundary.detach() / len(indices)
    return {"data_loss": data_loss, "physics_loss": physics_loss,
            "boundary_loss": boundary_loss, "loss": data_loss + physics_loss + boundary_loss}


class _RefinementEvaluationBudget(RuntimeError):
    """An attempted line search exhausted its actual closure-call allowance."""


def _bounded_lbfgs_step(optimizer: Any, parameters: tuple[Any, ...], closure: Callable[[], Any],
                       cancel: threading.Event, limit: int = 8) -> tuple[int, bool]:
    """Bound actual closure calls and roll back a discarded line-search trial."""
    torch = _torch()
    weights = [parameter.detach().clone() for parameter in parameters]
    gradients = [None if parameter.grad is None else parameter.grad.detach().clone()
                 for parameter in parameters]
    state = deepcopy(optimizer.state_dict())
    evaluations = 0

    def counted_closure():
        nonlocal evaluations
        _cancelled(cancel)
        if evaluations >= limit:
            raise _RefinementEvaluationBudget()
        evaluations += 1
        return closure()

    def restore():
        with torch.no_grad():
            for parameter, weight, gradient in zip(parameters, weights, gradients):
                parameter.copy_(weight)
                parameter.grad = gradient
        optimizer.load_state_dict(state)

    try:
        optimizer.step(counted_closure)
    except _RefinementEvaluationBudget:
        restore()
        return evaluations, True
    except BaseException as error:
        # PyTorch may still have trial weights installed when a closure fails.
        try:
            restore()
        except BaseException as rollback_error:
            # Retain the original failure when CUDA is too exhausted to restore.
            # The caller must discard this optimizer/model and may not publish it.
            if hasattr(error, "add_note"):
                error.add_note(f"L-BFGS rollback also failed: {type(rollback_error).__name__}: {rollback_error}")
        raise
    return evaluations, False


@dataclass(frozen=True)
class _RefinementStep:
    evaluations: int
    limited: bool
    outcome: str
    gradient_max: float
    delta_inf: float
    initial_loss: float


def _recovering_lbfgs_step(optimizer: Any, parameters: tuple[Any, ...], closure: Callable[[], Any],
                          cancel: threading.Event) -> _RefinementStep:
    """Recover useful training descent without spending another closure call."""
    torch = _torch()
    base = tuple(parameter.detach().clone() for parameter in parameters)
    initial_loss, initial_gradients, best = None, None, None

    def observed_closure():
        nonlocal initial_loss, initial_gradients, best
        loss = closure()
        value = float(loss.detach())
        gradients = tuple(torch.zeros_like(parameter) if parameter.grad is None
                          else parameter.grad.detach().clone() for parameter in parameters)
        if (not math.isfinite(value)
                or any(not bool(torch.isfinite(parameter).all()) for parameter in parameters)
                or any(not bool(torch.isfinite(gradient).all()) for gradient in gradients)):
            raise FloatingPointError("PINN refinement objective or gradient became non-finite.")
        if initial_loss is None:
            initial_loss, initial_gradients = value, gradients
        else:
            directional = float(sum((gradient * (parameter.detach() - weight)).sum()
                                    for gradient, parameter, weight in zip(initial_gradients, parameters, base)))
            if (directional < 0 and value < initial_loss
                    and value <= initial_loss + 1e-4 * directional
                    and (best is None or value < best[0])):
                # PyTorch restores base weights after this closure returns.
                best = (value, tuple(parameter.detach().clone() for parameter in parameters), gradients)
        return loss

    evaluations, limited = _bounded_lbfgs_step(optimizer, parameters, observed_closure, cancel)
    if initial_loss is None or initial_gradients is None:
        raise RuntimeError("Refinement did not evaluate its training objective.")
    gradient_max = max(float(gradient.abs().max()) for gradient in initial_gradients)
    outcome = "wolfe"
    if limited:
        if best is not None:
            with torch.no_grad():
                for parameter, weight, gradient in zip(parameters, best[1], best[2]):
                    parameter.copy_(weight)
                    parameter.grad = gradient
            outcome = "armijo_recovered"
        else:
            outcome = "rolled_back"
        optimizer.state.clear()
        for group in optimizer.param_groups:
            group["lr"] = 1. if best is not None else max(1e-3, float(group["lr"]) * .5)
    delta_inf = max(float((parameter.detach() - weight).abs().max())
                    for parameter, weight in zip(parameters, base))
    if not limited and delta_inf == 0.:
        if gradient_max <= 1e-7:
            outcome = "converged"
        else:
            outcome = "restarted_nochange"
            optimizer.state.clear()
            for group in optimizer.param_groups:
                group["lr"] = 1.
    return _RefinementStep(evaluations, limited, outcome, gradient_max, delta_inf, initial_loss)


def _refinement_iteration(run: TrainingRun, indices: np.ndarray, centers: np.ndarray,
                          parameters: tuple[Any, ...]) -> tuple[dict[str, Any], _RefinementStep]:
    torch = run.field.torch
    def closure():
        _cancelled(run.losses.cancel)
        run.optimizer.zero_grad(set_to_none=True)
        terms = _deterministic_objective(run, indices, centers, backward=True)
        if any(parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
               for parameter in parameters):
            raise FloatingPointError("PINN refinement gradient became non-finite.")
        run.metadata["function_evaluations"]["lbfgs_closures"] += 1
        return terms["loss"]
    step = _recovering_lbfgs_step(run.optimizer, parameters, closure, run.losses.cancel)
    run.metadata["refinement_budget_limited_steps"] += int(step.limited)
    run.metadata["refinement_step_outcomes"][step.outcome] += 1
    run.metadata["refinement_accepted_updates"] += int(step.delta_inf > 0.)
    run.metadata["refinement_nochange_steps"] += int(step.delta_inf == 0.)
    run.metadata["refinement_restarts"] += int(step.limited or step.outcome == "restarted_nochange")
    objective = _deterministic_objective(run, indices, centers, backward=False)
    run.metadata["function_evaluations"]["refinement_objective"] += 1
    run.metadata["refinement_last_step"] = {
        "outcome": step.outcome, "gradient_max": step.gradient_max,
        "parameter_delta_inf": step.delta_inf,
        "training_objective_change": float(objective["loss"]) - step.initial_loss,
        "learning_rate": float(run.optimizer.param_groups[0]["lr"])}
    return objective, step


def _train_corrections(run: TrainingRun, audit_indices: np.ndarray, centers: np.ndarray,
                       record: Callable[..., None], cancel: threading.Event) -> tuple[str, ...]:
    """Activate from training diagnostics and execute fixed correction stages."""
    field, data, torch = run.field, run.field.data, run.field.torch
    config = data.config
    run.metadata.setdefault("last_physics_evaluated_epoch", int(run.history["epoch"][-1]))
    run.metadata["correction_physics_snapshot_indices"] = audit_indices.tolist()
    groups = ()
    if config.pinn_epochs >= 50:
        diagnostics = _training_diagnostics(run, audit_indices)
        groups = _correction_groups(data, diagnostics)
        run.metadata["conditioning_activation"] = {"diagnostics": diagnostics,
            "active_groups": list(groups), "target": .05, "source": "training labels and raw equations only"}
        run.metadata["function_evaluations"]["activation_diagnostics"] = 1
    if groups:
        if data.thermal:
            parameters = tuple(field.model.parameters())
        else:
            # Each added 64x3 trunk retains another full accepted FE pair.
            _validate_correction_memory(data, groups)
            from .corrections import create_correction_model
            field.model = create_correction_model(field.model, groups, torch=torch)
            parameters = tuple(field.model.branch_parameters())
            run.metadata.update(architecture=field.model.architecture_metadata(), parameterization_version=3)
        run.optimizer = torch.optim.Adam(parameters, lr=1e-3)
        additional_adam, additional_lbfgs = min(2000, 2 * config.pinn_epochs), min(50, config.pinn_epochs // 20)
        run.metadata["conditioning_adam_objective"] = "original full sampled training labels plus raw physics and boundary losses, each with coefficient one"
        for index in range(additional_adam):
            _cancelled(cancel)
            field.model.train()
            run.optimizer.zero_grad(set_to_none=True)
            data_loss = run.losses.sampled_training_loss()
            physics_loss, boundary_loss = run.losses.thermal_physics() if data.thermal else run.losses.coupled_physics()
            loss = data_loss + physics_loss + boundary_loss
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("PINN correction joint loss became non-finite.")
            _cancelled(cancel)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 10., error_if_nonfinite=True)
            _cancelled(cancel)
            run.optimizer.step()
            run.metadata["additional_optimizer_steps_completed"]["adam"] += 1
            run.metadata["function_evaluations"]["adam"] += 1
            objective = {"loss": loss, "data_loss": data_loss,
                         "physics_loss": physics_loss, "boundary_loss": boundary_loss}
            run.metadata["last_physics_evaluated_epoch"] = int(run.history["epoch"][-1]) + 1
            record(objective, "Correction Adam", stage_end=index + 1 == additional_adam,
                   detail=f" · joint data+physics+boundary · physics@{run.metadata['last_physics_evaluated_epoch']}")
        run.optimizer = torch.optim.LBFGS(parameters, lr=1., max_iter=1, max_eval=8,
            history_size=20, line_search_fn="strong_wolfe")
        for index in range(additional_lbfgs):
            _cancelled(cancel)
            objective, step = _refinement_iteration(run, audit_indices, centers, parameters)
            run.metadata["additional_optimizer_steps_completed"]["lbfgs"] += 1
            record(objective, "Correction L-BFGS", stage_end=index + 1 == additional_lbfgs,
                detail=f" · {step.outcome} · delta_inf={step.delta_inf:.3g} · lr={run.optimizer.param_groups[0]['lr']:.3g}")
    run.metadata["conditioning_completed"] = bool(groups)
    return groups


def _train_model(run: TrainingRun, progress: Progress, cancel: threading.Event) -> float:
    field, data, torch = run.field, run.field.data, run.field.torch
    config = data.config
    initial_validation = run.losses.validation_loss()
    heldout_epoch = 0
    adam_steps, _ = _optimizer_steps(config.pinn_epochs)
    primary_indices = np.unique(np.linspace(1, data.snapshot_count - 1,
        min(4, data.snapshot_count - 1), dtype=np.int64))
    audit_indices = _audit_intervals(data)
    centers = np.unique(np.linspace(0, data.point_count - 1,
        min(field.maximum_batch, data.point_count), dtype=np.int64))
    run.metadata["refinement_physics_snapshot_indices"] = primary_indices.tolist()
    run.metadata["final_physics_snapshot_indices"] = audit_indices.tolist()
    run.metadata["refinement_center_ids"] = centers.tolist()
    run.metadata["refinement_labels"] = "all training labels, streamed; held-out labels never used by the optimizer"
    maximum_steps = config.pinn_epochs + (min(2000, 2 * config.pinn_epochs)
        + min(50, config.pinn_epochs // 20) if config.pinn_epochs >= 50 else 0)

    def record(objective, stage, *, stage_end=False, detail=""):
        nonlocal heldout_epoch
        epoch = len(run.history["epoch"]) + 1
        values = {"epoch": float(epoch), **{
            key: float(value.detach()) for key, value in objective.items()}}
        if epoch == 1 or epoch % 10 == 0 or stage_end:
            field.model.eval()
            current_validation = run.losses.validation_loss()
            heldout_epoch = epoch
        else:
            current_validation = run.history["validation_loss"][-1]
        values["validation_loss"] = current_validation
        run.metadata["last_physics_evaluated_epoch"] = epoch
        run.metadata.update(heldout_evaluated_epoch=heldout_epoch, actual_training_steps=epoch,
                            training_epochs=epoch)
        for key, value in values.items():
            run.history[key].append(value)
        stages = run.metadata["training_stages"]
        if not stages or stages[-1]["stage"] != stage:
            stages.append({"stage": stage, "first_step": epoch, "last_step": epoch, "steps": 1})
        else:
            stages[-1].update(last_step=epoch, steps=stages[-1]["steps"] + 1)
        sampled = "Adam" in stage
        label = "batch loss" if sampled else "training loss"
        display = "Adam" if stage.startswith("Adam") else stage
        progress(.9 * epoch / maximum_steps,
            f"Epoch {epoch} (budget ≤{maximum_steps}) · {display} · {label} {values['loss']:.6g} · held-out {current_validation:.6g}@{heldout_epoch}{detail}")
        if run.checkpoint_path is not None and (epoch % 50 == 0 or stage_end):
            _checkpoint(run.checkpoint_path, field.model, run.optimizer, epoch,
                        data.result, data.scaling, run.history, run.metadata)

    for epoch in range(config.pinn_epochs):
        _cancelled(cancel)
        field.model.train()
        if epoch < adam_steps:
            run.optimizer.zero_grad(set_to_none=True)
            data_loss = run.losses.sampled_training_loss()
            # Evaluate even during warmup so the original primary RNG draws
            # and raw equation diagnostics are preserved exactly.
            physics_loss, boundary_loss = run.losses.thermal_physics() if data.thermal else run.losses.coupled_physics()
            weight = _physics_weight(epoch, config.pinn_epochs, thermal=data.thermal)
            loss = data_loss + weight * (physics_loss + boundary_loss)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("PINN loss became non-finite; previous numerical results are unchanged.")
            _cancelled(cancel)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(field.model.parameters(), 10., error_if_nonfinite=True)
            _cancelled(cancel)
            run.optimizer.step()
            run.scheduler.step()
            run.metadata["optimizer_steps_completed"]["adam"] += 1
            run.metadata["function_evaluations"]["adam"] += 1
            objective = {"loss": loss, "data_loss": data_loss, "physics_loss": physics_loss,
                         "boundary_loss": boundary_loss}
            stage = "Adam data warmup" if weight == 0. else "Adam physics ramp" if weight < 1. else "Adam"
            detail = f" · physics weight={weight:.3g}" if weight < 1. else ""
        else:
            if epoch == adam_steps:
                run.optimizer = torch.optim.LBFGS(field.model.parameters(), lr=1., max_iter=1,
                    max_eval=8, history_size=20, line_search_fn="strong_wolfe")
            objective, step = _refinement_iteration(run, primary_indices, centers, tuple(field.model.parameters()))
            run.metadata["optimizer_steps_completed"]["lbfgs"] += 1
            stage = "L-BFGS"
            detail = f" · {step.outcome} · delta_inf={step.delta_inf:.3g} · lr={run.optimizer.param_groups[0]['lr']:.3g}"
        record(objective, stage, stage_end=epoch + 1 == config.pinn_epochs, detail=detail)

    _train_corrections(run, audit_indices, centers, record, cancel)
    # Qualification always uses final weights and raw unweighted equations.
    objective = _deterministic_objective(run, audit_indices, centers, backward=False)
    run.metadata["function_evaluations"]["final_objective"] += 1
    run.metadata["last_physics_evaluated_epoch"] = int(run.history["epoch"][-1])
    run.metadata["final_loss_evaluation"] = "final weights; all training labels and up to ten uniformly selected accepted-interval physics panels plus native charge-transition neighbors; original raw residual scales"
    for key, value in objective.items():
        run.history[key][-1] = float(value.detach())
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
        _checkpoint(run.checkpoint_path, field.model, run.optimizer, int(run.history["epoch"][-1]), result, data.scaling, run.history, run.metadata)
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
