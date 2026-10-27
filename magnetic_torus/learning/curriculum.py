"""Explicit training-only research curricula; the regular trainer is unchanged.

The helpers never read held-out labels or fit field scales. They receive the
prepared source-bound losses and leave validation and unweighted audits to the
caller. Torch is imported only when gradient balancing is actually requested.
"""
from __future__ import annotations

from copy import deepcopy
import math
from numbers import Real
from typing import TYPE_CHECKING, Any, Iterable, Mapping

import numpy as np

from .fields import _cancelled, _torch

if TYPE_CHECKING:
    from .training import TrainingLosses


_HOST_CACHE_LIMIT = 64 * 1024**2


class TimeLabelSampler:
    """Sample the original free training-label mixture at one accepted time.

    The optional stable grouped-ID cache shares the 64 MiB host-cache allowance
    with the existing normalized-label cache. Its conservative construction
    estimate includes bounded temporary grouping arrays; larger data stream instead.
    Both paths preserve the training pools' original within-time ordering.
    """

    def __init__(self, losses: TrainingLosses, seed: int, *,
                 cache_limit_bytes: int = _HOST_CACHE_LIMIT) -> None:
        if (not isinstance(seed, (int, np.integer)) or isinstance(seed, bool)
                or not isinstance(cache_limit_bytes, int) or isinstance(cache_limit_bytes, bool)
                or not 0 <= cache_limit_bytes <= _HOST_CACHE_LIMIT):
            raise ValueError("Time-label sampling needs an integer seed and a 0–64 MiB cache limit.")
        self.losses = losses
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)
        self._field = losses.field
        self._data = losses.field.data
        self._ids = None
        self._offsets = None
        self.cache_limit_bytes = cache_limit_bytes
        self.existing_cache_bytes = self._data.cache.nbytes if self._data.cache is not None else 0
        pools = losses.training_pools
        nt = self._data.snapshot_count
        sizes = [len(pool) for pool in pools]
        required = (sum(sizes) + len(pools) * (nt + 1)) * 8
        chunk_size = min(max(sizes, default=0), 4096)
        self.build_peak_estimate_bytes = required + chunk_size * 6 * 8 + nt * 3 * 8
        self.calls = 0
        self.sampled_labels = 0
        self.field_samples = dict.fromkeys(self._data.scaling.names, 0)
        self.last_interval = None
        _cancelled(losses.cancel)
        remaining = max(0, cache_limit_bytes - self.existing_cache_bytes)
        if self.build_peak_estimate_bytes <= remaining:
            ids = np.empty(sum(sizes), dtype=np.int64)
            offsets = np.empty((len(pools), nt + 1), dtype=np.int64)
            cursor = 0
            for index, pool in enumerate(pools):
                _cancelled(losses.cancel)
                counts = np.zeros(nt, dtype=np.int64)
                for start in range(0, len(pool), 4096):
                    _cancelled(losses.cancel)
                    times = pool[start:start + 4096] // self._data.point_count
                    counts += np.bincount(times, minlength=nt)
                offsets[index, 0] = cursor
                np.cumsum(counts, out=offsets[index, 1:])
                offsets[index, 1:] += cursor
                positions = offsets[index, :-1].copy()
                for start in range(0, len(pool), 4096):
                    _cancelled(losses.cancel)
                    chunk = pool[start:start + 4096]
                    times = chunk // self._data.point_count
                    for ti in np.unique(times):
                        selected = chunk[times == ti]
                        position = positions[ti]
                        ids[position:position + len(selected)] = selected
                        positions[ti] += len(selected)
                cursor += len(pool)
            _cancelled(losses.cancel)
            ids.setflags(write=False)
            offsets.setflags(write=False)
            self._ids, self._offsets = ids, offsets

    @property
    def cache_bytes(self) -> int:
        return 0 if self._ids is None else self._ids.nbytes + self._offsets.nbytes

    def _pool(self, index: int, ti: int) -> np.ndarray:
        if self._ids is not None:
            return self._ids[self._offsets[index, ti]:self._offsets[index, ti + 1]]
        pool = self.losses.training_pools[index]
        return pool[pool // self._data.point_count == ti]

    def components(self, ti: int) -> dict[str, Any]:
        """Return ordered per-field valid-component MSEs from training rows only."""
        _cancelled(self.losses.cancel)
        if self.losses.field is not self._field or self._field.data is not self._data:
            raise ValueError("Create a fresh time-label sampler for each prepared source.")
        if (not isinstance(ti, (int, np.integer)) or isinstance(ti, bool)
                or not 1 <= ti < self._data.snapshot_count):
            raise ValueError("Time-label sampling requires a noninitial accepted time.")
        eligible = []
        for index in range(len(self.losses.training_pools)):
            _cancelled(self.losses.cancel)
            pool = self._pool(index, int(ti))
            if len(pool):
                eligible.append((index, pool))
        if not eligible:
            self.calls += 1
            self.last_interval = int(ti)
            return {}
        order = self.rng.permutation(len(eligible))
        quota, remainder = divmod(self._field.maximum_batch, len(eligible))
        ids, fields = [], []
        for position, pool_index in enumerate(order):
            index, pool = eligible[pool_index]
            count = min(len(pool), quota + (position < remainder))
            if count:
                ids.append(self.rng.choice(pool, count, replace=False))
                fields.append(np.full(count, index, dtype=np.int64))
        _cancelled(self.losses.cancel)
        selected, chosen_fields = np.concatenate(ids), np.concatenate(fields)
        errors, validity = self.losses.errors_at(selected, training=True)
        self.calls += 1
        self.sampled_labels += len(selected)
        self.last_interval = int(ti)
        _cancelled(self.losses.cancel)
        result = {}
        column = 0
        for index, (name, width) in enumerate(zip(self._data.scaling.names, self._data.scaling.widths)):
            mask = validity[:, column:column + width] & (chosen_fields == index)[:, None]
            count = int(mask.sum())
            if count:
                loss = (errors[:, column:column + width] *
                    self._field.torch.as_tensor(mask, **self._field.kw)).sum() / count
                if not bool(self._field.torch.isfinite(loss)):
                    raise FloatingPointError("Time-conditioned training label loss became non-finite.")
                result[name] = loss
                self.field_samples[name] += int((chosen_fields == index).sum())
            column += width
        return result

    def metadata(self) -> dict[str, Any]:
        return {
            "kind": "accepted_time_training_labels", "version": 1, "seed": self.seed,
            "source": "existing training-only free-DOF pools; original per-field valid-component MSE",
            "cache_mode": "grouped" if self._ids is not None else "streaming",
            "existing_training_cache_bytes": self.existing_cache_bytes,
            "group_cache_bytes": self.cache_bytes, "cache_limit_bytes": self.cache_limit_bytes,
            "cache_build_peak_estimate_bytes": self.build_peak_estimate_bytes,
            "calls": self.calls, "sampled_labels": self.sampled_labels,
            "field_samples": dict(self.field_samples), "last_interval": self.last_interval,
            "rng_state": deepcopy(self.rng.bit_generator.state),
        }


class CausalWeights:
    """Detached exponential prefix weights from cached training losses.

    The caller bootstraps every accepted interval before optimization. Unseen
    losses start at zero, and observation counts expose incomplete bootstrap.
    The positive floor keeps every time interval in the training objective.
    """

    def __init__(self, snapshot_count: int, epsilon: float = 1., floor: float = .05) -> None:
        if (not isinstance(snapshot_count, int) or isinstance(snapshot_count, bool) or snapshot_count < 2
                or not isinstance(epsilon, Real) or isinstance(epsilon, bool)
                or not math.isfinite(float(epsilon)) or epsilon < 0
                or not isinstance(floor, Real) or isinstance(floor, bool)
                or not math.isfinite(float(floor)) or not 0 < floor <= 1):
            raise ValueError("Causal weighting needs at least two snapshots, finite epsilon≥0, and 0<floor≤1.")
        self.snapshot_count = snapshot_count
        self.epsilon, self.floor = float(epsilon), float(floor)
        self._errors = np.zeros(snapshot_count, dtype=np.float64)
        self._counts = np.zeros(snapshot_count, dtype=np.int64)
        self._cached_weights = None

    def _index(self, ti: int) -> int:
        if (not isinstance(ti, (int, np.integer)) or isinstance(ti, bool)
                or not 1 <= ti < self.snapshot_count):
            raise ValueError("Causal weights require a noninitial accepted interval.")
        return int(ti)

    def update(self, ti: int, raw_loss: float) -> None:
        ti = self._index(ti)
        if (not isinstance(raw_loss, Real) or isinstance(raw_loss, bool)
                or not math.isfinite(float(raw_loss)) or raw_loss < 0):
            raise ValueError("Causal losses must be detached finite nonnegative numbers.")
        self._errors[ti] = float(raw_loss)
        self._counts[ti] += 1
        self._cached_weights = None

    def _weights(self) -> np.ndarray:
        if self._cached_weights is None:
            weights = np.ones(self.snapshot_count, dtype=np.float64)
            cap = -math.log(self.floor)
            exponent = 0.
            for ti in range(1, self.snapshot_count):
                weights[ti] = self.floor if exponent >= cap else math.exp(-exponent)
                error = float(self._errors[ti])
                if self.epsilon and error:
                    if error >= (cap - exponent) / self.epsilon:
                        exponent = cap
                    else:
                        exponent += self.epsilon * error
            weights.setflags(write=False)
            self._cached_weights = weights
        return self._cached_weights

    def weight(self, ti: int) -> float:
        return float(self._weights()[self._index(ti)])

    def metadata(self) -> dict[str, Any]:
        return {
            "kind": "detached_training_prefix_weights", "version": 1,
            "epsilon": self.epsilon, "floor": self.floor,
            "weights": self._weights()[1:].tolist(), "last_training_losses": self._errors[1:].tolist(),
            "observation_counts": self._counts[1:].tolist(),
            "bootstrap_complete": bool(np.all(self._counts[1:] > 0)),
        }


class LabelGradientBalancer:
    """Bounded inverse-gradient weights for labels; physics weights stay external."""

    def __init__(self, refresh_steps: int = 50, *, cancel=None) -> None:
        if not isinstance(refresh_steps, int) or isinstance(refresh_steps, bool) or refresh_steps < 1:
            raise ValueError("Gradient balancing needs a positive refresh interval.")
        self.refresh_steps = refresh_steps
        self.cancel = cancel
        self._weights: dict[str, float] = {}
        self._norms: dict[str, float] = {}
        self._parameter_ids = ()
        self._step = -1
        self.gradient_evaluations = 0
        self.refreshes = 0

    def _cancelled(self) -> None:
        if self.cancel is not None:
            _cancelled(self.cancel)

    def _norm(self, loss: Any, parameters: tuple[Any, ...], torch: Any) -> float:
        if not parameters or not loss.requires_grad:
            return 0.
        gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
        self.gradient_evaluations += 1
        self._cancelled()
        gradients = tuple(value.detach() for value in gradients if value is not None)
        if not gradients:
            return 0.
        if not bool(torch.stack([torch.isfinite(value).all() for value in gradients]).all()):
            raise FloatingPointError("Training-label gradient became non-finite during balancing.")
        maximum = torch.stack([value.abs().max().double() for value in gradients]).max()
        if float(maximum) == 0.:
            return 0.
        squared = sum((value.double() / maximum).square().sum() for value in gradients)
        norm = float(maximum * squared.sqrt())
        if not math.isfinite(norm):
            raise FloatingPointError("Training-label gradient norm became non-finite.")
        return norm

    def combine(self, components: Mapping[str, Any], trainable_params: Iterable[Any],
                step: int) -> Any:
        """Return a detached-weight normalized mean, retaining its label gradients."""
        self._cancelled()
        if not isinstance(step, (int, np.integer)) or isinstance(step, bool) or step < 0:
            raise ValueError("Gradient balancing requires a nonnegative training step.")
        torch = _torch()
        all_parameters = tuple(trainable_params)
        parameters = tuple(value for value in all_parameters if value.requires_grad)
        if not components:
            if not all_parameters:
                raise ValueError("An empty label objective requires a parameter for its zero anchor.")
            return all_parameters[0].sum() * 0.
        for name, loss in components.items():
            if (not isinstance(name, str) or not isinstance(loss, torch.Tensor) or loss.ndim != 0
                    or not bool(torch.isfinite(loss)) or float(loss.detach()) < 0):
                raise ValueError("Label components must be named finite nonnegative scalar tensors.")
        names = tuple(components)
        parameter_ids = tuple(id(value) for value in parameters)
        if (tuple(self._weights) != names or parameter_ids != self._parameter_ids
                or step < self._step or step - self._step >= self.refresh_steps):
            norms = {}
            for name, loss in components.items():
                self._cancelled()
                norms[name] = self._norm(loss, parameters, torch)
            positive = [value for value in norms.values() if value > 0]
            maximum = max(positive, default=0.)
            reference = maximum * (sum(value / maximum for value in positive) / len(positive)) if positive else 0.
            weights = {name: max(.1, min(10., reference / norm)) if norm else 1.
                       for name, norm in norms.items()}
            self._cancelled()
            self._norms, self._weights = norms, weights
            self._parameter_ids, self._step = parameter_ids, int(step)
            self.refreshes += 1
        total_weight = sum(self._weights.values())
        return sum((self._weights[name] / total_weight) * loss for name, loss in components.items())

    def metadata(self) -> dict[str, Any]:
        return {
            "kind": "detached_inverse_label_gradient_norm", "version": 1,
            "refresh_steps": self.refresh_steps, "last_refresh_step": self._step,
            "relative_weight_bounds": [.1, 10.], "weights": dict(self._weights),
            "gradient_norms": dict(self._norms), "refreshes": self.refreshes,
            "gradient_evaluations": self.gradient_evaluations,
            "scope": "label components only; original physics and boundary coefficients remain external",
        }
