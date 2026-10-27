"""Training-only pressure sampling for explicit coupled PINN experiments."""
from __future__ import annotations

from typing import Any

import numpy as np

from .fields import _cancelled
from .training import TrainingLosses


class PressureHuberLosses(TrainingLosses):
    """Research-only Huber pressure labels, with all other losses unchanged."""

    def __init__(self, original: TrainingLosses, delta_pa: float = .003):
        super().__init__(original.field, original.field_validity, original.cancel)
        scaling = self.field.data.scaling
        if self.field.data.thermal or "pressure" not in scaling.names:
            raise ValueError("Pressure Huber loss requires a coupled pressure field.")
        index = scaling.names.index("pressure")
        if scaling.widths[index] != 1 or not np.isfinite(delta_pa) or delta_pa <= 0:
            raise ValueError("Pressure Huber requires a scalar pressure and positive finite transition.")
        self.column = sum(scaling.widths[:index])
        self.delta_pa = float(delta_pa)
        self.delta = self.delta_pa / float(scaling.scales[self.column])
        if not np.isfinite(self.delta) or self.delta <= 0:
            raise ValueError("Pressure training scale must be positive and finite.")

    def errors_at(self, ids: np.ndarray, *, training: bool = False,
                  forward_batch_size: int | None = None) -> tuple[Any, np.ndarray]:
        if not training:
            return super().errors_at(ids, training=False, forward_batch_size=forward_batch_size)
        field, data = self.field, self.field.data
        ti, pi = np.divmod(ids, data.point_count)
        if data.cache is not None:
            rows = data.cache.training_rows[ids]
            if np.any(rows < 0):
                raise ValueError("A held-out label cannot enter the training-target cache.")
            normalized = data.cache.targets[rows]
        else:
            if not np.isin(ids, data.train_ids).all():
                raise ValueError("A held-out label cannot enter pressure Huber training.")
            normalized = (data.reference[ti, pi] - data.scaling.offsets) / data.scaling.scales
        residual = field.forward(field.input_at(ti, pi), pi, ti,
                                 forward_batch_size=forward_batch_size) - field.torch.as_tensor(normalized, **field.kw)
        pressure = residual[:, self.column:self.column + 1]
        absolute = pressure.abs()
        # Twice the standard Huber loss equals squared error in its quadratic region.
        robust = field.torch.where(absolute <= self.delta, pressure.square(),
                                   2 * self.delta * absolute - self.delta ** 2)
        errors = field.torch.cat((residual[:, :self.column].square(), robust,
                                  residual[:, self.column + 1:].square()), dim=1)
        return errors, self.validity_at(ids, training=True)

    def metadata(self) -> dict[str, Any]:
        return {"kind": "pressure-only Huber", "delta_pa": self.delta_pa,
                "normalized_delta": self.delta, "normalization": "training labels only",
                "quadratic_region": "identical to squared normalized error"}


class PressureTemporalSampler:
    """Prefer pressure labels at changing times, with at most a 3:1 weight ratio.

    A temporal score uses only nodes whose pressure labels are in the training
    pool at both adjacent accepted times. Other fields retain the method-5
    free-DOF stratified sampling. No held-out target is read.
    """

    def __init__(self, losses: Any):
        self.losses = losses
        self.field = losses.field
        self.data = self.field.data
        if self.data.thermal or "pressure" not in self.data.scaling.names:
            raise ValueError("Temporal pressure sampling requires coupled pressure labels.")
        self.pressure_index = self.data.scaling.names.index("pressure")
        self.column = sum(self.data.scaling.widths[:self.pressure_index])
        pool = losses.training_pools[self.pressure_index]
        n, nt = self.data.point_count, self.data.snapshot_count
        self.pools = tuple(pool[pool // n == ti] % n for ti in range(nt))
        scores = np.zeros(nt, dtype=np.float64)
        pairs = np.zeros(nt, dtype=np.int64)
        for ti in range(1, nt):
            _cancelled(losses.cancel)
            common = np.intersect1d(self.pools[ti - 1], self.pools[ti], assume_unique=True)
            pairs[ti] = len(common)
            if len(common):
                dt = float(self.data.result.times[ti] - self.data.result.times[ti - 1])
                current = self.data.reference[ti, common, self.column]
                previous = self.data.reference[ti - 1, common, self.column]
                scores[ti] = float(np.mean(np.abs(current - previous))) / dt
        if not np.isfinite(scores).all():
            raise FloatingPointError("Training-only pressure variation became non-finite.")
        maximum = float(scores.max(initial=0.))
        self.weights = 1. + (2. * scores / maximum if maximum > 0. else scores)
        self.scores = scores
        self.pairs = pairs
        self.calls = 0

    def sampled_loss(self) -> Any:
        losses, field, data = self.losses, self.field, self.data
        _cancelled(losses.cancel)
        eligible = [(index, pool) for index, pool in enumerate(losses.training_pools) if len(pool)]
        if not eligible:
            return next(p for p in field.model.parameters() if p.requires_grad).sum() * 0.
        order = data.rng.permutation(len(eligible))
        quota, remainder = divmod(field.maximum_batch, len(eligible))
        samples, selections = [], []
        for position, pool_index in enumerate(order):
            index, pool = eligible[pool_index]
            count = min(len(pool), quota + (position < remainder))
            if not count:
                continue
            if index == self.pressure_index:
                weights = self.weights[pool // data.point_count]
                selected = data.rng.choice(pool, count, replace=False, p=weights / weights.sum())
            else:
                selected = data.rng.choice(pool, count, replace=False)
            samples.append(selected)
            selections.append(np.full(count, index, dtype=np.int64))
        ids, selected_fields = np.concatenate(samples), np.concatenate(selections)
        errors, validity = losses.errors_at(ids, training=True)
        column = 0
        for index, width in enumerate(data.scaling.widths):
            validity[:, column:column + width] &= (selected_fields == index)[:, None]
            column += width
        self.calls += 1
        return losses._label_loss(errors, validity)

    def metadata(self) -> dict[str, Any]:
        return {"strategy": "training-only adjacent pressure variation",
                "scores_pa_per_s": self.scores.tolist(),
                "matched_training_nodes": self.pairs.tolist(),
                "relative_time_weights": self.weights.tolist(),
                "maximum_ratio": 3., "calls": self.calls,
                "importance_correction": False}
