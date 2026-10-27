"""Training-only pressure sampling for explicit coupled PINN experiments."""
from __future__ import annotations

from typing import Any

import numpy as np

from .fields import _cancelled


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
