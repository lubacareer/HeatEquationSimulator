"""Explicit research-only front/residual sampling of the native discrete PDE.

The regular trainer does not construct this sampler. Half of every batch comes
from all source nodes, and half from a bounded candidate proposal. Importance
weights retain the original uniform-node physics and initial-condition means.
No numerical field labels, validation metrics, or global RNG are used to build
the proposal; residuals and phase fractions come from the current neural model.
"""
from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any

import numpy as np

from .fields import _cancelled

if TYPE_CHECKING:
    from .training import TrainingLosses


class FrontResidualSampler:
    """A bounded, detached proposal with an independent reproducible RNG.

    At least half of each batch remains uniform; odd sizes round the adaptive
    count down. A single last-interval proposal is retained. It is refreshed when the
    accepted interval changes or after ``refresh_steps`` training steps. The
    candidate proposal is an equal mixture of predicted mushy-zone membership
    (0.05 < liquid fraction < 0.95) and absolute normalized residual; absent
    front/residual information falls back to residual/uniform candidates.
    """

    def __init__(self, seed: int, *, candidate_cap: int = 4096,
                 refresh_steps: int = 50) -> None:
        if (not isinstance(seed, (int, np.integer)) or isinstance(seed, bool)
                or not isinstance(candidate_cap, int) or isinstance(candidate_cap, bool)
                or not 1 <= candidate_cap <= 4096
                or not isinstance(refresh_steps, int) or isinstance(refresh_steps, bool)
                or refresh_steps < 1):
            raise ValueError("Research sampling needs an integer seed, 1–4096 candidates, and a positive refresh interval.")
        self.seed = int(seed)
        self.candidate_cap = candidate_cap
        self.refresh_steps = refresh_steps
        self.rng = np.random.default_rng(seed)
        self._field = None
        self._point_count = 0
        self._candidates = np.empty(0, dtype=np.int64)
        self._proposal = np.empty(0, dtype=np.float64)
        self._interval = None
        self._step = -1
        self.proposal_evaluations = 0
        self.proposal_rows = 0
        self.training_evaluations = 0
        self.training_rows = 0
        self.sampled_rows = 0
        self.duplicate_rows = 0

    @property
    def cache_bytes(self) -> int:
        return self._candidates.nbytes + self._proposal.nbytes

    def _bind(self, losses: TrainingLosses) -> None:
        field = losses.field
        if not field.data.thermal:
            raise ValueError("Front/residual node sampling is only defined for the native thermal operator.")
        if self._field is not None and self._field is not field:
            raise ValueError("Create a fresh sampler for each training field/source.")
        if self._field is None:
            if field.data.point_count < 1:
                raise ValueError("Research sampling requires a nonempty source mesh.")
            self._field = field
            self._point_count = field.data.point_count
            self._candidates = np.unique(np.linspace(0, self._point_count - 1,
                min(self.candidate_cap, self._point_count), dtype=np.int64))

    def _refresh(self, losses: TrainingLosses, ti: int, step: int) -> None:
        _cancelled(losses.cancel)
        with losses.field.torch.no_grad():
            terms = losses.thermal_components(ti, self._candidates)
        self.proposal_evaluations += 1
        self.proposal_rows += len(self._candidates)
        _cancelled(losses.cancel)
        residual = np.abs(terms["residual"].detach().cpu().numpy())
        fraction = terms["liquid_fraction"].detach().cpu().numpy()
        if (residual.shape != self._candidates.shape or fraction.shape != residual.shape
                or not np.isfinite(residual).all() or not np.isfinite(fraction).all()):
            raise FloatingPointError("Research sampling received non-finite or malformed model scores.")
        maximum = float(residual.max())
        relative = residual / maximum if maximum else np.ones(len(residual))
        probability = relative / relative.sum()
        front = (fraction > .05) & (fraction < .95)
        if front.any():
            probability = .5 * probability + .5 * front / int(front.sum())
        self._proposal = np.asarray(probability / probability.sum(), dtype=np.float64)
        self._interval, self._step = ti, step

    def _probabilities(self, ids: np.ndarray, adaptive_fraction: float) -> np.ndarray:
        positions = np.searchsorted(self._candidates, ids)
        inside = positions < len(self._candidates)
        inside &= self._candidates[np.minimum(positions, len(self._candidates) - 1)] == ids
        adaptive = np.zeros(len(ids), dtype=np.float64)
        adaptive[inside] = self._proposal[positions[inside]]
        return (1. - adaptive_fraction) / self._point_count + adaptive_fraction * adaptive

    def _draw(self, batch: int) -> tuple[np.ndarray, np.ndarray]:
        adaptive = batch // 2
        uniform = batch - adaptive
        ids = np.concatenate((self.rng.integers(self._point_count, size=uniform, dtype=np.int64),
            self.rng.choice(self._candidates, size=adaptive, replace=True, p=self._proposal)))
        probability = self._probabilities(ids, adaptive / batch)
        return ids, 1. / (self._point_count * probability)

    def loss(self, losses: TrainingLosses, ti: int, step: int) -> tuple[Any, Any]:
        """Evaluate an importance-corrected joint physics/boundary minibatch.

        Draws use replacement; duplicate node IDs are evaluated once and then
        expanded before weighting. Passing duplicate centers directly to the
        conservative neighborhood operator would construct incorrect rows.
        """
        _cancelled(losses.cancel)
        self._bind(losses)
        if (not isinstance(ti, (int, np.integer)) or isinstance(ti, bool)
                or not 1 <= ti < losses.field.data.snapshot_count
                or not isinstance(step, (int, np.integer)) or isinstance(step, bool) or step < 0):
            raise ValueError("Research sampling requires an accepted interval and nonnegative step.")
        if self._interval != ti or step < self._step or step - self._step >= self.refresh_steps:
            self._refresh(losses, int(ti), int(step))
        _cancelled(losses.cancel)
        batch = min(losses.field.maximum_batch, self._point_count)
        ids, weights = self._draw(batch)
        centers, inverse = np.unique(ids, return_inverse=True)
        terms = losses.thermal_components(int(ti), centers)
        self.training_evaluations += 1
        self.training_rows += len(centers)
        self.sampled_rows += len(ids)
        self.duplicate_rows += len(ids) - len(centers)
        _cancelled(losses.cancel)
        field = losses.field
        tensor_indices = field.torch.as_tensor(inverse, device=field.device)
        importance = field.torch.as_tensor(weights, **field.kw)
        physics = (terms["residual"][tensor_indices].square() * importance).mean()
        boundary = (terms["initial_error"][tensor_indices].square() * importance).mean()
        if not bool(field.torch.isfinite(physics)) or not bool(field.torch.isfinite(boundary)):
            raise FloatingPointError("Research sampled physics/boundary objective became non-finite.")
        return physics, boundary

    def metadata(self) -> dict[str, Any]:
        """Include proposal maintenance in the experiment's operator-work ledger."""
        return {
            "kind": "native_front_residual_importance", "version": 1,
            "seed": self.seed, "candidate_cap": self.candidate_cap,
            "refresh_steps": self.refresh_steps, "uniform_fraction": .5,
            "rounding": "adaptive count=floor(batch/2); weights use actual fractions",
            "proposal": "equal predicted-front/absolute-residual mixture when a front exists; residual or uniform fallback",
            "objective": "importance-corrected original uniform-node residual and initial-error means",
            "cache_bytes": self.cache_bytes, "cache_limit_bytes": 4096 * 16,
            "cached_interval": self._interval, "cached_step": self._step,
            "proposal_evaluations": self.proposal_evaluations,
            "proposal_rows": self.proposal_rows,
            "training_evaluations": self.training_evaluations,
            "training_rows": self.training_rows,
            "sampled_rows": self.sampled_rows, "duplicate_rows": self.duplicate_rows,
            "rng_state": deepcopy(self.rng.bit_generator.state),
        }
