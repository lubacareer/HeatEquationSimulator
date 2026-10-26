"""Immutable particle results, SI constants, and cooperative cancellation."""
from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping
import numpy as np
from ..models import SimulationCancelled


C = 299792458.0
ELEMENTARY_CHARGE = 1.602176634e-19
ATOMIC_MASS = 1.66053906892e-27
PROTON_MASS = 1.67262192595e-27
MEV = ELEMENTARY_CHARGE * 1e6


def _cancelled(cancel: Any) -> bool:
    return bool(cancel() if callable(cancel) else cancel is not None and cancel.is_set())


def _check(cancel: Any) -> None:
    if _cancelled(cancel):
        raise SimulationCancelled("Radiation calculation cancelled; previous results retained.")


@dataclass(frozen=True)
class TrajectoryResult:
    source_run_id: str
    config_hash: str
    snapshot_index: int
    paths: np.ndarray
    baseline_paths: np.ndarray
    times: np.ndarray
    species: np.ndarray
    energies_mev_nucleon: np.ndarray
    outcomes: tuple[str, ...]
    baseline_outcomes: tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("paths", "baseline_paths", "times", "species", "energies_mev_nucleon"):
            value = np.array(getattr(self, name), copy=True)
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        if self.paths.ndim != 3 or self.paths.shape[2] != 3 or self.baseline_paths.shape != self.paths.shape:
            raise ValueError("Paired trajectories require matching (particle, frame, 3) arrays.")
        count, frames = self.paths.shape[:2]
        if self.times.shape != (count, frames) or self.species.shape != (count, 2) or self.energies_mev_nucleon.shape != (count,):
            raise ValueError("Trajectory times/species/energies do not match the particle paths.")
        if len(self.outcomes) != count or len(self.baseline_outcomes) != count:
            raise ValueError("Every primary requires field-on and field-off outcomes.")
        if any(not np.all(np.isfinite(getattr(self, key))) for key in ("paths", "baseline_paths", "times", "energies_mev_nucleon")):
            raise ValueError("Trajectory arrays must be finite.")


@dataclass(frozen=True)
class SourceParticles:
    positions: np.ndarray
    directions: np.ndarray
    z: np.ndarray
    a: np.ndarray
    energy_mev_nucleon: np.ndarray
    source_area_m2: float
    metadata: Mapping[str, Any] = field(default_factory=dict)
