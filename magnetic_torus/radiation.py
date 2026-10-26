"""Stable particle-diagnostic and quantitative dose-transport API."""
from __future__ import annotations

from .transport.common import (
    ATOMIC_MASS, C, ELEMENTARY_CHARGE, MEV, PROTON_MASS,
    SourceParticles, TrajectoryResult,
)
from .transport.sampling import load_spectrum, sample_source
from .transport.fields import AnalyticCoilField, FrozenField, field_source_run, frozen_field
from .transport.trajectories import relativistic_boris, trace_particles
from .transport.geant4 import calculate_dose, parse_dose_output


__all__ = [
    "ATOMIC_MASS", "C", "ELEMENTARY_CHARGE", "MEV", "PROTON_MASS",
    "AnalyticCoilField", "FrozenField", "SourceParticles", "TrajectoryResult",
    "calculate_dose", "field_source_run", "frozen_field", "load_spectrum",
    "parse_dose_output", "relativistic_boris", "sample_source", "trace_particles",
]
