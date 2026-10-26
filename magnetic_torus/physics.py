"""Stable numerical API; implementations live in the numerics package."""
from __future__ import annotations

from pathlib import Path
import sys
import threading
from .models import Progress, RunResult, ScenarioConfig, SimulationCancelled
from .numerics.constants import MU0, STEFAN_BOLTZMANN
from .numerics.coils import coil_magnetic_field, coil_vector_potential, imposed_coil
from .numerics.thermal import (
    _run_thermal,
    available_memory_bytes, enthalpy_from_temperature, liquid_fraction,
    native_memory_estimate, regularized_enthalpy, source_power,
    stefan_front_benchmark, temperature_from_enthalpy,
)
from .numerics.fenics import PhysicsBackendUnavailable, _run_fenics
from .numerics.weak_residual import evaluate_weak_residual_torch
from .numerics.validation import write_validation_report


__all__ = [
    "MU0", "STEFAN_BOLTZMANN", "PhysicsBackendUnavailable", "available_memory_bytes",
    "coil_magnetic_field", "coil_vector_potential", "enthalpy_from_temperature",
    "evaluate_weak_residual_torch", "imposed_coil", "liquid_fraction",
    "native_memory_estimate", "regularized_enthalpy", "run_simulation",
    "source_power", "stefan_front_benchmark", "temperature_from_enthalpy",
    "write_validation_report",
]


def run_simulation(config: ScenarioConfig, progress: Progress | None = None,
                   cancel: threading.Event | None = None) -> RunResult:
    config.validate()
    report = progress or (lambda _fraction, _message: None)
    stop = cancel or threading.Event()
    if stop.is_set():
        raise SimulationCancelled("Simulation cancelled before starting.")
    if config.solver == "thermal_benchmark":
        return _run_thermal(config, report, stop)
    return _run_fenics(config, report, stop)


if __name__ == "__main__":
    import argparse
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Reproducible Magnetic Torus physical checks")
    parser.add_argument("--benchmark-report", required=True, type=Path, help="Write analytical, conservation and pending-gate results as JSON.")
    parser.add_argument("--include-fenics", action="store_true", help="Run installed WSL/Linux coupled workers and analytical FE reductions.")
    args = parser.parse_args()
    write_validation_report(args.benchmark_report, args.include_fenics,
                            lambda fraction, message: print(f"{fraction:5.0%} {message}", flush=True))
    print(str(args.benchmark_report.resolve()))
