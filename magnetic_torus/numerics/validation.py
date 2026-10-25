"""Reproducible numerical checks and explicit outstanding qualification gates."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any
import numpy as np
from ..models import Progress, ScenarioConfig
from ..backend_environment import wsl_distribution, wsl_path
from .thermal import stefan_front_benchmark
from .weak_residual import evaluate_weak_residual_torch


def write_validation_report(destination: str | Path, include_fenics: bool = False,
                            progress: Progress | None = None) -> dict[str, Any]:
    """Re-run analytical checks and retain both passing and pending release gates."""
    from dataclasses import replace
    from ..physics import run_simulation
    report_progress = progress or (lambda _fraction, _message: None)
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"format_version": 1, "units": "SI", "overall_status": "experimental; release validation incomplete"}
    report_progress(.02, "Checking analytical Stefan-front refinement")
    cases = []
    for count in (32, 64, 128):
        case = stefan_front_benchmark(count, count * 16)
        cases.append({key: value for key, value in case.items() if not isinstance(value, np.ndarray)})
    report["stefan_reference"] = {"status": "passed" if max(case["relative_front_error"] for case in cases) < .005 else "failed",
                                   "cases": cases, "note": "Sharp FV enthalpy integrator. Subcell front position can give nonmonotone individual refinement errors; this does not validate smoothed 3D MHD."}
    report_progress(.12, "Checking conservative surface charging and thermal losses")
    benchmark = replace(ScenarioConfig(), solver="thermal_benchmark", resolution=3, time_steps=8,
                        heating_power=20000., discharge_power=5000., charge_fraction=.37,
                        duration=10., emissivity=.1)
    native = run_simulation(benchmark)
    balance = float(np.max(np.abs(native.diagnostics["energy_balance_error_j"])))
    report["thermal_energy_balance"] = {"status": "passed" if balance < .01 else "failed", "maximum_defect_j": balance, "config": benchmark.to_dict()}
    report["full_fe_stationary"] = {"status": "not run", "reason": "Add --include-fenics to run installed Linux/WSL workers."}
    report["full_fe_driven_vms"] = dict(report["full_fe_stationary"])
    report["full_fe_seed_removal"] = dict(report["full_fe_stationary"])
    report["mhd_taylor_green"] = {"status": "pending", "reason": "Published periodic B/psi validation has not validated this conductor/exterior A/phi adaptation."}
    report["coupled_manufactured_solution"] = {"status": "pending", "reason": "Full thermal/velocity/pressure/A/phi manufactured-solution convergence study is outstanding."}
    report["exterior_refinement"] = {"status": "pending", "reason": "Conductor/exterior mesh and enclosure-distance convergence are outstanding."}
    report["subgrid_total_energy"] = {"status": "pending", "reason": "Resolved plus modeled subgrid MHD total-energy budget has not been validated."}
    if include_fenics:
        base = replace(ScenarioConfig(), resolution=3, time_steps=2, duration=.02,
                       initial_temperature=1820., heating_power=0., coil_current=0., pump_force=0., turbulence=False)
        try:
            stationary = run_simulation(base, lambda f, message: report_progress(.15 + .15 * f, message))
            mask = np.asarray(stationary.metadata["conductor_mask"], dtype=bool)
            temp_error = float(np.max(np.abs(stationary.fields["temperature"][:, mask] - base.initial_temperature)))
            velocity_error = float(np.max(np.linalg.norm(stationary.fields["velocity"], axis=-1)))
            report["full_fe_stationary"] = {"status": "passed" if temp_error < 1.e-6 and velocity_error < 1.e-9 else "failed", "temperature_error_k": temp_error, "velocity_error_m_s": velocity_error, "artifact": stationary.metadata["weak_operator_path"], "config": base.to_dict()}
            driven_config = replace(base, duration=.1, time_steps=3, initial_temperature=1810.2,
                                     heating_power=1.e6, coil_current=10000., pump_force=1., turbulence=True)
            driven = run_simulation(driven_config, lambda f, message: report_progress(.30 + .2 * f, message))
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            predictions = {key: torch.tensor(value[:2], dtype=torch.float64, device=device, requires_grad=True) for key, value in driven.fields.items()}
            residuals = evaluate_weak_residual_torch(driven_config, driven, predictions, [0, 1])
            defects = {key: float(value.detach().abs().max().cpu()) for key, value in residuals.items() if value.numel()}
            sum(value.square().mean() for value in residuals.values() if value.numel()).backward()
            gradients_finite = all(value.grad is None or bool(torch.isfinite(value.grad).all()) for value in predictions.values())
            worst = max(defects[key] for key in ("momentum", "continuity", "induction", "charge", "enthalpy", "magnetic_consistency"))
            energy_error = float(np.max(np.abs(driven.diagnostics["energy_balance_error_j"])))
            report["full_fe_driven_vms"] = {"status": "passed" if worst < 1.e-7 and energy_error < .01 and gradients_finite else "failed", "dimensionless_weak_defects": defects, "gradients_finite": gradients_finite,
                                           "thermal_energy_defect_j": energy_error, "mesh_volume_relative_error": driven.metadata["volume_relative_error"],
                                           "artifact": driven.metadata["weak_operator_path"], "config": driven_config.to_dict(), "note": "Discrete equation agreement checks implementation; it does not establish physical validation of turbulence."}
            dynamo_config = replace(driven_config, time_steps=4, initial_temperature=1820., heating_power=0.,
                                     mode="dynamo", pump_force=100., seed_duration=.02)
            dynamo = run_simulation(dynamo_config, lambda f, message: report_progress(.50 + .2 * f, message))
            energies = dynamo.diagnostics["magnetic_energy_j"]
            report["full_fe_seed_removal"] = {"status": "passed" if np.all(np.isfinite(energies)) and energies[-1] < energies[0] else "failed", "magnetic_energy_j": energies.tolist(),
                                             "times_s": dynamo.times.tolist(), "config": dynamo_config.to_dict(), "artifact": dynamo.metadata["weak_operator_path"],
                                             "note": "Low-Rm pumped scenario decays after seed removal; self-exciting dynamo operation is not asserted."}
        except Exception as exc:
            report["full_fe_execution_error"] = {"status": "failed", "reason": str(exc)}
        report_progress(.75, "Running analytical FE Hartmann and magnetic-diffusion reductions")
        worker = Path(__file__).resolve().parent.parent / "workers" / "fenics_worker.py"
        case_file = destination.with_name(destination.stem + "_fenics.json")
        if sys.platform == "win32":
            command = ["wsl", "-d", wsl_distribution(), "--", "/usr/bin/python3", wsl_path(worker), "--benchmarks", wsl_path(case_file)]
        else:
            command = [os.environ.get("MAGNETIC_TORUS_FE_PYTHON", "/usr/bin/python3"), str(worker), "--benchmarks", str(case_file)]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)
            cases_data = json.loads(case_file.read_text(encoding="utf-8"))
            report.update(cases_data)
        except Exception as exc:
            report["fenics_special_cases"] = {"status": "failed", "reason": str(exc)}
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report_progress(1., "Validation report saved; pending gates remain explicit")
    return report
