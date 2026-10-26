"""Execute one isolated scientific case and retain its evidence.

Each invocation imports only the runtimes needed for that case. Numerical
sources are produced by separate invocations before any PINN process starts.
"""
from __future__ import annotations

import argparse
from dataclasses import replace

import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))


class Checks:
    def __init__(self):
        self.records = []

    def add(self, name, condition, **evidence):
        self.records.append({"name": name, "passed": bool(condition), **evidence})

    @property
    def failed(self):
        return [row["name"] for row in self.records if not row["passed"]]


def _load_dependency(request, dependency_id, artifact="source_path"):
    from magnetic_torus.store import RunStore
    entry = request["dependencies"][dependency_id]
    if "result" in entry:
        entry = entry["result"]
    path = entry.get("artifacts", {}).get(artifact)
    if not path:
        raise ValueError(f"Dependency {dependency_id} did not produce {artifact}.")
    return RunStore.load_run(path) if artifact == "source_path" else RunStore.load_pinn(path)


def _config(parameters):
    from magnetic_torus.models import preset, ScenarioConfig
    recipe = parameters.get("recipe", "")
    base = preset(parameters.get("preset", "Iron"))
    if recipe == "preset":
        return base
    if recipe.startswith("fe_") or recipe == "transport":
        config = replace(ScenarioConfig(), duration=.05, time_steps=3, resolution=3,
                         heating_power=100000., initial_temperature=1820., coil_current=1000.,
                         pump_force=1., seed_duration=.02, turbulence=parameters.get("closure", False))
        if recipe == "fe_matrix":
            initial = {"solid": 1800., "mushy": 1810.2, "liquid": 1820.}[parameters["phase"]]
            config = replace(config, initial_temperature=initial, mode=parameters["mode"])
            if config.mode == "dynamo":
                config = replace(config, coil_current=0., pump_force=10.)
        elif recipe == "fe_melting":
            config = replace(config, initial_temperature=1808., duration=.4, time_steps=8,
                             heating_power=4.e7, charge_fraction=1., pump_force=0., coil_current=0.)
        elif recipe == "fe_freezing":
            config = replace(config, initial_temperature=1812., duration=.4, time_steps=8,
                             heating_power=0., discharge_power=4.e7, charge_fraction=0., pump_force=0., coil_current=0.)
        elif recipe == "fe_stationary":
            config = replace(config, heating_power=0., pump_force=0., coil_current=0., gravity=0.)
        elif recipe == "transport":
            field = parameters["field"]
            config = replace(config, major_radius=1.6, minor_radius=.2, habitat_radius=1.,
                             duration=.02, time_steps=2, heating_power=0., gravity=0.,
                             ground_enabled=parameters["ground"], turbulence=False,
                             coil_current=0. if field in {"zero", "dynamo"} else 10000.,
                             pump_force=1. if field in {"driven", "dynamo"} else 0.,
                             mode="dynamo" if field == "dynamo" else "driven", seed_duration=.01)
        return config.validate()
    config = replace(base, solver="thermal_benchmark", turbulence=False, major_radius=.3,
                     minor_radius=.06, habitat_radius=.12, duration=2., time_steps=12,
                     initial_temperature=base.melting_temperature - 20., heating_power=20000.,
                     discharge_power=0., charge_fraction=1., emissivity=0., coil_current=0.)
    changes = {
        "solid_closed": {"heating_power": 0.},
        "solid_heating": {},
        "liquid_cooling": {"initial_temperature": base.melting_temperature + 20., "heating_power": 0., "discharge_power": 20000., "charge_fraction": 0.},
        "melting": {"initial_temperature": base.melting_temperature - 1., "heating_power": 1.1e7, "duration": 4., "time_steps": 16},
        "freezing": {"initial_temperature": base.melting_temperature + 1., "heating_power": 0., "discharge_power": 1.1e7, "charge_fraction": 0., "duration": 4., "time_steps": 16},
        "radiating": {"initial_temperature": base.melting_temperature + 20., "heating_power": 0., "emissivity": .2, "duration": .2},
        "zero_latent": {"latent_heat": 0., "initial_temperature": base.melting_temperature - 1., "heating_power": 100000.},
        "charge_zero": {"heating_power": 1000., "discharge_power": 500., "charge_fraction": 0.},
        "charge_one": {"heating_power": 1000., "discharge_power": 500., "charge_fraction": 1.},
        "charge_roundoff": {"duration": 1., "time_steps": 10, "charge_fraction": .3, "heating_power": 1000., "discharge_power": 500.},
        "efficiency_zero": {"efficiency": 0.}, "efficiency_one": {"efficiency": 1.}, "material": {},
    }
    return replace(config, **changes[recipe]).validate()


def _solve(request, checks, progress, cancel):
    import numpy as np
    from magnetic_torus.physics import run_simulation
    from magnetic_torus.store import RunStore
    from magnetic_torus.geometry import conductor_mask
    parameters = request["parameters"]
    config = _config(parameters)
    result = run_simulation(config, progress, cancel)
    store = RunStore(Path(request["case_dir"]) / "store")
    path = store.save_run(result)
    mask = conductor_mask(result)
    temperature = result.fields["temperature"][:, mask]
    fraction = result.fields["liquid_fraction"][:, mask]
    checks.add("positive_kelvin", np.all(temperature > 0), minimum_k=float(temperature.min()))
    checks.add("liquid_fraction_bounds", np.all((fraction >= -1e-10) & (fraction <= 1 + 1e-10)))
    checks.add("complete_timeline", result.times[0] == 0. and result.times[-1] == config.duration and np.all(np.diff(result.times) > 0))
    energy = result.diagnostics["energy_j"]
    balance = float(np.max(np.abs(result.diagnostics["energy_balance_error_j"])))
    tolerance = max(.01, float(np.max(np.abs(energy))) * 1e-9)
    checks.add("assembled_energy_balance", balance <= tolerance, error_j=balance, tolerance_j=tolerance)
    recoverable = result.diagnostics["recoverable_energy_j"]
    usable = result.diagnostics["usable_electric_energy_j"]
    checks.add("efficiency_accounting", np.allclose(usable, config.efficiency * recoverable, atol=.01, rtol=1e-12))
    recipe = parameters.get("recipe", "")
    if config.solver == "thermal_benchmark":
        volume = np.asarray(result.metadata["node_volumes"])
        mass = config.density * volume
        enthalpy = result.fields["enthalpy"]
        expected_t = np.where(enthalpy < 0, config.melting_temperature + enthalpy / config.heat_capacity_solid,
                              np.where(enthalpy <= config.latent_heat, config.melting_temperature,
                                       config.melting_temperature + (enthalpy - config.latent_heat) / config.heat_capacity_liquid))
        checks.add("sharp_constitutive_law", np.allclose(result.fields["temperature"], expected_t, rtol=1e-12, atol=1e-9))
        input_j = config.duration * (config.charge_fraction * config.heating_power - (1 - config.charge_fraction) * config.discharge_power)
        radiation_j = float(result.diagnostics["radiated_energy_j"][-1])
        defect = abs(float(energy[-1] - energy[0]) - input_j + radiation_j)
        checks.add("source_and_loss_integral", defect <= max(.01, abs(input_j) * 1e-9), error_j=defect, scheduled_input_j=input_j, radiated_j=radiation_j)
        checks.add("torus_volume", abs(float(volume.sum()) - config.volume) <= config.volume * 1e-12)
        checks.add("native_energy_weights", np.allclose(energy, enthalpy @ mass, atol=.01, rtol=1e-12))
        if recipe == "solid_closed":
            checks.add("closed_state_constant", np.max(np.abs(temperature - config.initial_temperature)) < 1e-8)
        if recipe in {"liquid_cooling", "radiating"}:
            checks.add("cooling_direction", float(temperature[-1].mean()) < float(temperature[0].mean()))
        if recipe == "solid_heating":
            checks.add("heating_direction", float(temperature[-1].mean()) > float(temperature[0].mean()))
    else:
        # FE energy is assembled with the worker's nonlinear quadrature, not a
        # nodal lumped-H approximation in the mushy region.
        checks.add("positive_fe_volume", float(result.metadata["mesh_volume_m3"]) > 0.)
        parts = sum(result.diagnostics[key] for key in ("imposed_field_energy_j", "induced_field_energy_j", "mutual_field_energy_j"))
        checks.add("magnetic_energy_decomposition", np.allclose(parts, result.diagnostics["magnetic_energy_j"], rtol=1e-10, atol=1e-8))
        pumping = np.r_[0., np.cumsum(np.diff(result.times) * result.diagnostics["pump_power_w"][1:])]
        checks.add("pumping_work", np.allclose(pumping, result.diagnostics["pump_energy_j"], rtol=1e-10, atol=1e-8))
        boundary = result.metadata.get("pinn_physics", {}).get("bc_nodes", {})
        no_slip = np.asarray(boundary.get("0", ()), dtype=int)
        checks.add("no_slip_boundaries", len(no_slip) > 0 and np.max(np.abs(result.fields["velocity"][:, no_slip])) < 1e-9)
        imposed = result.diagnostics["imposed_field_energy_j"]
        if config.mode == "dynamo" and config.duration >= config.seed_duration:
            checks.add("seed_removal_event", bool(np.any(np.isclose(result.times, config.seed_duration, rtol=0, atol=1e-12))))
            checks.add("imposed_seed_removed", abs(float(imposed[-1])) < 1e-12, final_imposed_energy_j=float(imposed[-1]))
        elif config.mode == "driven":
            checks.add("constant_independent_imposed_coil", np.allclose(imposed, imposed[0], rtol=1e-12, atol=1e-12))
        if recipe == "fe_matrix":
            initial_fraction = float(fraction[0].mean())
            phase = parameters["phase"]
            checks.add("requested_initial_phase", (initial_fraction < 1e-12 if phase == "solid" else
                       initial_fraction > 1 - 1e-12 if phase == "liquid" else 0 < initial_fraction < 1),
                       phase=phase, initial_liquid_fraction=initial_fraction)
        _weak_checks(result, checks, progress)
        if recipe == "fe_stationary":
            checks.add("stationary_temperature", float(np.max(np.abs(temperature - config.initial_temperature))) < 1e-6)
            checks.add("stationary_velocity", float(np.max(np.abs(result.fields["velocity"]))) < 1e-9)
    if recipe in {"melting", "fe_melting"}:
        checks.add("melting_front_evolves", float(fraction[-1].mean()) > float(fraction[0].mean()) + 1e-6)
    if recipe in {"freezing", "fe_freezing"}:
        checks.add("freezing_front_evolves", float(fraction[-1].mean()) < float(fraction[0].mean()) - 1e-6)
    return {"source_path": str(path)}, {"configuration": config.to_dict(), "configuration_hash": config.config_hash,
                                            "physics_hash": config.physics_hash, "seed": config.seed,
                                            "snapshots": len(result.times), "points": len(result.points),
                                            "initial_temperature_k": float(temperature[0].mean()), "final_temperature_k": float(temperature[-1].mean()),
                                            "energy_change_j": float(energy[-1] - energy[0])}


def _weak_checks(result, checks, progress):
    import numpy as np
    import torch
    from magnetic_torus.physics import evaluate_weak_residual_torch
    selected = np.unique(np.linspace(1, len(result.times) - 1, min(4, len(result.times) - 1), dtype=int))
    worst = {}
    for index in selected:
        values = {key: torch.tensor(value[index - 1:index + 1], dtype=torch.float64, requires_grad=True) for key, value in result.fields.items()}
        components = evaluate_weak_residual_torch(result.config, result, values, [index - 1, index])
        for key, value in components.items():
            if value.numel():
                worst[key] = max(worst.get(key, 0.), float(value.detach().abs().max()))
        sum(value.square().mean() for value in components.values() if value.numel()).backward()
        checks.add(f"weak_gradient_{index}", all(value.grad is None or bool(torch.isfinite(value.grad).all()) for value in values.values()))
        progress(.95, f"Independent weak-operator snapshot {index}")
    for name, defect in worst.items():
        checks.add("weak_" + name, math.isfinite(defect) and defect <= 1e-7, normalized_max=defect, limit=1e-7)


def _references(request, checks, progress, cancel):
    import numpy as np
    from magnetic_torus.models import ScenarioConfig
    from magnetic_torus.physics import stefan_front_benchmark, coil_magnetic_field
    errors = []
    for count in (32, 64, 128):
        row = stefan_front_benchmark(count, count * 16)
        errors.append(float(row["relative_front_error"]))
    checks.add("stefan_front_reference", max(errors) < .005, relative_errors=errors, limit=.005)
    config = replace(ScenarioConfig(), coil_current=1000.)
    from magnetic_torus.physics import imposed_coil, MU0
    coil = imposed_coil(config)
    point = np.array([[0., 0., .4]])
    observed = coil_magnetic_field(point, config)[0, 2]
    expected = MU0 * coil["ampere_turns"] * coil["radius_m"] ** 2 / (2 * (coil["radius_m"] ** 2 + .4 ** 2 + coil["wire_radius_m"] ** 2) ** 1.5)
    checks.add("coil_axis_reference", abs(observed / expected - 1) < 1e-10, observed_t=observed, expected_t=expected)
    return {}, {"stefan_relative_errors": errors}


def _fe_references(request, checks, progress, cancel):
    from magnetic_torus.backend_environment import wsl_path, wsl_distribution
    path = Path(request["case_dir"]) / "fe-references.json"
    worker = ROOT / "magnetic_torus" / "workers" / "fenics_worker.py"
    interpreter = os.environ.get("MAGNETIC_TORUS_FE_PYTHON", "/usr/bin/python3")
    if sys.platform == "win32":
        command = ["wsl", "-d", wsl_distribution(), "--", interpreter, wsl_path(worker), "--benchmarks", wsl_path(path)]
    else:
        command = [interpreter, str(worker), "--benchmarks", str(path)]
    completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)
    print(completed.stdout, flush=True)
    if completed.returncode:
        raise RuntimeError(completed.stderr[-4000:])
    report = json.loads(path.read_text(encoding="utf-8"))
    for name in ("hartmann", "magnetic_diffusion"):
        checks.add(name, report[name]["status"] == "passed", report=report[name])
    return {"reference_report": str(path)}, {"broader_pending_gates": {key: value for key, value in report.items() if isinstance(value, dict) and value.get("status") == "pending"}}


def _pinn(request, checks, progress, cancel):
    from magnetic_torus.pinn import train_pinn, validate_pinn_source
    from magnetic_torus.store import RunStore
    parameters = request["parameters"]
    source = _load_dependency(request, parameters["source"])
    source = replace(source, run_id="training-" + uuid.uuid4().hex,
                     config=replace(source.config, pinn_epochs=parameters["epochs"], pinn_batch_size=parameters["batch"]),
                     metadata={**source.metadata, "reused_fields_from": source.run_id})
    os.environ["MAGNETIC_TORUS_PINN_DEVICE"] = parameters["device"]
    import torch
    torch.set_num_threads(2)
    store = RunStore(Path(request["case_dir"]) / "store")
    source_path = store.save_run(source)
    original = {key: value.copy() for key, value in source.fields.items()}
    model = train_pinn(source, progress, cancel, output_dir=Path(request["case_dir"]) / "training")
    validate_pinn_source(model, source)
    path = store.save_pinn(model)
    roundtrip = store.load_pinn(path)
    validate_pinn_source(roundtrip, source)
    import numpy as np
    checks.add("source_immutable", all(np.array_equal(original[key], source.fields[key]) and not source.fields[key].flags.writeable for key in original))
    checks.add("finite_fields", all(np.isfinite(value).all() for value in model.fields.values()))
    checks.add("checkpoint_written", bool(model.metadata.get("checkpoint_path")) and Path(model.metadata["checkpoint_path"]).is_file())
    checks.add("heldout_and_conservation_gates", model.metadata.get("validation_status") == "passed",
               criteria=model.metadata.get("validation_criteria"), metrics=dict(model.metrics),
               missing_labels=list(model.metadata.get("fields_without_heldout_labels", ())))
    checks.add("requested_device", str(model.metadata.get("device")) == parameters["device"])
    return {"source_path": str(source_path), "pinn_path": str(path), "checkpoint_path": str(model.metadata["checkpoint_path"])}, {
        "configuration": source.config.to_dict(), "seed": source.config.seed, "metrics": dict(model.metrics),
        "model_validation_status": model.metadata.get("validation_status"), "actual_epochs": model.metrics.get("epochs"),
        "normalization_fitted_on": model.metadata.get("normalization_fitted_on")}


def _particle_config(source, parameters, directory):
    species = {"proton": (1, 1), "alpha": (2, 4), "iron": (26, 56), "mixed": (1, 1)}
    z, a = species[parameters["species"]]
    changes = {"particle_z": z, "particle_a": a, "particle_energy_mev_nucleon": parameters["energy"],
               "source_kind": parameters["source"], "particle_count": 32,
               "radiation_histories": parameters.get("histories", 128), "target": parameters.get("target", "water"),
               "transport_physics_list": parameters.get("physics_list", "Shielding"), "incident_fluence_m2": 1.}
    if parameters["source"] == "spectrum":
        bins = [{"z": z, "a": a, "energy_mev_nucleon": parameters["energy"], "weight": 1.}]
        if parameters["species"] == "mixed":
            bins = [{"z": zz, "a": aa, "energy_mev_nucleon": energy, "weight": weight} for zz, aa, energy, weight in
                    ((1, 1, 100., .4), (2, 4, 200., .3), (26, 56, 1000., .3))]
        path = directory / "spectrum.json"
        path.write_text(json.dumps({"schema_version": 1, "energy_unit": "MeV/nucleon", "angular_distribution": parameters.get("spectrum_angular", "beam"),
                                    "provenance": "Deterministic synthetic coverage spectrum; not a mission exposure model.", "bins": bins}), encoding="utf-8")
        changes["spectrum_path"] = str(path)
    return replace(source.config, **changes).validate()


def _radiation_source(request, parameters, directory):
    from magnetic_torus.models import RunResult, ScenarioConfig
    from simulation_suite.scientific_catalog import transport_source
    import numpy as np
    field = parameters["field"]
    if parameters["action"] == "boris" and field in {"zero", "imposed"}:
        config = replace(ScenarioConfig(), solver="thermal_benchmark", turbulence=False, major_radius=1.6,
                         minor_radius=.2, habitat_radius=1., ground_enabled=parameters["ground"],
                         coil_current=0. if field == "zero" else 10000.)
        source = RunResult("analytic-" + uuid.uuid4().hex, config, np.array([0.]), np.zeros((1, 3)), {}, {},
                           {"backend": "Analytic imposed-coil trajectory fixture; not an MHD solution"})
        return replace(source, config=_particle_config(source, parameters, directory))
    source_id = transport_source("zero" if field == "pinn" else field, parameters["ground"])
    if field == "pinn":
        model_id = f"pinn.transport.zero.{'ground' if parameters['ground'] else 'vacuum'}"
        # The PINN source records its actual training settings, not the earlier
        # numerical configuration used before training.
        source = _load_dependency(request, model_id)
        model = _load_dependency(request, model_id, "pinn_path")
        if model.metadata.get("validation_status") != "passed":
            raise ValueError("The prerequisite PINN did not pass; no substitute field is allowed.")
    else:
        source = _load_dependency(request, source_id)
        model = None
    config = _particle_config(source, parameters, directory)
    if config.physics_hash != source.config.physics_hash:
        raise ValueError("Radiation observer configuration cannot change accepted source physics.")
    if config.config_hash != source.config.config_hash:
        origin = source
        source = replace(source, run_id="observation-" + uuid.uuid4().hex, config=config,
                         metadata={**source.metadata, "reused_fields_from": source.run_id})
        if model is not None:
            model = replace(model, source_run_id=source.run_id, config_hash=config.config_hash,
                            metadata={**model.metadata, "explicit_prior_model_reuse": True,
                                      "origin_run_id": origin.run_id, "origin_config_hash": origin.config.config_hash,
                                      "source_run_id": source.run_id, "source_config_hash": config.config_hash})
    if model is not None:
        from magnetic_torus.radiation import field_source_run
        source = field_source_run(source, model, directory / "pinn-field")
    return source


def _boris_oracle(parameters, checks):
    import numpy as np
    from magnetic_torus.radiation import C, MEV, ELEMENTARY_CHARGE, PROTON_MASS, ATOMIC_MASS, relativistic_boris
    z, a = {"proton": (1, 1), "alpha": (2, 4), "iron": (26, 56)}[parameters["species"]]
    mass = PROTON_MASS if a == 1 else a * ATOMIC_MASS
    gamma = 1 + parameters["energy"] * a * MEV / (mass * C ** 2)
    speed = C * math.sqrt(1 - gamma ** -2)
    omega = z * ELEMENTARY_CHARGE / (gamma * mass)
    radius = speed / omega
    dt = .01 / omega
    momentum = np.array([[gamma * mass * speed, 0., 0.]])
    position = np.zeros((1, 3))
    original = float(np.linalg.norm(momentum))
    radius_error = 0.
    for _ in range(640):
        old = momentum.copy()
        momentum = relativistic_boris(momentum, np.array([z * ELEMENTARY_CHARGE]), np.array([mass]), np.array([[0., 0., 1.]]), dt)
        position += .5 * (old + momentum) / (gamma * mass) * dt
        radius_error = max(radius_error, abs(float(np.hypot(position[0, 0], position[0, 1] + radius)) - radius) / radius)
    checks.add("relativistic_momentum_norm", abs(float(np.linalg.norm(momentum)) / original - 1) < 1e-10)
    checks.add("uniform_field_gyro_radius", radius_error < 1e-6, relative_radius_error=radius_error)


def _boris(request, checks, progress, cancel):
    import numpy as np
    from magnetic_torus.radiation import trace_particles
    p = request["parameters"]
    source = _radiation_source(request, p, Path(request["case_dir"]))
    trace = trace_particles(source, len(source.times) - 1, progress, cancel)
    checks.add("identical_incident_samples", np.array_equal(trace.paths[:, 0], trace.baseline_paths[:, 0]))
    checks.add("finite_paths_and_times", np.isfinite(trace.paths).all() and np.isfinite(trace.baseline_paths).all() and np.isfinite(trace.times).all())
    checks.add("physical_flight_clock", np.all(np.diff(trace.times, axis=1) > 0))
    checks.add("outcome_partition", len(trace.outcomes) == source.config.particle_count and set(trace.outcomes) <= {"hardware", "habitat", "ground", "escaped", "window_end"})
    if p["field"] == "zero":
        checks.add("zero_field_pair_identity", np.array_equal(trace.paths, trace.baseline_paths) and trace.outcomes == trace.baseline_outcomes)
    checks.add("accepted_source_identity", trace.source_run_id == source.run_id and trace.config_hash == source.config.config_hash)
    if p["source"] == "spectrum":
        checks.add("versioned_spectrum_provenance", len(trace.metadata["source"].get("spectrum_sha256", "")) == 64)
    _boris_oracle(p, checks)
    output = Path(request["case_dir"]) / "trajectories.npz"
    np.savez_compressed(output, paths=trace.paths, baseline_paths=trace.baseline_paths, times=trace.times,
                        species=trace.species, energy_mev_nucleon=trace.energies_mev_nucleon)
    return {"trajectories": str(output)}, {"configuration": source.config.to_dict(), "seed": source.config.seed,
                                           "outcomes": {name: trace.outcomes.count(name) for name in sorted(set(trace.outcomes))},
                                           "field_sha256": trace.metadata["field_sha256"], "field_time_s": trace.metadata["field_time_s"]}


def _dose_checks(dose, source, checks, prefix=""):
    import numpy as np
    baseline = np.asarray(dose.metadata["field_off_dose_gy"])
    checks.add(prefix + "finite_nonnegative_dose", np.isfinite(dose.dose_gy).all() and np.all(dose.dose_gy >= 0) and np.isfinite(dose.standard_error_gy).all() and np.all(dose.standard_error_gy >= 0))
    masses = np.asarray(dose.metadata["organ_masses_kg"])
    checks.add(prefix + "positive_scoring_masses", np.isfinite(masses).all() and np.all(masses > 0))
    expected_mass = {"water": (63.999, 64.001), "male": (70., 76.), "female": (58., 62.)}[source.config.target]
    checks.add(prefix + "target_mass_reference", expected_mass[0] < float(masses.sum()) < expected_mass[1], total_mass_kg=float(masses.sum()))
    if source.config.target != "water":
        checks.add(prefix + "anatomical_regions", len(dose.organ_names) > 100, regions=len(dose.organ_names))
    checks.add(prefix + "immutable_field_source", dose.source_run_id == source.run_id and dose.config_hash == source.config.config_hash)
    checks.add(prefix + "secondary_contribution", np.all(np.asarray(dose.metadata["secondary_dose_gy"]) <= dose.dose_gy + 1e-30))
    reported = np.asarray(dose.metadata["statistical_precision_met"], dtype=bool)
    expected = (dose.dose_gy > 0) & (dose.standard_error_gy <= .05 * dose.dose_gy)
    checks.add(prefix + "honest_uncertainty_gate", np.array_equal(reported, expected))
    if source.config.coil_current == 0 and source.config.mode != "dynamo" and source.metadata.get("field_source") != "PINN":
        from magnetic_torus.radiation import frozen_field
        field = frozen_field(source, len(source.times) - 1)
        if np.max(np.abs(field.cell_values)) == 0:
            checks.add(prefix + "exact_zero_field_pairs", np.array_equal(dose.dose_gy, baseline))


def _dose(request, checks, progress, cancel):
    import numpy as np
    from magnetic_torus.radiation import calculate_dose
    from magnetic_torus.store import RunStore
    p = request["parameters"]
    source = _radiation_source(request, p, Path(request["case_dir"]))
    dose = calculate_dose(source, len(source.times) - 1, progress, cancel, Path(request["case_dir"]))
    _dose_checks(dose, source, checks)
    store = RunStore(Path(request["case_dir"]) / "store")
    path = store.save_dose(dose)
    return {"dose_path": str(path)}, {"configuration": source.config.to_dict(), "seed": source.config.seed,
                                      "histories": dose.histories, "max_dose_gy": float(dose.dose_gy.max()),
                                      "nonzero_regions": int(np.count_nonzero(dose.dose_gy)),
                                      "precision_regions": int(sum(dose.metadata["statistical_precision_met"])),
                                      "regions": len(dose.organ_names), "dose_gy": dose.dose_gy.tolist(),
                                      "standard_error_gy": dose.standard_error_gy.tolist()}


def _dose_reference(request, checks, progress, cancel):
    import numpy as np
    from magnetic_torus.radiation import calculate_dose
    from magnetic_torus.store import RunStore
    directory = Path(request["case_dir"])
    p = {**request["parameters"], "field": "zero", "ground": False, "species": "proton", "energy": 100.,
         "source": "beam", "target": "water", "physics_list": "Shielding"}
    source = _radiation_source(request, p, directory)
    attempts = []
    counts = [p["histories"]] if p["action"] == "dose_linearity" else [512, 2048, 8192, 32768, 131072, p["maximum_histories"]]
    store = RunStore(directory / "store")
    for count in dict.fromkeys(counts):
        source = replace(source, run_id="observation-" + uuid.uuid4().hex,
                         config=replace(source.config, radiation_histories=count),
                         metadata={**source.metadata, "reused_fields_from": source.run_id})
        dose = calculate_dose(source, len(source.times) - 1, progress, cancel, directory)
        relative = float(dose.standard_error_gy[0] / dose.dose_gy[0]) if dose.dose_gy[0] > 0 else None
        attempts.append({"histories": count, "relative_standard_error": relative, "dose_gy": float(dose.dose_gy[0]), "standard_error_gy": float(dose.standard_error_gy[0])})
        path = store.save_dose(dose)
        if p["action"] == "dose_linearity" or (relative is not None and relative <= .05):
            break
    _dose_checks(dose, source, checks)
    checks.add("nonzero_water_reference", bool(dose.dose_gy[0] > 0))
    if p["action"] == "dose_precision":
        checks.add("water_precision_5_percent", relative is not None and relative <= .05, relative_standard_error=relative, histories=dose.histories, limit=.05)
    else:
        doubled = replace(source, run_id="observation-" + uuid.uuid4().hex,
                          config=replace(source.config, incident_fluence_m2=2.),
                          metadata={**source.metadata, "reused_fields_from": source.run_id})
        other = calculate_dose(doubled, len(doubled.times) - 1, progress, cancel, directory)
        checks.add("linear_fluence_dose", np.allclose(other.dose_gy, 2 * dose.dose_gy, rtol=1e-12, atol=0))
        checks.add("linear_fluence_uncertainty", np.allclose(other.standard_error_gy, 2 * dose.standard_error_gy, rtol=1e-12, atol=0))
    return {"dose_path": str(path)}, {"attempts": attempts, "configuration": source.config.to_dict(), "seed": source.config.seed}


def _invalid(request, checks, progress, cancel):
    from contextlib import redirect_stdout
    from io import StringIO
    import numpy as np
    from magnetic_torus.models import RunResult, PINNResult, ScenarioConfig
    from magnetic_torus.radiation import calculate_dose, field_source_run, frozen_field
    directory = Path(request["case_dir"])
    recipe = request["parameters"]["recipe"]
    source = _load_dependency(request, next(iter(request["dependencies"]))) if request["dependencies"] else None
    if recipe == "native_dose":
        source = RunResult("native-no-map", replace(ScenarioConfig(), solver="thermal_benchmark", turbulence=False), np.array([0.]), np.zeros((1, 3)), {}, {}, {})
        def operation():
            return calculate_dose(source, 0, progress, cancel, directory)
        expected = "no exterior magnetic field"
    elif recipe == "failed_pinn":
        from magnetic_torus.worker import execute
        from magnetic_torus.store import RunStore
        store = RunStore(directory / "rejection-store")
        source_path = store.save_run(source)
        model = PINNResult("rejected-model", source.run_id, source.config.config_hash,
                           source.fields, {}, {}, {"validation_status": "failed"})
        pinn_path = store.save_pinn(model)
        worker_request = directory / "rejected-request.json"
        worker_request.write_text(json.dumps({"action": "trace", "config": source.config.to_dict(),
                                              "source_path": str(source_path), "pinn_path": str(pinn_path),
                                              "store": str(store.root), "snapshot_index": 0}), encoding="utf-8")
        captured = StringIO()
        with redirect_stdout(captured):
            code = execute(worker_request)
        events = [json.loads(line) for line in captured.getvalue().splitlines() if line.strip()]
        checks.add("failed_pinn_rejected_before_transport", code == 1 and any(
            event.get("event") == "error" and "did not pass" in event.get("message", "") for event in events),
            exit_code=code, events=events)
        checks.add("no_fallback_publication", not any(event.get("event") == "complete" for event in events))
        return {}, {"rejection_case": recipe}
    elif recipe == "stale_pinn":
        model = PINNResult("rejected-model", "stale",
                           source.config.config_hash, source.fields, {}, {}, {"validation_status": "failed"})
        def operation():
            return field_source_run(source, model, directory)
        expected = "another scenario/run"
    elif recipe == "small_anatomy":
        from scipy.spatial import Delaunay
        from magnetic_torus.artifacts import FieldArtifact
        # An independently generated exact zero field isolates the anatomical
        # geometry guard. It makes no claim of solving a coupled FE scenario.
        points = np.array([[x, y, z] for x in (-8., 8.) for y in (-8., 8.) for z in (-8., 8.)])
        field_path = directory / "zero-field.npz"
        np.savez(field_path, points=points, cells=Delaunay(points).simplices,
                 times=np.array([0.]), vector_potential=np.zeros((1, 8, 3)), magnetic_field=np.zeros((1, 8, 3)))
        config = replace(ScenarioConfig(), target="male")
        source = RunResult("small-phantom-habitat", config, np.array([0.]), points, {}, {},
                           {"exterior_radius_m": 7.2, "validation_status": "Exact zero-field diagnostic fixture"},
                           artifacts={"exterior_field_path": FieldArtifact.capture(field_path, root=directory)})
        def operation():
            return calculate_dose(source, 0, progress, cancel, directory)
        expected = "does not fit inside"
    else:
        field = frozen_field(source, len(source.times) - 1)
        def operation():
            return field(np.array([[100 * field.radius, 0., 0.]]))
        expected = "outside the accepted exterior FE mesh"
    try:
        operation()
    except (ValueError, RuntimeError) as error:
        checks.add("explicit_rejection", expected in str(error), reason=str(error), expected=expected)
    else:
        checks.add("explicit_rejection", False)
    return {}, {"rejection_case": recipe}


HANDLERS = {"solve": _solve, "references": _references, "fe_references": _fe_references, "pinn": _pinn,
            "boris": _boris, "dose": _dose, "dose_precision": _dose_reference, "dose_linearity": _dose_reference, "invalid": _invalid}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args(argv)
    request = json.loads(args.request.read_text(encoding="utf-8"))
    directory = Path(request["case_dir"])
    directory.mkdir(parents=True, exist_ok=True)
    os.environ["MAGNETIC_TORUS_RUNS"] = str(directory / "worker_artifacts")
    os.environ["MAGNETIC_TORUS_DATA"] = str(directory / "history")
    cancel = threading.Event()
    finished = threading.Event()

    def watch():
        while not finished.wait(.1):
            if (directory / "cancel").exists():
                cancel.set()
    threading.Thread(target=watch, daemon=True).start()

    def progress(fraction, message):
        if cancel.is_set():
            from magnetic_torus.models import SimulationCancelled
            raise SimulationCancelled("Scientific case cancelled")
        print(json.dumps({"progress": round(float(fraction), 4), "message": message}), flush=True)

    start = time.monotonic()
    checks = Checks()
    result = {"case_id": request.get("case_id", request.get("id")), "execution": "FAIL", "qualification": "NOT_APPLICABLE", "artifacts": {}}
    try:
        artifacts, metrics = HANDLERS[request["parameters"]["action"]](request, checks, progress, cancel)
        scientific = request["parameters"]["action"] not in {"boris", "dose", "invalid"}
        result.update(execution="PASS", qualification=("FAIL" if checks.failed else "PASS") if scientific else "NOT_APPLICABLE",
                      status="FAIL" if checks.failed else "PASS", reason="; ".join(checks.failed), artifacts=artifacts, metrics=metrics, checks=checks.records)
        exit_code = 3 if checks.failed else 0
    except Exception as error:
        result.update(status="ERROR", reason=str(error), traceback=traceback.format_exc(), checks=checks.records)
        print(result["traceback"], flush=True)
        exit_code = 130 if cancel.is_set() else 1
    finally:
        finished.set()
    result["elapsed_s"] = time.monotonic() - start
    from magnetic_torus.store import json_default
    args.result.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.result.with_suffix(".partial")
    temporary.write_text(json.dumps(result, indent=2, default=json_default, allow_nan=False), encoding="utf-8")
    os.replace(temporary, args.result)
    print(json.dumps({"status": result["status"], "reason": result["reason"], "elapsed_s": result["elapsed_s"]}), flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
