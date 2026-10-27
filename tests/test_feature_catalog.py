"""Independent representative-feature coverage, source identity and precision gates."""

from dataclasses import replace
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from magnetic_torus.models import DoseResult, RunResult, ScenarioConfig
from simulation_suite.catalog import all_cases, dependency_order
from simulation_suite.feature_catalog import (
    REQUIRED_FEATURES,
    feature_coverage,
)
from simulation_suite.runner import case_fingerprint
from simulation_suite import scientific
from simulation_suite.scientific_catalog import scientific_cases

ROOT = Path(__file__).resolve().parents[1]


class FeatureCatalogTests(unittest.TestCase):
    def setUp(self):
        self.cases, self.coverage = all_cases(ROOT, profile="features")
        self.by_id = {case.id: case for case in self.cases}

    def test_default_is_complete_representative_features_and_keeps_every_regression_module(
        self,
    ):
        cases, report = all_cases(ROOT)
        self.assertEqual([case.id for case in cases], [case.id for case in self.cases])
        proof = report["feature_matrix"]
        self.assertTrue(proof["complete"], proof)
        self.assertEqual(set(proof["covered_features"]), set(REQUIRED_FEATURES))
        self.assertEqual(proof["missing_features"], [])
        expected_units = {
            "unit." + path.stem for path in (ROOT / "tests").glob("test_*.py")
        }
        self.assertEqual(
            {case.id for case in cases if case.id.startswith("unit.")}, expected_units
        )
        self.assertTrue(proof["unselected_qualification"])
        self.assertNotIn("scientific_matrix", report)
        self.assertEqual(cases[-1].id, "feature.dose_water")
        self.assertFalse(
            any(
                case.id.startswith("preset.") or case.id.startswith("integration.")
                for case in cases
            )
        )

    def test_pressure_diagnostic_regression_tracks_its_script(self):
        case = self.by_id["unit.test_pressure_diagnostics"]
        self.assertIn("scripts/pinn_pressure_diagnostics.py", case.dependencies)
        pilot = self.by_id["unit.test_pinn_pressure_pilots"]
        self.assertIn("scripts/pinn_research.py", pilot.dependencies)
        self.assertIn("scripts/research_supervisor.py", pilot.dependencies)
        publication = self.by_id["unit.test_research_publication"]
        self.assertIn("scripts/pinn_research.py", publication.dependencies)
        self.assertIn("scripts/research_supervisor.py", publication.dependencies)
        parity = self.by_id["unit.test_pinn_fe_vjp"]
        self.assertIn("scripts/verify_pinn_fe_vjp.py", parity.dependencies)
        self.assertIn("scripts/profile_pinn_fe_vjp.py", parity.dependencies)

    def test_missing_duplicate_mislabeled_and_disconnected_providers_cannot_claim_coverage(
        self,
    ):
        required_ids = {
            name for providers in REQUIRED_FEATURES.values() for name in providers
        }
        for name in required_ids:
            with self.subTest(missing=name):
                proof = feature_coverage(
                    [case for case in self.cases if case.id != name], root=ROOT
                )
                self.assertFalse(proof["complete"])
                self.assertTrue(proof["missing_features"])
        self.assertFalse(
            feature_coverage(self.cases + [self.cases[0]], root=ROOT)["complete"]
        )
        water = self.by_id["feature.dose_water"]
        for malformed in (
            replace(water, kind="functional"),
            replace(water, parameters={**water.parameters, "histories": 128}),
            replace(water, capabilities=()),
            replace(water, depends_on=()),
            replace(water, depends_on=(water.id,)),
        ):
            with self.subTest(malformed=malformed):
                proof = feature_coverage(
                    [malformed if case.id == water.id else case for case in self.cases],
                    root=ROOT,
                )
                self.assertFalse(proof["complete"])
        performance = self.by_id["desktop.performance"]
        changed = replace(performance, parameters={"minimum_fps": 1.0})
        self.assertFalse(
            feature_coverage(
                [changed if case.id == performance.id else case for case in self.cases],
                root=ROOT,
            )["complete"]
        )
        unit = self.by_id["unit.test_configuration_matrix"]
        excluded = replace(
            unit, parameters={**unit.parameters, "exclude_ids": ["hidden.regression"]}
        )
        self.assertFalse(
            feature_coverage(
                [excluded if case.id == unit.id else case for case in self.cases],
                root=ROOT,
            )["complete"]
        )

    def test_full_matrix_and_fast_profile_are_retained_and_duplicate_heavy_work_is_explicitly_excluded(
        self,
    ):
        full, full_report = all_cases(ROOT, profile="full")
        legacy = {case.id: case for case in scientific_cases()}
        self.assertTrue(full_report["scientific_matrix"]["complete"])
        self.assertEqual({case.id for case in full if case.id in legacy}, set(legacy))
        self.assertIn("preset.lunar-habitat", {case.id for case in full})
        full_radiation = next(
            case for case in full if case.id == "unit.test_torus_radiation"
        )
        self.assertEqual(full_radiation.parameters["exclude_ids"], [])
        excluded = self.by_id["unit.test_torus_radiation"].parameters["exclude_ids"]
        self.assertEqual(
            set(excluded),
            {
                "test_torus_radiation.Geant4IntegrationTests.test_water_paired_zero_field_transport",
                "test_torus_radiation.Geant4IntegrationTests.test_ground_water_transport_uses_above_ground_incident_source",
                "test_torus_radiation.Geant4IntegrationTests.test_male_icrp110_transport",
                "test_torus_radiation.Geant4IntegrationTests.test_female_icrp110_transport",
            },
        )
        fast, _ = all_cases(ROOT, profile="fast")
        self.assertNotIn("unit.test_workspace_features", {case.id for case in fast})
        self.assertFalse(any(case.id.startswith("feature.") for case in fast))
        with self.assertRaises(ValueError):
            all_cases(ROOT, profile="typo")

    def test_sources_are_shared_and_pinn_and_precision_qualification_remain_scientific(
        self,
    ):
        self.assertEqual(
            self.by_id["feature.native_pinn"].depends_on, ("feature.native_source",)
        )
        for name in (
            "feature.coupled_pinn",
            "feature.dose_water",
            "invalid.failed_pinn",
            "invalid.stale_pinn",
            "invalid.out_of_map",
        ):
            self.assertEqual(self.by_id[name].depends_on, ("feature.zero_field",))
        for name in ("feature.boris", "feature.dose_anatomy"):
            self.assertEqual(self.by_id[name].depends_on, ("feature.mhd_driven",))
            self.assertTrue(self.by_id[name].parameters["ground"])
        for name in (
            "feature.native_pinn",
            "feature.coupled_pinn",
            "feature.dose_water",
        ):
            self.assertEqual(self.by_id[name].kind, "scientific")
        self.assertEqual(self.by_id["feature.native_pinn"].parameters["epochs"], 1000)
        self.assertEqual(self.by_id["feature.coupled_pinn"].parameters["epochs"], 250)
        self.assertEqual(
            self.by_id["feature.dose_water"].parameters["histories"], 1_048_576
        )
        self.assertEqual(
            self.by_id["feature.dose_water"].parameters["action"],
            "dose_fixed_precision",
        )
        self.assertEqual(self.by_id["feature.dose_water"].timeout, 1800.0)
        for name in ("feature.native_pinn", "feature.coupled_pinn"):
            self.assertEqual(self.by_id[name].timeout, 600.0)
        anatomy = self.by_id["feature.dose_anatomy"].parameters
        self.assertEqual(
            (
                anatomy["species"],
                anatomy["target"],
                anatomy["physics_list"],
                anatomy["histories"],
            ),
            ("mixed", "male", "Shielding", 128),
        )
        ordered = dependency_order(self.cases)
        positions = {case.id: index for index, case in enumerate(ordered)}
        self.assertTrue(
            all(
                positions[parent] < positions[case.id]
                for case in ordered
                for parent in case.depends_on
            )
        )

    def test_explicit_small_scenarios_have_the_declared_thermal_mhd_and_observer_choices(
        self,
    ):
        cycle = scientific._config(self.by_id["feature.thermal_cycle"].parameters)
        self.assertEqual(
            (
                cycle.major_radius,
                cycle.minor_radius,
                cycle.resolution,
                cycle.duration,
                cycle.time_steps,
            ),
            (0.3, 0.06, 4, 8.0, 32),
        )
        self.assertEqual(
            (
                cycle.heating_power,
                cycle.discharge_power,
                cycle.charge_fraction,
                cycle.emissivity,
            ),
            (11e6, 11e6, 0.5, 0.1),
        )
        self.assertEqual(cycle.initial_temperature, cycle.melting_temperature - 1.0)
        native = scientific._config(self.by_id["feature.native_source"].parameters)
        self.assertEqual(
            (native.solver, native.duration, native.time_steps, native.resolution),
            ("thermal_benchmark", 2.0, 12, 4),
        )
        driven = scientific._config(self.by_id["feature.mhd_driven"].parameters)
        self.assertEqual(
            (driven.mode, driven.turbulence, driven.ground_enabled),
            ("driven", False, True),
        )
        self.assertGreater(
            driven.initial_temperature,
            driven.melting_temperature + driven.transition_width / 2,
        )
        dynamo = scientific._config(self.by_id["feature.mhd_dynamo"].parameters)
        self.assertEqual(
            (dynamo.mode, dynamo.turbulence, dynamo.coil_current, dynamo.seed_duration),
            ("dynamo", True, 0.0, 0.02),
        )
        self.assertLess(
            abs(dynamo.initial_temperature - dynamo.melting_temperature),
            dynamo.transition_width / 2,
        )
        zero = scientific._config(self.by_id["feature.zero_field"].parameters)
        self.assertEqual(
            (
                zero.major_radius,
                zero.minor_radius,
                zero.habitat_radius,
                zero.resolution,
                zero.duration,
                zero.time_steps,
            ),
            (1.6, 0.2, 1.0, 3, 0.02, 2),
        )
        self.assertEqual(
            (
                zero.heating_power,
                zero.coil_current,
                zero.pump_force,
                zero.gravity,
                zero.ground_enabled,
            ),
            (0.0, 0.0, 0.0, 0.0, False),
        )

    def test_native_cycle_really_melts_freezes_radiates_and_preserves_existing_energy_checks(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            checks = scientific.Checks()
            artifacts, _ = scientific._solve(
                {
                    "case_dir": directory,
                    "parameters": self.by_id["feature.thermal_cycle"].parameters,
                },
                checks,
                lambda *_: None,
                threading.Event(),
            )
            self.assertEqual(checks.failed, [])
            self.assertTrue(Path(artifacts["source_path"]).is_file())
        names = {row["name"] for row in checks.records}
        self.assertTrue(
            {
                "melting_front_evolves",
                "freezing_front_evolves",
                "solid_mushy_liquid_states",
                "positive_radiation_loss",
                "source_and_loss_integral",
                "assembled_energy_balance",
                "sharp_constitutive_law",
            }
            <= names
        )

    def test_radiation_observers_reuse_the_declared_accepted_source_without_changing_physics(
        self,
    ):
        source = RunResult(
            "accepted-driven",
            scientific._config(self.by_id["feature.mhd_driven"].parameters),
            np.array([0.0, 0.05]),
            np.zeros((1, 3)),
            {"accepted": np.ones((2, 1))},
            {},
            {},
        )
        with tempfile.TemporaryDirectory() as directory:
            parameters = self.by_id["feature.dose_anatomy"].parameters
            with patch.object(
                scientific, "_load_dependency", return_value=source
            ) as load:
                observer = scientific._radiation_source({}, parameters, Path(directory))
            load.assert_called_once_with({}, "feature.mhd_driven")
            np.testing.assert_array_equal(
                observer.fields["accepted"], source.fields["accepted"]
            )
            self.assertEqual(observer.config.physics_hash, source.config.physics_hash)
            self.assertEqual(observer.config.target, "male")
            self.assertTrue(observer.config.ground_enabled)
            self.assertEqual(observer.metadata["reused_fields_from"], source.run_id)
            self.assertNotEqual(observer.run_id, source.run_id)
            self.assertTrue(Path(observer.config.spectrum_path).is_file())
            self.assertEqual(source.run_id, "accepted-driven")

    def test_fe_cache_is_independent_of_learning_and_display_but_invalidated_by_equations_and_oracles(
        self,
    ):
        case = self.by_id["feature.mhd_driven"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = (
                "magnetic_torus/ui/editor.py",
                "magnetic_torus/app.py",
                "magnetic_torus/learning/training.py",
                "magnetic_torus/learning/diagnostics.py",
                "magnetic_torus/workers/fenics_worker.py",
                "magnetic_torus/numerics/weak_residual.py",
                "tests/simulation_suite/scientific.py",
            )
            for name in files:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("original", encoding="utf-8")
            fingerprint = case_fingerprint(case, root, {})
            for name in files[:3]:
                (root / name).write_text("edited", encoding="utf-8")
                self.assertEqual(case_fingerprint(case, root, {}), fingerprint, name)
            for name in files[3:]:
                (root / name).write_text("edited", encoding="utf-8")
                self.assertNotEqual(case_fingerprint(case, root, {}), fingerprint, name)
                (root / name).write_text("original", encoding="utf-8")
            child = self.by_id["feature.native_pinn"]
            parent = {
                "feature.native_source": {
                    "status": "PASS",
                    "execution": "PASS",
                    "qualification": "PASS",
                    "fingerprint": "accepted",
                    "artifact_hashes": {"field.h5": "one"},
                }
            }
            first = case_fingerprint(child, root, {}, parent)
            parent["feature.native_source"]["artifact_hashes"]["field.h5"] = "changed"
            self.assertNotEqual(case_fingerprint(child, root, {}, parent), first)


class FixedWaterPrecisionTests(unittest.TestCase):
    def test_fixed_history_job_runs_once_and_requires_the_measured_precision_gate(self):
        config = replace(ScenarioConfig(), coil_current=0.0, radiation_histories=2)
        source = RunResult(
            "zero-source", config, np.array([0.0, 0.02]), np.zeros((1, 3)), {}, {}, {}
        )
        features, _ = all_cases(ROOT, profile="features")
        cases = {
            "features": next(
                case for case in features if case.id == "feature.dose_water"
            ),
            "full": next(
                case for case in scientific_cases() if case.id == "dose.water-precision"
            ),
        }
        for profile, dose, error, passed in (
            (profile, dose, error, passed)
            for profile in cases
            for dose, error, passed in (
                (1.0, 0.04, True),
                (1.0, 0.05, True),
                (1.0, 0.050855404763649886, False),
                (1.0, 0.06, False),
                (0.0, 0.0, False),
            )
        ):
            calls = []

            def calculate(actual, snapshot, _progress, _cancel, _directory):
                calls.append(actual.config.radiation_histories)
                return DoseResult(
                    "precision-fixture",
                    actual.run_id,
                    actual.config.config_hash,
                    snapshot,
                    "water",
                    ("Water target",),
                    np.array([dose]),
                    np.array([error]),
                    actual.config.radiation_histories,
                    {
                        "organ_masses_kg": [64.0],
                        "field_off_dose_gy": [dose],
                        "secondary_dose_gy": [0.0],
                        "statistical_precision_met": [
                            dose > 0 and error <= 0.05 * dose
                        ],
                    },
                )

            with (
                self.subTest(profile=profile, dose=dose, error=error),
                tempfile.TemporaryDirectory() as directory,
            ):
                checks = scientific.Checks()
                parameters = dict(cases[profile].parameters)
                with (
                    patch.object(scientific, "_radiation_source", return_value=source),
                    patch(
                        "magnetic_torus.radiation.calculate_dose", side_effect=calculate
                    ),
                    patch(
                        "magnetic_torus.radiation.frozen_field",
                        return_value=SimpleNamespace(cell_values=np.zeros((1, 3))),
                    ),
                ):
                    artifacts, metrics = scientific._dose_reference(
                        {"case_dir": directory, "parameters": parameters},
                        checks,
                        lambda *_: None,
                        threading.Event(),
                    )
                self.assertEqual(calls, [1_048_576])
                self.assertEqual(len(metrics["attempts"]), 1)
                self.assertEqual(metrics["attempts"][0]["histories"], 1_048_576)
                self.assertEqual(
                    metrics["configuration"]["radiation_histories"], 1_048_576
                )
                self.assertEqual(metrics["seed"], config.seed)
                self.assertTrue(Path(artifacts["dose_path"]).is_file())
                self.assertEqual(not checks.failed, passed, checks.records)
                self.assertEqual(
                    scientific.HANDLERS["dose_fixed_precision"],
                    scientific._dose_reference,
                )

    def test_both_profiles_keep_the_same_zero_field_geometry_and_incident_source(self):
        features, _ = all_cases(ROOT, profile="features")
        feature = {case.id: case for case in features}
        full = {case.id: case for case in scientific_cases()}
        configurations = (
            scientific._config(feature["feature.zero_field"].parameters),
            scientific._config(full["fe.transport.zero.vacuum"].parameters),
        )
        self.assertEqual(configurations[0].to_dict(), configurations[1].to_dict())
        for case, config in (
            (feature["feature.dose_water"], configurations[0]),
            (full["dose.water-precision"], configurations[1]),
        ):
            with self.subTest(case=case.id), tempfile.TemporaryDirectory() as directory:
                source = RunResult(
                    "accepted-zero",
                    config,
                    np.array([0.0, 0.02]),
                    np.zeros((1, 3)),
                    {},
                    {},
                    {},
                )
                parameters = {
                    **case.parameters,
                    "field": "zero",
                    "ground": False,
                    "species": "proton",
                    "energy": 100.0,
                    "source": "beam",
                    "target": "water",
                    "physics_list": "Shielding",
                }
                with patch.object(scientific, "_load_dependency", return_value=source):
                    observer = scientific._radiation_source(
                        {"parameters": parameters}, parameters, Path(directory)
                    )
                self.assertEqual(
                    observer.config.physics_hash, source.config.physics_hash
                )
                self.assertEqual(observer.metadata["reused_fields_from"], source.run_id)
                self.assertEqual(
                    (
                        observer.config.seed,
                        observer.config.ground_enabled,
                        observer.config.source_kind,
                        observer.config.target,
                        observer.config.particle_z,
                        observer.config.particle_a,
                        observer.config.particle_energy_mev_nucleon,
                        observer.config.transport_physics_list,
                        observer.config.incident_fluence_m2,
                    ),
                    (42, False, "beam", "water", 1, 1, 100.0, "Shielding", 1.0),
                )
                self.assertEqual(observer.config.radiation_histories, 1_048_576)
                self.assertEqual(source.config, config)


if __name__ == "__main__":
    unittest.main()
