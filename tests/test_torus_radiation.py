"""Physics and provenance checks independent of optional Geant4 installation."""
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from magnetic_torus.models import PINNResult, RunResult, ScenarioConfig
from magnetic_torus.radiation import (
    C, ELEMENTARY_CHARGE, MEV, PROTON_MASS, FrozenField, calculate_dose, load_spectrum,
    field_source_run, frozen_field, parse_dose_output, relativistic_boris, sample_source, trace_particles,
)


class RadiationTests(unittest.TestCase):
    def test_zero_field_straight_momentum(self):
        p = np.array([[1e-19, 2e-19, -3e-19]])
        np.testing.assert_array_equal(relativistic_boris(p, np.array([ELEMENTARY_CHARGE]),
                                                        np.array([PROTON_MASS]), np.zeros((1, 3)), 1e-8), p)

    def test_relativistic_gyro_radius_and_energy(self):
        energy = 100 * MEV
        gamma = 1 + energy / (PROTON_MASS * C ** 2)
        momentum = PROTON_MASS * C * math.sqrt(gamma ** 2 - 1)
        radius = momentum / ELEMENTARY_CHARGE
        omega = ELEMENTARY_CHARGE / (gamma * PROTON_MASS)
        dt = .001 / omega
        p = np.array([[momentum, 0., 0.]])
        position = np.zeros(3)
        for _ in range(1000):
            old = p.copy()
            p = relativistic_boris(p, np.array([ELEMENTARY_CHARGE]), np.array([PROTON_MASS]), np.array([[0., 0., 1.]]), dt)
            position += (old[0] + p[0]) * .5 * dt / (gamma * PROTON_MASS)
        self.assertAlmostEqual(np.linalg.norm(p) / momentum, 1., places=12)
        np.testing.assert_allclose(position[:2], (radius * np.sin(1.), radius * (np.cos(1.) - 1)), rtol=2e-7)

    def test_sources_reproducible_and_isotropic_incident(self):
        config = ScenarioConfig(source_kind="isotropic")
        one, two = sample_source(config, 200, 4.), sample_source(config, 200, 4.)
        np.testing.assert_array_equal(one.positions, two.positions)
        np.testing.assert_array_equal(one.directions, two.directions)
        self.assertTrue(np.all(np.sum(one.positions * one.directions, axis=1) < 0))
        np.testing.assert_allclose(np.linalg.norm(one.directions, axis=1), 1.)

    def test_heavy_ion_and_spectrum_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "spectrum.json"
            content = {"schema_version": 1, "energy_unit": "MeV/nucleon", "provenance": "unit test",
                       "angular_distribution": "beam", "bins": [{"z": 26, "a": 56, "energy_mev_nucleon": 500., "weight": 1.}]}
            path.write_text(json.dumps(content))
            source = sample_source(ScenarioConfig(source_kind="spectrum", spectrum_path=str(path)), 5, 4.)
            np.testing.assert_array_equal(source.z, 26)
            np.testing.assert_array_equal(source.a, 56)
            content["bins"][0]["weight"] = -1
            path.write_text(json.dumps(content))
            with self.assertRaisesRegex(ValueError, "positive"):
                load_spectrum(path)

    def test_original_tetrahedral_field_and_outside_rejection(self):
        points = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        evaluator = FrozenField(points, np.array([[0, 1, 2, 3]]), points * 2, 1.)
        np.testing.assert_allclose(evaluator(np.array([[.1, .2, .3]])), [[.2, .4, .6]])
        with self.assertRaisesRegex(ValueError, "outside"):
            evaluator(np.array([[2., 2., 2.]]))

    def test_cell_curl_vector_potential_ignores_display_projection(self):
        points = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        # A=(-By/2,Bx/2,0) has curl A=(0,0,B), exactly in P1.
        potential = np.column_stack((-points[:, 1], points[:, 0], np.zeros(4)))
        evaluator = FrozenField(points, np.array([[0, 1, 2, 3]]), np.ones((4, 3)) * 99., 1., vector_potential=potential)
        np.testing.assert_allclose(evaluator([[.1, .2, .3], [.3, .1, .1]]), [[0., 0., 2.], [0., 0., 2.]])
        self.assertEqual(evaluator.field_mode, "cell_curl_a")
        # A skew physical element checks the reference-to-world gradient map.
        skew = np.array([[1., 2., 3.], [2., 2.3, 3.1], [1.1, 3.2, 3.2], [1.3, 2.2, 4.1]])
        gradient = np.array([[1., .2, .3], [.4, 2., .6], [.7, .8, 3.]])
        affine_potential = skew @ gradient.T + np.array([3., -2., 1.])
        skew_field = FrozenField(skew, np.array([[2, 0, 3, 1]]), np.zeros((4, 3)), 10., vector_potential=affine_potential)
        point = np.array([.3, .2, .4, .1]) @ skew
        np.testing.assert_allclose(skew_field(point), [[.2, -.4, .2]], atol=1e-14)

    def test_pinn_source_binding_cannot_mix_scenarios(self):
        points = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        config = ScenarioConfig()
        numerical = RunResult("accepted", config, np.array([0.]), points, {}, {}, {"cells": [[0, 1, 2, 3]]})
        potential = np.column_stack((-points[:, 1], points[:, 0], np.zeros(4)))[None]
        model = PINNResult("model-1", "accepted", config.config_hash,
                           {"vector_potential": potential, "magnetic_field": np.ones((1, 4, 3)) * 99}, {}, {}, {})
        with tempfile.TemporaryDirectory() as directory:
            bound = field_source_run(numerical, model, Path(directory))
            np.testing.assert_allclose(frozen_field(bound, 0)([[.1, .2, .3]]), [[0, 0, 2]])
            self.assertEqual(bound.metadata["source_model_id"], "model-1")
            stale = PINNResult("wrong", "other-run", config.config_hash, model.fields, {}, {}, {})
            with self.assertRaisesRegex(ValueError, "another"):
                field_source_run(numerical, stale, Path(directory))

    def test_dose_parser_rejects_stale_result_and_preserves_uncertainty(self):
        config = ScenarioConfig()
        result = RunResult("accepted", config, np.array([0., 1.]), np.zeros((1, 3)), {}, {})
        data = {"schema_version": 1, "source_run_id": "accepted", "config_hash": config.config_hash,
                "snapshot_index": 1, "histories": config.radiation_histories,
                "metadata": {"physics_list": config.transport_physics_list + " + EM option 4", "production_cut_m": config.production_cut_m},
                "organs": [{"name": "water", "field_on_gy": 1e-9, "field_on_standard_error_gy": 1e-11,
                            "field_off_gy": 2e-9, "field_off_standard_error_gy": 1e-11,
                            "mass_kg": 1., "secondary_gy": .5e-9, "secondary_standard_error_gy": 1e-11}]}
        dose = parse_dose_output(data, result, 1)
        self.assertEqual(dose.dose_gy[0], 1e-9)
        np.testing.assert_array_equal(dose.metadata["field_off_dose_gy"], [2e-9])
        self.assertTrue(dose.metadata["statistical_precision_met"][0])
        data["source_run_id"] = "stale"
        with self.assertRaisesRegex(ValueError, "stale"):
            parse_dose_output(data, result, 1)

    def test_missing_exterior_field_cannot_silently_be_zero(self):
        result = RunResult("accepted", ScenarioConfig(), np.array([0., 1.]), np.zeros((1, 3)), {}, {})
        with self.assertRaisesRegex(RuntimeError, "exterior magnetic field"):
            trace_particles(result, 0, lambda *_: None, None)

    def test_accepted_zero_field_produces_identical_paired_paths(self):
        from scipy.spatial import Delaunay
        with tempfile.TemporaryDirectory() as directory:
            points = np.array([[x, y, z] for x in (-5., 5.) for y in (-5., 5.) for z in (-5., 5.)])
            cells = Delaunay(points).simplices
            path = Path(directory) / "exterior.npz"
            np.savez(path, points=points, cells=cells, magnetic_field=np.zeros((1, 8, 3)),
                     vector_potential=np.zeros((1, 8, 3)), times=np.array([0.]))
            result = RunResult("accepted", ScenarioConfig(particle_count=5), np.array([0.]),
                               np.zeros((1, 3)), {}, {}, {"exterior_field_path": str(path), "exterior_radius_m": 4.})
            trajectories = trace_particles(result, 0, lambda *_: None, None)
            np.testing.assert_array_equal(trajectories.paths, trajectories.baseline_paths)
            self.assertEqual(trajectories.paths.shape[:2], trajectories.times.shape)
            self.assertEqual(trajectories.outcomes, trajectories.baseline_outcomes)


@unittest.skipUnless(os.environ.get("TORUS_TEST_GEANT4") == "1", "Set TORUS_TEST_GEANT4=1 to run the installed transport worker.")
class Geant4IntegrationTests(unittest.TestCase):
    """Actual transport/geometry smoke checks, not physics-list qualification."""
    def _transport(self, target, histories):
        from scipy.spatial import Delaunay
        directory = Path(tempfile.mkdtemp(prefix=f"torus-geant4-test-{target}-"))
        points = np.array([[x, y, z] for x in (-8., 8.) for y in (-8., 8.) for z in (-8., 8.)])
        cells = Delaunay(points).simplices
        path = directory / "zero-field-fixture.npz"
        np.savez(path, points=points, cells=cells, times=np.array([0.]),
                 vector_potential=np.zeros((1, 8, 3)), magnetic_field=np.zeros((1, 8, 3)))
        config = ScenarioConfig(major_radius=1.6, minor_radius=.2, habitat_radius=1.,
                                target=target, radiation_histories=histories, coil_current=0., seed_field=0.)
        numerical = RunResult("transport-fixture-" + target, config, np.array([0.]), points, {}, {},
                              {"exterior_field_path": str(path), "exterior_radius_m": 7.2,
                               "validation_status": "Synthetic zero-field transport fixture, not an MHD solution."})
        dose = calculate_dose(numerical, 0, lambda *_: None, None, directory)
        np.testing.assert_allclose(dose.dose_gy, dose.metadata["field_off_dose_gy"], rtol=1e-12, atol=0)
        self.assertTrue(Path(dose.metadata["voxel_dose_path"]).is_file())
        self.assertTrue(np.all(np.isfinite(dose.standard_error_gy)))
        return dose

    def test_water_paired_zero_field_transport(self):
        dose = self._transport("water", 1024)
        self.assertAlmostEqual(sum(dose.metadata["organ_masses_kg"]), 64., places=5)
        self.assertGreater(dose.dose_gy[0], 0.)

    def test_male_icrp110_transport(self):
        dose = self._transport("male", 64)
        self.assertGreater(len(dose.organ_names), 100)
        self.assertTrue(70 < sum(dose.metadata["organ_masses_kg"]) < 76)

    def test_female_icrp110_transport(self):
        dose = self._transport("female", 64)
        self.assertGreater(len(dose.organ_names), 100)
        self.assertTrue(58 < sum(dose.metadata["organ_masses_kg"]) < 62)


if __name__ == "__main__":
    unittest.main()
