"""Physical invariants of the native reference and optional Linux FE integration."""
from dataclasses import replace
import os
import io
from io import BytesIO
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from magnetic_torus.models import ScenarioConfig, SimulationCancelled, RunResult, DoseResult
from magnetic_torus.physics import (MU0, _thermal_graph, coil_magnetic_field,
                                    enthalpy_from_temperature, liquid_fraction,
                                    regularized_enthalpy, run_simulation,
                                    temperature_from_enthalpy)


class ThermalPhysicsTests(unittest.TestCase):
    def setUp(self):
        self.config = replace(ScenarioConfig(), solver="thermal_benchmark", resolution=3,
                              time_steps=5, duration=10., turbulence=False,
                              coil_current=0., pump_force=0., charge_fraction=1.)

    def test_sharp_plateau_and_si_material_units(self):
        c = self.config
        h = np.array([-449., 0., .5 * c.latent_heat, c.latent_heat, c.latent_heat + 820.])
        np.testing.assert_allclose(temperature_from_enthalpy(h, c), c.melting_temperature + np.array([-1, 0, 0, 0, 1]))
        np.testing.assert_allclose(liquid_fraction(h, c), [0, 0, .5, 1, 1])
        np.testing.assert_allclose(enthalpy_from_temperature([c.melting_temperature - 1, c.melting_temperature + 1], c), [-449, c.latent_heat + 820])

    def test_regularized_enthalpy_continuous_and_monotone(self):
        c = self.config
        temperatures = np.linspace(c.melting_temperature - 3, c.melting_temperature + 3, 10001)
        enthalpy = regularized_enthalpy(temperatures, c)
        self.assertTrue(np.all(np.diff(enthalpy) > 0))
        self.assertAlmostEqual(float(enthalpy[0]), -3 * c.heat_capacity_solid)
        self.assertAlmostEqual(float(enthalpy[-1]), c.latent_heat + 3 * c.heat_capacity_liquid)

    def test_stefan_front_against_neumann_similarity_solution(self):
        from magnetic_torus.physics import stefan_front_benchmark
        coarse = stefan_front_benchmark(32, 512)
        fine = stefan_front_benchmark(64, 1024)
        self.assertLess(fine["relative_front_error"], .005)
        self.assertLess(fine["relative_front_error"], coarse["relative_front_error"])

    def test_torus_volume_and_periodic_conductance(self):
        points, graph = _thermal_graph(self.config)
        self.assertEqual(points.shape[1], 3)
        self.assertAlmostEqual(float(graph["node_volume"].sum()), self.config.volume, places=12)
        self.assertAlmostEqual(float(graph["boundary_area"].sum()), 4 * np.pi ** 2 * self.config.major_radius * self.config.minor_radius, places=12)
        self.assertTrue(np.all(graph["conductance_w_k"] > 0))
        self.assertAlmostEqual(float(graph["source_weights"].sum()), 1., places=12)

    def test_impossible_native_snapshot_budget_is_rejected_before_allocation(self):
        from magnetic_torus.physics import native_memory_estimate
        ordinary = replace(self.config, resolution=8, time_steps=80)
        enormous = replace(self.config, resolution=128, time_steps=100000)
        self.assertLess(native_memory_estimate(ordinary), 64 * 1024 ** 2)
        with patch("magnetic_torus.physics.available_memory_bytes", return_value=512 * 1024 ** 2), \
             patch("magnetic_torus.physics._thermal_graph") as allocation, \
             patch.dict(os.environ, {"MAGNETIC_TORUS_NATIVE_MEMORY_MB": "256"}):
            with self.assertRaisesRegex(ValueError, "Reduce mesh resolution or output snapshots"):
                run_simulation(enormous)
            allocation.assert_not_called()

    def test_surface_heating_conserves_energy_and_crosses_plateau(self):
        c = replace(self.config, initial_temperature=1809., heating_power=2.e8)
        r = run_simulation(c)
        expected = c.heating_power * c.duration
        self.assertAlmostEqual(float(r.diagnostics["energy_j"][-1] - r.diagnostics["energy_j"][0]) / expected, 1., places=12)
        self.assertLess(np.max(np.abs(r.diagnostics["energy_balance_error_j"])), 1.e-5)
        self.assertTrue(np.any((r.fields["liquid_fraction"] > 0) & (r.fields["liquid_fraction"] < 1)))
        self.assertGreater(np.max(r.fields["temperature"][-1]), c.melting_temperature)
        self.assertGreater(np.ptp(r.fields["temperature"][-1]), 0.)
        np.testing.assert_allclose(r.fields["velocity"], 0)
        np.testing.assert_allclose(r.diagnostics["imposed_field_energy_j"], r.diagnostics["magnetic_energy_j"])
        np.testing.assert_allclose(r.diagnostics["induced_field_energy_j"], 0.)
        np.testing.assert_allclose(r.diagnostics["pump_energy_j"], 0.)

    def test_charge_discharge_transition_integrates_exactly(self):
        c = replace(self.config, charge_fraction=.37, heating_power=20000., discharge_power=5000.)
        r = run_simulation(c)
        self.assertTrue(np.any(np.isclose(r.times, 3.7)))
        expected = c.duration * (.37 * c.heating_power - .63 * c.discharge_power)
        self.assertAlmostEqual(float(r.diagnostics["energy_j"][-1] - r.diagnostics["energy_j"][0]), expected, places=6)

    def test_radiation_cools_surface_and_closes_energy_balance(self):
        c = replace(self.config, heating_power=0., emissivity=.1, duration=1.)
        r = run_simulation(c)
        self.assertLess(float(np.min(r.fields["temperature"][-1])), c.initial_temperature)
        self.assertGreater(float(np.ptp(r.fields["temperature"][-1])), 0.)
        self.assertLess(float(np.max(np.abs(r.diagnostics["energy_balance_error_j"]))), .01)

    def test_cancellation_does_not_publish_result(self):
        cancel = threading.Event(); cancel.set()
        with self.assertRaises(SimulationCancelled):
            run_simulation(self.config, cancel=cancel)

    def test_biot_savart_matches_circular_coil_axis(self):
        from magnetic_torus.physics import imposed_coil
        c = replace(self.config, coil_current=1000.)
        coil = imposed_coil(c)
        a, z = coil["radius_m"], .4
        b = coil_magnetic_field(np.array([[0., 0., z]]), c)[0]
        expected = MU0 * coil["ampere_turns"] * a * a / (2 * (a * a + z * z + coil["wire_radius_m"] ** 2) ** 1.5)
        np.testing.assert_allclose(b[:2], 0, atol=1.e-15)
        self.assertAlmostEqual(b[2] / expected, 1., places=12)

    def test_exited_fe_worker_retains_reader_thread_final_traceback(self):
        from magnetic_torus.physics import PhysicsBackendUnavailable
        class DelayedOutput(io.StringIO):
            def __iter__(self):
                time.sleep(.08)
                yield "Traceback: delayed final numerical error\n"
        process = SimpleNamespace(stdout=DelayedOutput(), poll=lambda: 1,
                                  wait=lambda **_kwargs: 1, terminate=lambda: None)
        with tempfile.TemporaryDirectory() as directory, \
             patch.dict(os.environ, {"MAGNETIC_TORUS_RUNS": directory}), \
             patch("magnetic_torus.physics.subprocess.Popen", return_value=process):
            with self.assertRaisesRegex(PhysicsBackendUnavailable, "delayed final numerical error"):
                run_simulation(replace(self.config, solver="fenics"))
            logs = list(Path(directory).glob("*/worker.log"))
            self.assertEqual(len(logs), 1)
            self.assertIn("delayed final numerical error", logs[0].read_text())

    def test_recovered_timesteps_do_not_exhaust_the_next_step_retry_budget(self):
        from magnetic_torus.workers.fenics_worker import _TimestepRejections
        retries = _TimestepRejections()
        for _ in range(35):
            self.assertFalse(retries.rejected(.5, 1 / 256))
            retries.accepted()
        self.assertEqual(retries.total, 35)
        self.assertEqual(retries.consecutive, 0)
        for _ in range(20):
            self.assertFalse(retries.rejected(.5, 1 / 256))
        self.assertTrue(retries.rejected(.5, 1 / 256))
        self.assertEqual(retries.total, 56)
        retries.accepted()
        self.assertFalse(retries.rejected(1 / 256, 1 / 256))
        self.assertTrue(retries.rejected(1 / 512, 1 / 256))

    def assert_complete_fe_boundary_nodes(self, result):
        """Compare exported PINN constraints with topology, not locator shape assumptions."""
        with np.load(BytesIO(result.artifact_bytes("weak_operator_path")), allow_pickle=False) as operators:
            cells, labels = operators["cells"], operators["cell_labels"]
        all_nodes = np.arange(len(result.points))
        conductor = np.unique(cells[labels == 1])
        outside = np.setdiff1d(all_nodes, conductor)
        local_faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
        faces = np.sort(cells[:, local_faces].reshape(-1, 3), axis=1)
        unique_faces, counts = np.unique(faces, axis=0, return_counts=True)
        exterior = np.unique(unique_faces[counts == 1])
        conductor_faces = {tuple(face) for face in np.sort(cells[labels == 1][:, local_faces].reshape(-1, 3), axis=1)}
        vacuum_faces = {tuple(face) for face in np.sort(cells[labels != 1][:, local_faces].reshape(-1, 3), axis=1)}
        interface = np.unique(np.asarray(list(conductor_faces & vacuum_faces)))
        exported = {key: np.sort(value) for key, value in result.metadata["pinn_physics"]["bc_nodes"].items()}
        np.testing.assert_array_equal(exported["0"], np.union1d(outside, interface))
        np.testing.assert_array_equal(exported["2"], exterior)
        for index in ("3", "4"):
            np.testing.assert_array_equal(exported[index], outside)
        np.testing.assert_array_equal(np.setdiff1d(exported["1"], conductor), outside)
        self.assertEqual(len(np.intersect1d(exported["1"], conductor)), 1)

    @unittest.skipUnless(os.environ.get("TORUS_TEST_FENICS") == "1", "Set TORUS_TEST_FENICS=1 to run the installed WSL FE worker.")
    def test_full_fe_stationary_closed_system_and_weak_operator(self):
        c = replace(self.config, solver="fenics", duration=.02, time_steps=2,
                    initial_temperature=1820., heating_power=0., discharge_power=0.)
        r = run_simulation(c)
        fluid = np.asarray(r.metadata["conductor_mask"])
        np.testing.assert_allclose(r.fields["temperature"][:, fluid], c.initial_temperature, atol=1.e-6)
        np.testing.assert_allclose(r.fields["velocity"], 0, atol=1.e-9)
        self.assertTrue(Path(r.metadata["weak_operator_path"]).is_file())
        self.assertTrue(Path(r.metadata["exterior_field_path"]).is_file())
        self.assertEqual(r.metadata["pinn_physics"]["kind"], "fenics_a_phi_vms_v1")
        self.assertGreaterEqual(r.metadata["mumps_memory_limit_mb"], 128)
        self.assert_complete_fe_boundary_nodes(r)
        np.testing.assert_allclose(r.diagnostics["magnetic_energy_j"],
                                   r.diagnostics["imposed_field_energy_j"] + r.diagnostics["induced_field_energy_j"]
                                   + r.diagnostics["mutual_field_energy_j"], atol=1.e-10)

    @unittest.skipUnless(os.environ.get("TORUS_TEST_FENICS") == "1", "Set TORUS_TEST_FENICS=1 for coupled WSL integration.")
    def test_full_fe_driven_vms_matches_differentiable_equations(self):
        import torch
        from magnetic_torus.physics import evaluate_weak_residual_torch
        c = replace(self.config, solver="fenics", duration=.1, time_steps=3, turbulence=True,
                    initial_temperature=1810.2, heating_power=1.e6, pump_force=1., coil_current=10000.)
        r = run_simulation(c)
        self.assertGreater(float(np.max(np.linalg.norm(r.fields["magnetic_field"], axis=-1))), .01)
        self.assertGreater(float(np.max(np.linalg.norm(r.fields["velocity"], axis=-1))), 0.)
        self.assertLess(float(np.max(np.abs(r.diagnostics["energy_balance_error_j"]))), .01)
        self.assertGreaterEqual(r.metadata["mumps_memory_limit_mb"], 128)
        self.assertEqual(r.metadata["mumps_memory_limit_verified_mb"], r.metadata["mumps_memory_limit_mb"])
        np.testing.assert_allclose(r.diagnostics["magnetic_energy_j"],
                                   r.diagnostics["imposed_field_energy_j"] + r.diagnostics["induced_field_energy_j"]
                                   + r.diagnostics["mutual_field_energy_j"], atol=1.e-10)
        np.testing.assert_allclose(np.diff(r.diagnostics["pump_energy_j"]),
                                   np.diff(r.times) * r.diagnostics["pump_power_w"][1:], atol=1.e-12)
        predictions = {key: torch.tensor(value[:2], dtype=torch.float64, requires_grad=True)
                       for key, value in r.fields.items()}
        residuals = evaluate_weak_residual_torch(c, r, predictions, [0, 1])
        for name in ("momentum", "continuity", "induction", "charge", "enthalpy", "magnetic_consistency"):
            self.assertLess(float(residuals[name].detach().abs().max()), 1.e-7, name)
        sum(value.square().mean() for value in residuals.values()).backward()
        self.assertTrue(all(value.grad is None or bool(torch.isfinite(value.grad).all()) for value in predictions.values()))


@unittest.skipUnless(os.environ.get("TORUS_TEST_DESKTOP") == "1", "Set TORUS_TEST_DESKTOP=1 for the real Qt/VTK interpolation check.")
class SceneInterpolationTests(unittest.TestCase):
    def test_cutaway_slice_and_result_identity(self):
        """A linear field must remain linear at newly clipped/sliced vertices."""
        from PyQt6 import QtWidgets
        from magnetic_torus.scene import SceneWidget
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        scene = SceneWidget(); scene.setWindowTitle("Smoke test - field interpolation")
        scene.resize(800, 600); scene.show(); app.processEvents()
        try:
            config = replace(ScenarioConfig(), minor_radius=.35)
            points = np.array([[.8, -.2, -.2], [1.2, -.2, -.2], [1., .2, -.2],
                               [1., 0., .2], [1.4, .2, .2]])
            temperature = 1800 + points @ np.array([10., 20., 30.])
            fields = {"temperature": np.tile(temperature, (2, 1))}
            result = RunResult("linear-test", config, np.array([0., 1.]), points, fields, {},
                               {"cells": [[0, 1, 2, 3], [1, 2, 3, 4]], "conductor_cells": [[0, 1, 2, 3]],
                                "conductor_mask": [True, True, True, True, False], "backend": "Linear interpolation verification"})
            scene.set_result(result)
            self.assertEqual(scene._grid.n_cells, 1)
            expected = 1800 + scene._surface.points @ np.array([10., 20., 30.])
            np.testing.assert_allclose(scene._surface["display"], expected, atol=1.e-8, rtol=0.)
            scene.slice_enabled = True; scene.slice_position = 1.; scene.update_index(0)
            section = scene.plotter.actors["slice"].mapper.dataset
            np.testing.assert_allclose(section["slice_value"], 1800 + section.points @ np.array([10., 20., 30.]), atol=1.e-8, rtol=0.)
            scene.update_index(0, {"temperature": np.full((2, 5), -5.)}, difference=True)
            np.testing.assert_allclose(scene.plotter.actors["slice"].mapper.scalar_range, (-5., 5.))
            scene.preview_geometry(replace(config, major_radius=1.2))
            self.assertIn("draft_geometry", scene.plotter.actors)
            scene.preview_geometry(config)
            self.assertNotIn("draft_geometry", scene.plotter.actors)
            scene.set_dose(DoseResult("dose", result.run_id, config.config_hash, 1, "water", ("water",), np.array([1.]), np.array([.1]), 10))
            scene.set_result(replace(result, run_id="new-run", times=np.array([0.]),
                                     fields={"temperature": fields["temperature"][:1]}))
            self.assertIsNone(scene.dose)
        finally:
            scene.close_scene(); scene.close(); app.processEvents()


if __name__ == "__main__":
    unittest.main()
