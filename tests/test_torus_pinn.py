"""Physical operator and lifecycle checks for scenario-bound PINN training."""
from dataclasses import replace
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np
from magnetic_torus.artifacts import FieldArtifact

try:
    import torch
except ImportError:
    torch = None

from magnetic_torus.models import RunResult, ScenarioConfig, SimulationCancelled
from magnetic_torus.pinn import (ThermalOperator, UnsupportedPINNConfiguration,
                                 enthalpy_to_state_torch,
                                 regularized_enthalpy_to_state_torch, train_pinn,
                                 validate_pinn_source)


def uniform_heating_run(**overrides):
    values = dict(solver="thermal_benchmark", turbulence=False,
                  heat_capacity_solid=1000., heat_capacity_liquid=1200.,
                  density=2000., melting_temperature=500., initial_temperature=490.,
                  duration=2., time_steps=4, charge_fraction=.5,
                  heating_power=4000., discharge_power=1000.,
                  pinn_epochs=8, pinn_batch_size=4)
    values.update(overrides)
    config = replace(ScenarioConfig(), **values)
    times = np.linspace(0., config.duration, 5)
    points = np.array([[.9, 0., 0.], [1.1, 0., 0.]])
    supplied = (config.heating_power * np.minimum(times, config.charge_fraction * config.duration)
                - config.discharge_power * np.maximum(times - config.charge_fraction * config.duration, 0))
    h0 = config.heat_capacity_solid * (config.initial_temperature - config.melting_temperature)
    h = np.repeat((h0 + supplied / config.density)[:, None], 2, axis=1)
    temperature = config.melting_temperature + h / config.heat_capacity_solid
    physics = {"kind": "thermal_p1_lumped_v1", "edge_i": [0], "edge_j": [1],
               "conductance_w_k": [3.], "node_volume": [.4, .6],
               "boundary_area": [0., 0.], "source_weights": [.4, .6]}
    return RunResult("manufactured-uniform-heating", config, times, points,
                     {"enthalpy": h, "temperature": temperature, "liquid_fraction": np.zeros_like(h)},
                     {}, {"pinn_physics": physics, "validation_status": "manufactured analytic solution"})


class PINNImportTests(unittest.TestCase):
    def test_public_api_import_keeps_scientific_runtimes_lazy(self):
        # The desktop must be able to inspect saved PINNs without initializing
        # CUDA or the separate SciPy/OpenMP runtime used by numerical workers.
        result = subprocess.run(
            [sys.executable, "-B", "-c",
             "import sys; import magnetic_torus.pinn; "
             "assert 'torch' not in sys.modules; assert 'scipy' not in sys.modules"],
            cwd=Path(__file__).resolve().parent.parent,
            capture_output=True, text=True, check=False, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


@unittest.skipIf(torch is None, "PyTorch is required")
class TorusPINNTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_sharp_enthalpy_plateau_and_gradients(self):
        config = ScenarioConfig()
        enthalpy = torch.tensor([-config.heat_capacity_solid * 20, 0., config.latent_heat / 2,
                                 config.latent_heat, config.latent_heat + config.heat_capacity_liquid * 30],
                                dtype=torch.float64, requires_grad=True)
        temperature, fraction = enthalpy_to_state_torch(enthalpy, config)
        np.testing.assert_allclose(temperature.detach().numpy(), [1790., 1810., 1810., 1810., 1840.])
        np.testing.assert_allclose(fraction.detach().numpy(), [0., 0., .5, 1., 1.])
        derivative = torch.autograd.grad(temperature.sum(), enthalpy)[0]
        np.testing.assert_allclose(derivative.numpy(), [1 / 449, 0., 0., 0., 1 / 820])

    def test_regularized_inverse_matches_worker_apparent_capacity_law(self):
        config = ScenarioConfig()
        temperature = np.array([1750., 1809., 1809.5, 1810., 1810.5, 1811., 1900.])
        delta = temperature - config.melting_temperature
        width = config.transition_width
        fraction = np.clip(delta / width + .5, 0., 1.)
        middle = (config.heat_capacity_solid * delta + config.latent_heat * fraction
                  + .5 * (config.heat_capacity_liquid - config.heat_capacity_solid) * width * fraction ** 2)
        h = np.where(delta < -width / 2, config.heat_capacity_solid * delta,
                     np.where(delta > width / 2, config.latent_heat + config.heat_capacity_liquid * delta, middle))
        values = torch.tensor(h, dtype=torch.float64, requires_grad=True)
        actual_t, actual_f = regularized_enthalpy_to_state_torch(values, config)
        np.testing.assert_allclose(actual_t.detach().numpy(), temperature, atol=1e-10)
        np.testing.assert_allclose(actual_f.detach().numpy(), fraction, atol=1e-12)
        self.assertTrue(bool(torch.isfinite(torch.autograd.grad(actual_t.sum(), values)[0]).all()))

    def test_zero_latent_heat_uses_the_reference_solid_endpoint(self):
        config = replace(ScenarioConfig(), latent_heat=0.)
        temperature, fraction = enthalpy_to_state_torch(torch.tensor([-449., 0., 820.], dtype=torch.float64), config)
        np.testing.assert_array_equal(fraction.numpy(), [0., 0., 1.])
        np.testing.assert_allclose(temperature.numpy(), [1809., 1810., 1811.])

    def test_conduction_is_conservative_and_residual_matches_analytic_rates(self):
        run = uniform_heating_run(heating_power=0., discharge_power=0.)
        operator = ThermalOperator.from_result(run)
        temperature = torch.tensor([400., 600.], dtype=torch.float64)
        transfer = 3. * (400. - 600.)
        enthalpy_rate = torch.tensor([-transfer / (2000. * .4), transfer / (2000. * .6)], dtype=torch.float64)
        np.testing.assert_allclose(operator.residual(enthalpy_rate, temperature, .25, run.config).numpy(), 0., atol=1e-10)
        self.assertAlmostEqual(float(np.dot(operator.volumes, enthalpy_rate.numpy())), 0.)

    def test_radiation_and_source_are_balanced_in_watts(self):
        run = uniform_heating_run(emissivity=.7)
        raw = dict(run.metadata["pinn_physics"])
        raw["boundary_area"] = [.1, .2]
        run = RunResult(run.run_id, run.config, run.times, run.points, run.fields, {}, {"pinn_physics": raw})
        operator = ThermalOperator.from_result(run)
        temperature = torch.tensor([490., 491.], dtype=torch.float64)
        radiation = .7 * 5.670374419e-8 * np.array([.1, .2]) * (
            temperature.numpy() ** 4 - run.config.ambient_temperature ** 4)
        conduction = np.array([-3., 3.])
        rates = (4000. * np.array([.4, .6]) - radiation - conduction) / (2000. * np.array([.4, .6]))
        residual = operator.residual(torch.tensor(rates), temperature, .1, run.config)
        np.testing.assert_allclose(residual.numpy(), 0., atol=1e-10)

    def test_local_stencil_equals_global_conduction(self):
        run = uniform_heating_run(heating_power=0., discharge_power=0.)
        operator = ThermalOperator.from_result(run)
        support, center_positions, indices, values = operator.neighborhood(np.array([1]))
        matrix = np.zeros((1, len(support)))
        np.add.at(matrix, (indices[0], indices[1]), values)
        temperature = torch.tensor([400., 600.], dtype=torch.float64)
        local = np.sum(matrix * temperature.numpy()[support], axis=1)
        global_residual = operator.residual(torch.zeros_like(temperature), temperature, .5, run.config)
        self.assertEqual(int(support[center_positions[0]]), 1)
        np.testing.assert_allclose(local, global_residual[1:2].numpy())

    def test_training_preserves_source_and_has_disjoint_validation_and_checkpoint(self):
        run = uniform_heating_run()
        original = run.fields["enthalpy"].copy()
        updates = []
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"MAGNETIC_TORUS_PINN_DEVICE": "cpu"}):
            result = train_pinn(run, lambda fraction, message: updates.append((fraction, message)),
                                threading.Event(), Path(directory))
            checkpoint = Path(result.metadata["checkpoint_path"])
            self.assertTrue(checkpoint.is_file())
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            self.assertEqual(saved["source_run_id"], run.run_id)
            self.assertEqual(saved["config_hash"], run.config.config_hash)
        self.assertEqual(result.source_run_id, run.run_id)
        self.assertEqual(result.config_hash, run.config.config_hash)
        self.assertEqual(result.metadata["device"], "cpu")
        self.assertEqual(result.metadata["training_samples"] + result.metadata["heldout_samples"], 10)
        self.assertLess(result.metrics["enthalpy_rmse"], .1)
        self.assertLess(result.metrics["energy_defect_relative"], .01)
        self.assertEqual(result.metadata["validation_status"], "passed")
        self.assertTrue(all(np.isfinite(v) for v in result.metrics.values()))
        self.assertTrue(all(np.isfinite(v).all() for v in result.history.values()))
        np.testing.assert_array_equal(run.fields["enthalpy"], original)
        self.assertFalse(run.fields["enthalpy"].flags.writeable)
        self.assertFalse(result.fields["enthalpy"].flags.writeable)
        self.assertEqual(result.fields["temperature"].shape, (5, 2))
        np.testing.assert_array_equal(result.fields["enthalpy"][0], run.fields["enthalpy"][0])
        self.assertTrue(all(updates[i][0] <= updates[i + 1][0] for i in range(len(updates) - 1)))
        self.assertEqual(updates[-1][0], 1.)

    def test_cancelled_training_does_not_publish_a_partial_result(self):
        run = uniform_heating_run()
        cancel = threading.Event()
        def update(fraction, _message):
            if fraction > .1:
                cancel.set()
        with patch.dict(os.environ, {"MAGNETIC_TORUS_PINN_DEVICE": "cpu"}):
            with self.assertRaises(SimulationCancelled):
                train_pinn(run, update, cancel)
        np.testing.assert_allclose(run.fields["temperature"][0], 490.)

    def test_staged_training_is_seeded_and_preserves_the_callers_torch_rng(self):
        run = uniform_heating_run(pinn_epochs=2)
        with torch.random.fork_rng(devices=[]), patch.dict(os.environ, {"MAGNETIC_TORUS_PINN_DEVICE": "cpu"}):
            torch.manual_seed(431)
            before_first = torch.random.get_rng_state().clone()
            first = train_pinn(run, lambda *_: None, threading.Event())
            self.assertTrue(torch.equal(torch.random.get_rng_state(), before_first))
            torch.manual_seed(962)
            before_second = torch.random.get_rng_state().clone()
            second = train_pinn(run, lambda *_: None, threading.Event())
            self.assertTrue(torch.equal(torch.random.get_rng_state(), before_second))
        for key in first.fields:
            np.testing.assert_array_equal(first.fields[key], second.fields[key])
        for key in first.history:
            np.testing.assert_array_equal(first.history[key], second.history[key])
        self.assertEqual(dict(first.metrics), dict(second.metrics))

    def test_cancellation_during_prediction_does_not_assemble_a_result(self):
        run = uniform_heating_run(pinn_epochs=1)
        cancel = threading.Event()
        updates = []

        def update(fraction, _message):
            updates.append(fraction)
            if fraction > .9:
                cancel.set()

        with patch.dict(os.environ, {"MAGNETIC_TORUS_PINN_DEVICE": "cpu"}):
            with self.assertRaises(SimulationCancelled):
                train_pinn(run, update, cancel)
        self.assertGreater(max(updates), .9)
        self.assertNotIn(1., updates)
        np.testing.assert_allclose(run.fields["temperature"][0], 490.)

    def test_saved_prediction_cannot_attach_to_a_different_run_or_edited_config(self):
        run = uniform_heating_run(pinn_epochs=1)
        with patch.dict(os.environ, {"MAGNETIC_TORUS_PINN_DEVICE": "cpu"}):
            pinn = train_pinn(run, lambda *_: None, threading.Event())
        validate_pinn_source(pinn, run)
        other_run = RunResult("another-run", run.config, run.times, run.points, run.fields, {}, run.metadata)
        with self.assertRaisesRegex(ValueError, "different numerical run"):
            validate_pinn_source(pinn, other_run)
        edited = RunResult(run.run_id, replace(run.config, heating_power=5000.),
                           run.times, run.points, run.fields, {}, run.metadata)
        with self.assertRaisesRegex(ValueError, "different numerical run"):
            validate_pinn_source(pinn, edited)
        provenance_mismatch = replace(pinn, metadata={**pinn.metadata, "source_run_id": "wrong-provenance"})
        with self.assertRaisesRegex(ValueError, "provenance disagrees"):
            validate_pinn_source(provenance_mismatch, run)
        # Explicit model reuse is allowed for observer changes with the same
        # physical field; training identity remains the immutable origin.
        alias = RunResult("observer-alias", replace(run.config, particle_count=32),
                          run.times, run.points, run.fields, {}, run.metadata)
        reused = replace(pinn, source_run_id=alias.run_id, config_hash=alias.config.config_hash,
                         metadata={**pinn.metadata, "explicit_prior_model_reuse": True})
        validate_pinn_source(reused, alias)
        self.assertEqual(reused.metadata["training_source_run_id"], run.run_id)
        changed_physics = RunResult("bad-physics-alias", replace(run.config, coil_current=200.),
                                    run.times, run.points, run.fields, {}, run.metadata)
        falsely_reused = replace(reused, source_run_id=changed_physics.run_id,
                                 config_hash=changed_physics.config.config_hash)
        with self.assertRaisesRegex(ValueError, "observer-only"):
            validate_pinn_source(falsely_reused, changed_physics)

    def test_cuda_training_preserves_field_shapes_and_finite_metrics_when_available(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is unavailable")
        run = uniform_heating_run(pinn_epochs=2)
        with patch.dict(os.environ, {"MAGNETIC_TORUS_PINN_DEVICE": "cuda"}):
            result = train_pinn(run, lambda *_: None, threading.Event())
        self.assertEqual(result.metadata["device"], "cuda")
        self.assertTrue(all(np.isfinite(v) for v in result.metrics.values()))
        self.assertEqual(result.fields["enthalpy"].shape, run.fields["enthalpy"].shape)
        validate_pinn_source(result, run)

    def test_missing_operator_does_not_silently_fit_a_data_only_network(self):
        run = uniform_heating_run()
        missing = RunResult(run.run_id, run.config, run.times, run.points, run.fields, {}, {})
        with self.assertRaisesRegex(UnsupportedPINNConfiguration, "physics operator"):
            train_pinn(missing, lambda *_: None, threading.Event())

    def test_turbulent_benchmark_is_not_mislabeled_as_coupled_mhd(self):
        run = uniform_heating_run()
        turbulent = RunResult(run.run_id, replace(run.config, turbulence=True), run.times,
                              run.points, run.fields, {}, run.metadata)
        with self.assertRaisesRegex(UnsupportedPINNConfiguration, "cannot model turbulence"):
            train_pinn(turbulent, lambda *_: None, threading.Event())

    def test_coupled_vms_weak_operator_is_differentiable_and_conserves_internal_heat(self):
        from magnetic_torus.physics import evaluate_weak_residual_torch
        config = replace(ScenarioConfig(), initial_temperature=1820., pump_force=0., gravity=0.,
                         heating_power=0., discharge_power=0., coil_current=0., seed_field=0.,
                         duration=.02, time_steps=2, resolution=3, turbulence=True,
                         pinn_epochs=2, pinn_batch_size=8)
        points = np.array([[.9, 0., 0.], [1.1, 0., 0.], [1., .1, 0.], [1., 0., .1]])
        # Analytic barycentric gradients avoid initializing a second vendor's
        # LAPACK/OpenMP runtime inside the PyTorch test process on Windows.
        gradients = np.array([[-5., -5., -5.], [5., -5., -5.], [0., 10., 0.], [0., 0., 10.]])
        volume = .2 * .1 * .1 / 6
        times = np.array([0., .01, .02])
        temperature = np.full((3, 4), 1820.)
        h = np.full((3, 4), config.latent_heat + config.heat_capacity_liquid * 10.)
        vector = np.zeros((3, 4, 3))
        scalar = np.zeros((3, 4))
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "weak.npz"
            np.savez(artifact, points=points, cells=np.array([[0, 1, 2, 3]]), gradients=gradients[None],
                     volumes=np.array([volume]), cell_labels=np.array([1]), cell_diameter=np.array([.2]),
                     boundary_faces=np.empty((0, 3), dtype=np.int64), boundary_areas=np.empty(0),
                     imposed_vector_potential=vector, node_volumes=np.full(4, volume / 4))
            run = RunResult("weak-form-manufactured-cell", config, times, points,
                             {"temperature": temperature, "enthalpy": h, "liquid_fraction": np.ones_like(scalar),
                              "velocity": vector, "pressure": scalar, "magnetic_field": vector,
                              "vector_potential": vector, "electric_potential": scalar}, {},
                             {"weak_operator_path": str(artifact), "conductor_mask": [True] * 4,
                              "pinn_physics": {"kind": "fenics_a_phi_vms_v1", "bc_nodes": {}},
                              "validation_status": "manufactured single-cell weak-form benchmark"},
                             artifacts={"weak_operator_path": FieldArtifact.capture(artifact, root=directory)})
            fields = {name: torch.tensor(values[:2], dtype=torch.float64, requires_grad=True)
                      for name, values in run.fields.items()}
            residuals = evaluate_weak_residual_torch(config, run, fields, [0, 1])
            for name, value in residuals.items():
                self.assertLess(float(value.detach().abs().max()), 1e-10, name)
            # A stationary temperature gradient creates conductive redistribution,
            # with no net creation of energy in an insulated element.
            perturbation = torch.tensor([[0., 1., 0., 0.], [0., 1., 0., 0.]], dtype=torch.float64)
            fields["temperature"] = fields["temperature"] + perturbation
            perturbed = evaluate_weak_residual_torch(config, run, fields, [0, 1])
            self.assertGreater(float(perturbed["enthalpy"].detach().abs().max()), 0.)
            self.assertAlmostEqual(float(perturbed["enthalpy"].detach().sum()), 0., places=10)
            gradient = torch.autograd.grad(perturbed["enthalpy"].square().sum(), fields["temperature"])[0]
            self.assertTrue(bool(torch.isfinite(gradient).all()))
            self.assertGreater(float(gradient.abs().max()), 0.)
            # Exercise actual weak-form training, independent conservation metrics,
            # checkpoint identity and held-out labels without requiring WSL in CI.
            with patch.dict(os.environ, {"MAGNETIC_TORUS_PINN_DEVICE": "cpu"}):
                trained = train_pinn(run, lambda *_: None, threading.Event(), Path(directory) / "training")
            self.assertEqual(trained.metadata["physics"], "coupled_fe_weak_operator")
            for name in ("momentum", "continuity", "induction", "charge", "enthalpy", "gauge"):
                self.assertTrue(np.isfinite(trained.metrics[name + "_residual_rms"]))
            self.assertTrue(Path(trained.metadata["checkpoint_path"]).is_file())
            validate_pinn_source(trained, run)


if __name__ == "__main__":
    unittest.main()
