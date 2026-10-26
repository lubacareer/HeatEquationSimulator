"""Physics-derived trial functions and bounded optimizer conditioning."""

from dataclasses import replace
from copy import deepcopy
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from magnetic_torus.learning.fields import FieldScaling, _make_network
from magnetic_torus.learning.training import (
    NeuralField,
    TrainingData,
    TrainingLosses,
    _bounded_lbfgs_step,
    _optimizer_steps,
    _prepare_data,
    _prepare_training,
    _train_model,
)
from magnetic_torus.models import SimulationCancelled
from magnetic_torus.pinn import train_pinn
from test_torus_pinn import uniform_heating_run
from workspace_fixtures import accepted_run

SUITE_CAPABILITIES = ("torch_cpu",)


@unittest.skipIf(torch is None, "PyTorch is required")
class PINNConditioningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def prepare(self, source=None, **changes):
        source = uniform_heating_run(**changes) if source is None else source
        cancel = threading.Event()
        with patch.dict(os.environ, {"MAGNETIC_TORUS_PINN_DEVICE": "cpu"}):
            data = _prepare_data(source, cancel)
            return _prepare_training(data, lambda *_: None, cancel, None)

    def coupled_field(self):
        # This is a trial-envelope fixture, not an FE solve or trained model.
        source = accepted_run(times=(0.0, 0.1, 5.0))
        names = (
            "enthalpy",
            "velocity",
            "magnetic_field",
            "vector_potential",
            "pressure",
            "electric_potential",
        )
        scaling = FieldScaling(names, (1, 3, 3, 3, 1, 1), np.zeros(12), np.ones(12))
        ids = np.arange(len(source.times) * len(source.points), dtype=np.int64)
        data = TrainingData(
            source,
            False,
            None,
            None,
            np.random.default_rng(42),
            ids[:-1],
            ids[-1:],
            scaling,
            scaling.references(source),
        )
        model = _make_network(torch, 12, 21).to(dtype=torch.float64)
        with torch.no_grad():
            model[-1].weight.zero_()
            model[-1].bias.fill_(1.0)
        free = np.ones((len(source.points), 12))
        free[0, 1:4] = 0.0
        kw = {"device": torch.device("cpu"), "dtype": torch.float64}
        return NeuralField(
            data,
            torch,
            kw["device"],
            model,
            8,
            kw,
            torch.zeros(12, **kw),
            torch.ones(12, **kw),
            torch.as_tensor(2 * np.pi * 2.0 ** np.arange(6), **kw),
            1.35,
            0.0,
            5.0,
            False,
            np.zeros((len(source.times), len(source.points), 12)),
            free,
        )

    def test_native_trial_obeys_total_mass_energy_without_local_deposition(self):
        source = uniform_heating_run()
        physics = {
            **source.metadata["pinn_physics"],
            "node_volume": [0.1, 0.9],
            "source_weights": [1.0, 0.0],
        }
        source = replace(source, metadata={**source.metadata, "pinn_physics": physics})
        run = self.prepare(source)
        data, field = run.field.data, run.field
        ti = np.repeat(np.arange(len(source.times)), 2)
        pi = np.tile(np.arange(2), len(source.times))
        predicted = (
            field.predict(field.input_at(ti, pi), pi, ti)["enthalpy"]
            .detach()
            .numpy()
            .reshape(-1, 2)
        )
        energy = 4000 * np.minimum(source.times, 1.0) - 1000 * np.maximum(
            source.times - 1.0, 0.0
        )
        expected = -10000 + energy / 2000
        np.testing.assert_allclose(
            predicted, np.repeat(expected[:, None], 2, axis=1), atol=1e-10
        )
        np.testing.assert_array_equal(predicted[0], source.fields["enthalpy"][0])
        self.assertEqual(data.config.config_hash, source.config.config_hash)

    def test_native_bulk_radiation_has_correct_initial_global_energy_rate(self):
        source = uniform_heating_run(emissivity=0.2)
        physics = {**source.metadata["pinn_physics"], "boundary_area": [1.0, 3.0]}
        source = replace(source, metadata={**source.metadata, "pinn_physics": physics})
        run = self.prepare(source)
        field = run.field
        ti, pi = np.zeros(2, dtype=np.int64), np.arange(2)
        inputs = field.input_at(ti, pi, grad=True)
        h = field.predict(inputs, pi, ti)["enthalpy"]
        rates = (
            torch.autograd.grad(h.sum(), inputs)[0][:, 3].detach().numpy()
            / field.time_scale
        )
        expected = (4000 - 0.2 * 5.670374419e-8 * 4 * (490**4 - 293**4)) / 2000
        np.testing.assert_allclose(rates, expected, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(h.detach().numpy(), -10000, atol=1e-10)

    def test_native_particular_does_not_use_future_reference_labels(self):
        source = uniform_heating_run()
        changed = dict(source.fields)
        changed["enthalpy"] = np.array(source.fields["enthalpy"], copy=True)
        changed["enthalpy"][1:] += 10000
        altered = replace(source, fields=changed)
        fields = [self.prepare(item).field for item in (source, altered)]
        ti, pi = np.array([1, 4]), np.array([0, 1])
        predictions = [
            field.predict(field.input_at(ti, pi), pi, ti)["enthalpy"].detach().numpy()
            for field in fields
        ]
        np.testing.assert_allclose(predictions[0], predictions[1], atol=1e-10)

    def test_coupled_envelope_preserves_initial_conditions_and_releases_at_first_step(
        self,
    ):
        field = self.coupled_field()
        ti, pi = np.array([0, 1, 2, 1]), np.array([1, 1, 1, 0])
        values = field.forward(field.input_at(ti, pi), pi, ti).detach().numpy()
        np.testing.assert_array_equal(values[0, :10], np.zeros(10))
        self.assertEqual(values[0, 11], 0.0)
        self.assertEqual(values[0, 10], 1.0)
        np.testing.assert_array_equal(values[1], np.ones(12))
        np.testing.assert_array_equal(values[2], np.ones(12))
        np.testing.assert_array_equal(values[3, 1:4], np.zeros(3))

    def test_training_free_dof_mask_does_not_change_heldout_domain_or_split(self):
        field = self.coupled_field()
        validity = np.ones((5, 12), dtype=bool)
        validity[-1, [0, 1, 2, 3, 10, 11]] = False
        losses = TrainingLosses(field, validity, threading.Event())
        train_ids = field.data.train_ids.copy()
        heldout_ids = field.data.validation_ids.copy()
        ids = np.array([0, 5, 6])
        expected_heldout = validity[[0, 0, 1]].copy()
        expected_heldout[0, 10] = False
        np.testing.assert_array_equal(losses.validity_at(ids), expected_heldout)
        training = losses.validity_at(ids, training=True)
        self.assertFalse(training[0].any())
        self.assertFalse(training[1, 1:4].any())
        self.assertTrue(training[2, 1:4].all())
        np.testing.assert_array_equal(field.data.train_ids, train_ids)
        np.testing.assert_array_equal(field.data.validation_ids, heldout_ids)
        self.assertTrue(
            all(np.isin(pool, train_ids).all() for pool in losses.training_pools)
        )
        self.assertTrue(
            all(not np.isin(pool, heldout_ids).any() for pool in losses.training_pools)
        )

    def test_streamed_training_gradient_matches_full_label_objective(self):
        run = self.prepare(pinn_batch_size=2)
        with torch.no_grad():
            run.field.model[-1].bias.fill_(0.3)
        full = run.losses.sampled_loss(run.field.data.train_ids)
        full.backward()
        expected = [
            parameter.grad.clone() for parameter in run.field.model.parameters()
        ]
        run.optimizer.zero_grad(set_to_none=True)
        streamed = run.losses.stream_training_loss(backward=True)
        self.assertAlmostEqual(float(streamed), float(full.detach()), places=12)
        for parameter, gradient in zip(run.field.model.parameters(), expected):
            torch.testing.assert_close(parameter.grad, gradient, rtol=1e-12, atol=1e-12)

    def test_streamed_training_checks_cancellation_between_chunks(self):
        run = self.prepare(pinn_batch_size=2)
        original = run.losses.errors_at

        def cancel_after_first(*args, **kwargs):
            result = original(*args, **kwargs)
            run.losses.cancel.set()
            return result

        with patch.object(run.losses, "errors_at", side_effect=cancel_after_first):
            with self.assertRaises(SimulationCancelled):
                run.losses.stream_training_loss(backward=True)

    def test_optimizer_stage_budget_and_checkpoint_metadata(self):
        for budget, expected in (
            (1, (1, 0)),
            (49, (49, 0)),
            (50, (40, 10)),
            (1000, (800, 200)),
        ):
            self.assertEqual(_optimizer_steps(budget), expected)
        source = uniform_heating_run(pinn_epochs=50)
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(os.environ, {"MAGNETIC_TORUS_PINN_DEVICE": "cpu"}),
        ):
            result = train_pinn(
                source, lambda *_: None, threading.Event(), Path(directory)
            )
            saved = torch.load(
                result.metadata["checkpoint_path"],
                map_location="cpu",
                weights_only=True,
            )
        self.assertEqual(len(result.history["epoch"]), 50)
        self.assertEqual(
            result.metadata["optimizer_steps_completed"], {"adam": 40, "lbfgs": 10}
        )
        self.assertGreaterEqual(
            result.metadata["function_evaluations"]["lbfgs_closures"], 10
        )
        self.assertEqual(result.metadata["parameterization_version"], 2)
        self.assertEqual(saved["metadata"]["training_method_version"], 2)
        self.assertEqual(saved["source_run_id"], source.run_id)
        self.assertEqual(saved["config_hash"], source.config.config_hash)

    def test_refinement_line_search_has_strict_closure_budget_and_rolls_back(self):
        parameter = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
        optimizer = torch.optim.LBFGS(
            [parameter],
            lr=1.0,
            max_iter=1,
            max_eval=8,
            history_size=20,
            line_search_fn="strong_wolfe",
        )
        def warmup():
            optimizer.zero_grad(set_to_none=True)
            loss = (parameter - 1).square()
            loss.backward()
            return loss
        optimizer.step(warmup)
        original_weight = parameter.detach().clone()
        original_gradient = parameter.grad.clone()
        original_state = deepcopy(optimizer.state_dict())
        calls = []

        def unbounded_training_objective():
            calls.append(True)
            optimizer.zero_grad(set_to_none=True)
            loss = -parameter
            loss.backward()
            return loss

        evaluations, limited = _bounded_lbfgs_step(
            optimizer, (parameter,), unbounded_training_objective, threading.Event()
        )
        self.assertTrue(limited)
        self.assertEqual(evaluations, 8)
        self.assertEqual(len(calls), 8)
        self.assertTrue(torch.equal(parameter.detach(), original_weight))
        self.assertEqual(optimizer.state_dict(), original_state)
        self.assertTrue(torch.equal(parameter.grad, original_gradient))

    def test_refinement_does_not_swallow_cancellation_or_nonfinite_errors(self):
        parameter = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
        optimizer = torch.optim.LBFGS([parameter], max_iter=1, max_eval=8)
        for error in (
            SimulationCancelled("cancelled"),
            FloatingPointError("non-finite"),
        ):
            with self.subTest(error=type(error).__name__):
                with self.assertRaises(type(error)):
                    _bounded_lbfgs_step(
                        optimizer,
                        (parameter,),
                        lambda: (_ for _ in ()).throw(error),
                        threading.Event(),
                    )

    def test_heldout_monitor_values_do_not_adapt_training_or_model_selection(self):
        runs = [self.prepare(pinn_epochs=50) for _ in range(2)]
        for run, heldout in zip(runs, (0.0, 1e6)):
            with patch.object(run.losses, "validation_loss", return_value=heldout):
                _train_model(run, lambda *_: None, run.losses.cancel)
        for left, right in zip(
            runs[0].field.model.parameters(), runs[1].field.model.parameters()
        ):
            self.assertTrue(torch.equal(left, right))
        self.assertEqual(runs[0].history["loss"], runs[1].history["loss"])


if __name__ == "__main__":
    unittest.main()
