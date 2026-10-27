"""Native research proposal tests; tiny field fixtures do not qualify a solve."""
from copy import deepcopy
from dataclasses import replace
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import threading
import unittest
from unittest.mock import patch

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from magnetic_torus.learning.sampling import FrontResidualSampler
from magnetic_torus.learning import training
from magnetic_torus.models import SimulationCancelled
from test_pinn_batching import label_fixture, prepared_labels
from test_torus_pinn import uniform_heating_run

SUITE_CAPABILITIES = ("torch_cpu",)


class SamplingImportTests(unittest.TestCase):
    def test_import_keeps_torch_and_scipy_lazy(self):
        result = subprocess.run([sys.executable, "-B", "-c",
            "import sys; import magnetic_torus.learning.sampling; "
            "assert 'torch' not in sys.modules; assert 'scipy' not in sys.modules"],
            cwd=Path(__file__).resolve().parent.parent, capture_output=True, text=True,
            timeout=30, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)


@unittest.skipIf(torch is None, "PyTorch is required")
class FrontResidualSamplingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_components_match_independent_full_node_physical_operator_and_gradients(self):
        cancel = threading.Event()
        with patch.dict(os.environ, {"MAGNETIC_TORUS_PINN_DEVICE": "cpu"}):
            run = training._prepare_training(training._prepare_data(
                uniform_heating_run(), cancel), lambda *_: None, cancel, None)
        field, data = run.field, run.field.data
        with torch.no_grad():
            field.model[-1].weight.fill_(.01)
            field.model[-1].bias.fill_(.1)
        nodes = np.arange(data.point_count)
        terms = run.losses.thermal_components(1, nodes)
        tids = np.repeat([0, 1], data.point_count)
        pids = np.tile(nodes, 2)
        prediction = field.predict(field.input_at(tids, pids), pids, tids)
        h = prediction["enthalpy"].reshape(2, data.point_count)
        temperature = prediction["temperature"].reshape(2, data.point_count)
        dt = data.result.times[1] - data.result.times[0]
        watts = data.operator.residual((h[1]-h[0])/dt, temperature[1], dt/2, data.config)
        normalization = torch.as_tensor(data.config.density * data.operator.volumes
            * data.scaling.scales[0] / field.time_scale, **field.kw)
        expected = watts / normalization
        torch.testing.assert_close(terms["residual"], expected, rtol=0, atol=0)
        torch.testing.assert_close(terms["initial_error"], torch.zeros_like(expected), rtol=0, atol=0)
        parameters = tuple(field.model.parameters())
        actual_grad = torch.autograd.grad(terms["residual"].square().mean(), parameters)
        expected_grad = torch.autograd.grad(expected.square().mean(), parameters)
        for actual, reference in zip(actual_grad, expected_grad):
            torch.testing.assert_close(actual, reference, rtol=1e-13, atol=1e-13)

    def test_default_wrapper_preserves_uniform_rng_draws_and_means(self):
        run = prepared_labels(label_fixture(35))
        rng = np.random.default_rng()
        rng.bit_generator.state = deepcopy(run.field.data.rng.bit_generator.state)
        ti = int(rng.integers(1, run.field.data.snapshot_count))
        centers = rng.choice(run.field.data.point_count,
            min(run.field.maximum_batch, run.field.data.point_count), replace=False)
        expected = run.losses.thermal_components(ti, centers)
        actual = run.losses.thermal_physics()
        self.assertEqual(run.field.data.rng.bit_generator.state, rng.bit_generator.state)
        torch.testing.assert_close(actual[0], expected["residual"].square().mean(), rtol=0, atol=0)
        torch.testing.assert_close(actual[1], expected["initial_error"].square().mean(), rtol=0, atol=0)

    def fake_losses(self, point_count=3, batch=2):
        """Differentiable finite terms isolate sampling algebra, not physics."""
        parameter = torch.nn.Parameter(torch.tensor(.7, dtype=torch.float64))
        cancel = threading.Event()
        calls = []
        field = SimpleNamespace(torch=torch, device=torch.device("cpu"),
            kw={"dtype": torch.float64}, maximum_batch=batch,
            data=SimpleNamespace(thermal=True, point_count=point_count, snapshot_count=4))
        def components(ti, centers):
            self.assertEqual(len(centers), len(np.unique(centers)))
            calls.append((ti, centers.copy()))
            ids = torch.as_tensor(centers, dtype=torch.float64)
            return {"residual": parameter * (ids + 1),
                "initial_error": .25 * parameter * (ids + 2),
                "liquid_fraction": torch.where(ids == 1, .5, 0.)}
        return SimpleNamespace(field=field, cancel=cancel, thermal_components=components), parameter, calls

    def test_stratified_importance_expectation_and_gradient_equal_original_uniform_means(self):
        losses, parameter, calls = self.fake_losses()
        sampler = FrontResidualSampler(42)
        sampler._bind(losses)
        sampler._refresh(losses, 1, 0)
        q = sampler._proposal.copy()
        expected_physics = parameter.square() * torch.arange(1, 4, dtype=torch.float64).square().mean()
        expected_boundary = .25**2 * parameter.square() * torch.arange(2, 5, dtype=torch.float64).square().mean()
        total_physics = torch.zeros((), dtype=torch.float64)
        total_boundary = torch.zeros((), dtype=torch.float64)
        for uniform in range(3):
            for adaptive in range(3):
                class EnumeratedDraw:
                    def integers(self, *args, **kwargs):
                        return np.array([uniform], dtype=np.int64)
                    def choice(self, *args, **kwargs):
                        return np.array([adaptive], dtype=np.int64)
                sampler.rng = EnumeratedDraw()
                physics, boundary = sampler.loss(losses, 1, 0)
                total_physics = total_physics + q[adaptive] / 3 * physics
                total_boundary = total_boundary + q[adaptive] / 3 * boundary
        torch.testing.assert_close(total_physics, expected_physics, rtol=1e-14, atol=1e-14)
        torch.testing.assert_close(total_boundary, expected_boundary, rtol=1e-14, atol=1e-14)
        actual_gradient = torch.autograd.grad(total_physics + total_boundary, parameter)[0]
        expected_gradient = torch.autograd.grad(expected_physics + expected_boundary, parameter)[0]
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-14, atol=1e-14)
        self.assertEqual(sampler.duplicate_rows, 3)
        self.assertTrue(all(len(nodes) <= 2 for _, nodes in calls[1:]))

    def test_full_support_floor_and_odd_batch_use_actual_stratified_probability(self):
        losses, _, _ = self.fake_losses(point_count=7, batch=3)
        sampler = FrontResidualSampler(42, candidate_cap=3)
        sampler._bind(losses)
        sampler._refresh(losses, 1, 0)
        ids = np.arange(7)
        probability = sampler._probabilities(ids, 1/3)
        dense_proposal = np.zeros(7)
        dense_proposal[sampler._candidates] = sampler._proposal
        np.testing.assert_allclose(probability, (2/3)/7 + dense_proposal/3, rtol=3e-16, atol=0)
        self.assertTrue(np.all(probability > 0))
        self.assertAlmostEqual(float(probability.sum()), 1.)
        drawn, weights = sampler._draw(3)
        np.testing.assert_allclose(weights, 1 / (7 * probability[drawn]), rtol=0, atol=0)
        self.assertLessEqual(float(weights.max()), 1.5)

    def test_proposal_uses_predictions_only_and_leaves_label_rng_unchanged(self):
        source = label_fixture(40)
        first = prepared_labels(source)
        h = np.array(source.fields["enthalpy"], copy=True)
        ti, pi = np.divmod(first.field.data.validation_ids, len(source.points))
        h[ti[ti > 0], pi[ti > 0]] += 1e6
        second = prepared_labels(replace(source, fields={**source.fields, "enthalpy": h}))
        samplers = [FrontResidualSampler(42, candidate_cap=20) for _ in range(2)]
        states = [deepcopy(run.field.data.rng.bit_generator.state) for run in (first, second)]
        outputs = [sampler.loss(run.losses, 2, 0) for sampler, run in zip(samplers, (first, second))]
        for a, b in zip(outputs[0], outputs[1]):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        np.testing.assert_array_equal(samplers[0]._proposal, samplers[1]._proposal)
        self.assertEqual(samplers[0].metadata(), samplers[1].metadata())
        for run, state in zip((first, second), states):
            self.assertEqual(run.field.data.rng.bit_generator.state, state)

    def test_bounded_proposal_cache_and_forward_chunks_with_explicit_work_ledger(self):
        run = prepared_labels(label_fixture(5000))
        sampler = FrontResidualSampler(42)
        calls = []
        handle = run.field.model.register_forward_pre_hook(lambda _module, args: calls.append(len(args[0])))
        try:
            for ti, step in ((1, 0), (1, 49), (1, 50), (2, 51)):
                physics, boundary = sampler.loss(run.losses, ti, step)
                self.assertTrue(torch.isfinite(physics + boundary))
                self.assertLessEqual(sampler.cache_bytes, 4096 * 16)
        finally:
            handle.remove()
        report = sampler.metadata()
        self.assertEqual(report["proposal_evaluations"], 3)
        self.assertEqual(report["proposal_rows"], 3 * 4096)
        self.assertEqual(report["training_evaluations"], 4)
        self.assertEqual(report["sampled_rows"], 4 * run.field.maximum_batch)
        self.assertLessEqual(report["training_rows"], report["sampled_rows"])
        self.assertLessEqual(max(calls), 4096)
        self.assertEqual(report["cached_interval"], 2)

    def test_cancellation_before_and_during_proposal_or_training_preserves_work_counts(self):
        for when in (0, 1, 2):
            with self.subTest(when=when):
                losses, parameter, calls = self.fake_losses()
                sampler = FrontResidualSampler(42)
                before = parameter.detach().clone()
                original = losses.thermal_components
                def cancelling(ti, centers):
                    result = original(ti, centers)
                    if len(calls) == when:
                        losses.cancel.set()
                    return result
                losses.thermal_components = cancelling
                if when == 0:
                    losses.cancel.set()
                with self.assertRaises(SimulationCancelled):
                    sampler.loss(losses, 1, 0)
                self.assertTrue(torch.equal(parameter, before))
                self.assertEqual(sampler.proposal_evaluations, int(when > 0))
                self.assertEqual(sampler.training_evaluations, int(when == 2))

    def test_degenerate_scores_fall_back_and_nonfinite_scores_are_rejected(self):
        losses, parameter, _ = self.fake_losses()
        with torch.no_grad():
            parameter.zero_()
        original = losses.thermal_components
        def no_front(ti, centers):
            terms = original(ti, centers)
            terms["liquid_fraction"] = terms["liquid_fraction"] * 0
            return terms
        losses.thermal_components = no_front
        sampler = FrontResidualSampler(42)
        sampler.loss(losses, 1, 0)
        np.testing.assert_allclose(sampler._proposal, np.full(3, 1/3), rtol=0, atol=0)
        def invalid(ti, centers):
            terms = original(ti, centers)
            terms["residual"] = terms["residual"] * float("nan")
            return terms
        losses.thermal_components = invalid
        with self.assertRaises(FloatingPointError):
            FrontResidualSampler(42).loss(losses, 1, 0)

    def test_sampler_source_binding_and_configuration_boundaries(self):
        losses, _, _ = self.fake_losses()
        sampler = FrontResidualSampler(42)
        sampler.loss(losses, 1, 0)
        other, _, _ = self.fake_losses()
        with self.assertRaises(ValueError):
            sampler.loss(other, 1, 1)
        for kwargs in ({"candidate_cap": 0}, {"candidate_cap": 4097}, {"refresh_steps": 0}):
            with self.assertRaises(ValueError):
                FrontResidualSampler(42, **kwargs)
        for ti, step in ((0, 0), (4, 0), (1, -1)):
            with self.assertRaises(ValueError):
                sampler.loss(losses, ti, step)
        with self.assertRaisesRegex(ValueError, "unique"):
            prepared_labels().losses.thermal_components(1, np.array([0, 0]))


if __name__ == "__main__":
    unittest.main()
