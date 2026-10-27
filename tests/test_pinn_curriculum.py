"""Training-only time curricula and label balancing; fixtures are not FE solves."""
from copy import deepcopy
from dataclasses import replace
import math
import subprocess
import sys
from pathlib import Path
import threading
import unittest
from unittest.mock import patch

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from magnetic_torus.learning.curriculum import (
    CausalWeights, LabelGradientBalancer, TimeLabelSampler,
)
from magnetic_torus.learning import training
from magnetic_torus.learning.fields import FieldScaling
from magnetic_torus.models import SimulationCancelled
from test_pinn_batching import label_fixture, prepared_labels
import test_pinn_conditioning as conditioning

SUITE_CAPABILITIES = ("torch_cpu",)


class CausalWeightTests(unittest.TestCase):
    def test_prefix_excludes_current_future_errors_and_records_observations(self):
        weights = CausalWeights(5)
        self.assertEqual(weights.weight(1), 1.)
        weights.update(1, .5)
        weights.update(2, .25)
        self.assertAlmostEqual(weights.weight(2), math.exp(-.5))
        self.assertAlmostEqual(weights.weight(3), math.exp(-.75))
        before = [weights.weight(ti) for ti in (1, 2, 3)]
        weights.update(4, 1e300)
        self.assertEqual(before, [weights.weight(ti) for ti in (1, 2, 3)])
        self.assertFalse(weights.metadata()["bootstrap_complete"])
        weights.update(3, .1)
        self.assertTrue(weights.metadata()["bootstrap_complete"])
        weights.update(1, 0.)
        self.assertEqual(weights.metadata()["observation_counts"], [2, 1, 1, 1])
        self.assertEqual(weights.weight(2), 1.)

    def test_floor_zero_error_zero_epsilon_and_extreme_finite_values_are_stable(self):
        weights = CausalWeights(5, epsilon=1e308)
        for ti in range(1, 5):
            weights.update(ti, 1e308)
        self.assertEqual(weights.metadata()["weights"], [1., .05, .05, .05])
        for epsilon, floor in ((0., .05), (1., 1.), (1e-300, .05)):
            weights = CausalWeights(4, epsilon=epsilon, floor=floor)
            for ti in range(1, 4):
                weights.update(ti, 0.)
            self.assertEqual(weights.metadata()["weights"], [1., 1., 1.])

    def test_invalid_values_and_tensor_inputs_cannot_enter_weight_state(self):
        for values in ((1,), (3, -1.), (3, float("inf")), (3, 1., 0.)):
            with self.assertRaises(ValueError):
                CausalWeights(*values)
        weights = CausalWeights(3)
        for loss in (-1., float("nan"), float("inf"), True, "1"):
            with self.assertRaises(ValueError):
                weights.update(1, loss)
        for ti in (0, 3, True):
            with self.assertRaises(ValueError):
                weights.weight(ti)

    def test_import_keeps_torch_and_scipy_lazy(self):
        result = subprocess.run([sys.executable, "-B", "-c",
            "import sys; import magnetic_torus.learning.curriculum; "
            "assert 'torch' not in sys.modules; assert 'scipy' not in sys.modules"],
            cwd=Path(__file__).resolve().parent.parent, capture_output=True, text=True,
            timeout=30, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)


@unittest.skipIf(torch is None, "PyTorch is required")
class CurriculumTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_time_sampler_matches_all_conditional_training_labels_and_keeps_default_rng(self):
        run = prepared_labels(label_fixture(20))
        run.field.maximum_batch = 256
        before = deepcopy(run.field.data.rng.bit_generator.state)
        sampler = TimeLabelSampler(run.losses, 42)
        for ti in range(1, run.field.data.snapshot_count):
            components = sampler.components(ti)
            pool = run.losses.training_pools[0]
            ids = pool[pool // run.field.data.point_count == ti]
            errors, valid = run.losses.errors_at(ids, training=True)
            expected = (errors[:, 0] * torch.as_tensor(valid[:, 0])).sum() / int(valid[:, 0].sum())
            self.assertEqual(list(components), ["enthalpy"])
            torch.testing.assert_close(components["enthalpy"], expected, rtol=1e-14, atol=1e-14)
        self.assertEqual(run.field.data.rng.bit_generator.state, before)
        self.assertLessEqual(sampler.cache_bytes + sampler.existing_cache_bytes, 64 * 1024**2)

    def test_cached_and_streaming_paths_match_labels_gradients_rng_and_batch_bound(self):
        run = prepared_labels(label_fixture(45))
        run.field.maximum_batch = 9
        cached = TimeLabelSampler(run.losses, 42)
        streamed = TimeLabelSampler(run.losses, 42, cache_limit_bytes=run.field.data.cache.nbytes)
        self.assertEqual(cached.metadata()["cache_mode"], "grouped")
        self.assertEqual(streamed.metadata()["cache_mode"], "streaming")
        self.assertEqual(streamed.cache_bytes, 0)
        for ti in (1, 4, 2, 3):
            values = [sampler.components(ti) for sampler in (cached, streamed)]
            a, b = values[0]["enthalpy"], values[1]["enthalpy"]
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            params = tuple(run.field.model.parameters())
            for ga, gb in zip(torch.autograd.grad(a, params), torch.autograd.grad(b, params)):
                torch.testing.assert_close(ga, gb, rtol=0, atol=0)
        self.assertEqual(cached.metadata()["rng_state"], streamed.metadata()["rng_state"])
        self.assertLessEqual(cached.sampled_labels, 4 * run.field.maximum_batch)

    def coupled_labels(self, *, change_heldout=False):
        with torch.random.fork_rng():
            torch.manual_seed(42)
            field = conditioning.PINNConditioningTests().coupled_field()
        source = field.data.result
        values = {name: np.zeros_like(value) for name, value in source.fields.items()}
        values["temperature"][:] = source.config.melting_temperature
        for name, amplitude in (("enthalpy", 50.), ("pressure", .0002), ("electric_potential", .0003)):
            values[name][1:] = amplitude
        heldout = np.array([7, 13], dtype=np.int64)
        train = np.setdiff1d(np.arange(15), heldout)
        if change_heldout:
            ti, pi = np.divmod(heldout, 5)
            for name in ("enthalpy", "pressure", "electric_potential"):
                values[name][ti, pi] += 1e6
        source = replace(source, fields=values)
        scaling = FieldScaling.from_result(source, coupled=True, training_ids=train)
        reference = scaling.references(source)
        cache = training._fixed_training_cache(source, scaling, reference, train, 64 * 1024**2)
        field.data = replace(field.data, result=source, scaling=scaling, reference=reference,
            train_ids=train, validation_ids=heldout, cache=cache, rng=np.random.default_rng(42))
        field.offsets = torch.as_tensor(scaling.offsets, **field.kw)
        field.scales = torch.as_tensor(scaling.scales, **field.kw)
        field.coupled_free[0, 0] = 0.
        field.coupled_free[2, 10] = 0.
        field.coupled_free[4, 11] = 0.
        with torch.no_grad():
            field.model[-1].weight.zero_()
            field.model[-1].bias.fill_(.1)
        return training.TrainingLosses(field, np.ones((5, 12), dtype=bool), threading.Event())

    def test_heldout_enthalpy_pressure_potential_cannot_change_cache_weights_or_updates(self):
        losses = [self.coupled_labels(change_heldout=changed) for changed in (False, True)]
        np.testing.assert_array_equal(losses[0].field.data.cache.targets, losses[1].field.data.cache.targets)
        samplers = [TimeLabelSampler(item, 42) for item in losses]
        samplers_data, balance_data, causal_data, values = [], [], [], []
        for item, sampler in zip(losses, samplers):
            before = deepcopy(item.field.data.rng.bit_generator.state)
            balancer = LabelGradientBalancer(refresh_steps=2)
            causal = CausalWeights(item.field.data.snapshot_count)
            params = tuple(item.field.model.parameters())
            optimizer = torch.optim.SGD(params, lr=1e-3)
            for ti in (1, 2):
                components = sampler.components(ti)
                causal.update(ti, float(torch.stack(tuple(components.values())).mean().detach()))
            history = []
            for step, ti in enumerate((1, 2, 1, 2)):
                optimizer.zero_grad(set_to_none=True)
                components = sampler.components(ti)
                raw = torch.stack(tuple(components.values())).mean()
                causal.update(ti, float(raw.detach()))
                loss = causal.weight(ti) * balancer.combine(components, params, step)
                history.append(float(loss.detach()))
                loss.backward()
                optimizer.step()
            self.assertEqual(item.field.data.rng.bit_generator.state, before)
            samplers_data.append(sampler.metadata())
            balance_data.append(balancer.metadata())
            causal_data.append(causal.metadata())
            values.append(history)
        self.assertEqual(samplers_data[0], samplers_data[1])
        self.assertEqual(balance_data[0], balance_data[1])
        self.assertEqual(causal_data[0], causal_data[1])
        self.assertEqual(values[0], values[1])
        for name, value in losses[0].field.model.state_dict().items():
            self.assertTrue(torch.equal(value, losses[1].field.model.state_dict()[name]), name)

    def test_time_sampling_excludes_initial_heldout_and_prescribed_unknowns(self):
        losses = self.coupled_labels()
        sampler = TimeLabelSampler(losses, 42)
        captured = []
        original = losses.errors_at
        def observe(ids, **kwargs):
            captured.append(ids.copy())
            return original(ids, **kwargs)
        with patch.object(losses, "errors_at", side_effect=observe):
            values = sampler.components(1)
        self.assertEqual(list(values), list(losses.field.data.scaling.names))
        ids = np.concatenate(captured)
        self.assertTrue(np.all(ids // losses.field.data.point_count == 1))
        self.assertTrue(np.all(np.isin(ids, losses.field.data.train_ids)))
        self.assertFalse(np.any(np.isin(ids, losses.field.data.validation_ids)))
        self.assertLessEqual(len(ids), losses.field.maximum_batch)
        with self.assertRaises(ValueError):
            sampler.components(0)

    def test_cancellation_before_cache_and_during_label_evaluation_prevents_loss_return(self):
        losses = self.coupled_labels()
        losses.cancel.set()
        with self.assertRaises(SimulationCancelled):
            TimeLabelSampler(losses, 42)
        losses.cancel.clear()
        sampler = TimeLabelSampler(losses, 42)
        original = losses.errors_at
        def cancelling(*args, **kwargs):
            values = original(*args, **kwargs)
            losses.cancel.set()
            return values
        with patch.object(losses, "errors_at", side_effect=cancelling):
            with self.assertRaises(SimulationCancelled):
                sampler.components(1)
        self.assertEqual(sampler.calls, 1)
        self.assertGreater(sampler.sampled_labels, 0)

    def test_gradient_weights_balance_known_label_derivatives_and_preserve_gradient_graph(self):
        parameter = torch.nn.Parameter(torch.tensor(1., dtype=torch.float64))
        parameter.grad = torch.tensor(7., dtype=torch.float64)
        components = {"enthalpy": parameter.square(), "pressure": 10 * parameter.square()}
        balancer = LabelGradientBalancer()
        value = balancer.combine(components, (parameter,), 0)
        state = balancer.metadata()
        self.assertEqual(state["gradient_norms"], {"enthalpy": 2., "pressure": 20.})
        self.assertEqual(state["weights"], {"enthalpy": 5.5, "pressure": .55})
        self.assertEqual(float(parameter.grad), 7.)
        self.assertTrue(all(isinstance(weight, float) for weight in state["weights"].values()))
        value.backward()
        expected = (5.5 * 2 + .55 * 20) / (5.5 + .55)
        self.assertAlmostEqual(float(parameter.grad) - 7, expected)
        self.assertEqual(state["gradient_evaluations"], 2)

    def test_zero_and_frozen_label_tasks_keep_unit_weights_and_relative_weights_are_bounded(self):
        parameter = torch.nn.Parameter(torch.tensor(1., dtype=torch.float64))
        frozen = torch.nn.Parameter(torch.tensor(2., dtype=torch.float64), requires_grad=False)
        balancer = LabelGradientBalancer()
        value = balancer.combine({"tiny": 1e-12 * parameter.square(),
            "large": 1e12 * parameter.square(), "zero": parameter * 0,
            "frozen": frozen.square()}, (parameter, frozen), 0)
        self.assertTrue(torch.isfinite(value))
        self.assertEqual(balancer.metadata()["weights"],
            {"tiny": 10., "large": .5, "zero": 1., "frozen": 1.})

    def test_balance_refresh_reuses_detached_weights_and_checks_cancellation(self):
        parameter = torch.nn.Parameter(torch.tensor(1., dtype=torch.float64))
        cancel = threading.Event()
        balancer = LabelGradientBalancer(refresh_steps=2, cancel=cancel)
        for step in range(3):
            balancer.combine({"label": parameter.square()}, (parameter,), step)
        self.assertEqual(balancer.metadata()["refreshes"], 2)
        self.assertEqual(balancer.metadata()["gradient_evaluations"], 2)
        before = balancer.metadata()
        original = torch.autograd.grad
        def cancel_gradient(*args, **kwargs):
            result = original(*args, **kwargs)
            cancel.set()
            return result
        with patch.object(torch.autograd, "grad", side_effect=cancel_gradient):
            with self.assertRaises(SimulationCancelled):
                balancer.combine({"label": parameter.square()}, (parameter,), 4)
        self.assertEqual(balancer.metadata()["refreshes"], before["refreshes"])
        self.assertEqual(balancer.metadata()["weights"], before["weights"])
        self.assertEqual(balancer.metadata()["gradient_evaluations"], 3)

    def test_nonfinite_labels_gradients_and_causal_tensor_values_are_rejected(self):
        parameter = torch.nn.Parameter(torch.tensor(0., dtype=torch.float64))
        with self.assertRaises(ValueError):
            CausalWeights(3).update(1, parameter.square())
        with self.assertRaises(ValueError):
            LabelGradientBalancer().combine({"label": parameter * float("nan")}, (parameter,), 0)
        with self.assertRaises(FloatingPointError):
            LabelGradientBalancer().combine({"label": parameter.sqrt()}, (parameter,), 0)


if __name__ == "__main__":
    unittest.main()
