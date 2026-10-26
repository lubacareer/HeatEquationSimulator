"""Equivalent bounded full-label gradients with unchanged stochastic batches."""

from copy import deepcopy
from dataclasses import replace
import os
import threading
import unittest
from unittest.mock import patch

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from magnetic_torus.learning import training
from magnetic_torus.models import RunResult, SimulationCancelled
from test_torus_pinn import uniform_heating_run
import test_pinn_conditioning as conditioning_fixtures

SUITE_CAPABILITIES = ("torch_cpu",)


def label_fixture(point_count=1300):
    """Nonuniform finite labels for gradient checks, never a solver oracle."""
    original = uniform_heating_run(pinn_epochs=2, pinn_batch_size=256)
    points = np.column_stack(
        (np.linspace(0.8, 1.2, point_count), np.zeros((point_count, 2)))
    )
    h = np.repeat(original.fields["enthalpy"][:, :1], point_count, axis=1)
    h += (
        original.times[:, None]
        / original.config.duration
        * 50
        * (points[None, :, 0] - 1.0) ** 2
    )
    physics = {
        "kind": "thermal_p1_lumped_v1",
        "edge_i": [],
        "edge_j": [],
        "conductance_w_k": [],
        "node_volume": np.full(point_count, 1.0 / point_count),
        "boundary_area": np.zeros(point_count),
        "source_weights": np.full(point_count, 1.0 / point_count),
    }
    return RunResult(
        "nonuniform-label-fixture",
        original.config,
        original.times,
        points,
        {
            "enthalpy": h,
            "temperature": original.config.melting_temperature
            + h / original.config.heat_capacity_solid,
            "liquid_fraction": np.zeros_like(h),
        },
        {},
        {
            "pinn_physics": physics,
            "validation_status": "Label fixture; not a numerical solution",
        },
    )


def prepared_labels(source=None, *, cached=True):
    source = label_fixture() if source is None else source
    cancel = threading.Event()
    budget = 64 * 1024**2 if cached else 0
    with (
        patch.object(training, "_HOST_TRAINING_CACHE_BYTES", budget),
        patch.dict(os.environ, {"MAGNETIC_TORUS_PINN_DEVICE": "cpu"}),
    ):
        data = training._prepare_data(source, cancel)
        run = training._prepare_training(data, lambda *_: None, cancel, None)
    with torch.no_grad():
        run.field.model[-1].weight.fill_(0.01)
        run.field.model[-1].bias.fill_(0.2)
    return run


@unittest.skipIf(torch is None, "PyTorch is required")
class PINNBatchingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_host_cache_contains_training_targets_only_with_exact_budget_and_fallback(
        self,
    ):
        run = prepared_labels()
        data, cache = run.field.data, run.field.data.cache
        self.assertIsNotNone(cache)
        self.assertLessEqual(cache.nbytes, 64 * 1024**2)
        self.assertEqual(cache.targets.shape, (len(data.train_ids), data.output_width))
        np.testing.assert_array_equal(cache.training_rows[data.validation_ids], -1)
        ti, pi = np.divmod(data.train_ids, data.point_count)
        np.testing.assert_array_equal(
            cache.targets,
            (data.reference[ti, pi] - data.scaling.offsets) / data.scaling.scales,
        )
        for array in (cache.targets, cache.training_rows, cache.points, cache.times):
            self.assertFalse(array.flags.writeable)
        self.assertIsNone(
            training._fixed_training_cache(
                data.result,
                data.scaling,
                data.reference,
                data.train_ids,
                cache.nbytes - 1,
            )
        )
        exact = training._fixed_training_cache(
            data.result, data.scaling, data.reference, data.train_ids, cache.nbytes
        )
        self.assertEqual(exact.nbytes, cache.nbytes)
        fallback = prepared_labels(cached=False)
        self.assertIsNone(fallback.field.data.cache)
        with self.assertRaisesRegex(ValueError, "held-out label"):
            run.losses.errors_at(data.validation_ids[:1], training=True)

    def test_full_4096_batches_match_256_loss_gradient_and_use_larger_forward_calls(
        self,
    ):
        runs = [prepared_labels(cached=False), prepared_labels(cached=True)]
        results, batches = [], []
        for run, size in zip(runs, (256, 4096)):
            run.field.full_label_batch_size = size
            calls = []
            handle = run.field.model.register_forward_pre_hook(
                lambda _module, args: calls.append(len(args[0]))
            )
            try:
                loss = run.losses.stream_training_loss(backward=True)
            finally:
                handle.remove()
            results.append(
                (
                    loss,
                    [
                        parameter.grad.clone()
                        for parameter in run.field.model.parameters()
                    ],
                )
            )
            batches.append(calls)
        torch.testing.assert_close(results[0][0], results[1][0], rtol=1e-11, atol=1e-12)
        for old, new in zip(results[0][1], results[1][1]):
            torch.testing.assert_close(old, new, rtol=1e-11, atol=1e-12)
        self.assertEqual(max(batches[0]), 256)
        self.assertEqual(max(batches[1]), 4096)
        self.assertLess(len(batches[1]), len(batches[0]))
        self.assertEqual(runs[1].field.maximum_batch, 256)

    def test_cache_and_full_batch_setting_do_not_change_adam_draws_or_rng(self):
        runs = [prepared_labels(cached=False), prepared_labels(cached=True)]
        snapshots, sampled_ids, losses = [], [], []
        for run in runs:
            snapshots.append(deepcopy(run.field.data.rng.bit_generator.state))
            ids = []
            original = run.losses.errors_at

            def record(batch, **kwargs):
                ids.append(batch.copy())
                return original(batch, **kwargs)

            with patch.object(run.losses, "errors_at", side_effect=record):
                losses.append(
                    [run.losses.sampled_training_loss().detach() for _ in range(3)]
                )
            sampled_ids.append(ids)
        self.assertEqual(snapshots[0], snapshots[1])
        self.assertEqual(
            runs[0].field.data.rng.bit_generator.state,
            runs[1].field.data.rng.bit_generator.state,
        )
        for old, new in zip(sampled_ids[0], sampled_ids[1]):
            np.testing.assert_array_equal(old, new)
            self.assertEqual(len(new), 256)
        for old, new in zip(losses[0], losses[1]):
            self.assertTrue(torch.equal(old, new))

    def test_cache_construction_is_cancellable_and_forward_override_is_bounded(self):
        run = prepared_labels()
        data = run.field.data
        cancel = threading.Event()
        original = np.divide
        calls = []

        def cancel_after_chunk(*args, **kwargs):
            result = original(*args, **kwargs)
            calls.append(True)
            cancel.set()
            return result

        with patch.object(training.np, "divide", side_effect=cancel_after_chunk):
            with self.assertRaises(SimulationCancelled):
                training._fixed_training_cache(
                    data.result,
                    data.scaling,
                    data.reference,
                    data.train_ids,
                    64 * 1024**2,
                    cancel,
                )
        self.assertEqual(len(calls), 1)
        ti, pi = np.array([1]), np.array([0])
        inputs = run.field.input_at(ti, pi)
        for size in (0, 4097):
            with self.subTest(size=size), self.assertRaises(ValueError):
                run.field.forward(inputs, pi, ti, forward_batch_size=size)

    def test_heldout_label_changes_cannot_change_training_cache_loss_or_gradient(self):
        source = label_fixture()
        first = prepared_labels(source)
        h = np.array(source.fields["enthalpy"], copy=True)
        ti, pi = np.divmod(first.field.data.validation_ids, len(source.points))
        h[ti[ti > 0], pi[ti > 0]] += 1e6
        second = prepared_labels(
            replace(source, fields={**source.fields, "enthalpy": h})
        )
        np.testing.assert_array_equal(
            first.field.data.train_ids, second.field.data.train_ids
        )
        np.testing.assert_array_equal(
            first.field.data.cache.targets, second.field.data.cache.targets
        )
        values = [
            run.losses.stream_training_loss(backward=True) for run in (first, second)
        ]
        self.assertTrue(torch.equal(values[0], values[1]))
        for old, new in zip(
            first.field.model.parameters(), second.field.model.parameters()
        ):
            self.assertTrue(torch.equal(old.grad, new.grad))
        self.assertGreater(
            second.losses.validation_loss(), first.losses.validation_loss()
        )

    def test_cached_coordinates_preserve_grad_enabled_input_derivatives(self):
        runs = [prepared_labels(cached=False), prepared_labels(cached=True)]
        results = []
        ti, pi = np.array([0, 1, 4]), np.array([0, 512, 1299])
        for run in runs:
            inputs = run.field.input_at(ti, pi, grad=True)
            h = run.field.predict(inputs, pi, ti)["enthalpy"]
            derivative = torch.autograd.grad(h.sum(), inputs)[0]
            self.assertTrue(inputs.requires_grad)
            results.append((inputs.detach(), h.detach(), derivative))
        for old, new in zip(results[0], results[1]):
            torch.testing.assert_close(old, new, rtol=1e-12, atol=1e-12)
        self.assertGreater(float(results[1][2][:, 3].abs().max()), 0.0)

    def test_validation_reduces_all_fields_with_one_final_cpu_transfer(self):
        field = conditioning_fixtures.PINNConditioningTests().coupled_field()
        heldout = np.array([0, 1, 5, 6, 14])
        field.data = replace(
            field.data,
            validation_ids=heldout,
            train_ids=np.setdiff1d(np.arange(15), heldout),
        )
        field.maximum_batch = 2
        validity = np.ones((5, 12), dtype=bool)
        validity[-1, [0, 1, 2, 3, 10, 11]] = False
        losses = training.TrainingLosses(field, validity, threading.Event())
        totals, counts = np.zeros(6), np.zeros(6, dtype=int)
        with torch.no_grad():
            for start in range(0, len(heldout), 2):
                errors, valid = losses.errors_at(heldout[start : start + 2])
                column = 0
                for index, width in enumerate(field.data.scaling.widths):
                    totals[index] += float(
                        (
                            errors[:, column : column + width]
                            * torch.as_tensor(valid[:, column : column + width])
                        ).sum()
                    )
                    counts[index] += int(valid[:, column : column + width].sum())
                    column += width
        expected = float(np.mean(totals[counts > 0] / counts[counts > 0]))
        original_cpu = torch.Tensor.cpu
        transfers = []

        def record_cpu(tensor, *args, **kwargs):
            transfers.append(tensor.numel())
            return original_cpu(tensor, *args, **kwargs)

        with patch.object(torch.Tensor, "cpu", new=record_cpu):
            actual = losses.validation_loss()
        np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-12)
        self.assertEqual(transfers, [6])

    def test_streaming_checks_cancellation_after_the_last_chunk_and_rejects_nonfinite_loss(
        self,
    ):
        run = prepared_labels()
        calls = []
        original = run.losses.errors_at

        def cancel_last(*args, **kwargs):
            calls.append(True)
            result = original(*args, **kwargs)
            if len(calls) == 2:
                run.losses.cancel.set()
            return result

        with patch.object(run.losses, "errors_at", side_effect=cancel_last):
            with self.assertRaises(SimulationCancelled):
                run.losses.stream_training_loss(backward=True)
        self.assertEqual(len(calls), 2)
        run.losses.cancel.clear()
        weights = [
            parameter.detach().clone() for parameter in run.field.model.parameters()
        ]
        original_loss = run.losses._label_loss
        with patch.object(
            run.losses,
            "_label_loss",
            side_effect=lambda *args, **kwargs: (
                original_loss(*args, **kwargs) * float("inf")
            ),
        ):
            with self.assertRaises(FloatingPointError):
                run.losses.stream_training_loss(backward=True)
        for parameter, weight in zip(run.field.model.parameters(), weights):
            self.assertTrue(torch.equal(parameter.detach(), weight))


if __name__ == "__main__":
    unittest.main()
