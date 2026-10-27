"""The global FE loss and its neural gradients survive bounded recomputation."""

from dataclasses import replace
import os
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from magnetic_torus.learning import training
from magnetic_torus.learning.corrections import create_correction_model
from magnetic_torus.models import SimulationCancelled
from test_pinn_research_scaling import tiny_fe_source


SUITE_CAPABILITIES = ("torch_cpu", "torch_cuda")


@unittest.skipIf(torch is None, "PyTorch is required")
class GlobalFEGradientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def make_run(self, directory, device, correction=False):
        source = tiny_fe_source(directory)
        config = replace(source.config, initial_temperature=1810., pinn_batch_size=4)
        physics = {**source.metadata["pinn_physics"],
                   "bc_nodes": {"0": [0], "1": [0], "2": [0], "3": [1], "4": [2]}}
        source = replace(source, config=config, metadata={**source.metadata, "pinn_physics": physics})
        with patch.dict(os.environ, {"MAGNETIC_TORUS_PINN_DEVICE": device}):
            data = training._prepare_data(source, threading.Event())
            run = training._prepare_training(data, lambda *_: None, threading.Event(), None)
        with torch.no_grad():
            generator = torch.Generator(device=device).manual_seed(4)
            for parameter in run.field.model.parameters():
                parameter.copy_(torch.randn(parameter.shape, generator=generator,
                                            device=device, dtype=torch.float64) * .003)
        if correction:
            run.field.model = create_correction_model(run.field.model, ("elliptic", "thermofluid"), torch=torch)
        return run

    def compare_pair(self, device, *, correction=False, chunk=3, offload=False):
        with tempfile.TemporaryDirectory() as directory:
            run = self.make_run(directory, device, correction)
            model, losses = run.field.model, run.losses
            parameters = tuple(p for p in model.parameters() if p.requires_grad)
            ti, pi = np.array([1]), np.array([1])
            with torch.no_grad():
                fraction = run.field.predict(run.field.input_at(ti, pi), pi, ti)["liquid_fraction"]
            self.assertTrue(0. < float(fraction[0]) < 1., "The parity fixture must exercise mushy derivatives.")
            for p in parameters:
                p.grad = None
            p_full, b_full = losses.coupled_physics(1)
            (.37 * (p_full + b_full)).backward()
            expected = [None if p.grad is None else p.grad.detach().clone() for p in parameters]
            for p in parameters:
                p.grad = None
            p_vjp, b_vjp = losses.coupled_physics_backprop(
                1, weight=.37, batch_size=chunk, offload_saved_tensors=offload)
            rtol, atol = ((1e-6, 1e-9) if device == "cuda" else (1e-8, 1e-10))
            torch.testing.assert_close(p_vjp, p_full.detach(), rtol=rtol, atol=atol)
            torch.testing.assert_close(b_vjp, b_full.detach(), rtol=rtol, atol=atol)
            for p, reference in zip(parameters, expected):
                actual = torch.zeros_like(p) if p.grad is None else p.grad
                reference = torch.zeros_like(p) if reference is None else reference
                torch.testing.assert_close(actual, reference, rtol=rtol, atol=atol)

    def test_real_fe_global_loss_and_gradients_match_for_small_chunks(self):
        for chunk in (1, 3, 8):
            with self.subTest(chunk=chunk):
                self.compare_pair("cpu", chunk=chunk)

    def test_frozen_core_and_both_correction_branches_match(self):
        self.compare_pair("cpu", correction=True)

    def test_multiple_interval_objective_and_label_gradients_match(self):
        with tempfile.TemporaryDirectory() as directory:
            run = self.make_run(directory, "cpu", correction=True)
            parameters = tuple(p for p in run.field.model.parameters() if p.requires_grad)
            indices, centers = np.array([1, 2]), np.array([0, 1, 2, 3])
            full = training._deterministic_objective(run, indices, centers, backward=True, coupled_vjp=False)
            reference = [None if p.grad is None else p.grad.detach().clone() for p in parameters]
            for p in parameters:
                p.grad = None
            vjp = training._deterministic_objective(run, indices, centers, backward=True, coupled_vjp=True)
            for key in full:
                torch.testing.assert_close(vjp[key], full[key], rtol=1e-8, atol=1e-10)
            for p, expected in zip(parameters, reference):
                actual = torch.zeros_like(p) if p.grad is None else p.grad
                expected = torch.zeros_like(p) if expected is None else expected
                torch.testing.assert_close(actual, expected, rtol=1e-8, atol=1e-10)

    def test_cancellation_after_global_fe_evaluation_does_not_backpropagate_chunks(self):
        with tempfile.TemporaryDirectory() as directory:
            run = self.make_run(directory, "cpu")
            original = run.field.predict
            def cancel_after_snapshot(*args, **kwargs):
                values = original(*args, **kwargs)
                run.losses.cancel.set()
                return values
            with patch.object(run.field, "predict", side_effect=cancel_after_snapshot):
                with self.assertRaises(SimulationCancelled):
                    run.losses.coupled_physics_backprop(1, batch_size=2)
            self.assertTrue(all(p.grad is None for p in run.field.model.parameters()))

    def test_primary_failure_is_preserved_if_optimizer_rollback_also_fails(self):
        parameter = torch.nn.Parameter(torch.tensor([1.], dtype=torch.float64))
        optimizer = torch.optim.LBFGS([parameter], max_iter=1)
        def failed_closure():
            raise torch.cuda.OutOfMemoryError("first failure")
        with patch.object(optimizer, "load_state_dict", side_effect=RuntimeError("restore failure")):
            with self.assertRaisesRegex(torch.cuda.OutOfMemoryError, "first failure") as context:
                training._bounded_lbfgs_step(optimizer, (parameter,), failed_closure, threading.Event())
        self.assertTrue(any("restore failure" in note for note in getattr(context.exception, "__notes__", ())))

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_fe_gradients_and_cpu_saved_tensor_offload(self):
        for correction in (False, True):
            for offload in (False, True):
                with self.subTest(correction=correction, offload=offload):
                    self.compare_pair("cuda", correction=correction, offload=offload)


if __name__ == "__main__":
    unittest.main()
