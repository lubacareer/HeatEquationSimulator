"""Small fixed-architecture research checks; these do not qualify evolving MHD."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from magnetic_torus.learning.corrections import restore_model_from_checkpoint
from magnetic_torus.learning.modulation import create_film_model
import test_pinn_conditioning as conditioning
from test_pinn_correction_model import checkpoint_for, source_run

try:
    import torch
except ImportError:
    torch = None


SUITE_CAPABILITIES = ("torch_cpu",)


@unittest.skipIf(torch is None, "PyTorch is required")
class FiLMResearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.original_threads)

    def setUp(self):
        torch.manual_seed(42)
        self.model = create_film_model(torch=torch)
        self.inputs = torch.randn(11, 21, dtype=torch.float64)

    def activate(self):
        with torch.no_grad():
            self.model.temporal[-1].weight.fill_(0.01)
            self.model.temporal[-1].bias.fill_(0.02)
            self.model.output[-1].weight.fill_(0.025)
            self.model.output[-1].bias.fill_(0.03)

    def test_fresh_model_starts_at_zero_correction_and_fixed_declared_layout(self):
        self.assertTrue(all(p.dtype == torch.float64 for p in self.model.parameters()))
        self.assertEqual(self.model(self.inputs).shape, (11, 12))
        self.assertEqual(torch.count_nonzero(self.model(self.inputs)).item(), 0)
        self.assertEqual(torch.count_nonzero(self.model.temporal[-1].weight).item(), 0)
        self.assertEqual(torch.count_nonzero(self.model.temporal[-1].bias).item(), 0)
        metadata = self.model.architecture_metadata()
        self.assertEqual(metadata["spatial_columns"], list(range(8)))
        self.assertEqual(metadata["temporal_columns"], list(range(8, 21)))
        self.assertEqual(metadata["spatial_widths"], [8, 64, 64])
        self.assertEqual(metadata["temporal_widths"], [13, 32, 128])
        self.assertTrue(metadata["experimental"])
        self.assertFalse(any("label" in key or "rank" in key for key in metadata))
        for bad in (self.inputs[:, :20], self.inputs[None]):
            with self.assertRaisesRegex(ValueError, "shape"):
                self.model(bad)

    def test_time_modulation_changes_outputs_and_keeps_finite_spacetime_gradients(self):
        self.activate()
        inputs = self.inputs.clone()
        inputs[1, :8] = inputs[0, :8]
        inputs[1, 8:] = inputs[0, 8:] + 0.5
        inputs.requires_grad_(True)
        prediction = self.model(inputs)
        self.assertFalse(torch.equal(prediction[0], prediction[1]))
        gradient = torch.autograd.grad(prediction.sum(), inputs)[0]
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient[:, :8].abs().sum()), 0)
        self.assertGreater(float(gradient[:, 8:].abs().sum()), 0)

    def test_training_reaches_temporal_and_spatial_parameters_after_zero_head_start(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        target = 2.0 + self.inputs[:, 8:9]
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            loss = (self.model(self.inputs) - target).square().mean()
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            optimizer.step()
        self.assertGreater(float(self.model.temporal[-1].weight.grad.abs().sum()), 0)
        self.assertGreater(float(self.model.spatial[0].weight.grad.abs().sum()), 0)

    def test_outer_field_preserves_initial_and_dirichlet_constraints_after_updates(self):
        # This fixture tests the outer envelope and masks, not an FE solution.
        field = conditioning.PINNConditioningTests().coupled_field()
        field.model = self.model
        field.coupled_free[0] = 0.0
        ti, pi = np.array([0, 1, 2, 1]), np.array([1, 1, 1, 0])
        initial = field.forward(field.input_at(ti, pi), pi, ti)
        torch.testing.assert_close(initial, torch.zeros_like(initial), rtol=0, atol=0)
        self.activate()
        prediction = field.forward(field.input_at(ti, pi), pi, ti)
        dynamic = list(range(10)) + [11]
        torch.testing.assert_close(
            prediction[0, dynamic], torch.zeros(11, dtype=torch.float64), rtol=0, atol=0
        )
        self.assertNotEqual(float(prediction[0, 10].detach()), 0.0)
        torch.testing.assert_close(
            prediction[-1], torch.zeros(12, dtype=torch.float64), rtol=0, atol=0
        )
        self.assertGreater(float(prediction[1].detach().abs().sum()), 0)
        prediction.square().sum().backward()
        self.assertTrue(
            all(p.grad is None or torch.isfinite(p.grad).all() for p in self.model.parameters())
        )

    def test_restricted_checkpoint_roundtrip_preserves_rng_and_explicit_source_binding(self):
        self.activate()
        source = source_run()
        payload = checkpoint_for(self.model, source, self.model.architecture_metadata())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            torch.save(payload, path)
            state = torch.random.get_rng_state().clone()
            restored = restore_model_from_checkpoint(path, source)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        self.assertEqual(restored.architecture_metadata(), self.model.architecture_metadata())
        torch.testing.assert_close(restored(self.inputs), self.model(self.inputs), rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "different source"):
            restore_model_from_checkpoint(payload, replace(source, run_id="different"))

    def test_checkpoint_rejects_unknown_architecture_fields_versions_and_bad_weights(self):
        source = source_run()
        original = checkpoint_for(self.model, source, self.model.architecture_metadata())
        for mutate in (
            lambda d: d["metadata"]["architecture"].update(kind="untrusted.Model"),
            lambda d: d["metadata"]["architecture"].update(version=999),
            lambda d: d["metadata"]["architecture"].update(temporal_widths=[13, 64, 128]),
            lambda d: d["metadata"]["architecture"].update(unrecorded_option=True),
            lambda d: d["metadata"]["architecture"].update(experimental=False),
            lambda d: d["model_state"].pop("spatial.0.weight"),
            lambda d: d["model_state"]["spatial.0.weight"].fill_(float("nan")),
        ):
            payload = deepcopy(original)
            mutate(payload)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                restore_model_from_checkpoint(payload, source)


class FiLMLazyImportTests(unittest.TestCase):
    def test_import_does_not_load_torch_or_scientific_worker(self):
        code = (
            "import sys; import magnetic_torus.learning.modulation; "
            "assert 'torch' not in sys.modules; "
            "assert 'scipy' not in sys.modules; "
            "assert 'magnetic_torus.physics' not in sys.modules"
        )
        completed = subprocess.run(
            [sys.executable, "-B", "-c", code], capture_output=True, text=True, timeout=30
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)


if __name__ == "__main__":
    unittest.main()
