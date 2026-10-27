"""Research-only phi gauge bias compatibility and hard-constraint checks."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from magnetic_torus.learning.fields import _make_network
from magnetic_torus.learning.gauge import create_temporal_gauge_model
from magnetic_torus.learning.corrections import restore_model_from_checkpoint
from test_pinn_correction_model import source_run, checkpoint_for
from test_pinn_research_scaling import tiny_fe_source
import test_pinn_conditioning as conditioning

SUITE_CAPABILITIES = ("torch_cpu",)


@unittest.skipIf(torch is None, "PyTorch is required")
class TemporalGaugeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.original_threads)

    def core(self, *, zero=False):
        model = _make_network(torch, 12, 21).double()
        if zero:
            with torch.no_grad():
                model[-1].weight.zero_()
                model[-1].bias.zero_()
        return model

    def inputs(self, times):
        values = torch.zeros((len(times), 21), dtype=torch.float64)
        values[:, 8] = torch.tensor(times, dtype=torch.float64)
        return values

    def test_zero_initialization_preserves_predictions_weights_and_rng(self):
        core = self.core()
        original = deepcopy(core.state_dict())
        values = torch.randn((17, 21), dtype=torch.float64)
        values[:, 8] = torch.linspace(0, 1, len(values), dtype=torch.float64)
        rng = torch.random.get_rng_state().clone()
        model = create_temporal_gauge_model(core, [0.0, 0.17, 1.0], torch=torch)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertTrue(torch.equal(model(values), core(values)))
        self.assertEqual(torch.count_nonzero(model.phi_bias).item(), 0)
        self.assertEqual(list(model.branch_parameters()), [model.phi_bias])
        self.assertTrue(
            all(not parameter.requires_grad for parameter in model.core.parameters())
        )
        self.assertTrue(all(parameter.requires_grad for parameter in core.parameters()))
        for name, value in original.items():
            self.assertTrue(torch.equal(value, core.state_dict()[name]))
            self.assertTrue(torch.equal(value, model.core.state_dict()[name]))

    def test_linear_interpolation_preserves_initial_bias_and_extends_endpoints(self):
        model = create_temporal_gauge_model(self.core(zero=True), [0.0, 0.25, 1.0])
        with torch.no_grad():
            model.phi_bias.copy_(torch.tensor([2.0, -1.0], dtype=torch.float64))
        values = model(self.inputs([-0.2, 0.0, 0.125, 0.25, 0.625, 1.0, 1.2]))
        torch.testing.assert_close(
            values[:, 11],
            torch.tensor([0.0, 0.0, 1.0, 2.0, 0.5, -1.0, -1.0], dtype=torch.float64),
            rtol=0,
            atol=0,
        )
        self.assertTrue(torch.equal(values[:, :11], torch.zeros_like(values[:, :11])))

    def test_interpolation_connects_parameter_and_time_derivatives(self):
        model = create_temporal_gauge_model(self.core(zero=True), [0.0, 0.25, 1.0])
        with torch.no_grad():
            model.phi_bias.copy_(torch.tensor([2.0, -1.0], dtype=torch.float64))
        values = self.inputs([0.125, 0.625]).requires_grad_(True)
        prediction = model(values)[:, 11].sum()
        parameter_gradient, input_gradient = torch.autograd.grad(
            prediction, (model.phi_bias, values)
        )
        torch.testing.assert_close(
            parameter_gradient,
            torch.tensor([1.0, 0.5], dtype=torch.float64),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            input_gradient[:, 8],
            torch.tensor([8.0, -4.0], dtype=torch.float64),
            rtol=0,
            atol=0,
        )
        self.assertTrue(
            torch.equal(input_gradient[:, :8], torch.zeros_like(input_gradient[:, :8]))
        )

    def test_training_bias_preserves_all_other_fields_and_frozen_core(self):
        model = create_temporal_gauge_model(self.core(), [0.0, 0.2, 1.0])
        values = self.inputs([0.0, 0.1, 0.2, 0.7, 1.0])
        before = model(values).detach().clone()
        original = deepcopy(model.core.state_dict())
        optimizer = torch.optim.SGD(model.branch_parameters(), lr=0.2)
        optimizer.zero_grad()
        (model(values)[:, 11] - 4).square().mean().backward()
        optimizer.step()
        after = model(values).detach()
        self.assertTrue(torch.equal(before[:, :11], after[:, :11]))
        self.assertTrue(torch.equal(before[0], after[0]))
        self.assertFalse(torch.equal(before[1:, 11], after[1:, 11]))
        self.assertTrue(
            all(parameter.grad is None for parameter in model.core.parameters())
        )
        for name, value in original.items():
            self.assertTrue(torch.equal(value, model.core.state_dict()[name]))

    def test_outer_field_preserves_initial_vacuum_and_pressure_constraints(self):
        # Trial-envelope fixture, not an accepted FE solution or trained model.
        field = conditioning.PINNConditioningTests().coupled_field()
        field.coupled_free[0, 10:12] = 0.0
        normalized = (field.data.result.times - field.start_time) / field.time_scale
        field.model = create_temporal_gauge_model(field.model, normalized)
        time_ids = np.repeat(
            np.arange(field.data.snapshot_count), field.data.point_count
        )
        point_ids = np.tile(
            np.arange(field.data.point_count), field.data.snapshot_count
        )
        before = field.predict(field.input_at(time_ids, point_ids), point_ids, time_ids)
        before = {name: value.detach().clone() for name, value in before.items()}
        optimizer = torch.optim.SGD(field.model.branch_parameters(), lr=0.2)
        optimizer.zero_grad()
        values = field.predict(field.input_at(time_ids, point_ids), point_ids, time_ids)
        (values["electric_potential"] - 3).square().mean().backward()
        optimizer.step()
        after = field.predict(field.input_at(time_ids, point_ids), point_ids, time_ids)
        initial, constrained = time_ids == 0, point_ids == 0
        self.assertTrue(
            torch.equal(
                after["electric_potential"][initial],
                before["electric_potential"][initial],
            )
        )
        self.assertTrue(
            torch.equal(
                after["electric_potential"][constrained],
                before["electric_potential"][constrained],
            )
        )
        self.assertTrue(
            torch.equal(after["pressure"][constrained], before["pressure"][constrained])
        )
        for name in before:
            if name != "electric_potential":
                self.assertTrue(torch.equal(after[name], before[name]), name)
        self.assertFalse(
            torch.equal(after["electric_potential"], before["electric_potential"])
        )

    def test_invalid_times_architecture_and_nonfinite_weights_are_rejected(self):
        for times in (
            None,
            [],
            [0.0],
            [[0.0, 1.0]],
            [0.1, 1.0],
            [0.0, 0.9],
            [0.0, 0.0, 1.0],
            [0.0, 0.8, 0.4, 1.0],
            [0.0, float("nan"), 1.0],
            [0.0, float("inf"), 1.0],
        ):
            with self.subTest(times=times), self.assertRaises(ValueError):
                create_temporal_gauge_model(self.core(), times)
        for core in (_make_network(torch, 12, 4).double(), self.core().float()):
            with self.subTest(core=core), self.assertRaises(ValueError):
                create_temporal_gauge_model(core, [0.0, 1.0])
        core = self.core()
        with torch.no_grad():
            core[-1].bias[0] = float("inf")
        with self.assertRaises(ValueError):
            create_temporal_gauge_model(core, [0.0, 1.0])

    def test_metadata_and_state_restore_only_known_temporal_bias_layout(self):
        times = [0.0, 0.17, 1.0]
        model = create_temporal_gauge_model(self.core(), times)
        with torch.no_grad():
            model.phi_bias.copy_(torch.tensor([1.25, -3.0], dtype=torch.float64))
        metadata = model.architecture_metadata()
        self.assertEqual(metadata["kind"], "coupled_temporal_phi")
        self.assertEqual(metadata["normalized_times"], times)
        self.assertEqual(metadata["bias_output"], 11)
        self.assertEqual(metadata["version"], 1)
        restored = create_temporal_gauge_model(
            self.core(), metadata["normalized_times"]
        )
        restored.load_state_dict(model.state_dict(), strict=True)
        values = self.inputs([0.0, 0.1, 0.2, 0.9, 1.0])
        self.assertTrue(torch.equal(restored(values), model(values)))
        metadata["normalized_times"][1] = 0.5
        self.assertEqual(model.architecture_metadata()["normalized_times"], times)

    def test_empty_batches_and_invalid_input_times(self):
        model = create_temporal_gauge_model(self.core(), [0.0, 1.0])
        self.assertEqual(
            model(torch.empty((0, 21), dtype=torch.float64)).shape, (0, 12)
        )
        with self.assertRaises(ValueError):
            model(torch.empty((2, 4), dtype=torch.float64))
        values = self.inputs([float("nan")])
        with self.assertRaises(ValueError):
            model(values)


    def test_restricted_source_bound_checkpoint_roundtrip_and_knot_tamper_rejection(self):
        source = source_run()
        model = create_temporal_gauge_model(self.core(), [0., 1.])
        with torch.no_grad():
            model.phi_bias.fill_(.7)
        payload = checkpoint_for(model, source, model.architecture_metadata())
        inputs = self.inputs([0., .3, 1.])
        state = torch.random.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            torch.save(payload, path)
            restored = restore_model_from_checkpoint(path, source)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        self.assertTrue(torch.equal(restored(inputs), model(inputs)))
        for location in ("metadata", "state"):
            altered = deepcopy(payload)
            if location == "metadata":
                altered["metadata"]["architecture"]["normalized_times"][-1] = .5
            else:
                altered["model_state"]["normalized_times"][-1] = .5
            with self.subTest(location=location), self.assertRaises(ValueError):
                restore_model_from_checkpoint(altered, source)

    def test_research_warm_start_reuses_its_temporal_checkpoint(self):
        from magnetic_torus.learning import research, training
        from magnetic_torus.learning.corrections import create_correction_model

        with tempfile.TemporaryDirectory() as directory, patch.dict(
            "os.environ", {"MAGNETIC_TORUS_PINN_DEVICE": "cpu"}
        ):
            root = Path(directory)
            source = tiny_fe_source(root)
            data = training._prepare_data(source, threading.Event())
            run = training._prepare_training(data, lambda *_: None, threading.Event(), None)
            initial = root / "core.pt"
            training._checkpoint(initial, run.field.model, run.optimizer, 0, source,
                                 data.scaling, run.history, run.metadata)
            options = research.ResearchOptions(architecture="temporal-phi", adam_steps=2,
                                                lbfgs_steps=0, seconds=30.)
            copied, first, evidence = research.run_research(
                source, options, lambda *_: None, threading.Event(), root / "first", initial
            )
            self.assertTrue(evidence["completed"])
            generated_checkpoint = Path(first.metadata["checkpoint_path"])
            # The tiny all-zero FE fixture can have no potential gradient. Give
            # the produced checkpoint a nonzero prior correction so the retry
            # cannot silently reset it and still pass this regression.
            payload = torch.load(generated_checkpoint, map_location="cpu", weights_only=True)
            payload["model_state"]["phi_bias"] = torch.tensor([.25, -.4], dtype=torch.float64)
            checkpoint = root / "warm-start.pt"
            torch.save(payload, checkpoint)
            original_bytes = checkpoint.read_bytes()
            model = restore_model_from_checkpoint(checkpoint, copied)
            self.assertTrue(torch.count_nonzero(model.phi_bias).item())
            sample = self.inputs([0., .25, .5, .75, 1.])
            expected_prediction = model(sample).detach().clone()
            expected_bias = model.phi_bias.detach().clone()
            expected_core = deepcopy(model.core.state_dict())
            captured = {}
            original_validation = training.TrainingLosses.validation_loss

            def capture_pre_training(losses):
                if not captured:
                    candidate = losses.field.model
                    captured.update(prediction=candidate(sample).detach().clone(),
                                    bias=candidate.phi_bias.detach().clone(),
                                    architecture=candidate.architecture_metadata(),
                                    core_frozen=all(not parameter.requires_grad for parameter in candidate.core.parameters()))
                return original_validation(losses)

            with patch.object(training.TrainingLosses, "validation_loss", capture_pre_training):
                resumed_source, second, resumed = research.run_research(
                    copied, options, lambda *_: None, threading.Event(), root / "second", checkpoint
                )
            self.assertTrue(resumed["completed"])
            self.assertTrue(torch.equal(captured["prediction"], expected_prediction))
            self.assertTrue(torch.equal(captured["bias"], expected_bias))
            self.assertEqual(captured["architecture"], model.architecture_metadata())
            self.assertTrue(captured["core_frozen"])
            self.assertEqual(second.metadata["research_warm_start"]["source_run_id"], copied.run_id)
            self.assertEqual(second.metadata["optimizer_steps_completed"]["adam"], 2)
            self.assertEqual(checkpoint.read_bytes(), original_bytes)
            restored = restore_model_from_checkpoint(second.metadata["checkpoint_path"], resumed_source)
            for name, value in expected_core.items():
                self.assertTrue(torch.equal(restored.core.state_dict()[name], value), name)

            with self.assertRaisesRegex(ValueError, "different source"):
                research.run_research(source, options, lambda *_: None, threading.Event(),
                                      root / "wrong-source", checkpoint)
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            payload["metadata"]["architecture"]["normalized_times"][1] += .01
            bad_knots = root / "bad-knots.pt"
            torch.save(payload, bad_knots)
            with self.assertRaisesRegex(ValueError, "knots"):
                research.run_research(copied, options, lambda *_: None, threading.Event(),
                                      root / "bad-knots", bad_knots)
            correction = create_correction_model(run.field.model, ("elliptic",), torch=torch)
            correction_path = root / "correction.pt"
            training._checkpoint(correction_path, correction, torch.optim.Adam(correction.parameters()),
                                 0, source, data.scaling, run.history,
                                 {**run.metadata, "architecture": correction.architecture_metadata()})
            with self.assertRaisesRegex(ValueError, "source-bound core or temporal phi checkpoint"):
                research.run_research(source, options, lambda *_: None, threading.Event(),
                                      root / "wrong-architecture", correction_path)


class LazyGaugeImportTests(unittest.TestCase):
    def test_module_import_does_not_load_torch(self):
        command = [
            sys.executable,
            "-B",
            "-c",
            (
                "import json,sys; import magnetic_torus.learning.gauge; "
                "print(json.dumps({'torch': 'torch' in sys.modules}))"
            ),
        ]
        process = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        self.assertFalse(json.loads(process.stdout)["torch"])


if __name__ == "__main__":
    unittest.main()
