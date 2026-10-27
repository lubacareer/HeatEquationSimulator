"""Bounded research-driver lifecycle and unchanged qualification regressions."""

from dataclasses import replace
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

import numpy as np

from magnetic_torus.learning import research, training
from magnetic_torus.learning.corrections import restore_model_from_checkpoint
from magnetic_torus.models import SimulationCancelled
from magnetic_torus.store import RunStore
from scripts.pinn_research import main as research_main
from test_pinn_research_scaling import tiny_fe_source
from test_torus_pinn import uniform_heating_run

try:
    import torch
except ImportError:
    torch = None


SUITE_CAPABILITIES = ("torch_cpu",)


@unittest.skipIf(torch is None, "PyTorch is required")
class ResearchDriverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def options(self, **changes):
        return replace(research.ResearchOptions(adam_steps=2, lbfgs_steps=1, seconds=30.), **changes)

    def run_experiment(self, source, directory, *, options=None, cancel=None):
        with patch.dict(os.environ, {"MAGNETIC_TORUS_PINN_DEVICE": "cpu"}):
            return research.run_research(
                source, options or self.options(), lambda *_: None,
                cancel or threading.Event(), Path(directory),
            )

    def checkpoint(self, model):
        return torch.load(model.metadata["checkpoint_path"], map_location="cpu", weights_only=True)

    def test_declared_options_reject_invalid_strategies_architectures_and_budgets(self):
        for changes in (
            {"strategy": "unknown"}, {"architecture": "unknown"},
            {"adam_steps": -1}, {"adam_steps": True}, {"lbfgs_steps": True},
            {"adam_steps": 0, "lbfgs_steps": 0}, {"adam_steps": 10000, "lbfgs_steps": 1},
            {"seconds": 0.}, {"seconds": float("nan")}, {"causal_epsilon": 0.},
            {"causal_floor": 0.}, {"causal_floor": 1.1},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.options(**changes).validate()

    def test_all_strategies_preserve_source_and_record_actual_steps_and_checkpoint_identity(self):
        source = uniform_heating_run()
        original = {name: values.copy() for name, values in source.fields.items()}
        source_ids = []
        with tempfile.TemporaryDirectory() as directory:
            for strategy in ("uniform", "front", "causal", "front-causal"):
                with self.subTest(strategy=strategy):
                    copied, model, evidence = self.run_experiment(
                        source, directory, options=self.options(strategy=strategy)
                    )
                    source_ids.append(copied.run_id)
                    self.assertNotEqual(copied.run_id, source.run_id)
                    self.assertEqual(copied.metadata["reused_fields_from"], source.run_id)
                    self.assertEqual(model.source_run_id, copied.run_id)
                    self.assertEqual(model.config_hash, copied.config.config_hash)
                    self.assertEqual(model.metadata["originating_run_id"], source.run_id)
                    self.assertEqual(model.metadata["originating_config_hash"], source.config.config_hash)
                    self.assertEqual(model.metadata["actual_training_steps"], 3)
                    self.assertEqual(model.metadata["training_epochs"], 3)
                    self.assertEqual(model.metadata["requested_training_steps"], 3)
                    self.assertEqual(model.metrics["epochs"], 3.)
                    self.assertEqual(model.metrics["requested_epochs"], 3.)
                    self.assertEqual(model.metadata["optimizer_steps_completed"], {"adam": 2, "lbfgs": 1})
                    self.assertEqual(model.metadata["research_options"]["strategy"], strategy)
                    self.assertTrue(evidence["completed"])
                    self.assertEqual(evidence["status"], "PASS")
                    self.assertEqual(model.metadata["validation_status"], "passed")
                    self.assertIn("≤ 0.01", model.metadata["validation_criteria"])
                    self.assertIn("≤ 0.1", model.metadata["validation_criteria"])
                    payload = self.checkpoint(model)
                    self.assertEqual(payload["epoch"], 3)
                    self.assertEqual(payload["source_run_id"], copied.run_id)
                    self.assertEqual(payload["config_hash"], copied.config.config_hash)
                    self.assertTrue(all(value.dtype == torch.float64 for value in payload["model_state"].values()))
                    restored = restore_model_from_checkpoint(payload, copied)
                    self.assertIsNotNone(restored)
                    with self.assertRaisesRegex(ValueError, "different source"):
                        restore_model_from_checkpoint(payload, source)
                    for name, values in source.fields.items():
                        np.testing.assert_array_equal(values, original[name])
                        self.assertFalse(values.flags.writeable)
                    self.assertTrue(all(not values.flags.writeable for values in copied.fields.values()))
        self.assertEqual(len(set(source_ids)), 4)

    def test_changed_noninitial_heldout_enthalpy_cannot_change_optimized_weights_or_curricula(self):
        source = uniform_heating_run()
        # A nonuniform label fixture supplies a real optimization gradient;
        # it is not claimed to be an accepted numerical solution.
        enthalpy = source.fields["enthalpy"].copy()
        enthalpy[1:] += np.arange(1, len(source.times))[:, None] * np.array([10., -10.])
        source = replace(source, fields={**source.fields, "enthalpy": enthalpy,
            "temperature": source.config.melting_temperature + enthalpy / source.config.heat_capacity_solid},
            metadata={**source.metadata, "validation_status": "Nonuniform training-label fixture; not a solver benchmark"})
        data = training._prepare_data(source, threading.Event())
        ti, pi = np.divmod(data.validation_ids, len(source.points))
        h = source.fields["enthalpy"].copy()
        h[ti[ti > 0], pi[ti > 0]] += 50.
        changed = replace(source, fields={**source.fields, "enthalpy": h})
        models = []
        with tempfile.TemporaryDirectory() as directory:
            for item in (source, changed):
                _, model, evidence = self.run_experiment(
                    item, directory, options=self.options(strategy="front-causal", balance_labels=True)
                )
                self.assertTrue(evidence["completed"])
                models.append(model)
            states = [self.checkpoint(model)["model_state"] for model in models]
        self.assertEqual(states[0].keys(), states[1].keys())
        self.assertGreater(torch.count_nonzero(states[0]["6.weight"]).item(), 0)
        self.assertGreater(models[0].metadata["research_label_balancer"]["gradient_norms"]["enthalpy"], 0)
        for name in states[0]:
            self.assertTrue(torch.equal(states[0][name], states[1][name]), name)
        for name in models[0].fields:
            np.testing.assert_array_equal(models[0].fields[name], models[1].fields[name])
        for key in ("research_causal", "research_front_sampler", "research_label_sampler",
                    "research_label_balancer", "research_initial_label_gradient_norms",
                    "research_physics_evaluations", "function_evaluations", "training_stages"):
            self.assertEqual(models[0].metadata[key], models[1].metadata[key], key)
        self.assertNotEqual(models[0].metrics["validation_loss"], models[1].metrics["validation_loss"])

    def test_cancellation_before_preparation_does_not_start_or_publish_an_experiment(self):
        cancel = threading.Event()
        cancel.set()
        with tempfile.TemporaryDirectory() as directory, patch.object(research.base, "_prepare_training") as prepare:
            with self.assertRaises(SimulationCancelled):
                self.run_experiment(uniform_heating_run(), directory, cancel=cancel)
            prepare.assert_not_called()
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_coupled_heldout_vectors_do_not_change_weights_caches_or_training_curricula(self):
        def weak(_config, _reference, fields, _indices, *, normalization_scales=None):
            # Differentiable operator fixture with no access to reference labels.
            scales = normalization_scales or {"velocity": 1.e-4, "magnetic_field": 1.e-6,
                                                "vector_potential": 2.e-7}
            velocity = fields["velocity"][1] / scales["velocity"]
            magnetic = fields["magnetic_field"][1] / scales["magnetic_field"]
            potential = fields["vector_potential"] / scales["vector_potential"]
            return {"momentum": velocity, "continuity": velocity.sum(-1),
                    "induction": potential[1] - potential[0],
                    "charge": potential[1].sum(-1), "gauge": magnetic.sum(-1),
                    "enthalpy": (fields["enthalpy"][1] - fields["enthalpy"][0]) / 1000.}

        caches, models = [], []
        prepare = training._prepare_training

        def capture_cache(data, *args, **kwargs):
            self.assertIsNotNone(data.cache)
            caches.append(tuple(value.copy() for value in (
                data.cache.targets, data.cache.training_rows, data.cache.points, data.cache.times)))
            return prepare(data, *args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            source = tiny_fe_source(directory)
            fields = dict(source.fields)
            for name, amplitude in (("velocity", 1.e-5), ("magnetic_field", 2.e-6),
                                    ("vector_potential", 3.e-6)):
                fields[name] = fields[name].copy()
                fields[name][1:] = amplitude * np.array([1., 2., -1.])
            source = replace(source, fields=fields, metadata={**source.metadata,
                "validation_status": "Nonuniform vector label fixture; not a solver benchmark"})
            with patch.object(training, "_weak_adapter", return_value=weak):
                data = training._prepare_data(source, threading.Event())
            ti, pi = np.divmod(data.validation_ids, len(source.points))
            changed = dict(source.fields)
            for name in ("velocity", "magnetic_field", "vector_potential"):
                changed[name] = changed[name].copy()
                changed[name][ti[ti > 0], pi[ti > 0]] += 10.
            altered = replace(source, fields=changed)
            with (patch.object(training, "_weak_adapter", return_value=weak),
                  patch.object(training, "_prepare_training", side_effect=capture_cache)):
                for item in (source, altered):
                    _, model, evidence = self.run_experiment(item, directory,
                        options=self.options(strategy="causal", balance_labels=True))
                    self.assertTrue(evidence["completed"])
                    models.append(model)
            states = [self.checkpoint(model)["model_state"] for model in models]
        self.assertGreater(torch.count_nonzero(states[0]["6.weight"]).item(), 0)
        for name in states[0]:
            self.assertTrue(torch.equal(states[0][name], states[1][name]), name)
        for first, second in zip(*caches):
            np.testing.assert_array_equal(first, second)
        for name in ("research_training_weak_scales", "research_causal", "research_label_sampler",
                     "research_label_balancer", "research_initial_label_gradient_norms",
                     "research_physics_evaluations", "function_evaluations", "training_stages"):
            self.assertEqual(models[0].metadata[name], models[1].metadata[name], name)
        for name in models[0].fields:
            np.testing.assert_array_equal(models[0].fields[name], models[1].fields[name])
        self.assertNotEqual(models[0].metrics["validation_loss"], models[1].metrics["validation_loss"])

    def test_cancellation_in_last_training_progress_cannot_turn_into_qualification(self):
        cancel = threading.Event()
        requested = []

        def progress(fraction, _message):
            if not requested and abs(fraction - .9) < 1.e-12:
                requested.append(True)
                cancel.set()

        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {"MAGNETIC_TORUS_PINN_DEVICE": "cpu"}
        ):
            _, model, evidence = research.run_research(uniform_heating_run(), self.options(),
                progress, cancel, Path(directory))
            payload = self.checkpoint(model)
        self.assertTrue(requested)
        self.assertFalse(evidence["completed"])
        self.assertEqual(evidence["status"], "INTERRUPTED")
        self.assertEqual(model.metadata["validation_status"], "failed")
        self.assertEqual(model.metadata["actual_training_steps"], 3)
        self.assertEqual(model.metrics["epochs"], 3.)
        self.assertEqual(payload["epoch"], 3)
        self.assertTrue(cancel.is_set())

    def test_cancellation_during_physics_applies_no_update_and_records_zero_actual_epochs(self):
        cancel = threading.Event()
        original = training.TrainingLosses.thermal_physics
        states = []

        def cancel_first_physics(losses, *args, **kwargs):
            value = original(losses, *args, **kwargs)
            if not states:
                states.append({name: value.detach().clone() for name, value in losses.field.model.state_dict().items()})
                losses.cancel.set()
            return value

        with tempfile.TemporaryDirectory() as directory, patch.object(
            training.TrainingLosses, "thermal_physics", new=cancel_first_physics
        ):
            _, model, evidence = self.run_experiment(uniform_heating_run(), directory, cancel=cancel)
            payload = self.checkpoint(model)
        self.assertFalse(evidence["completed"])
        self.assertEqual(evidence["status"], "INTERRUPTED")
        self.assertEqual(model.metadata["validation_status"], "failed")
        self.assertEqual(model.metadata["actual_training_steps"], 0)
        self.assertEqual(model.metadata["training_epochs"], 0)
        self.assertEqual(model.metrics["epochs"], 0.)
        self.assertEqual(payload["epoch"], 0)
        self.assertEqual(model.metadata["optimizer_steps_completed"], {"adam": 0, "lbfgs": 0})
        self.assertTrue(cancel.is_set())
        for name, value in payload["model_state"].items():
            self.assertTrue(torch.equal(value, states[0][name]), name)

    def test_deadline_expiring_in_preparation_cannot_launch_the_first_optimizer_step(self):
        calls = []

        def clock():
            calls.append(True)
            return 0. if len(calls) == 1 else 2.

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(research.time, "monotonic", side_effect=clock),
            patch.object(research.threading, "Timer", return_value=Mock()),
        ):
            _, model, evidence = self.run_experiment(
                uniform_heating_run(), directory, options=self.options(seconds=1.)
            )
        self.assertFalse(evidence["completed"])
        self.assertTrue(evidence["timed_out"])
        self.assertEqual(evidence["status"], "TIMEOUT")
        self.assertEqual(model.metadata["actual_training_steps"], 0)
        self.assertEqual(model.metadata["training_epochs"], 0)
        self.assertEqual(model.metrics["epochs"], 0.)
        self.assertEqual(model.metadata["validation_status"], "failed")

    def test_training_only_weak_scales_are_restored_to_original_oracle_before_final_qualification(self):
        calls = []

        def weak(_config, _reference, fields, _indices, *, normalization_scales=None):
            calls.append(normalization_scales)
            anchor = fields["enthalpy"].sum() * 0.
            return {"momentum": anchor + (.01 if normalization_scales is not None else .2),
                    **{name: anchor for name in ("continuity", "induction", "charge", "enthalpy", "gauge")}}

        with tempfile.TemporaryDirectory() as directory:
            source = tiny_fe_source(directory)
            with patch.object(research.base, "_weak_adapter", return_value=weak):
                _, model, evidence = self.run_experiment(source, directory)
        self.assertTrue(evidence["completed"])
        self.assertEqual(evidence["status"], "FAIL")
        self.assertEqual(model.metadata["validation_status"], "failed")
        self.assertAlmostEqual(model.metrics["momentum_residual_rms"], .2)
        self.assertAlmostEqual(model.metrics["physics_loss"], .2 ** 2 / 6)
        first_original = next(index for index, value in enumerate(calls) if value is None)
        self.assertGreater(first_original, 0)
        self.assertTrue(all(value is None for value in calls[first_original:]))
        expected = {"velocity", "magnetic_field", "vector_potential"}
        self.assertTrue(all(set(value) == expected for value in calls[:first_original]))


    def test_cli_saves_strict_report_and_private_source_without_overwriting_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = uniform_heating_run(pinn_epochs=1)
            path = RunStore(root / "original").save_run(source)
            original = path.read_bytes()
            output = root / "experiment"
            args = ["--source", str(path), "--output", str(output), "--adam-steps", "1",
                    "--lbfgs-steps", "0", "--seconds", "30", "--device", "cpu"]
            with redirect_stdout(StringIO()):
                status = research_main(args)
            report = json.loads((output / "report.json").read_text())
            self.assertEqual(status, 0)
            self.assertEqual(report["status"], "PASS")
            self.assertTrue(report["originals_unchanged"])
            self.assertTrue(report["implementation_unchanged"])
            fingerprinted = {Path(key).as_posix() for key in report["implementation_sha256"]}
            self.assertTrue({"magnetic_torus/physics.py", "magnetic_torus/numerics/constants.py",
                             "magnetic_torus/numerics/thermal.py", "magnetic_torus/store.py",
                             "magnetic_torus/artifacts.py"}.issubset(fingerprinted))
            self.assertFalse(report["canonical"])
            self.assertFalse(report["desktop_modes_enabled"])
            self.assertEqual(report["metrics"]["epochs"], 1.)
            self.assertEqual(path.read_bytes(), original)
            previous = (output / "report.json").read_bytes()
            with redirect_stderr(StringIO()), self.assertRaises(SystemExit) as error:
                research_main(args)
            self.assertEqual(error.exception.code, 2)
            self.assertEqual((output / "report.json").read_bytes(), previous)

    def test_cli_records_unreadable_source_as_execution_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "experiment"
            with redirect_stdout(StringIO()):
                status = research_main(["--source", str(root / "absent.h5"), "--output", str(output)])
            report = json.loads((output / "report.json").read_text())
        self.assertEqual(status, 1)
        self.assertEqual(report["status"], "ERROR")
        self.assertIn("FileNotFoundError", report["error"])
        self.assertIsNone(report["originals_unchanged"])

    def test_warm_start_cannot_silently_change_requested_mlp_architecture(self):
        source = uniform_heating_run(pinn_epochs=3)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "input.pt"
            data = training._prepare_data(source, threading.Event())
            torch.save({"field_offsets": data.scaling.offsets.tolist(),
                        "field_scales": data.scaling.scales.tolist(),
                        "field_names": data.scaling.names, "field_widths": data.scaling.widths}, checkpoint)
            with (
                patch.dict(os.environ, {"MAGNETIC_TORUS_PINN_DEVICE": "cpu"}),
                patch("magnetic_torus.learning.corrections.restore_model_from_checkpoint",
                      return_value=torch.nn.Linear(4, 1).double()),
                self.assertRaisesRegex(ValueError, "legacy Sequential"),
            ):
                research.run_research(source, self.options(), lambda *_: None,
                                      threading.Event(), Path(directory) / "output", checkpoint)

    def test_warm_start_hash_describes_the_same_bytes_as_loaded_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, model, _ = self.run_experiment(uniform_heating_run(), root / "first")
            checkpoint = Path(model.metadata["checkpoint_path"])
            original = checkpoint.read_bytes()
            read_bytes = Path.read_bytes
            reads = []

            def changing_checkpoint(path):
                if path == checkpoint:
                    reads.append(True)
                    return original if len(reads) == 1 else b"a different file"
                return read_bytes(path)

            with patch.object(Path, "read_bytes", new=changing_checkpoint), patch.dict(
                os.environ, {"MAGNETIC_TORUS_PINN_DEVICE": "cpu"}
            ):
                _, candidate, _ = research.run_research(source, self.options(), lambda *_: None,
                    threading.Event(), root / "second", checkpoint)
        self.assertEqual(len(reads), 1)
        self.assertEqual(candidate.metadata["research_warm_start"]["checkpoint_sha256"],
                         sha256(original).hexdigest())


if __name__ == "__main__":
    unittest.main()
