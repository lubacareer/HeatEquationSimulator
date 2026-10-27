"""Pressure experiments preserve source binding and training-only selection."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

from magnetic_torus.learning import research, training
from magnetic_torus.learning.corrections import (
    create_correction_model, create_pressure_fourier_model, restore_model_from_checkpoint,
)
from magnetic_torus.learning.fields import _make_network
from magnetic_torus.learning.pressure_pilots import PressureHuberLosses, PressureTemporalSampler
from magnetic_torus.models import SimulationCancelled
from magnetic_torus.store import RunStore
from scripts.pinn_research import main as research_main, stage_source
from scripts.research_supervisor import supervise
from test_pinn_correction_model import checkpoint_for, source_run
from test_pinn_research_scaling import tiny_fe_source

try:
    import torch
except ImportError:
    torch = None

SUITE_CAPABILITIES = ("torch_cpu",)


@unittest.skipIf(torch is None, "PyTorch is required")
class PressurePilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_pressure_fourier_preserves_initial_predictions_and_other_outputs(self):
        torch.manual_seed(17)
        core = _make_network(torch, 12, 21).double()
        baseline = create_correction_model(core, ("elliptic", "thermofluid"), torch=torch)
        inputs = torch.randn(19, 21, dtype=torch.float64)
        expected = baseline(inputs).detach().clone()
        rng = torch.random.get_rng_state().clone()
        pilot = create_pressure_fourier_model(baseline, torch=torch)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertTrue(torch.equal(expected, pilot(inputs)))
        with torch.no_grad():
            pilot.pressure_weight.fill_(.2)
        changed = pilot(inputs).detach()
        self.assertTrue(torch.equal(changed[:, :10], expected[:, :10]))
        self.assertTrue(torch.equal(changed[:, 11:], expected[:, 11:]))
        self.assertFalse(torch.equal(changed[:, 10], expected[:, 10]))

        source = source_run()
        payload = checkpoint_for(pilot, source, pilot.architecture_metadata())
        restored = restore_model_from_checkpoint(payload, source)
        self.assertTrue(torch.equal(restored(inputs), changed))
        with self.assertRaisesRegex(ValueError, "different source"):
            restore_model_from_checkpoint(payload, replace(source, run_id="other"))
        corrupt = deepcopy(payload)
        corrupt["metadata"]["architecture"]["frequency"] = 32
        with self.assertRaisesRegex(ValueError, "architecture metadata"):
            restore_model_from_checkpoint(corrupt, source)

    def source_pair(self, directory):
        source = tiny_fe_source(directory)
        pressure = np.arange(source.fields["pressure"].size, dtype=np.float64).reshape(
            source.fields["pressure"].shape) * .01
        source = replace(source, fields={**source.fields, "pressure": pressure})
        prepared = training._prepare_data(source, threading.Event())
        ti, pi = np.divmod(prepared.validation_ids, prepared.point_count)
        altered = pressure.copy()
        altered[ti[ti > 0], pi[ti > 0]] += 1000.
        self.assertTrue(np.any(ti > 0))
        changed_fields = {**source.fields, "pressure": altered}
        for name in ("velocity", "magnetic_field", "vector_potential"):
            values = source.fields[name].copy()
            values[ti[ti > 0], pi[ti > 0]] += 1000.
            changed_fields[name] = values
        return source, replace(source, fields=changed_fields)

    def source_checkpoint(self, source, directory):
        data = training._prepare_data(source, threading.Event())
        run = training._prepare_training(data, lambda *_: None, threading.Event(), None)
        checkpoint = Path(directory) / "source-checkpoint.pt"
        training._checkpoint(checkpoint, run.field.model, run.optimizer, 0, source,
                             data.scaling, run.history, run.metadata)
        return checkpoint

    def test_huber_values_gradients_and_unchanged_validation(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
                "os.environ", {"MAGNETIC_TORUS_PINN_DEVICE": "cpu"}):
            source, _ = self.source_pair(directory)
            data = training._prepare_data(source, threading.Event())
            run = training._prepare_training(data, lambda *_: None, threading.Event(), None)
            original = run.losses
            huber = PressureHuberLosses(original)
            self.assertEqual(huber.delta, .003 / data.scaling.scales[huber.column])
            self.assertEqual(huber.validation_loss(), original.validation_loss())
            ids = data.train_ids[:4]
            ti, pi = np.divmod(ids, data.point_count)
            targets = (data.reference[ti, pi] - data.scaling.offsets) / data.scaling.scales
            normalized = torch.as_tensor(targets, **run.field.kw)
            offsets = torch.zeros_like(normalized)
            offsets[:, huber.column] = torch.tensor([0., .5, -2., 3.], dtype=torch.float64) * huber.delta
            offsets[:, 0] = .123
            offsets.requires_grad_()
            with patch.object(run.field, "forward", return_value=normalized + offsets):
                errors, _ = huber.errors_at(ids, training=True)
            pressure = errors[:, huber.column]
            expected = torch.tensor([0., .25, 3., 5.], dtype=torch.float64) * huber.delta ** 2
            self.assertTrue(torch.allclose(pressure, expected, atol=1e-12))
            pressure.sum().backward()
            expected_gradient = torch.tensor([0., 1., -2., 2.], dtype=torch.float64) * huber.delta
            self.assertTrue(torch.allclose(offsets.grad[:, huber.column], expected_gradient, atol=1e-12))
            self.assertTrue(torch.allclose(errors[:, 0], offsets[:, 0].square()))
            with self.assertRaisesRegex(ValueError, "held-out"):
                huber.errors_at(data.validation_ids[:1], training=True)

    def test_huber_refinement_and_selection_ignore_heldout_labels(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
                "os.environ", {"MAGNETIC_TORUS_PINN_DEVICE": "cpu"}):
            sources = self.source_pair(directory)
            checkpoint = self.source_checkpoint(sources[0], directory)
            payloads = []
            for index, source in enumerate(sources):
                options = research.ResearchOptions(architecture="method5", pressure_huber=True,
                    adam_steps=2, lbfgs_steps=1, seconds=30.)
                _, model, evidence = research.run_research(source, options, lambda *_: None,
                    threading.Event(), Path(directory) / f"huber-{index}", checkpoint)
                self.assertTrue(evidence["completed"])
                self.assertEqual(model.metadata["research_pressure_loss"]["delta_pa"], .003)
                self.assertEqual(model.metadata["optimizer_steps_completed"]["lbfgs"], 1)
                payloads.append((model, torch.load(model.metadata["checkpoint_path"],
                                                  map_location="cpu", weights_only=True)))
            self.assertEqual(payloads[0][0].metadata["research_selection"], payloads[1][0].metadata["research_selection"])
            for name, value in payloads[0][1]["model_state"].items():
                self.assertTrue(torch.equal(value, payloads[1][1]["model_state"][name]), name)
            self.assertNotEqual(payloads[0][0].metrics["validation_loss"], payloads[1][0].metrics["validation_loss"])

    def test_pressure_time_weights_are_bounded_and_independent_of_heldout_labels(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
                "os.environ", {"MAGNETIC_TORUS_PINN_DEVICE": "cpu"}):
            samplers = []
            for source in self.source_pair(directory):
                data = training._prepare_data(source, threading.Event())
                run = training._prepare_training(data, lambda *_: None, threading.Event(), None)
                samplers.append(PressureTemporalSampler(run.losses))
            first, second = samplers
            np.testing.assert_array_equal(first.weights, second.weights)
            np.testing.assert_array_equal(first.scores, second.scores)
            self.assertLessEqual(first.weights.max() / first.weights.min(), 3.)
            self.assertTrue(np.any(first.pairs > 0))
            self.assertTrue(torch.equal(first.sampled_loss(), second.sampled_loss()))

    def test_method5_experiment_uses_training_only_selection_and_explicit_source_binding(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
                "os.environ", {"MAGNETIC_TORUS_PINN_DEVICE": "cpu"}):
            results = []
            sources = self.source_pair(directory)
            checkpoint = self.source_checkpoint(sources[0], directory)
            for index, source in enumerate(sources):
                options = research.ResearchOptions(strategy="temporal-pressure",
                    architecture="method5", adam_steps=2, lbfgs_steps=0, seconds=30.)
                copied, model, evidence = research.run_research(source, options, lambda *_: None,
                    threading.Event(), Path(directory) / f"trial-{index}", checkpoint)
                self.assertTrue(evidence["completed"])
                self.assertEqual(model.metadata["research_weak_normalization"],
                    "training labels only; original oracle restored for final qualification")
                self.assertEqual(model.metadata["research_selection"]["source"],
                    "training labels and training-normalized raw equations only")
                self.assertEqual(model.metadata["research_pressure_sampler"]["maximum_ratio"], 3.)
                payload = torch.load(model.metadata["checkpoint_path"], map_location="cpu", weights_only=True)
                restore_model_from_checkpoint(payload, copied)
                with self.assertRaisesRegex(ValueError, "different source"):
                    restore_model_from_checkpoint(payload, source)
                results.append((model, payload))
            self.assertEqual(results[0][0].metadata["research_selection"],
                             results[1][0].metadata["research_selection"])
            self.assertEqual(results[0][0].metadata["research_pressure_sampler"],
                             results[1][0].metadata["research_pressure_sampler"])
            for name, value in results[0][1]["model_state"].items():
                self.assertTrue(torch.equal(value, results[1][1]["model_state"][name]), name)
            self.assertNotEqual(results[0][0].metrics["validation_loss"],
                                results[1][0].metrics["validation_loss"])

    def test_source_bound_method5_checkpoint_warm_starts_both_pressure_candidates(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
                "os.environ", {"MAGNETIC_TORUS_PINN_DEVICE": "cpu"}):
            source, _ = self.source_pair(directory)
            initial_checkpoint = self.source_checkpoint(source, directory)
            options = research.ResearchOptions(architecture="method5", adam_steps=1,
                                                lbfgs_steps=0, seconds=30.)
            with self.assertRaisesRegex(ValueError, "explicit source-bound pretrained checkpoint"):
                research.run_research(source, options, lambda *_: None, threading.Event(),
                                      Path(directory) / "no-checkpoint")
            copied, baseline, _ = research.run_research(source, options, lambda *_: None,
                threading.Event(), Path(directory) / "initial", initial_checkpoint)
            checkpoint = Path(baseline.metadata["checkpoint_path"])
            for name, candidate in (
                ("temporal", replace(options, strategy="temporal-pressure")),
                ("fourier", replace(options, architecture="pressure-fourier")),
            ):
                _, model, evidence = research.run_research(copied, candidate, lambda *_: None,
                    threading.Event(), Path(directory) / name, checkpoint)
                self.assertTrue(evidence["completed"])
                self.assertEqual(model.metadata["research_warm_start"]["source_run_id"], copied.run_id)
                self.assertIn("pressure_normalized_rmse", evidence["training_only_selection"])
            with self.assertRaisesRegex(ValueError, "different source"):
                research.run_research(source, options, lambda *_: None, threading.Event(),
                                      Path(directory) / "wrong-source", checkpoint)

    def test_pressure_deadline_does_not_publish_incomplete_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
                "os.environ", {"MAGNETIC_TORUS_PINN_DEVICE": "cpu"}):
            source, _ = self.source_pair(directory)
            checkpoint = self.source_checkpoint(source, directory)
            options = research.ResearchOptions(architecture="method5", adam_steps=1,
                                                lbfgs_steps=0, seconds=5.)
            def progress(_fraction, message):
                if message.startswith("Research"):
                    time.sleep(5.1)
            with self.assertRaises(research.ResearchPilotTimeout) as error:
                research.run_research(source, options, progress, threading.Event(),
                                      Path(directory) / "deadline", checkpoint)
            self.assertEqual(error.exception.steps, 1)
            self.assertFalse(list((Path(directory) / "deadline").rglob("checkpoint.pt")))

    def test_late_cancellation_removes_newly_published_pilot_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
                "os.environ", {"MAGNETIC_TORUS_PINN_DEVICE": "cpu"}):
            source, _ = self.source_pair(directory)
            checkpoint = self.source_checkpoint(source, directory)
            output = Path(directory) / "late-cancel"
            cancel = threading.Event()
            original = training._assemble_result
            def publish_then_cancel(*args, **kwargs):
                candidate = original(*args, **kwargs)
                cancel.set()
                return candidate
            options = research.ResearchOptions(architecture="method5", adam_steps=1,
                                                lbfgs_steps=0, seconds=30.)
            with patch.object(research.base, "_assemble_result", side_effect=publish_then_cancel):
                with self.assertRaises(SimulationCancelled):
                    research.run_research(source, options, lambda *_: None, cancel, output, checkpoint)
            self.assertTrue(checkpoint.is_file())
            self.assertFalse(list(output.rglob("checkpoint.pt")))

    def test_refined_screen_clones_only_its_source_bound_pretrained_core(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
                "os.environ", {"MAGNETIC_TORUS_PINN_DEVICE": "cpu"}):
            source, _ = self.source_pair(directory)
            source = replace(source, config=replace(source.config, seed=123))
            primary = research.ResearchOptions(architecture="mlp", adam_steps=2,
                                                lbfgs_steps=0, seconds=30.)
            copied, base_model, pretrain = research.run_research(source, primary, lambda *_: None,
                threading.Event(), Path(directory) / "pretrain")
            self.assertTrue(pretrain["completed"])
            checkpoint = Path(base_model.metadata["checkpoint_path"])
            candidate = replace(primary, architecture="method5", strategy="temporal-pressure")
            candidate_source, model, evidence = research.run_research(copied, candidate, lambda *_: None,
                threading.Event(), Path(directory) / "candidate", checkpoint)
            self.assertTrue(evidence["completed"])
            self.assertTrue(model.metadata["research_warm_start"]["cloned_sequential_core"])
            self.assertEqual(model.metadata["research_warm_start"]["source_run_id"], copied.run_id)
            self.assertEqual(candidate_source.config.seed, 123)
            with self.assertRaisesRegex(ValueError, "different source"):
                research.run_research(source, candidate, lambda *_: None, threading.Event(),
                                      Path(directory) / "mismatch", checkpoint)

    def test_supervised_cli_preserves_the_worker_report_and_source_hashes(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
                "os.environ", {"MAGNETIC_TORUS_PINN_DEVICE": "cpu"}):
            source, _ = self.source_pair(directory)
            checkpoint = self.source_checkpoint(source, directory)
            source_path = RunStore(Path(directory) / "source-store").save_run(source)
            original_files = {path.relative_to(source_path.parent): path.read_bytes()
                              for path in source_path.parent.rglob("*") if path.is_file()}
            output = Path(directory) / "supervised"
            command = [sys.executable, "-B", "scripts/pinn_research.py", "--source", str(source_path),
                       "--checkpoint", str(checkpoint), "--output", str(output),
                       "--architecture", "method5", "--adam-steps", "1", "--lbfgs-steps", "0",
                       "--seconds", "30", "--device", "cpu", "--cpu-threads", "1"]
            result = subprocess.run(command, cwd=Path(__file__).resolve().parents[1],
                                    capture_output=True, text=True, timeout=40)
            self.assertIn(result.returncode, (0, 3), result.stdout + result.stderr)
            final = json.loads((output / "report.json").read_text(encoding="utf-8"))
            worker = json.loads((output / "worker-report.json").read_text(encoding="utf-8"))
            self.assertEqual(final["status"], worker["status"])
            self.assertTrue(final["originals_unchanged"])
            self.assertTrue(final["implementation_unchanged"])
            self.assertTrue(final["supervisor"]["worker_provenance_matches"])
            self.assertEqual(final["supervisor"]["child_returncode"], result.returncode)
            self.assertEqual(final["original_source_path"], str(source_path.resolve()))
            self.assertEqual(Path(final["private_source_path"]), output / "input" / "source.h5")
            self.assertEqual(Path(final["private_source_path"]).read_bytes(), source_path.read_bytes())
            self.assertTrue(list((output / "input" / "artifacts").glob("weak_operator_path-*.npz")))
            self.assertEqual(original_files, {path.relative_to(source_path.parent): path.read_bytes()
                for path in source_path.parent.rglob("*") if path.is_file()})

    def test_private_source_copy_rejects_a_mismatched_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "accepted.h5"
            source.write_bytes(b"original accepted source")
            output = root / "experiment"
            output.mkdir()
            with self.assertRaisesRegex(ValueError, "source changed"):
                stage_source(source, output, "0" * 64)
            self.assertEqual(source.read_bytes(), b"original accepted source")
            self.assertFalse((output / "input" / "source.h5").exists())
            self.assertFalse((output / "input" / "source.h5.partial").exists())


class ResearchSupervisorTests(unittest.TestCase):
    WORKER = r"""
import json, os, pathlib, subprocess, sys, time
directory, mode = pathlib.Path(sys.argv[1]), sys.argv[2]
if mode == 'error-hang':
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
    (directory / 'descendant.pid').write_text(str(child.pid))
report = {'options': {'pilot': 1}, 'selected_inputs': ['selected-source'],
    'canonical': False, 'original_sha256': {}, 'implementation_sha256': {},
    'originals_unchanged': True, 'implementation_unchanged': True,
    'status': 'ERROR' if mode == 'error-hang' else 'PASS',
    'completed': mode != 'error-hang', 'error': 'CUDA OOM' if mode == 'error-hang' else ''}
if mode == 'wrong-source':
    report['selected_inputs'] = ['another-source']
    report['status'] = 'ERROR'
    report['completed'] = False
if mode == 'malformed':
    (directory / 'worker-report.json').write_text('{')
elif mode == 'partial':
    (directory / 'worker-report.json.partial').write_text(json.dumps(report))
else:
    partial = directory / 'worker-report.json.partial'
    partial.write_text(json.dumps(report))
    os.replace(partial, directory / 'worker-report.json')
if mode == 'pass-exit':
    sys.exit(0)
if mode == 'pass-nonzero':
    sys.exit(3)
time.sleep(60)
"""

    def invoke(self, directory, mode, *, seconds=3., stop=None, grace=.2):
        root = Path(directory)
        provenance = {"options": {"pilot": 1}, "selected_inputs": ["selected-source"],
                      "canonical": False, "original_sha256": {}, "implementation_sha256": {}}
        command = [sys.executable, "-B", "-c", self.WORKER, str(root), mode]
        return supervise(command, root, started=time.monotonic(), seconds=seconds,
                         provenance=provenance, stop=stop or threading.Event(),
                         cwd=Path(__file__).resolve().parents[1], failure_grace_s=grace)

    def test_pass_requires_successful_exit_and_a_late_pass_never_overrides_deadline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for mode, expected, expected_code in (("pass-exit", "PASS", 0),
                                                   ("pass-nonzero", "ERROR", 1),
                                                   ("pass-hang", "TIMEOUT", 124)):
                output = root / mode
                output.mkdir()
                with self.subTest(mode=mode):
                    code, result = self.invoke(output, mode)
                    self.assertEqual((code, result["status"]), (expected_code, expected))
                    self.assertEqual(json.loads((output / "worker-report.json").read_text())["status"], "PASS")
                    self.assertEqual(json.loads((output / "report.json").read_text())["status"], expected)
                    if mode == "pass-hang":
                        self.assertFalse(result["completed"])

    def test_error_report_is_preserved_and_owned_descendant_is_stopped(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
                start_new_session=os.name != "nt")
            try:
                started = time.monotonic()
                code, result = self.invoke(output, "error-hang", seconds=8.)
                self.assertEqual((code, result["status"]), (1, "ERROR"))
                self.assertEqual(result["error"], "CUDA OOM")
                self.assertLess(time.monotonic() - started, 7.)
                pid = int((output / "descendant.pid").read_text())
                deadline = time.monotonic() + 3.
                while self.process_running(pid) and time.monotonic() < deadline:
                    time.sleep(.05)
                self.assertFalse(self.process_running(pid))
                self.assertIsNone(unrelated.poll())
            finally:
                unrelated.terminate()
                unrelated.wait(timeout=5)

    @staticmethod
    def process_running(pid):
        if os.name != "nt":
            try:
                status = Path(f"/proc/{pid}/stat")
                if status.is_file() and status.read_text().rsplit(")", 1)[1].split()[0] == "Z":
                    return False
                os.kill(pid, 0)
                return True
            except (ProcessLookupError, FileNotFoundError):
                return False
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
        kernel.GetExitCodeProcess.restype = wintypes.BOOL
        kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        try:
            exit_code = wintypes.DWORD()
            return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(exit_code)) and exit_code.value == 259)
        finally:
            kernel.CloseHandle(handle)

    def test_incomplete_malformed_and_wrong_source_reports_do_not_end_the_run(self):
        with tempfile.TemporaryDirectory() as directory:
            for mode in ("partial", "malformed", "wrong-source"):
                output = Path(directory) / mode
                output.mkdir()
                with self.subTest(mode=mode):
                    code, result = self.invoke(output, mode, seconds=2.)
                    self.assertEqual((code, result["status"]), (124, "TIMEOUT"))

    def test_user_interruption_overrides_a_written_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            stop = threading.Event()
            timer = threading.Timer(.5, stop.set)
            timer.start()
            try:
                code, result = self.invoke(directory, "pass-hang", seconds=8., stop=stop)
            finally:
                timer.cancel()
            self.assertEqual((code, result["status"]), (130, "INTERRUPTED"))
            self.assertFalse(result["completed"])

    def test_stalled_final_hash_verification_cannot_turn_a_late_pass_into_success(self):
        with tempfile.TemporaryDirectory() as directory:
            def slow_hashes(_values, _root):
                time.sleep(3.)
                return True
            with patch("scripts.research_supervisor._hashes_match", side_effect=slow_hashes):
                started = time.monotonic()
                code, result = self.invoke(directory, "pass-exit", seconds=2.)
            self.assertEqual((code, result["status"]), (124, "TIMEOUT"))
            self.assertLess(time.monotonic() - started, 3.)
            self.assertIsNone(result["supervisor"]["implementation_hashes_match"])

    def test_stalled_preflight_hashing_cannot_start_a_child_or_exceed_the_deadline(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "preflight"
            args = ["--source", str(Path(directory) / "source.h5"),
                    "--checkpoint", str(Path(directory) / "checkpoint.pt"),
                    "--output", str(output), "--architecture", "method5",
                    "--adam-steps", "1", "--lbfgs-steps", "0", "--seconds", "1"]
            def stalled_hash(_path):
                time.sleep(3.)
                return "a" * 64
            with patch("scripts.pinn_research.file_hash", side_effect=stalled_hash):
                started = time.monotonic()
                code = research_main(args)
            result = json.loads((output / "report.json").read_text(encoding="utf-8"))
            self.assertEqual((code, result["status"]), (124, "TIMEOUT"))
            self.assertFalse(result["supervisor_preflight_complete"])
            self.assertIsNone(result["supervisor"]["child_returncode"])
            self.assertLess(time.monotonic() - started, 2.5)


if __name__ == "__main__":
    unittest.main()
