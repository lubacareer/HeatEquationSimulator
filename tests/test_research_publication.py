"""Research evidence is exposed only as a complete source/PINN bundle."""
from __future__ import annotations

from dataclasses import replace
from contextlib import redirect_stdout
import errno
from io import StringIO
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from magnetic_torus.models import PINNResult, RunResult, SimulationCancelled, preset
from magnetic_torus.store import RunStore
from scripts.pinn_research import _rename_directory_noreplace, main, publish_research_bundle
from scripts.research_supervisor import _remove_owned_stages, supervise
from simulation_suite.host_skips import host_skip_reason, is_host_limitation


def create_host_symlink(source, destination):
    """Create a test link or classify an explicit host limitation."""
    try:
        os.symlink(source, destination, target_is_directory=True)
    except OSError as error:
        if not is_host_limitation(error):
            raise
        raise unittest.SkipTest(host_skip_reason("symlink", error.errno, getattr(error, "winerror", None))) from error


class ResearchPublicationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.store = RunStore(self.root / "research-store")
        self.original_store = RunStore(self.root / "original-store")
        config = preset("Thermal benchmark")
        fields = {"temperature": np.array([[1800.], [1810.]])}
        self.original = RunResult("original", config, np.array([0., 1.]), np.array([[.3, 0., 0.]]),
                                  fields, {"energy_j": np.array([0., 4.])}, {"validation_status": "test"})
        self.original_path = self.original_store.save_run(self.original)
        self.original_bytes = self.original_path.read_bytes()
        self.source = replace(self.original, run_id="research-source")
        self.checkpoint = self.root / "training" / "checkpoint.pt"
        self.checkpoint.parent.mkdir()
        self.checkpoint.write_bytes(b"preserved checkpoint")
        self.model = PINNResult("pilot", self.source.run_id, config.config_hash, fields,
                                {"loss": np.array([1., .1])}, {"heldout": .02},
                                {"checkpoint_path": str(self.checkpoint), "validation_status": "failed"})
        self.cancel = threading.Event()

    def assert_no_partial_publication(self):
        self.assertEqual(self.store.list_runs(), [])
        self.assertFalse((self.store.root / "runs" / self.source.run_id).exists())
        self.assertEqual(list(self.store.root.rglob("*.partial")), [])
        self.assertEqual(list(self.store.root.glob(".research-stage-*")), [])
        self.assertEqual(self.original_path.read_bytes(), self.original_bytes)
        self.assertEqual(self.checkpoint.read_bytes(), b"preserved checkpoint")

    def test_success_publishes_roundtrippable_pair_and_final_manifest(self):
        source_path, model_path = publish_research_bundle(self.store, self.source, self.model, self.cancel)
        expected = self.store.root.resolve() / "runs" / self.source.run_id
        self.assertEqual(source_path.parent, expected)
        self.assertEqual(model_path.parent, expected)
        self.assertEqual(json.loads((expected / "manifest.json").read_text())["path"], str(source_path))
        self.assertEqual(self.store.list_runs()[0]["run_id"], self.source.run_id)
        loaded_source = RunStore.load_run(source_path)
        loaded_model = RunStore.load_pinn(model_path)
        self.assertEqual(loaded_source.run_id, loaded_model.source_run_id)
        self.assertEqual(loaded_model.metadata["checkpoint_path"], str(self.checkpoint))
        self.assertEqual(loaded_model.metadata["validation_status"], "failed")
        np.testing.assert_array_equal(loaded_model.fields["temperature"], self.model.fields["temperature"])
        self.assertEqual(self.original_path.read_bytes(), self.original_bytes)
        self.assertEqual(list(self.store.root.glob(".research-stage-*")), [])
        self.assertEqual(list(self.store.root.rglob("*.partial")), [])

    def test_source_and_model_save_failures_remove_all_staging(self):
        for method in ("save_run", "save_pinn"):
            with self.subTest(method=method):
                def fail_with_partial(stage, result):
                    directory = stage.directory(self.source.run_id)
                    (directory / "fixture.h5.partial").write_bytes(b"incomplete")
                    raise OSError("injected disk full")
                with patch.object(RunStore, method, fail_with_partial):
                    with self.assertRaisesRegex(OSError, "injected disk full"):
                        publish_research_bundle(self.store, self.source, self.model, self.cancel)
                self.assert_no_partial_publication()

    def test_rename_failure_cleans_stage_and_preserves_original_error(self):
        with patch("scripts.pinn_research._rename_directory_noreplace", side_effect=OSError("rename failed")):
            with self.assertRaisesRegex(OSError, "rename failed"):
                publish_research_bundle(self.store, self.source, self.model, self.cancel)
        self.assert_no_partial_publication()

    def test_cancellation_after_model_save_does_not_publish(self):
        save = RunStore.save_pinn
        def save_then_cancel(stage, model):
            path = save(stage, model)
            self.cancel.set()
            return path
        with patch.object(RunStore, "save_pinn", save_then_cancel):
            with self.assertRaisesRegex(SimulationCancelled, "before publishing"):
                publish_research_bundle(self.store, self.source, self.model, self.cancel)
        self.assert_no_partial_publication()

    def test_collision_preserves_existing_destination(self):
        destination = self.store.root / "runs" / self.source.run_id
        destination.mkdir(parents=True)
        marker = destination / "existing"
        marker.write_bytes(b"do not replace")
        with self.assertRaises(FileExistsError):
            publish_research_bundle(self.store, self.source, self.model, self.cancel)
        self.assertEqual(marker.read_bytes(), b"do not replace")
        self.assertEqual(list(self.store.root.glob(".research-stage-*")), [])
        self.assertEqual(self.original_path.read_bytes(), self.original_bytes)

    def test_collision_appearing_during_staging_does_not_replace_it(self):
        destination = self.store.root / "runs" / self.source.run_id
        save = RunStore.save_pinn
        def save_then_collide(stage, model):
            path = save(stage, model)
            destination.mkdir(parents=True)
            (destination / "existing").write_bytes(b"do not replace")
            return path
        with patch.object(RunStore, "save_pinn", save_then_collide):
            with self.assertRaises(FileExistsError):
                publish_research_bundle(self.store, self.source, self.model, self.cancel)
        self.assertEqual((destination / "existing").read_bytes(), b"do not replace")
        self.assertEqual(list(self.store.root.glob(".research-stage-*")), [])
        self.assertEqual(list(self.store.root.rglob("*.partial")), [])
        self.assertEqual(self.original_path.read_bytes(), self.original_bytes)

    def test_atomic_rename_rejects_empty_destination_created_after_check(self):
        destination = self.store.root / "runs" / self.source.run_id
        def race(source, target):
            destination.mkdir(parents=True)
            _rename_directory_noreplace(source, target)
        with patch("scripts.pinn_research._rename_directory_noreplace", side_effect=race):
            with self.assertRaises(FileExistsError):
                publish_research_bundle(self.store, self.source, self.model, self.cancel)
        self.assertTrue(destination.is_dir())
        self.assertEqual(list(destination.iterdir()), [])
        self.assertEqual(list(self.store.root.glob(".research-stage-*")), [])
        self.assertEqual(self.original_path.read_bytes(), self.original_bytes)

    def test_mismatched_source_cannot_publish(self):
        with self.assertRaisesRegex(ValueError, "must belong"):
            publish_research_bundle(self.store, self.source, replace(self.model, source_run_id="other"), self.cancel)
        self.assert_no_partial_publication()

    def test_cli_reports_save_failure_for_supervised_and_unsupervised_workers(self):
        from magnetic_torus.learning import research
        def fail_with_partial(stage, _model):
            directory = stage.directory(self.source.run_id)
            (directory / "pinn.h5.partial").write_bytes(b"partial")
            raise OSError("injected disk full")
        evidence = {"status": "FAIL", "completed": True}
        for supervised in (False, True):
            with self.subTest(supervised=supervised):
                output = self.root / ("supervised" if supervised else "unsupervised")
                args = ["--source", str(self.original_path), "--output", str(output), "--adam-steps", "1"]
                if supervised:
                    args += ["--architecture", "method5", "--_supervised-worker"]
                with (patch.object(research, "run_research", return_value=(self.source, self.model, evidence)),
                      patch.object(RunStore, "save_pinn", fail_with_partial),
                      patch.dict(sys.modules, {"torch": SimpleNamespace(set_num_threads=lambda _: None)}),
                      redirect_stdout(StringIO())):
                    code = main(args)
                report_path = output / ("worker-report.json" if supervised else "report.json")
                report = json.loads(report_path.read_text(encoding="utf-8"))
                self.assertEqual(code, 1)
                self.assertEqual(report["status"], "ERROR")
                self.assertIn("injected disk full", report["error"])
                self.assertTrue(report["originals_unchanged"])
                self.assertNotIn("source_path", report)
                self.assertNotIn("pinn_path", report)
                self.assertNotIn("completed", report)
                self.assertNotIn("metrics", report)
                self.assertNotIn("metadata", report)
                self.assertEqual(RunStore(output / "store").list_runs(), [])
                self.assertEqual(list((output / "store").rglob("*.partial")), [])
                self.assertEqual(list((output / "store").glob(".research-stage-*")), [])
        self.assertEqual(self.original_path.read_bytes(), self.original_bytes)
        self.assertEqual(self.checkpoint.read_bytes(), b"preserved checkpoint")

    def test_timeout_retains_timeout_status_without_publishing(self):
        from magnetic_torus.learning import research
        output = self.root / "timeout"
        def timed_out(_source, _options, _progress, cancel, *_args):
            cancel.set()
            return self.source, self.model, {"status": "TIMEOUT", "completed": False}
        with (patch.object(research, "run_research", side_effect=timed_out),
              patch.dict(sys.modules, {"torch": SimpleNamespace(set_num_threads=lambda _: None)}),
              redirect_stdout(StringIO())):
            code = main(["--source", str(self.original_path), "--output", str(output), "--adam-steps", "1"])
        report = json.loads((output / "report.json").read_text(encoding="utf-8"))
        self.assertEqual((code, report["status"]), (124, "TIMEOUT"))
        self.assertEqual(RunStore(output / "store").list_runs(), [])
        self.assertNotIn("source_path", report)

    def test_supervisor_cleans_only_the_forcibly_terminated_workers_stage(self):
        output = self.root / "forced"
        output.mkdir()
        worker = "\n".join((
            "from pathlib import Path",
            "import os, sys, time",
            "root = Path(sys.argv[1]) / 'store'",
            "root.mkdir()",
            "pid = os.getpid()",
            "mine = root / ('.research-stage-%s-own' % pid)",
            "other = root / ('.research-stage-%s-other' % (pid + 1))",
            "mine.mkdir(); other.mkdir()",
            "(mine / 'incomplete.h5.partial').write_bytes(b'partial')",
            "(other / 'keep').write_bytes(b'keep')",
            "(root / 'pid').write_text(str(pid))",
            "while True: time.sleep(.05)",
        ))
        provenance = {"options": {}, "selected_inputs": [], "canonical": False,
                      "implementation_sha256": {}, "original_sha256": {}}
        code, report = supervise([sys.executable, "-B", "-c", worker, str(output)], output,
                                 started=time.monotonic(), seconds=6., provenance=provenance,
                                 stop=threading.Event(), cwd=self.root)
        self.assertEqual((code, report["status"]), (124, "TIMEOUT"))
        store = output / "store"
        pid = int((store / "pid").read_text())
        self.assertFalse((store / f".research-stage-{pid}-own").exists())
        self.assertEqual((store / f".research-stage-{pid + 1}-other" / "keep").read_bytes(), b"keep")
        self.assertEqual(report["supervisor"]["cleanup_notes"], [])

    def test_cleanup_errors_are_reported_without_touching_other_stages(self):
        output = self.root / "cleanup"
        store = output / "store"
        store.mkdir(parents=True)
        owned = store / ".research-stage-77-own"
        other = store / ".research-stage-78-other"
        owned.mkdir()
        other.mkdir()
        with patch("scripts.research_supervisor.shutil.rmtree", side_effect=OSError("cleanup denied")):
            notes = _remove_owned_stages(output.resolve(), 77)
        self.assertTrue(any("cleanup denied" in note for note in notes))
        self.assertTrue(owned.is_dir())
        self.assertTrue(other.is_dir())

    def test_supervisor_records_cleanup_failure(self):
        output = self.root / "cleanup-notes"
        output.mkdir()
        provenance = {"options": {}, "selected_inputs": [], "canonical": False,
                      "implementation_sha256": {}, "original_sha256": {}}
        with patch("scripts.research_supervisor._remove_owned_stages", return_value=["injected cleanup failure"]):
            code, report = supervise([sys.executable, "-B", "-c", "pass"], output,
                                     started=time.monotonic(), seconds=6., provenance=provenance,
                                     stop=threading.Event(), cwd=self.root)
        self.assertEqual((code, report["status"]), (1, "ERROR"))
        self.assertEqual(report["supervisor"]["cleanup_notes"], ["injected cleanup failure"])
        written = json.loads((output / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(written["supervisor"]["cleanup_notes"], ["injected cleanup failure"])

    def test_cleanup_does_not_follow_stage_symlink(self):
        output = self.root / "symlink-output"
        store = output / "store"
        store.mkdir(parents=True)
        external = self.root / "external"
        external.mkdir()
        (external / "keep").write_bytes(b"keep")
        link = store / ".research-stage-77-link"
        create_host_symlink(external, link)
        notes = _remove_owned_stages(output.resolve(), 77)
        self.assertTrue(link.is_symlink())
        self.assertEqual((external / "keep").read_bytes(), b"keep")
        self.assertTrue(any("skipped linked" in note for note in notes))

    def test_symlink_unavailability_distinguishes_host_and_io_errors(self):
        with patch("os.symlink", side_effect=OSError(errno.EPERM, "not permitted")):
            with self.assertRaisesRegex(unittest.SkipTest, r"\[host-inapplicable:symlink:"):
                create_host_symlink("source", "destination")
        with patch("os.symlink", side_effect=OSError(errno.ENOSPC, "disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                create_host_symlink("source", "destination")


if __name__ == "__main__":
    unittest.main()
