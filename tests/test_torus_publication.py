from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
import json
import os
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from magnetic_torus.models import DoseResult, PINNResult, RunResult, preset
from magnetic_torus.publication import request_cancel
from magnetic_torus.publication import JobPublication
from magnetic_torus.store import RunStore
from magnetic_torus.worker import execute


class JobPublicationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.store = RunStore(self.root / "store")
        self.config = preset("Thermal benchmark")
        self.source = RunResult("original", self.config, np.array([0., 1.]), np.array([[.3, 0., 0.]]),
                                {"temperature": np.array([[1800.], [1801.]])}, {})
        self.source_path = self.store.save_run(self.source)
        self.original = self.source_path.read_bytes()
        self.request = self.root / "request.json"

    def fake_pinn(self, source, progress, cancel, output_dir=None):
        checkpoint = Path(output_dir) / "checkpoint.pt"
        checkpoint.write_bytes(b"checkpoint fixture")
        return PINNResult("trained", source.run_id, source.config.config_hash, source.fields,
                          {"loss": np.array([1.])}, {}, {"checkpoint_path": str(checkpoint)})

    def fake_dose(self, source, index, progress, cancel, output_dir=None):
        voxel = Path(output_dir) / "transport" / "voxels.npz"
        voxel.parent.mkdir()
        np.savez(voxel, points=np.zeros((1, 3)), dose_gy=np.array([1e-9]))
        return DoseResult("dose", source.run_id, source.config.config_hash, index, "water", ("Water",),
                          np.array([1e-9]), np.array([1e-10]), 10, {"voxel_dose_path": str(voxel)})

    def run_job(self, action, *, alias=False, cancel_save=None, failure=False):
        config = replace(self.config, particle_count=8) if alias else self.config
        self.request.write_text(json.dumps({"action": action, "store": str(self.store.root),
                                           "source_path": str(self.source_path), "config": config.to_dict(),
                                           "snapshot_index": 0}), encoding="utf-8")
        def simulation(config, progress, cancel):
            return replace(self.source, run_id="new-simulation", config=config)
        def training(*args, **kwargs):
            if failure:
                raise RuntimeError("deliberate analysis failure")
            return self.fake_pinn(*args, **kwargs)
        originals = {name: getattr(RunStore, name) for name in ("save_run", "save_pinn", "save_dose")}
        def save(name):
            def wrapped(store, result):
                value = originals[name](store, result)
                if name == cancel_save and store._target_root is not None:
                    self.assertEqual(len(self.store.list_runs()), 1)  # Even serialized aliases stay hidden.
                    self.assertTrue(request_cancel(self.request))
                return value
            return wrapped
        output = StringIO()
        modules = {"magnetic_torus.physics": SimpleNamespace(run_simulation=simulation),
                   "magnetic_torus.pinn": SimpleNamespace(train_pinn=training),
                   "magnetic_torus.radiation": SimpleNamespace(calculate_dose=self.fake_dose)}
        with patch.dict("sys.modules", modules), redirect_stdout(output), \
             patch.object(RunStore, "save_run", save("save_run")), \
             patch.object(RunStore, "save_pinn", save("save_pinn")), \
             patch.object(RunStore, "save_dose", save("save_dose")):
            code = execute(self.request)
        events = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(self.source_path.read_bytes(), self.original)
        self.assertFalse(list((self.store.root / ".pending").iterdir()))
        return code, events

    def test_cancel_during_serialization_publishes_no_result_or_alias(self):
        for action, saver, alias in (("simulate", "save_run", False), ("pinn", "save_pinn", False),
                                     ("dose", "save_dose", False), ("pinn", "save_run", True),
                                     ("pinn", "save_pinn", True), ("dose", "save_dose", True)):
            with self.subTest(action=action, saver=saver, alias=alias):
                self.request.with_suffix(".cancel").unlink(missing_ok=True)
                code, events = self.run_job(action, alias=alias, cancel_save=saver)
                self.assertEqual(code, 2)
                self.assertFalse(any(event["event"] == "complete" for event in events))
                self.assertEqual(len(self.store.list_runs()), 1)
                self.assertFalse(list(self.source_path.parent.rglob("pinn-*.h5")))
                self.assertFalse(list(self.source_path.parent.rglob("dose-*.json")))

    def test_analysis_failure_does_not_publish_observer_alias(self):
        code, events = self.run_job("pinn", alias=True, failure=True)
        self.assertEqual(code, 1)
        self.assertTrue(any(event["event"] == "error" for event in events))
        self.assertEqual(len(self.store.list_runs()), 1)

    def test_successful_existing_analysis_publishes_complete_bundle(self):
        code, events = self.run_job("pinn")
        self.assertEqual(code, 0)
        complete = next(event for event in events if event["event"] == "complete")
        model = self.store.load_pinn(complete["path"])
        self.assertTrue(Path(model.metadata["checkpoint_path"]).is_file())
        self.assertNotIn(".pending", complete["path"])
        self.assertEqual(Path(complete["source_path"]), self.source_path)
        self.assertEqual(len(self.store.list_runs()), 1)
        self.assertFalse(request_cancel(self.request))  # Publication already completed.
        self.assertFalse(self.request.with_suffix(".cancel").exists())

    def test_successful_alias_is_published_with_its_analysis(self):
        code, events = self.run_job("pinn", alias=True)
        self.assertEqual(code, 0)
        complete = next(event for event in events if event["event"] == "complete")
        alias = self.store.load_run(complete["source_path"])
        model = self.store.load_pinn(complete["path"])
        self.assertEqual(model.source_run_id, alias.run_id)
        self.assertEqual(alias.config.particle_count, 8)
        self.assertEqual(len(self.store.list_runs()), 2)
        self.assertTrue(Path(model.metadata["checkpoint_path"]).is_file())

    def test_successful_dose_keeps_its_published_spatial_artifact(self):
        code, events = self.run_job("dose")
        self.assertEqual(code, 0)
        complete = next(event for event in events if event["event"] == "complete")
        dose = self.store.load_dose(complete["path"])
        voxel = Path(dose.metadata["voxel_dose_path"])
        self.assertTrue(voxel.is_relative_to(Path(complete["path"]).parent))
        self.assertTrue(voxel.is_file())
        self.assertNotIn(".pending", str(voxel))

    @unittest.skipUnless(os.name == "nt", "Windows/WSL path representation")
    def test_transport_request_paths_survive_publication(self):
        self.request.write_text("{}", encoding="utf-8")
        publication = JobPublication(self.store, self.request, lambda: False)
        try:
            output = publication.store.directory(self.source.run_id)
            particle_file = output / "primaries.bin"
            particle_file.write_bytes(b"primary fixture")
            def linux(path):
                return "/mnt/" + path.drive[0].lower() + "/" + path.as_posix()[3:]
            request = output / "transport-request.json"
            request.write_text(json.dumps({"primary_file": linux(particle_file)}), encoding="utf-8")
            publication.commit()
            final = publication.store.published_path(request)
            recorded = json.loads(final.read_text())["primary_file"]
            self.assertEqual(recorded, linux(publication.store.published_path(particle_file)))
            self.assertNotIn(".pending", recorded)
        finally:
            publication.close()

    def test_cancel_and_atomic_publish_share_the_same_lock(self):
        self.request.write_text("{}", encoding="utf-8")
        publication = JobPublication(self.store, self.request, lambda: False)
        publication.store.save_run(replace(self.source, run_id="completed"))
        entered, release, cancelled = threading.Event(), threading.Event(), threading.Event()
        accepted = []
        failures = []
        original_replace = os.replace
        def pause_rename(source, destination):
            entered.set()
            if not release.wait(5):
                raise TimeoutError("Publication test synchronization failed")
            original_replace(source, destination)
        def commit():
            try:
                publication.commit()
            except Exception as error:
                failures.append(error)
        def cancel():
            accepted.append(request_cancel(self.request))
            cancelled.set()
        try:
            with patch("magnetic_torus.publication.os.replace", pause_rename):
                publishing = threading.Thread(target=commit)
                publishing.start()
                self.assertTrue(entered.wait(5))
                cancelling = threading.Thread(target=cancel)
                cancelling.start()
                time.sleep(.02)
                self.assertFalse(cancelled.is_set())
                release.set()
                publishing.join(5)
                cancelling.join(5)
            self.assertFalse(failures)
            self.assertEqual(accepted, [False])
            self.assertEqual(len(self.store.list_runs()), 2)
            self.assertFalse(self.request.with_suffix(".cancel").exists())
        finally:
            release.set()
            publication.close()


if __name__ == "__main__":
    unittest.main()
