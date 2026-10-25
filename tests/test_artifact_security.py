from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np

from magnetic_torus.artifacts import FieldArtifact
from magnetic_torus.models import DoseResult, RunResult, preset
from magnetic_torus.store import RunStore


class ArtifactSecurityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.store = RunStore(self.root / "store")
        self.result = RunResult("source", preset("Thermal benchmark"), np.array([0., 1.]),
                                np.array([[.3, 0., 0.]]), {"temperature": np.array([[1800.], [1801.]])}, {})

    def captured(self, name, value):
        path = self.root / name
        np.savez(path, value=np.array([value]))
        return path, FieldArtifact.capture(path, root=self.root)

    def test_metadata_cannot_authorize_file_read_or_resave(self):
        secret = self.root / "private.txt"
        secret.write_bytes(b"private fixture, never embed")
        path = self.store.save_run(self.result)
        with h5py.File(path, "r+") as file:
            file.attrs["metadata"] = json.dumps({key: str(secret) for key in RunStore.FIELD_ARTIFACTS})
        original_read = Path.read_bytes
        def read_guard(path):
            if path == secret:
                self.fail("Imported metadata caused an arbitrary file read")
            return original_read(path)
        with patch.object(Path, "read_bytes", read_guard):
            imported = self.store.load_run(path)
            self.assertFalse(imported.artifacts)
            self.assertFalse(any(key in imported.metadata for key in RunStore.FIELD_ARTIFACTS))
            copied = self.store.save_run(replace(imported, run_id="copy"))
            with h5py.File(copied, "r") as file:
                self.assertEqual(len(file["artifacts"]), 0)
        with self.assertRaisesRegex(ValueError, "uncaptured"):
            self.store.save_run(replace(self.result, metadata={"exterior_field_path": str(secret)}))

    def test_two_imports_keep_distinct_immutable_artifacts(self):
        exported = self.root / "exports"
        exported.mkdir()
        runs = []
        for name, value in (("first", 1), ("second", 2)):
            source, captured = self.captured(name + ".npz", value)
            result = replace(self.result, run_id=name, metadata={"exterior_field_path": str(source)},
                             artifacts={"exterior_field_path": captured})
            path = self.store.save_run(result)
            export = exported / f"{name}.h5"
            export.write_bytes(path.read_bytes())
            runs.append(self.store.load_run(export))
        first_path, second_path = (Path(run.metadata["exterior_field_path"]) for run in runs)
        self.assertNotEqual(first_path, second_path)
        self.assertEqual(first_path.read_bytes(), runs[0].artifact_bytes("exterior_field_path"))
        self.assertNotEqual(runs[0].artifact_bytes("exterior_field_path"), runs[1].artifact_bytes("exterior_field_path"))
        first_path.write_bytes(b"changed after accepted snapshot")
        self.assertNotEqual(first_path.read_bytes(), runs[0].artifact_bytes("exterior_field_path"))

    def test_cache_hit_does_not_skip_embedded_checksum(self):
        source, captured = self.captured("field.npz", 1)
        path = self.store.save_run(replace(self.result, metadata={"exterior_field_path": str(source)},
                                          artifacts={"exterior_field_path": captured}))
        self.store.load_run(path)
        with h5py.File(path, "r+") as file:
            values = file["artifacts/exterior_field_path"]
            values[0] = int(values[0]) ^ 1
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.store.load_run(path)

    def test_digest_cannot_become_a_path(self):
        path = self.store.save_run(self.result)
        with h5py.File(path, "r+") as file:
            dataset = file["artifacts"].create_dataset("exterior_field_path", data=np.arange(3, dtype=np.uint8))
            dataset.attrs["sha256"] = "../../outside"
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            self.store.load_run(path)

    def test_hdf_links_external_storage_and_virtual_arrays_are_rejected(self):
        external = self.root / "other.h5"
        with h5py.File(external, "w") as file:
            file.create_dataset("data", data=np.arange(4, dtype=np.uint8))
        for kind in ("external_link", "soft_link", "external_storage", "virtual"):
            with self.subTest(kind=kind):
                path = self.store.save_run(replace(self.result, run_id=kind))
                with h5py.File(path, "r+") as file:
                    group = file["artifacts"]
                    if kind == "external_link":
                        group["exterior_field_path"] = h5py.ExternalLink(str(external), "data")
                    elif kind == "soft_link":
                        file.create_dataset("hidden", data=np.arange(4, dtype=np.uint8))
                        group["exterior_field_path"] = h5py.SoftLink("/hidden")
                    elif kind == "external_storage":
                        group.create_dataset("exterior_field_path", shape=(4,), dtype="u1", external=[(str(self.root / "raw"), 0, 4)])
                    else:
                        layout = h5py.VirtualLayout(shape=(4,), dtype="u1")
                        layout[:] = h5py.VirtualSource(str(external), "data", shape=(4,))
                        group.create_virtual_dataset("exterior_field_path", layout)
                    # The digest is valid-shaped; storage/link rejection must
                    # happen before any external dataset bytes are requested.
                    if kind in {"external_storage", "virtual"}:
                        group["exterior_field_path"].attrs["sha256"] = sha256(bytes(range(4))).hexdigest()
                with self.assertRaisesRegex(ValueError, "External|soft"):
                    self.store.load_run(path)

    def test_capture_rejects_a_file_outside_its_producer(self):
        source, _ = self.captured("outside.npz", 1)
        producer = self.root / "producer"
        producer.mkdir()
        with self.assertRaisesRegex(ValueError, "producing directory"):
            FieldArtifact.capture(source, root=producer)

    def test_nested_thermal_operator_metadata_cannot_open_a_file(self):
        from magnetic_torus.pinn import ThermalOperator, UnsupportedPINNConfiguration
        path = self.store.save_run(self.result)
        with h5py.File(path, "r+") as file:
            file.attrs["metadata"] = json.dumps({"pinn_physics": {"kind": "thermal_fv_lumped_v1",
                                                                       "operator_path": str(self.root / "private.npz")}})
        imported = self.store.load_run(path)
        with patch("numpy.load", side_effect=AssertionError("Untrusted metadata authorized np.load")):
            with self.assertRaises(UnsupportedPINNConfiguration):
                ThermalOperator.from_result(imported)

    def test_imported_dose_cannot_link_unrelated_local_voxel_files(self):
        private = self.root / "unrelated.npz"
        dose = DoseResult("dose", self.result.run_id, self.result.config.config_hash, 0, "water", ("water",),
                          np.array([1e-9]), np.array([1e-10]), 10, {"voxel_dose_path": str(private)})
        imported = self.store.load_dose(self.store.save_dose(dose))
        self.assertNotIn("voxel_dose_path", imported.metadata)
        self.assertEqual(imported.dose_gy[0], dose.dose_gy[0])


if __name__ == "__main__":
    unittest.main()
