"""Verify portable input identity and fail-closed container lock maintenance."""
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Event, local
import unittest
from unittest.mock import patch

CHECKER = Path(__file__).resolve().parents[1] / "docker/check_container_lock.py"
spec = importlib.util.spec_from_file_location("container_lock_checker", CHECKER)
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)
generator_spec = importlib.util.spec_from_file_location("container_lock_generator", CHECKER.with_name("generate_container_lock.py"))
generator = importlib.util.module_from_spec(generator_spec)
with patch.dict(sys.modules, {"check_container_lock": checker}):
    generator_spec.loader.exec_module(generator)


class ContainerLockTests(unittest.TestCase):
    def fixture(self, root):
        for name in checker.INPUTS:
            (root / name).write_bytes(b"# fixture\nnumpy==2.5.4\n")
        record = {"source_requirements_normalized_sha256": {
            name: checker.normalized_digest(root / name) for name in checker.INPUTS}}
        generator.publish_lock_set(root, "pip==26.3\n", "numpy==2.5.4\n", record)
        return root / "requirements-lock-provenance.json", record

    def test_lf_and_crlf_inputs_match_but_a_requirement_update_fails(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            provenance, _ = self.fixture(root)
            checker.check(root, provenance)
            for name in (*checker.INPUTS, *checker.LOCK_FILES):
                path = root / name
                path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
            checker.check(root, provenance)
            (root / "requirements-gpu.txt").write_bytes(b"tensorflow==2.22.0\n")
            result = subprocess.run([sys.executable, "-I", str(CHECKER), "--requirements-root", str(root),
                                     "--provenance", str(provenance)], capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 2)
            self.assertIn("Container lock is stale for requirements-gpu.txt", result.stderr)
            self.assertIn("regenerate", result.stderr)

    def test_missing_or_extra_provenance_inputs_are_rejected(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            provenance = root / "provenance.json"
            for expected in ({}, {"requirements.txt": "0" * 64, "other.txt": "0" * 64}):
                provenance.write_text(json.dumps({"source_requirements_normalized_sha256": expected}), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "unexpected requirement inputs"):
                    checker.check(root, provenance)

    def test_each_modified_or_missing_lock_is_rejected(self):
        for name in checker.LOCK_FILES:
            for action in ("modify", "remove"):
                with self.subTest(lock=name, action=action), TemporaryDirectory() as directory:
                    root = Path(directory)
                    provenance, _ = self.fixture(root)
                    if action == "modify":
                        (root / name).write_text("unexpected-package==1.0\n", encoding="utf-8")
                    else:
                        (root / name).unlink()
                    result = subprocess.run([sys.executable, "-I", str(CHECKER), "--requirements-root", str(root),
                                             "--provenance", str(provenance)], capture_output=True, text=True, timeout=10)
                    self.assertEqual(result.returncode, 2)
                    self.assertIn(name, result.stderr)

    def test_interrupted_publication_is_rejected_before_installation(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            provenance, record = self.fixture(root)
            previous_record = provenance.read_bytes()
            replace = generator.os.replace

            def interrupt_commit(source, destination):
                if Path(destination) == provenance:
                    raise OSError("fixture interruption before provenance commit")
                return replace(source, destination)

            with patch.object(generator.os, "replace", side_effect=interrupt_commit):
                with self.assertRaisesRegex(OSError, "fixture interruption"):
                    generator.publish_lock_set(root, "pip==26.2\n", "numpy==2.5.3\n", record)
            self.assertEqual(provenance.read_bytes(), previous_record)
            with self.assertRaisesRegex(ValueError, "stale"):
                checker.check(root, provenance)
            self.assertFalse(list(root.glob(".requirements-staging-*")))

    def test_competing_publications_cannot_accept_a_mixed_inventory(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            provenance, record = self.fixture(root)
            replaced = {name: Event() for name in ("a-bootstrap", "b-application", "a-provenance")}
            publisher = local()
            replace = generator.os.replace

            def interleave(source, destination):
                name, label = Path(destination).name, publisher.label
                prerequisite = None
                if label == "b" and name == checker.LOCK_FILES[0]:
                    prerequisite = "a-bootstrap"
                elif label == "a" and name == checker.LOCK_FILES[1]:
                    prerequisite = "b-application"
                elif label == "b" and name == provenance.name:
                    prerequisite = "a-provenance"
                if prerequisite and not replaced[prerequisite].wait(5):
                    raise RuntimeError("Competing publication fixture did not progress")
                result = replace(source, destination)
                if label == "a" and name == checker.LOCK_FILES[0]:
                    replaced["a-bootstrap"].set()
                elif label == "b" and name == checker.LOCK_FILES[1]:
                    replaced["b-application"].set()
                elif label == "a" and name == provenance.name:
                    replaced["a-provenance"].set()
                return result

            def publish(label):
                publisher.label = label
                generator.publish_lock_set(root, f"pip=={label}\n", f"numpy=={label}\n",
                                           {**record, "fixture_generation": label})

            with patch.object(generator.os, "replace", side_effect=interleave):
                with ThreadPoolExecutor(max_workers=2) as workers:
                    list(workers.map(publish, ("a", "b")))
            self.assertEqual((root / checker.LOCK_FILES[0]).read_text(encoding="utf-8"), "pip==b\n")
            self.assertEqual((root / checker.LOCK_FILES[1]).read_text(encoding="utf-8"), "numpy==a\n")
            self.assertEqual(json.loads(provenance.read_text(encoding="utf-8"))["fixture_generation"], "b")
            with self.assertRaisesRegex(ValueError, "stale"):
                checker.check(root, provenance)
            self.assertFalse(list(root.glob(".requirements-staging-*")))


if __name__ == "__main__":
    unittest.main()
