"""Verify portable input identity and fail-closed container lock maintenance."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

CHECKER = Path(__file__).resolve().parents[1] / "docker/check_container_lock.py"
spec = importlib.util.spec_from_file_location("container_lock_checker", CHECKER)
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


class ContainerLockTests(unittest.TestCase):
    def test_lf_and_crlf_inputs_match_but_a_requirement_update_fails(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for name in checker.INPUTS:
                (root / name).write_bytes(b"# fixture\nnumpy==2.5.4\n")
            expected = {name: checker.normalized_digest(root / name) for name in checker.INPUTS}
            provenance = root / "provenance.json"
            provenance.write_text(json.dumps({"source_requirements_normalized_sha256": expected}), encoding="utf-8")
            checker.check(root, provenance)
            for name in checker.INPUTS:
                (root / name).write_bytes(b"# fixture\r\nnumpy==2.5.4\r\n")
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


if __name__ == "__main__":
    unittest.main()
