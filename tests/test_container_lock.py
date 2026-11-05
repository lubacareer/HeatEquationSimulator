"""Verify portable input identity and fail-closed container lock maintenance."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import importlib.util
import json
import os
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
    def inventory_fixture(self, root):
        (root / "requirements.txt").write_text("numpy>=1.26,<3\ntensorflow==2.21.0\n", encoding="utf-8")
        (root / "requirements-gpu.txt").write_text(
            '-r requirements.txt\ntensorflow[and-cuda]==2.21.0; platform_system == "Linux"\n', encoding="utf-8")
        environment = {"implementation_name": "cpython", "implementation_version": "3.12.3", "os_name": "posix",
                       "platform_machine": "x86_64", "platform_python_implementation": "CPython",
                       "platform_release": "fixture-release", "platform_system": "Linux", "platform_version": "fixture-version",
                       "python_full_version": "3.12.3", "python_version": "3.12", "sys_platform": "linux"}
        packages = [{"name": name, "version": version, "requires_dist": [], "provides_extra": []}
                    for name, version in (("pip", "26.3"), ("numpy", "2.5.4"), ("tensorflow", "2.21.0"),
                                          ("nvidia-cuda-runtime-cu12", "12.9.79"))]
        packages[2].update({"provides_extra": ["and-cuda"], "requires_dist": [
            "numpy>=1.26", 'nvidia-cuda-runtime-cu12>=12.9,<13; extra == "and-cuda" and platform_system == "Linux"']})
        return {"python": "3.12.3 (fixture)", "environment": environment, "distributions": packages,
                "tags": ["cp312-cp312-linux_x86_64"]}

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

    def test_validation_uses_recorded_linux_target_and_requested_gpu_extra(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            inventory = self.inventory_fixture(root)
            host = {**inventory["environment"], "platform_system": "Windows", "os_name": "nt", "sys_platform": "win32",
                    "python_version": "3.11", "python_full_version": "3.11.0"}
            with patch("pip._vendor.packaging.markers.default_environment", return_value=host):
                result = generator.validate_inventory(root, inventory)
            self.assertIn("nvidia-cuda-runtime-cu12", result["required_distributions"])
            self.assertEqual(result["requested_extras"], {"tensorflow": ["and-cuda"]})

    def test_validation_rejects_missing_or_incompatible_direct_and_gpu_requirements(self):
        for change in ("missing-direct", "incompatible-direct", "missing-gpu", "incompatible-gpu", "missing-extra"):
            with self.subTest(change=change), TemporaryDirectory() as directory:
                root = Path(directory)
                inventory = deepcopy(self.inventory_fixture(root))
                packages = inventory["distributions"]
                if change == "missing-direct":
                    packages[:] = [package for package in packages if package["name"] != "numpy"]
                elif change == "incompatible-direct":
                    packages[2]["version"] = "2.20.0"
                elif change == "missing-gpu":
                    packages[:] = [package for package in packages if package["name"] != "nvidia-cuda-runtime-cu12"]
                elif change == "incompatible-gpu":
                    packages[3]["version"] = "11.8.0"
                else:
                    packages[2]["provides_extra"] = []
                with self.assertRaises(ValueError):
                    generator.validate_inventory(root, inventory)

    def test_validation_rejects_inconsistent_target_python_versions(self):
        for change in ("wrong-full-version", "wrong-implementation-version"):
            with self.subTest(change=change), TemporaryDirectory() as directory:
                root = Path(directory)
                inventory = self.inventory_fixture(root)
                inventory["environment"]["implementation_version"] = "3.13.0"
                if change == "wrong-full-version":
                    inventory["python"] = "3.13.0 (fixture)"
                    inventory["environment"]["python_full_version"] = "3.13.0"
                with self.assertRaisesRegex(ValueError, "inconsistent"):
                    generator.validate_inventory(root, inventory)

    def test_requirement_inputs_reject_external_links_and_internal_aliases(self):
        for kind in ("external-file", "internal-file-alias", "external-directory", "hardlink-alias"):
            with self.subTest(kind=kind), TemporaryDirectory() as directory, TemporaryDirectory() as outside:
                root, external = Path(directory), Path(outside)
                inventory = self.inventory_fixture(root)
                target = external / "requirements.txt"
                target.write_text("owned external fixture; must not be read", encoding="utf-8")
                try:
                    if kind == "external-directory":
                        (root / "linked-inputs").symlink_to(external, target_is_directory=True)
                        (root / "requirements-gpu.txt").write_text("-r linked-inputs/requirements.txt\n", encoding="utf-8")
                    else:
                        (root / "requirements-gpu.txt").unlink()
                        if kind == "external-file":
                            (root / "requirements-gpu.txt").symlink_to(target)
                        elif kind == "internal-file-alias":
                            (root / "requirements-gpu.txt").symlink_to(root / "requirements.txt")
                        else:
                            os.link(root / "requirements.txt", root / "requirements-gpu.txt")
                except OSError:
                    self.skipTest("This host does not permit the synthetic link fixture")
                read_text = Path.read_text

                def guarded_read(path, *args, **kwargs):
                    if path.resolve() == target.resolve():
                        raise AssertionError("External synthetic contents were read")
                    return read_text(path, *args, **kwargs)

                with patch.object(Path, "read_text", guarded_read), self.assertRaises(ValueError):
                    generator.validate_inventory(root, inventory)

    def test_invalid_inputs_and_inventory_fail_before_network_or_publication(self):
        marker = "DO_NOT_PRINT_FIXTURE_MARKER"
        cases = (("direct-url", f"tensorflow @ https://fixture:{marker}@example.invalid/package.whl"),
                 ("unknown-option", f"--index-url https://fixture:{marker}@example.invalid/simple"),
                 ("remote-include", f"-r https://example.invalid/{marker}"),
                 ("incompatible-inventory", None))
        for kind, line in cases:
            with self.subTest(kind=kind), TemporaryDirectory() as directory:
                root = Path(directory)
                inventory = self.inventory_fixture(root)
                if line is not None:
                    (root / "requirements-gpu.txt").write_text(line + "\n", encoding="utf-8")
                else:
                    inventory["distributions"][2]["version"] = "2.20.0"
                docker = root / "docker"
                docker.mkdir()
                for source in (CHECKER, CHECKER.with_name("generate_container_lock.py")):
                    (docker / source.name).write_bytes(source.read_bytes())
                outputs = {name: b"previous owned publication" for name in (*checker.LOCK_FILES, "requirements-lock-provenance.json")}
                for name, content in outputs.items():
                    (docker / name).write_bytes(content)
                inventory_file = root / "inventory.json"
                inventory_file.write_text(json.dumps(inventory), encoding="utf-8")
                # Any attempted network request fails the control before it can
                # leave the process, even when this unit test runs on a host.
                launcher = ("import runpy,sys,urllib.request; sys.path.insert(0,sys.argv[1]); "
                            "urllib.request.urlopen=lambda *a,**k: (_ for _ in ()).throw(AssertionError('network forbidden')); "
                            "script=sys.argv[2]; sys.argv=sys.argv[2:]; runpy.run_path(script,run_name='__main__')")
                result = subprocess.run([sys.executable, "-I", "-c", launcher, str(docker),
                                         str(docker / "generate_container_lock.py"), "--inventory", str(inventory_file),
                                         "--source-image", "sha256:" + "a" * 64, "--metadata-dir", str(root / "metadata")],
                                        capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 2)
                self.assertNotIn(marker, result.stderr + result.stdout)
                self.assertFalse((root / "metadata").exists())
                self.assertEqual({name: (docker / name).read_bytes() for name in outputs}, outputs)


if __name__ == "__main__":
    unittest.main()
