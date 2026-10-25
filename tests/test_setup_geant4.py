"""Exercise installer trust checks with local archives and harmless tool stubs."""
from hashlib import sha256
import io
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest


REPO = Path(__file__).resolve().parent.parent
ARCHIVES = ("micromamba-2.3.3-0.tar.bz2", "geant4-11.4.3.tar.gz", "ICRPdata.tar.gz")


class Geant4InstallerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if sys.platform == "win32":
            if not shutil.which("wsl"):
                raise unittest.SkipTest("Installer tests need Bash in Linux/WSL.")
            cls.runner = ["wsl", "-d", os.environ.get("TORUS_WSL_DISTRO", "Ubuntu"), "--"]
        elif shutil.which("bash"):
            cls.runner = []
        else:
            raise unittest.SkipTest("Installer tests need Bash.")
        probe = subprocess.run(cls.runner + ["bash", "-c", "command -v tar sha256sum mktemp"],
                               capture_output=True, text=True, timeout=30)
        if probe.returncode:
            raise unittest.SkipTest("Installer tests need Bash, tar, and sha256sum in Linux/WSL.")

    @staticmethod
    def shell_path(path):
        path = Path(path).resolve()
        if sys.platform == "win32":
            return "/mnt/" + path.drive[0].lower() + path.as_posix()[2:]
        return str(path)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="torus-installer-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.repo = self.root / "repo"
        (self.repo / "scripts").mkdir(parents=True)
        (self.repo / "env").mkdir()
        self.script = self.repo / "scripts" / "setup_geant4.sh"
        self.script.write_text((REPO / "scripts" / "setup_geant4.sh").read_text(), newline="\n")
        (self.repo / "env" / "geant4-linux-64.explicit.txt").write_text("@EXPLICIT\n")
        self.cache, self.incoming, self.stub = (self.root / name for name in ("cache", "incoming", "stub"))
        for directory in (self.cache, self.incoming, self.stub):
            directory.mkdir()
        self.prefix = self.root / "geant4"
        self.torus = self.root / "torus"
        self.log = self.root / "tools.log"
        self.log.touch()
        mamba = b'''#!/usr/bin/env bash
set -eu
printf 'micromamba %s\n' "$*" >> "$TASK_TEST_LOG"
mkdir -p "$TASK_TEST_PREFIX/conda-meta" "$TASK_TEST_PREFIX/include" "$TASK_TEST_PREFIX/lib/cmake/Geant4"
touch "$TASK_TEST_PREFIX/include/expat.h" "$TASK_TEST_PREFIX/lib/cmake/Geant4/Geant4Config.cmake"
'''
        self.archive(ARCHIVES[0], {"bin/micromamba": mamba})
        self.archive(ARCHIVES[1], {"geant4-v11.4.3/CMakeLists.txt": b"verified source\n"})
        self.archive(ARCHIVES[2], {"ICRPdata/MaleData.dat": b"verified phantom\n"})
        manifest = [f"{sha256((self.incoming / name).read_bytes()).hexdigest()}  {name}" for name in ARCHIVES]
        manifest.append(f"{sha256(mamba).hexdigest()}  micromamba-2.3.3-0")
        (self.repo / "env" / "geant4-downloads.sha256").write_text("\n".join(manifest) + "\n", newline="\r\n")
        self.tool("sudo", '''printf 'sudo %s\n' "$*" >> "$TASK_TEST_LOG"
if [[ "$1" == apt-get ]]; then exit 0; fi
exec "$@"
''')
        self.tool("curl", '''printf 'curl %s\n' "$*" >> "$TASK_TEST_LOG"
case "$*" in
  *micromamba*) name=micromamba-2.3.3-0.tar.bz2 ;;
  *geant4-v11.4.3*) name=geant4-11.4.3.tar.gz ;;
  *ICRPdata*) name=ICRPdata.tar.gz ;;
  *) exit 90 ;;
esac
while [[ "$1" != -o ]]; do shift; done
cp -- "$TASK_TEST_INCOMING/$name" "$2"
''')
        self.tool("tar", '''printf 'tar %s\n' "$*" >> "$TASK_TEST_LOG"
exec /usr/bin/tar "$@"
''')
        self.tool("cmake", '''printf 'cmake %s\n' "$*" >> "$TASK_TEST_LOG"
mkdir -p "$TASK_TEST_PREFIX/lib/cmake/Geant4" "$TASK_TEST_PREFIX/bin" "$TASK_TEST_TORUS/bin"
touch "$TASK_TEST_PREFIX/lib/cmake/Geant4/Geant4Config.cmake" "$TASK_TEST_PREFIX/bin/geant4.sh"
''')
        self.tool("python3", '''printf 'python3 %s\n' "$*" >> "$TASK_TEST_LOG"
''')

    def archive(self, name, members):
        with tarfile.open(self.incoming / name, "w:bz2" if name.endswith("bz2") else "w:gz") as archive:
            for path, content in members.items():
                entry = tarfile.TarInfo(path)
                entry.size, entry.mode = len(content), 0o755 if path == "bin/micromamba" else 0o644
                archive.addfile(entry, io.BytesIO(content))

    def tool(self, name, body):
        path = self.stub / name
        path.write_text("#!/usr/bin/env bash\nset -eu\n" + body, newline="\n")
        path.chmod(0o755)

    def run_installer(self, mode="prebuilt"):
        variables = {
            "PATH": self.shell_path(self.stub) + ":/usr/bin:/bin",
            "GEANT4_PREFIX": self.shell_path(self.prefix),
            "TORUS_PREFIX": self.shell_path(self.torus),
            "TORUS_BUILD_ROOT": self.shell_path(self.cache),
            "TORUS_BUILD_JOBS": "2",
            "TORUS_GEANT4_PREBUILT": "0" if mode == "source" else "1",
            "TASK_TEST_LOG": self.shell_path(self.log),
            "TASK_TEST_INCOMING": self.shell_path(self.incoming),
            "TASK_TEST_PREFIX": self.shell_path(self.prefix),
            "TASK_TEST_TORUS": self.shell_path(self.torus),
        }
        command = self.runner + ["env"] + [f"{key}={value}" for key, value in variables.items()]
        command += ["bash", self.shell_path(self.script)]
        if mode == "phantom":
            command.append("--phantom-data")
        return subprocess.run(command, capture_output=True, text=True, timeout=30)

    def assert_no_archive_use(self):
        log = self.log.read_text()
        self.assertNotIn("\ntar ", "\n" + log)
        self.assertNotIn("\nmicromamba ", "\n" + log)
        self.assertNotIn("\ncmake ", "\n" + log)
        self.assertNotIn("sudo env ", log)
        self.assertFalse(list(self.cache.glob(".setup.*")))

    def test_corrupt_downloads_and_cached_archives_fail_before_extraction_or_execution(self):
        cases = (("prebuilt", ARCHIVES[0]), ("source", ARCHIVES[1]), ("phantom", ARCHIVES[2]))
        for mode, name in cases:
            original = (self.incoming / name).read_bytes()
            for cached in (False, True):
                with self.subTest(mode=mode, cached=cached):
                    self.log.write_text("")
                    if cached:
                        (self.incoming / name).write_bytes(original)
                        (self.cache / name).write_bytes(b"tampered cached archive")
                    else:
                        (self.incoming / name).write_bytes(b"tampered download")
                    result = self.run_installer(mode)
                    self.assertNotEqual(result.returncode, 0, result.stdout)
                    self.assertIn("SHA256 verification failed", result.stderr)
                    self.assert_no_archive_use()
                    if cached:
                        self.assertNotIn("\ncurl ", "\n" + self.log.read_text())
                        (self.cache / name).unlink()
            (self.incoming / name).write_bytes(original)

    def test_verified_prebuilt_bootstrap_ignores_cached_binary_and_supports_create_and_repair(self):
        (self.cache / "bin").mkdir()
        (self.cache / "bin" / "micromamba").write_text("#!/bin/bash\nexit 99\n")
        (self.cache / "bin" / "micromamba").chmod(0o755)
        for repairing in (False, True):
            with self.subTest(repairing=repairing):
                self.log.write_text("")
                if repairing:
                    (self.prefix / "include" / "expat.h").unlink()
                result = self.run_installer()
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                log = self.log.read_text()
                self.assertIn("micromamba " + ("install" if repairing else "create"), log)
                self.assertIn(".setup.", log)
                self.assertNotIn(self.shell_path(self.cache / "bin" / "micromamba"), log)
                self.assertNotIn("sudo tar", log)
                self.assertIn("--parallel 2", log)
                self.assertTrue((self.torus / "bin" / "torus-dose-run").is_file())
                self.assertEqual((self.torus / "data" / "ICRPdata" / "MaleData.dat").read_bytes(), b"verified phantom\n")
                self.assertFalse(list(self.cache.glob(".setup.*")))

    def test_verified_source_uses_fresh_extraction_and_build_tree(self):
        previous = self.cache / "geant4-v11.4.3"
        previous.mkdir()
        (previous / "CMakeLists.txt").write_text("untrusted old source")
        (self.cache / ARCHIVES[1]).write_bytes((self.incoming / ARCHIVES[1]).read_bytes())
        result = self.run_installer("source")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        log = self.log.read_text()
        self.assertIn("/geant4-v11.4.3 -B ", log)
        self.assertIn(".setup.", log)
        self.assertNotIn(self.shell_path(previous), log)
        self.assertNotIn("sudo tar", log)
        self.assertNotIn("micromamba ", log)

    def test_phantom_only_verifies_and_installs_without_toolkit_actions(self):
        result = self.run_installer("phantom")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        log = self.log.read_text()
        self.assertNotIn("apt-get", log)
        self.assertNotIn("cmake ", log)
        self.assertNotIn("micromamba", log)
        self.assertNotIn("sudo tar", log)
        self.assertTrue((self.torus / "data" / "ICRPdata" / "MaleData.dat").is_file())

    def test_prebuilt_fails_closed_when_package_lock_is_missing(self):
        (self.repo / "env" / "geant4-linux-64.explicit.txt").unlink()
        result = self.run_installer()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Missing pinned Geant4 package lock", result.stderr)
        self.assertNotIn("\ncurl ", "\n" + self.log.read_text())
        self.assert_no_archive_use()

    def test_extracted_bootstrap_binary_must_match_its_own_pin(self):
        manifest = self.repo / "env" / "geant4-downloads.sha256"
        lines = manifest.read_text().splitlines()
        lines[-1] = "0" * 64 + "  micromamba-2.3.3-0"
        manifest.write_text("\n".join(lines) + "\n")
        result = self.run_installer()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("SHA256 verification failed: micromamba-2.3.3-0", result.stderr)
        self.assertNotIn("sudo env ", self.log.read_text())

    def test_cached_archive_symlink_is_rejected(self):
        name = ARCHIVES[2]
        linked = subprocess.run(self.runner + ["ln", "-s", self.shell_path(self.incoming / name),
                                               self.shell_path(self.cache / name)], capture_output=True, text=True)
        if linked.returncode:
            self.skipTest("This filesystem does not support Linux symlinks.")
        result = self.run_installer("phantom")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Invalid cached download", result.stderr)
        self.assert_no_archive_use()


if __name__ == "__main__":
    unittest.main()
