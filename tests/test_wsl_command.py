"""WSL launches must pass arguments literally rather than evaluate a shell."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from magnetic_torus.backend_environment import wsl_command, wsl_distribution, wsl_path


class WSLDistributionSelectionTests(unittest.TestCase):
    def test_diagnostics_and_installer_use_shared_distribution_precedence(self):
        from magnetic_torus import diagnostics
        from test_setup_geant4 import Geant4InstallerTests

        variants = [
            ({"MAGNETIC_TORUS_WSL_DISTRO": "Selected Ubuntu", "TORUS_WSL_DISTRO": "Old Ubuntu"}, "Selected Ubuntu"),
            ({"TORUS_WSL_DISTRO": "Old Ubuntu"}, "Old Ubuntu"),
            ({}, "Ubuntu"),
        ]
        for variables, expected in variants:
            with self.subTest(variables=variables), patch.dict(os.environ, variables, clear=True):
                completed = SimpleNamespace(returncode=0, stdout="", stderr="")
                with (patch.object(diagnostics.os, "name", "nt"),
                      patch.object(diagnostics.importlib.metadata, "version", return_value="test"),
                      patch.object(diagnostics.subprocess, "run", return_value=completed) as run):
                    diagnostics.inspect_environment()
                self.assertEqual([call.args[0][2] for call in run.call_args_list[1:]],
                                 [expected, expected])
                with (patch.object(sys, "platform", "win32"),
                      patch("test_setup_geant4.shutil.which", return_value="wsl"),
                      patch("test_setup_geant4.subprocess.run", return_value=completed),
                      patch.object(Geant4InstallerTests, "runner", create=True)):
                    Geant4InstallerTests.setUpClass()
                    self.assertEqual(Geant4InstallerTests.runner[2], expected)


@unittest.skipUnless(sys.platform == "win32" and shutil.which("wsl"), "Requires Windows WSL")
class WSLCommandTests(unittest.TestCase):
    def test_real_wsl_preserves_hostile_and_normal_arguments(self):
        distro = wsl_distribution()
        listing = subprocess.run(["wsl", "--list", "--quiet"], capture_output=True, timeout=30)
        if listing.returncode:
            self.skipTest("WSL distributions are unavailable")
        names = listing.stdout.decode("utf-16-le", errors="replace").replace("\x00", "").splitlines()
        if distro.casefold() not in {name.strip().casefold() for name in names}:
            self.skipTest("Configured WSL distribution is unavailable")
        with tempfile.TemporaryDirectory(prefix="torus-wsl-") as directory:
            # Spaces, quotes and command substitutions are legal filename characters
            # on Windows and must never be interpreted by a Linux shell.
            path = Path(directory) / "space 'quote' $(printf INJECTED)"
            path.mkdir()
            linux_path = wsl_path(path)
            payloads = [linux_path, "SAFE$(printf INJECTED)", "`printf INJECTED`",
                        "two words", "a'quote", 'a"quote', "\u96ea \u00e9", "", "line\nbreak"]
            program = (
                "import json,os,sys; "
                "print(json.dumps({'args':sys.argv[1:],'exists':os.path.isdir(sys.argv[1])},ensure_ascii=True))"
            )
            result = subprocess.run(
                wsl_command("/usr/bin/python3", "-c", program, *payloads),
                capture_output=True, text=True, encoding="utf-8", timeout=60,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            data = json.loads(result.stdout)
            self.assertTrue(data["exists"])
            self.assertEqual(data["args"], payloads)


if __name__ == "__main__":
    unittest.main()
