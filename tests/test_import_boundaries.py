"""Public APIs must remain usable without initializing optional runtimes."""
from pathlib import Path
import subprocess
import sys
import unittest


class ImportBoundaryTests(unittest.TestCase):
    def assert_isolated_import(self, imports, forbidden):
        program = (
            "import sys\n"
            + "\n".join(f"import {module}" for module in imports)
            + f"\nloaded = sorted(set(sys.modules).intersection({forbidden!r}))\n"
            + "assert not loaded, f'Unexpected optional runtimes: {loaded}'\n"
        )
        result = subprocess.run([sys.executable, "-B", "-c", program],
                                cwd=Path(__file__).resolve().parent.parent,
                                capture_output=True, text=True, timeout=45)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_scientific_facades_preserve_lazy_process_boundaries(self):
        self.assert_isolated_import(
            ["magnetic_torus.physics", "magnetic_torus.radiation", "magnetic_torus.pinn"],
            ["scipy", "torch", "PyQt6", "vtkmodules"],
        )

    def test_workspace_state_has_no_graphics_or_training_dependency(self):
        self.assert_isolated_import(
            ["magnetic_torus.ui.session", "magnetic_torus.display", "magnetic_torus.geometry"],
            ["scipy", "torch", "PyQt6", "vtkmodules", "pyvista", "matplotlib"],
        )


if __name__ == "__main__":
    unittest.main()
