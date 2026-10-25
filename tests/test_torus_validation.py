"""External benchmark failures must leave a complete, honest validation report."""

import json
import math
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from magnetic_torus.numerics.validation import write_validation_report


class ValidationReportTests(unittest.TestCase):
    def test_fe_special_case_timeout_is_bounded_and_written_to_report(self):
        native = SimpleNamespace(diagnostics={"energy_balance_error_j": np.array([0.])})

        def timeout(command, **options):
            self.assertTrue(math.isfinite(options["timeout"]))
            self.assertGreater(options["timeout"], 0.)
            raise subprocess.TimeoutExpired(command, options["timeout"])

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "validation.json"
            with patch("magnetic_torus.numerics.validation.stefan_front_benchmark",
                       return_value={"relative_front_error": 0.}), \
                 patch("magnetic_torus.physics.run_simulation",
                       side_effect=[native, RuntimeError("FE fixture unavailable")]), \
                 patch("magnetic_torus.numerics.validation.subprocess.run", side_effect=timeout) as run:
                report = write_validation_report(destination, include_fenics=True)
            run.assert_called_once()
            self.assertEqual(report["fenics_special_cases"]["status"], "failed")
            self.assertIn("timed out", report["fenics_special_cases"]["reason"])
            persisted = json.loads(destination.read_text(encoding="utf-8"))
            self.assertEqual(persisted["fenics_special_cases"], report["fenics_special_cases"])
            self.assertEqual(persisted["thermal_energy_balance"]["status"], "passed")


if __name__ == "__main__":
    unittest.main()
