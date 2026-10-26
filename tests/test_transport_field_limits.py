"""Finite-map geometry guards and diagnostic reporting for Geant4 transport."""

import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from magnetic_torus.backend_environment import wsl_path
from magnetic_torus.models import RunResult, ScenarioConfig
from magnetic_torus.transport import geant4

SUITE_CAPABILITIES = ("geant4",)

FATAL = """-------- EEEE ------- G4Exception-START -------- EEEE -------
*** G4Exception : OutsideAcceptedMap
      issued by : FrozenMagneticField
Query [m] = (9.25, 0, 0); world radius [m] = 6.48
*** Fatal Exception *** core dump ***
-------- EEEE -------- G4Exception-END --------- EEEE -------
"""


class TransportFailureDiagnosticTests(unittest.TestCase):
    def test_first_fatal_survives_long_initialization_output(self):
        detail = geant4._transport_error_detail(
            FATAL + "Unrelated physics initialization\n" * 2000
        )
        self.assertIn("OutsideAcceptedMap", detail)
        self.assertIn("Query [m] = (9.25, 0, 0)", detail)
        self.assertNotIn("Unrelated physics initialization", detail)

    def test_initialization_warning_does_not_hide_a_later_fatal(self):
        warning = FATAL.replace("OutsideAcceptedMap", "InitializationWarning").replace(
            "*** Fatal Exception *** core dump ***",
            "*** This is just a warning message. ***",
        )
        detail = geant4._transport_error_detail(warning + "startup output\n" + FATAL)
        self.assertIn("OutsideAcceptedMap", detail)
        self.assertNotIn("InitializationWarning", detail)

    def test_preflight_errors_survive_initialization_and_unknown_errors_keep_a_bounded_tail(
        self,
    ):
        message = "Dose transport error: Incident primary starts outside the validated transport world"
        self.assertEqual(
            geant4._transport_error_detail(message + "\n" + "irrelevant\n" * 2000),
            message,
        )
        unknown = "start\n" + "unknown output\n" * 2000 + "last numerical error"
        detail = geant4._transport_error_detail(unknown)
        self.assertLessEqual(len(detail), 3000)
        self.assertTrue(detail.endswith("last numerical error"))

    def test_calculate_dose_reports_the_fatal_query_without_an_installation_diagnosis(
        self,
    ):
        config = ScenarioConfig(radiation_histories=2)
        source = RunResult(
            "diagnostic-fixture", config, np.array([0.0]), np.zeros((1, 3)), {}, {}, {}
        )
        field = SimpleNamespace(
            points=np.zeros((4, 3)),
            cells=np.array([[0, 1, 2, 3]]),
            values=np.zeros((4, 3)),
            cell_values=np.zeros((1, 3)),
            radius=10.0,
            fingerprint="fixture",
        )

        def failed_worker(_command, **options):
            options["stdout"].write(FATAL + "Unrelated physics initialization\n" * 2000)
            return SimpleNamespace(returncode=6, poll=lambda: 6)

        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.object(geant4, "frozen_field", return_value=field),
                patch.object(geant4.subprocess, "Popen", side_effect=failed_worker),
            ):
                with self.assertRaises(RuntimeError) as error:
                    geant4.calculate_dose(
                        source,
                        0,
                        lambda *_: None,
                        SimpleNamespace(is_set=lambda: False),
                        Path(directory),
                    )
        self.assertIn("OutsideAcceptedMap", str(error.exception))
        self.assertIn("Query [m]", str(error.exception))
        self.assertNotIn("Install its worker", str(error.exception))
        self.assertNotIn("Unrelated physics initialization", str(error.exception))


@unittest.skipUnless(
    os.environ.get("TORUS_TEST_GEANT4") == "1",
    "Installed Geant4 worker required for finite-map diagnostics",
)
class InstalledTransportFieldLimitTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="torus field limits ")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.points = np.array(
            [[x, y, z] for x in (-8.0, 8.0) for y in (-8.0, 8.0) for z in (-8.0, 8.0)]
        )
        self.cells = np.array(
            [
                [0, 4, 6, 7],
                [0, 4, 5, 7],
                [0, 2, 6, 7],
                [0, 2, 3, 7],
                [0, 1, 5, 7],
                [0, 1, 3, 7],
            ]
        )

    def fixture(
        self,
        *,
        values=None,
        points=None,
        cells=None,
        world=6.0,
        primary_position=5.0,
        extra_primary_bytes=b"",
        maximum_step=None,
        field_mode=1,
    ):
        points = self.points if points is None else points
        cells = self.cells if cells is None else cells
        values = np.zeros((len(cells), 3)) if values is None else values
        field = self.directory / "frozen field.bin"
        with field.open("wb") as stream:
            np.array([len(points), len(cells), field_mode], dtype="<i8").tofile(stream)
            np.asarray(points, dtype="<f8").tofile(stream)
            np.asarray(cells, dtype="<i8").tofile(stream)
            np.asarray(values, dtype="<f8").tofile(stream)
        primary = self.directory / "incident samples.bin"
        with primary.open("wb") as stream:
            np.array(
                [[1.0, 1.0, 100.0, primary_position, 0.0, 0.0, -1.0, 0.0, 0.0]] * 2,
                dtype="<f8",
            ).tofile(stream)
            stream.write(extra_primary_bytes)
        request = {
            "schema_version": 1,
            "world_radius_m": world,
            "field_file": wsl_path(field) if os.name == "nt" else str(field),
            "primary_file": wsl_path(primary) if os.name == "nt" else str(primary),
            "config": {"radiation_histories": 2},
        }
        if maximum_step is not None:
            request["maximum_field_step_m"] = maximum_step
        path = self.directory / "inspect request.json"
        path.write_text(json.dumps(request), encoding="utf-8")
        command = geant4._geant4_command(path)
        command.insert(-1, "--inspect-field")
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=45,
        )

    def report(self, **options):
        completed = self.fixture(**options)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        return json.loads(completed.stdout.strip().splitlines()[-1])

    def test_exact_zero_and_signed_zero_are_fast_only_after_geometry_validation(self):
        for zero in (0.0, -0.0):
            with self.subTest(zero=zero):
                report = self.report(values=np.full((len(self.cells), 3), zero))
                self.assertTrue(report["exact_zero_field_fast_path"])
                self.assertEqual(report["field_transport_mode"], "straight-exact-zero")
                self.assertAlmostEqual(report["field_map_support_radius_m"], 8.0)
                self.assertAlmostEqual(report["field_map_clearance_m"], 2.0)
                self.assertAlmostEqual(report["maximum_field_step_m"], 0.1)
                self.assertEqual(report["primary_count"], 2)

    def test_tiny_nonzero_fields_use_the_bounded_manager_and_steps_can_only_decrease(
        self,
    ):
        values = np.zeros((len(self.cells), 3))
        values[0, 2] = 1e-30
        report = self.report(values=values, maximum_step=0.05)
        self.assertFalse(report["exact_zero_field_fast_path"])
        self.assertEqual(report["field_transport_mode"], "bounded-frozen-map")
        self.assertAlmostEqual(report["maximum_field_step_m"], 0.05)
        rejected = self.fixture(values=values, maximum_step=0.2)
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn("clearance limit", rejected.stderr)

    def test_world_boundary_and_primary_bounds_are_checked_even_for_an_exact_zero_map(
        self,
    ):
        for options, message in (
            ({"world": 8.0}, "positive clearance"),
            ({"primary_position": 6.5}, "outside the validated transport world"),
            ({"points": self.points + 20.0}, "world origin"),
            ({"extra_primary_bytes": b"broken"}, "truncated record"),
        ):
            with self.subTest(options=options):
                rejected = self.fixture(**options)
                self.assertNotEqual(rejected.returncode, 0)
                self.assertIn(message, rejected.stderr)

    def test_nonfinite_values_and_invalid_tetrahedra_are_rejected_before_fast_path(
        self,
    ):
        values = np.zeros((len(self.cells), 3))
        values[0, 0] = np.nan
        degenerate = self.cells.copy()
        degenerate[0, 3] = degenerate[0, 0]
        for options, message in (
            ({"values": values}, "must be finite"),
            ({"cells": degenerate}, "degenerate tetrahedron"),
            ({"field_mode": 2}, "Unsupported frozen magnetic field storage mode"),
            ({"cells": np.repeat(self.cells[:1], 3, axis=0)}, "same side"),
        ):
            with self.subTest(message=message):
                rejected = self.fixture(**options)
                self.assertNotEqual(rejected.returncode, 0)
                self.assertIn(message, rejected.stderr)

    def test_overlapping_tetrahedra_cannot_hide_a_true_map_boundary(self):
        points = np.array(
            [
                [-10.0, -10.0, -0.1],
                [10.0, -10.0, -0.1],
                [0.0, 10.0, -0.1],
                [0.0, 0.0, 10.0],
                [0.0, 0.0, 11.0],
            ]
        )
        cells = np.array([[0, 1, 2, 3], [0, 1, 2, 4]])
        rejected = self.fixture(
            points=points, cells=cells, world=1.0, primary_position=0.5
        )
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn("same side of a shared face", rejected.stderr)


if __name__ == "__main__":
    unittest.main()
