"""FE solves and benchmark launches must select the preflight interpreter."""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from magnetic_torus.models import ScenarioConfig
from magnetic_torus.numerics import fenics, validation
from simulation_suite import capabilities, scientific


class FELaunchConfigurationTests(unittest.TestCase):
    @contextmanager
    def launch_configuration(self, platform, interpreter, directory):
        environment = {
            "MAGNETIC_TORUS_RUNS": directory,
            "MAGNETIC_TORUS_WSL_DISTRO": "Test Ubuntu",
        }
        if interpreter is not None:
            environment["MAGNETIC_TORUS_FE_PYTHON"] = interpreter
        with (
            patch.dict(os.environ, environment),
            patch.object(sys, "platform", platform),
            patch.object(fenics.subprocess, "CREATE_NO_WINDOW", 0, create=True),
        ):
            if interpreter is None:
                os.environ.pop("MAGNETIC_TORUS_FE_PYTHON", None)
            yield

    def assert_interpreter(self, command, platform, configured):
        expected = configured or "/usr/bin/python3"
        if platform == "win32":
            self.assertEqual(command[:5], ["wsl", "-d", "Test Ubuntu", "--", expected])
            worker_index = 5
        else:
            self.assertEqual(command[0], expected)
            worker_index = 1
        self.assertTrue(command[worker_index].endswith("fenics_worker.py"))
        return worker_index

    def configurations(self):
        for platform in ("win32", "linux"):
            for interpreter in (None, "/opt/FE environment/bin/python3"):
                yield platform, interpreter

    def test_preflight_probes_the_default_or_selected_fe_interpreter(self):
        for platform, interpreter in self.configurations():
            with self.subTest(platform=platform, interpreter=interpreter):
                with tempfile.TemporaryDirectory() as directory:
                    with (
                        self.launch_configuration(platform, interpreter, directory),
                        patch.object(
                            capabilities, "_run_json", return_value=(None, "probe")
                        ) as probe,
                        patch.object(
                            capabilities.importlib.metadata,
                            "version",
                            return_value="test",
                        ),
                    ):
                        detected = capabilities.detect_capabilities()
                    fe_calls = [
                        call
                        for call in probe.call_args_list
                        if call.kwargs.get("input_text") == capabilities._FENICS_PROBE
                    ]
                    self.assertEqual(len(fe_calls), 1)
                    selected = interpreter or "/usr/bin/python3"
                    expected = (
                        ["wsl", "-d", "Test Ubuntu", "--", selected, "-"]
                        if platform == "win32"
                        else [selected, "-"]
                    )
                    self.assertEqual(fe_calls[0].args[0], expected)
                    self.assertFalse(detected["fenics"]["available"])
                    self.assertEqual(detected["fenics"]["reason"], "probe")

    def test_simulation_launch_uses_default_or_selected_interpreter_on_both_platforms(
        self,
    ):
        for platform, interpreter in self.configurations():
            with self.subTest(platform=platform, interpreter=interpreter):
                with tempfile.TemporaryDirectory() as directory:
                    with (
                        self.launch_configuration(platform, interpreter, directory),
                        patch.object(
                            fenics.subprocess,
                            "Popen",
                            side_effect=OSError("launch probe"),
                        ) as launch,
                    ):
                        with self.assertRaisesRegex(
                            fenics.PhysicsBackendUnavailable, "launch probe"
                        ):
                            fenics._run_fenics(
                                ScenarioConfig(), lambda *_: None, threading.Event()
                            )
                    launch.assert_called_once()
                    command = launch.call_args.args[0]
                    worker_index = self.assert_interpreter(
                        command, platform, interpreter
                    )
                    self.assertEqual(len(command), worker_index + 3)
                    self.assertTrue(command[-2].endswith("request.json"))
                    requests = list(Path(directory).glob("*/request.json"))
                    self.assertEqual(len(requests), 1)
                    self.assertEqual(
                        json.loads(requests[0].read_text(encoding="utf-8")),
                        json.loads(json.dumps(ScenarioConfig().to_dict())),
                    )

    def test_validation_report_uses_the_same_interpreter_and_retains_launch_failures(
        self,
    ):
        native = SimpleNamespace(
            diagnostics={"energy_balance_error_j": np.array([0.0])}
        )
        for platform, interpreter in self.configurations():
            with self.subTest(platform=platform, interpreter=interpreter):
                with tempfile.TemporaryDirectory() as directory:
                    destination = Path(directory) / "validation.json"
                    with (
                        self.launch_configuration(platform, interpreter, directory),
                        patch.object(
                            validation,
                            "stefan_front_benchmark",
                            return_value={"relative_front_error": 0.0},
                        ),
                        patch(
                            "magnetic_torus.physics.run_simulation",
                            side_effect=[
                                native,
                                RuntimeError("FE fixture unavailable"),
                            ],
                        ),
                        patch.object(
                            validation.subprocess,
                            "run",
                            side_effect=OSError("benchmark probe"),
                        ) as launch,
                    ):
                        report = validation.write_validation_report(
                            destination, include_fenics=True
                        )
                    launch.assert_called_once()
                    command = launch.call_args.args[0]
                    worker_index = self.assert_interpreter(
                        command, platform, interpreter
                    )
                    self.assertEqual(command[worker_index + 1], "--benchmarks")
                    self.assertTrue(command[-1].endswith("validation_fenics.json"))
                    self.assertGreater(launch.call_args.kwargs["timeout"], 0.0)
                    self.assertEqual(report["fenics_special_cases"]["status"], "failed")
                    persisted = json.loads(destination.read_text(encoding="utf-8"))
                    self.assertEqual(
                        persisted["fenics_special_cases"],
                        report["fenics_special_cases"],
                    )

    def test_suite_references_use_the_same_interpreter_and_preserve_benchmark_checks(
        self,
    ):
        for platform, interpreter in self.configurations():
            with self.subTest(platform=platform, interpreter=interpreter):
                with tempfile.TemporaryDirectory() as directory:
                    output = Path(directory) / "fe-references.json"
                    benchmarks = {
                        name: {"status": "passed"}
                        for name in ("hartmann", "magnetic_diffusion")
                    }
                    output.write_text(json.dumps(benchmarks), encoding="utf-8")
                    checks = scientific.Checks()
                    with (
                        self.launch_configuration(platform, interpreter, directory),
                        patch.object(
                            scientific.subprocess,
                            "run",
                            return_value=SimpleNamespace(
                                returncode=0, stdout="", stderr=""
                            ),
                        ) as launch,
                    ):
                        artifacts, _ = scientific._fe_references(
                            {"case_dir": directory},
                            checks,
                            lambda *_: None,
                            threading.Event(),
                        )
                    launch.assert_called_once()
                    command = launch.call_args.args[0]
                    worker_index = self.assert_interpreter(
                        command, platform, interpreter
                    )
                    self.assertEqual(command[worker_index + 1], "--benchmarks")
                    self.assertTrue(command[-1].endswith("fe-references.json"))
                    self.assertGreater(launch.call_args.kwargs["timeout"], 0.0)
                    self.assertEqual(checks.failed, [])
                    self.assertEqual(len(checks.records), 2)
                    self.assertEqual(artifacts["reference_report"], str(output))


if __name__ == "__main__":
    unittest.main()
