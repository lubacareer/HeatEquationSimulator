"""The default command is bounded; old report profiles remain explicit evidence."""

import importlib.util
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_all_tests.py"
spec = importlib.util.spec_from_file_location("suite_entrypoint", SCRIPT)
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


class ProfilePolicyTests(unittest.TestCase):
    def test_fresh_default_and_explicit_profiles(self):
        self.assertEqual(cli.resolve_profile(None, None), "features")
        for name in cli.PROFILES:
            self.assertEqual(cli.resolve_profile(name, None), name)

    def test_resume_inherits_profile_and_rejects_mismatch_or_unknown_reports(self):
        for name in cli.PROFILES:
            self.assertEqual(cli.resolve_profile(None, {"profile": name}), name)
            self.assertEqual(cli.resolve_profile(name, {"profile": name}), name)
        with self.assertRaisesRegex(ValueError, "differs"):
            cli.resolve_profile("features", {"profile": "full"})
        for previous in ({}, {"profile": "unknown"}):
            with self.assertRaisesRegex(ValueError, "unknown"):
                cli.resolve_profile(None, previous)

    def test_default_delegates_deadline_to_supervisor_without_starting_cases(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "private-run"
            with (
                redirect_stdout(io.StringIO()) as output,
                patch("simulation_suite.budget.supervise", return_value=124) as supervisor,
                patch.object(cli, "_execute", side_effect=AssertionError("unsupervised cases")),
            ):
                code = cli.main(["--output", str(directory)])
            self.assertEqual(code, 124)
            self.assertIn("BUDGET_EXHAUSTED", output.getvalue())
            self.assertIn(str(directory / "report.json"), output.getvalue())
            command = supervisor.call_args.args[0]
            self.assertIn("--_suite-worker", command)
            self.assertEqual(command[command.index("--profile") + 1], "features")
            self.assertEqual(supervisor.call_args.kwargs["run_dir"], directory.resolve())
            self.assertFalse(directory.exists())

    def test_explicit_full_keeps_unbounded_execution_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "private-full"
            with (
                patch("simulation_suite.budget.supervise", side_effect=AssertionError("unexpected cap")),
                patch.object(cli, "_execute", return_value=0) as execute,
            ):
                code = cli.main(["--profile", "full", "--output", str(directory)])
            self.assertEqual(code, 0)
            self.assertIsNone(execute.call_args.args[-1])


class CatalogCommandTests(unittest.TestCase):
    def command(self, *arguments):
        environment = {key: value for key, value in os.environ.items()
                       if not key.startswith("TORUS_SUITE_")}
        return subprocess.run(
            [sys.executable, "-B", str(SCRIPT), *arguments], cwd=ROOT,
            env=environment, capture_output=True, text=True, encoding="utf-8", timeout=30,
        )

    def catalog(self, output):
        return json.loads(output[output.index("{\n"):])

    def test_default_list_is_feature_complete_and_creates_no_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "must-not-exist"
            result = self.command("--list", "--output", str(directory))
            self.assertEqual(result.returncode, 0, result.stderr)
            data = self.catalog(result.stdout)
            self.assertEqual(data["coverage"]["profile"], "features")
            self.assertTrue(data["coverage"]["feature_matrix"]["complete"])
            self.assertNotIn("scientific_matrix", data["coverage"])
            self.assertFalse(directory.exists())

    def test_old_full_resume_list_preserves_report_and_profile(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            report = directory / "report.json"
            original = json.dumps({"profile": "full", "results": [{"id": "old", "status": "PASS"}]})
            report.write_text(original, encoding="utf-8")
            result = self.command("--list", "--resume", str(directory))
            self.assertEqual(result.returncode, 0, result.stderr)
            data = self.catalog(result.stdout)
            self.assertEqual(data["coverage"]["profile"], "full")
            self.assertTrue(data["coverage"]["scientific_matrix"]["complete"])
            self.assertEqual(report.read_text(encoding="utf-8"), original)
            mismatch = self.command("--list", "--resume", str(directory), "--profile", "features")
            self.assertEqual(mismatch.returncode, 2)
            self.assertIn("differs", mismatch.stderr)
            self.assertEqual(report.read_text(encoding="utf-8"), original)

    def test_fast_profile_and_invalid_selection_remain_explicit(self):
        result = self.command("--profile", "fast", "--list")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.catalog(result.stdout)["coverage"]["profile"], "fast")
        invalid = self.command("--list", "--case", "does-not-exist")
        self.assertEqual(invalid.returncode, 2)


class PreparationStopTests(unittest.TestCase):
    def test_deadline_and_user_interrupt_during_preflight_preserve_old_evidence(self):
        from simulation_suite.budget import BudgetExpired, InvocationBudget, InvocationInterrupted
        from simulation_suite.spec import CaseSpec

        case = CaseSpec("sample.required", "functional", "Required case", ("unused",))
        for reason, error, code in (("budget", BudgetExpired(), 124),
                                    ("user", InvocationInterrupted(), 130)):
            with self.subTest(reason=reason), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary) / "run"
                directory.mkdir()
                old = {"profile": "features", "results": [{"id": "old.pass", "status": "PASS"}],
                       "retained_results": [{"id": "old.unvisited", "status": "PASS"}]}
                stop = threading.Event()
                budget = InvocationBudget(0., 60., 5., stop_event=stop, clock=lambda: 0.)
                args = SimpleNamespace(profile="features", case=[], preflight=False)
                with (
                    redirect_stdout(io.StringIO()),
                    patch.object(cli, "ROOT", Path(temporary)),
                    patch("simulation_suite.catalog.all_cases", return_value=([case], {})),
                    patch("simulation_suite.capabilities.detect_capabilities", side_effect=error),
                    patch("simulation_suite.runner.execute", side_effect=AssertionError("case launched")),
                ):
                    returned = cli._execute(args, directory, old, stop, budget)
                self.assertEqual(returned, code)
                report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
                self.assertEqual(report["results"], [])
                self.assertEqual({row["id"] for row in report["retained_results"]},
                                 {"old.pass", "old.unvisited"})
                self.assertEqual(report["pending_case_ids"], [case.id])
                self.assertEqual(report["stop_reason"], reason)
                self.assertFalse(report["summary"]["selection_passed"])
                self.assertFalse(report["summary"]["feature_coverage"])


if __name__ == "__main__":
    unittest.main()
