"""Short real processes and fake clocks exercise the complete invocation cap."""

from __future__ import annotations

import copy
import ctypes
import errno
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from simulation_suite import budget, capabilities, host_skips, reporting, runner
from simulation_suite.spec import CaseSpec


class FakeClock:
    def __init__(self, now=0.0):
        self.now = now

    def __call__(self):
        return self.now


class InvocationBudgetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="torus budget ")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.run_dir = self.root / "run"
        self.run_dir.mkdir()

    def case(self, name="fixture", **kwargs):
        return CaseSpec(
            name,
            "fixture",
            "bounded fixture",
            (sys.executable, "-B", "-c", "pass"),
            **kwargs,
        )

    def cached(self, case):
        directory = self.run_dir / "cases" / case.id / "attempt-0001"
        directory.mkdir(parents=True)
        artifact = directory / "accepted.dat"
        artifact.write_bytes(b"accepted source")
        result = {
            "id": case.id,
            "kind": case.kind,
            "status": "PASS",
            "execution": "PASS",
            "qualification": "PASS",
            "elapsed_s": 0.1,
            "fingerprint": runner.case_fingerprint(case, self.root, {}),
            "case_dir": str(directory),
            "artifacts": {"source": str(artifact)},
            "artifact_hashes": {
                str(artifact): runner._artifact_hash(artifact, self.run_dir)
            },
        }
        (directory / "runner-result.json").write_text(
            json.dumps(result), encoding="utf-8"
        )
        return result

    def test_monotonic_work_cutoff_reserves_cleanup_and_keeps_first_stop_reason(self):
        clock = FakeClock(100.0)
        limit = budget.InvocationBudget(100.0, 60.0, 5.0, clock=clock)
        self.assertEqual(limit.remaining_seconds(), 55.0)
        clock.now = 155.0
        with self.assertRaises(budget.BudgetExpired):
            limit.poll()
        self.assertEqual(limit.remaining_seconds(hard=True), 5.0)
        limit.request_stop("user")
        self.assertEqual(limit.stop_reason, "budget")
        stop = threading.Event()
        stop.set()
        cancelled = budget.InvocationBudget(
            100.0, 60.0, 5.0, stop_event=stop, clock=clock
        )
        with self.assertRaises(budget.InvocationInterrupted):
            cancelled.poll()
        for total, reserve in (
            (0.0, 0.0),
            (10.0, 10.0),
            (float("nan"), 0.0),
            (10.0, -1.0),
        ):
            with (
                self.subTest(total=total, reserve=reserve),
                self.assertRaises(ValueError),
            ):
                budget.InvocationBudget(0.0, total, reserve)

    def test_environment_and_markers_roundtrip_without_becoming_case_inputs(self):
        names = {
            "TORUS_SUITE_STARTED_MONOTONIC": "100",
            "TORUS_SUITE_WORK_DEADLINE": "155",
            "TORUS_SUITE_HARD_DEADLINE": "160",
            "TORUS_SUITE_CANCEL_FILE": str(self.run_dir / ".control.cancel"),
        }
        with patch.dict(os.environ, names):
            limit = budget.InvocationBudget.from_env()
        self.assertEqual(limit.work_deadline, 155.0)
        self.assertEqual(limit.hard_deadline, 160.0)
        clock = FakeClock()
        first = budget.InvocationBudget(0.0, 60.0, 5.0, clock=clock)
        second = budget.InvocationBudget(0.0, 3600.0, 300.0, clock=clock)
        case = self.case()
        self.assertEqual(
            runner.case_fingerprint(case, self.root, {}, budget=first),
            runner.case_fingerprint(case, self.root, {}, budget=second),
        )
        self.assertEqual(case.timeout, 300.0)
        marker = self.run_dir / "cancel-marker"
        marker.write_text("user\n", encoding="ascii")
        with self.assertRaises(budget.InvocationInterrupted):
            budget.InvocationBudget(
                0.0, 60.0, 5.0, clock=clock, cancel_path=marker
            ).poll()

    def test_hashing_observes_deadline_between_blocks(self):
        clock = FakeClock()
        limit = budget.InvocationBudget(0.0, 2.0, 1.0, clock=clock)

        class AdvancingFile(io.BytesIO):
            def read(self, size=-1):
                value = super().read(size)
                clock.now = 1.0
                return value

        path = SimpleNamespace(open=lambda mode: AdvancingFile(b"x" * 100))
        with self.assertRaises(budget.BudgetExpired):
            runner._sha(path, limit)

    def test_budget_exhausted_cache_check_preserves_unvisited_passing_evidence(self):
        first, second = self.case("first"), self.case("second")
        cached = [self.cached(first), self.cached(second)]
        previous = {"results": copy.deepcopy(cached)}
        clock = FakeClock()
        limit = budget.InvocationBudget(0.0, 2.0, 1.0, clock=clock)
        original = runner._artifact_hash

        def expire_during_validation(path, run_dir, budget=None):
            clock.now = 1.0
            return original(path, run_dir, budget)

        with (
            patch.object(
                runner, "_artifact_hash", side_effect=expire_during_validation
            ),
            patch.object(runner, "run_case") as launch,
        ):
            report = runner.execute(
                [first, second],
                root=self.root,
                run_dir=self.run_dir,
                capabilities={},
                coverage={"feature_matrix": {"complete": True}},
                profile="features",
                previous=previous,
                budget=limit,
            )
        launch.assert_not_called()
        self.assertEqual(report["results"], [])
        self.assertEqual(report["retained_results"], cached)
        self.assertEqual(report["pending_case_ids"], [first.id, second.id])
        self.assertTrue(report["budget_exhausted"])
        self.assertFalse(report["summary"]["selection_passed"])
        self.assertFalse(report["summary"]["feature_coverage"])
        self.assertFalse(report["summary"]["full_coverage"])
        persisted = json.loads(
            (self.run_dir / "report.json").read_text(encoding="utf-8")
        )
        fresh = budget.InvocationBudget(time.monotonic(), 10.0, 1.0)
        with patch.object(runner, "run_case") as launch:
            resumed = runner.execute(
                [first, second],
                root=self.root,
                run_dir=self.run_dir,
                capabilities={},
                coverage={"feature_matrix": {"complete": True}},
                profile="features",
                previous=persisted,
                budget=fresh,
            )
        launch.assert_not_called()
        self.assertTrue(resumed["summary"]["feature_coverage"])
        self.assertFalse(resumed["summary"]["full_coverage"])

    def test_probe_budget_is_not_reported_as_missing_dependency(self):
        limit = budget.InvocationBudget(time.monotonic(), 0.8, 0.4)
        started = time.monotonic()
        with self.assertRaises(budget.BudgetExpired):
            capabilities._run_json(
                [sys.executable, "-B", "-c", "import time;time.sleep(30)"],
                timeout=30.0,
                budget=limit,
            )
        self.assertLess(time.monotonic() - started, 2.0)
        clock = FakeClock(2.0)
        expired = budget.InvocationBudget(0.0, 2.0, 1.0, clock=clock)
        with (
            patch.object(capabilities.subprocess, "Popen") as launch,
            self.assertRaises(budget.BudgetExpired),
        ):
            capabilities.detect_capabilities(budget=expired)
        launch.assert_not_called()

    def test_active_case_budget_cancellation_is_not_a_scientific_failure(self):
        case = CaseSpec(
            "active",
            "scientific",
            "cooperative timeout fixture",
            (sys.executable, "-B", "-c", "import time;time.sleep(30)"),
            timeout=30.0,
            kind="scientific",
        )
        limit = budget.InvocationBudget(time.monotonic(), 1.0, 0.5)
        started = time.monotonic()
        result = runner.run_case(
            case,
            root=self.root,
            run_dir=self.run_dir,
            capabilities={},
            profile="features",
            budget=limit,
        )
        self.assertEqual(result["status"], "INTERRUPTED")
        self.assertEqual(result["stop_reason"], "budget")
        self.assertEqual(result["qualification"], "NOT_RUN")
        self.assertLess(time.monotonic() - started, 3.0)
        self.assertEqual(case.timeout, 30.0)

    def test_cleanup_waits_use_remaining_hard_budget(self):
        clock = FakeClock(0.9)
        limit = budget.InvocationBudget(0.0, 1.0, 0.5, clock=clock)
        process = Mock(pid=12345)
        process.wait.side_effect = [subprocess.TimeoutExpired("owned", 1), 0]
        with (
            patch.object(runner, "stop_owned_wsl", return_value=None) as wsl,
            patch.object(runner.subprocess, "run") as terminate,
            patch.object(runner.os, "killpg", create=True),
        ):
            runner._stop_owned_process(process, self.root, external=True, budget=limit)
        self.assertTrue(
            all(
                call.kwargs["timeout"] <= 0.100001
                for call in process.wait.call_args_list
            )
        )
        self.assertLessEqual(wsl.call_args.kwargs["timeout"], 0.100001)
        if os.name == "nt":
            self.assertLessEqual(terminate.call_args.kwargs["timeout"], 0.100001)

    def test_private_ownership_registry_never_selects_unrelated_paths(self):
        marker = self.run_dir / ".invocation-specific.cancel"
        limit = budget.InvocationBudget(0.0, 60.0, 5.0, cancel_path=marker)
        limit.register_preflight()
        case_dir = self.run_dir / "cases" / "only-ours" / "attempt-0001"
        case_dir.mkdir(parents=True)
        limit.register_owned_path(case_dir)
        self.assertEqual(
            budget._owned_wsl_paths(self.run_dir, limit),
            [limit.preflight_token, case_dir],
        )
        marker.with_suffix(".owned").write_text(
            json.dumps({"case_dir": str(self.root)}), encoding="utf-8"
        )
        self.assertEqual(
            budget._owned_wsl_paths(self.run_dir, limit), [limit.preflight_token]
        )

    def test_supervisor_hard_backstop_stops_owned_native_descendants(self):
        script = self.root / "ignores-budget.py"
        script.write_text(
            "import json,os,subprocess,sys,time\nfrom pathlib import Path\n"
            "root=Path(sys.argv[1]); root.mkdir(parents=True,exist_ok=True)\n"
            "child=subprocess.Popen([sys.executable,'-B','-c','import time;time.sleep(30)'])\n"
            "(root/'descendant.pid').write_text(str(child.pid))\n"
            "report=dict(profile='features',manifest=[dict(id='pending')],results=[],retained_results=[dict(id='previous',status='PASS')])\n"
            "(root/'report.json').write_text(json.dumps(report))\ntime.sleep(30)\n",
            encoding="utf-8",
        )
        started = time.monotonic()
        with patch(
            "simulation_suite.processes.stop_owned_wsl", return_value=None
        ) as wsl:
            code = budget.supervise(
                [sys.executable, "-B", str(script), str(self.run_dir)],
                run_dir=self.run_dir,
                started_at=started,
                total_seconds=1.2,
                cleanup_reserve_seconds=0.8,
            )
        self.assertEqual(code, 124)
        self.assertLess(time.monotonic() - started, 3.0)
        report = json.loads((self.run_dir / "report.json").read_text(encoding="utf-8"))
        self.assertTrue(report["budget_exhausted"])
        self.assertEqual(report["pending_case_ids"], ["pending"])
        self.assertEqual(
            report["retained_results"], [{"id": "previous", "status": "PASS"}]
        )
        self.assertFalse(report["summary"]["selection_passed"])
        self.assertTrue(
            all(call.args[0] != self.run_dir for call in wsl.call_args_list)
        )
        pid = int((self.run_dir / "descendant.pid").read_text())
        if os.name == "nt":
            from ctypes import wintypes

            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.OpenProcess.argtypes = (
                wintypes.DWORD,
                wintypes.BOOL,
                wintypes.DWORD,
            )
            kernel.OpenProcess.restype = wintypes.HANDLE
            kernel.GetExitCodeProcess.argtypes = (
                wintypes.HANDLE,
                ctypes.POINTER(wintypes.DWORD),
            )
            kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
            handle = kernel.OpenProcess(0x1000, False, pid)
            if handle:
                try:
                    code = wintypes.DWORD()
                    self.assertTrue(
                        kernel.GetExitCodeProcess(handle, ctypes.byref(code))
                    )
                    self.assertNotEqual(code.value, 259)
                finally:
                    kernel.CloseHandle(handle)

    def test_supervisor_preserves_normal_exit_and_user_cancellation_codes(self):
        started = time.monotonic()
        self.assertEqual(
            budget.supervise(
                [sys.executable, "-B", "-c", "raise SystemExit(3)"],
                run_dir=self.run_dir,
                started_at=started,
                total_seconds=5.0,
                cleanup_reserve_seconds=1.0,
            ),
            3,
        )
        stop = threading.Event()
        stop.set()
        with patch.object(budget.subprocess, "Popen") as launch:
            code = budget.supervise(
                [sys.executable, "-B", "-c", "pass"],
                run_dir=self.run_dir,
                started_at=time.monotonic(),
                total_seconds=5.0,
                cleanup_reserve_seconds=1.0,
                stop_event=stop,
            )
        launch.assert_not_called()
        self.assertEqual(code, 130)

    def test_features_enables_required_integrations_and_rejects_required_skips(self):
        case_dir = self.root / "environment"
        case_dir.mkdir()
        environment = runner._environment(case_dir, "features")
        for name in ("TORUS_TEST_DESKTOP", "TORUS_TEST_FENICS", "TORUS_TEST_GEANT4"):
            self.assertEqual(environment[name], "1")
        path = Path(__file__).parent / "simulation_suite" / "unittest_worker.py"
        spec = importlib.util.spec_from_file_location("budget_unittest_worker", path)
        worker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(worker)
        for profile, expected in (("features", 1), ("full", 1), ("fast", 0)):
            with self.subTest(profile=profile):
                test = unittest.FunctionTestCase(lambda: None)
                test.setUp = lambda: self.skipTest("deliberate fixture skip")
                request, output = self.root / "request.json", self.root / "result.json"
                request.write_text(
                    json.dumps(
                        {"parameters": {"pattern": "fixture.py"}, "profile": profile}
                    ),
                    encoding="utf-8",
                )
                with (
                    patch.object(
                        sys,
                        "argv",
                        ["worker", "--request", str(request), "--result", str(output)],
                    ),
                    patch.object(sys, "path", list(sys.path)),
                    patch.object(
                        unittest.defaultTestLoader,
                        "discover",
                        return_value=unittest.TestSuite([test]),
                    ),
                    patch.object(sys, "stderr", io.StringIO()),
                ):
                    self.assertEqual(worker.main(), expected)
                record = json.loads(output.read_text(encoding="utf-8"))
                self.assertEqual(record["metrics"]["skipped"], 1)

    def test_worker_classifies_only_explicit_host_limitations(self):
        path = Path(__file__).parent / "simulation_suite" / "unittest_worker.py"
        spec = importlib.util.spec_from_file_location("applicability_unittest_worker", path)
        worker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(worker)
        request, output = self.root / "host-request.json", self.root / "host-result.json"
        symlink_id = "test_research_publication.ResearchPublicationTests.test_cleanup_does_not_follow_stage_symlink"
        reason = host_skips.host_skip_reason("symlink", errno.EPERM)

        class Fixture(unittest.TestCase):
            def __init__(self, identifier, *, skip=None, fail=False):
                super().__init__("runTest")
                self.identifier, self.skip_reason, self.should_fail = identifier, skip, fail

            def id(self):
                return self.identifier

            def runTest(self):
                if self.skip_reason:
                    self.skipTest(self.skip_reason)
                if self.should_fail:
                    self.fail("intentional fixture failure")

        def run(profile, cases):
            request.write_text(json.dumps({"parameters": {"pattern": "fixture.py"}, "profile": profile}), encoding="utf-8")
            with (patch.object(sys, "argv", ["worker", "--request", str(request), "--result", str(output)]),
                  patch.object(sys, "path", list(sys.path)),
                  patch.object(unittest.defaultTestLoader, "discover", return_value=unittest.TestSuite(cases)),
                  patch.object(sys, "stderr", io.StringIO())):
                code = worker.main()
            return code, json.loads(output.read_text(encoding="utf-8"))

        for profile in ("features", "full"):
            with self.subTest(profile=profile):
                code, record = run(profile, [Fixture("fixture.pass"), Fixture(symlink_id, skip=reason)])
                self.assertEqual(code, 0)
                self.assertEqual((record["status"], record["execution"], record["qualification"]),
                                 ("PASS", "PASS", "NOT_APPLICABLE"))
                self.assertEqual(record["metrics"]["tests_executed"], 1)
                self.assertEqual(record["skipped"][0]["classification"], "NOT_APPLICABLE")
                for skipped in (Fixture("fixture.unknown", skip="unavailable"),
                                Fixture("fixture.cuda", skip="CUDA is unavailable"),
                                Fixture("fixture.fenics", skip="FE is unavailable"),
                                Fixture("fixture.geant4", skip="Geant4 is unavailable"),
                                Fixture("fixture.desktop", skip="Desktop is unavailable"),
                                Fixture(symlink_id, skip="unexpected I/O failure")):
                    code, record = run(profile, [Fixture("fixture.pass"), skipped])
                    self.assertEqual(code, 1)
                    self.assertEqual(record["skipped"][0]["classification"], "MISSING")
                self.assertEqual(run(profile, [Fixture(symlink_id, skip=reason)])[0], 1)
                self.assertEqual(run(profile, [Fixture("fixture.fail", fail=True), Fixture(symlink_id, skip=reason)])[0], 1)
                self.assertEqual(run(profile, [])[0], 1)
        self.assertEqual(run("fast", [Fixture(symlink_id, skip=reason)])[0], 0)
        self.assertEqual(run("fast", [Fixture("fixture.unknown", skip="unavailable")])[0], 0)

        class WithSkippedSubtest(Fixture):
            def runTest(self):
                with self.subTest(sample=1):
                    self.skipTest("an optional subtest")

        code, record = run("fast", [WithSkippedSubtest("fixture.subtest")])
        self.assertEqual(code, 0)
        self.assertEqual(record["metrics"]["tests_executed"], 1)

    def test_host_skip_rules_and_html_keep_reasons_visible(self):
        windows_id = "test_torus_publication.JobPublicationTests.test_transport_request_paths_survive_publication"
        self.assertTrue(host_skips.is_host_inapplicable(windows_id, "Windows/WSL path representation", os_name="posix"))
        self.assertFalse(host_skips.is_host_inapplicable(windows_id, "Windows/WSL path representation", os_name="nt"))
        self.assertFalse(host_skips.is_host_limitation(OSError(errno.ENOSPC, "full")))
        self.assertTrue(host_skips.is_host_limitation(OSError(errno.EACCES, "permission")))
        with self.assertRaises(ValueError):
            host_skips.host_skip_reason("symlink", errno.ENOSPC)
        symlink_id = "test_setup_geant4.Geant4InstallerTests.test_cached_archive_symlink_is_rejected"
        for name, number in (("EPERM", 1), ("EACCES", 13), ("ENOSYS", 38),
                             ("ENOTSUP", 95), ("EOPNOTSUPP", 95)):
            self.assertTrue(host_skips.is_host_limitation(OSError(number, "fixture"), origin="linux", error_name=name))
            reason = host_skips.host_skip_reason("symlink", number, origin="linux", error_name=name)
            self.assertTrue(host_skips.is_host_inapplicable(symlink_id, reason, os_name="nt"))
        self.assertFalse(host_skips.is_host_limitation(OSError(38, "fixture"), origin="linux", error_name="ENOTSUP"))
        with self.assertRaises(ValueError):
            host_skips.host_skip_reason("symlink", 38, origin="linux", error_name="ENOTSUP")
        self.assertFalse(host_skips.is_host_inapplicable(
            symlink_id, "[host-inapplicable:symlink:38:none:linux:ENOTSUP] invalid", os_name="nt"))
        self.assertFalse(host_skips.is_host_inapplicable(
            symlink_id, "[host-inapplicable:symlink:38:1314:linux:ENOSYS] invalid", os_name="nt"))
        html = reporting.render_html({"results": [{"id": "fixture", "status": "PASS", "execution": "PASS",
            "qualification": "NOT_APPLICABLE", "skipped": [{"test": "fixture", "classification": "NOT_APPLICABLE",
            "reason": "host <not supported>"}]}]})
        self.assertIn("Skipped checks", html)
        self.assertIn("NOT_APPLICABLE", html)
        self.assertIn("host &lt;not supported&gt;", html)


if __name__ == "__main__":
    unittest.main()
