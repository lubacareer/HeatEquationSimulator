"""Real short child processes verify orchestration, resume, and honest reports."""

from __future__ import annotations

import ast
import copy
from dataclasses import replace
import ctypes
import hashlib
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
from unittest.mock import patch

from simulation_suite import capabilities, reporting
from simulation_suite.catalog import validate_cases
from simulation_suite.spec import CaseSpec


class CapabilityProbeTests(unittest.TestCase):
    def test_dataset_metadata_fingerprint_is_deterministic_and_tracks_input_changes(
        self,
    ):
        # Exercise the actual helper sent to WSL, without starting workers or
        # loading either scientific runtime in this process.
        tree = ast.parse(capabilities._TRANSPORT_PROBE)
        helper = next(
            item
            for item in tree.body
            if isinstance(item, ast.FunctionDef) and item.name == "stat_manifest"
        )
        namespace = {"hashlib": hashlib, "json": json}
        exec(
            compile(
                ast.Module(body=[helper], type_ignores=[]), "<stat manifest>", "exec"
            ),
            namespace,
        )
        manifest = namespace["stat_manifest"]
        with tempfile.TemporaryDirectory() as temporary:
            first, second = Path(temporary) / "voxel.dat", Path(temporary) / "table.dat"
            first.write_bytes(b"1234")
            second.write_bytes(b"organ names")
            original = manifest([first, second])
            self.assertEqual(manifest([second, first, first]), original)
            stat = first.stat()
            os.utime(first, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1000000000))
            self.assertNotEqual(manifest([first, second]), original)
            touched = manifest([first, second])
            first.write_bytes(b"12345")
            self.assertNotEqual(manifest([first, second]), touched)
            resized = manifest([first, second])
            relocated = first.with_name("moved.dat")
            first.rename(relocated)
            self.assertNotEqual(manifest([relocated, second]), resized)
            with self.assertRaises(OSError):
                manifest([first, second])

    def test_backend_code_is_sent_on_stdin_not_quoted_through_wsl_shell(self):
        code = 'print({"reason": "quotes and `backticks` remain data"})'
        completed = SimpleNamespace(
            returncode=0, stdout='startup message\n{"available": true}\n', stderr=""
        )
        with patch.object(
            capabilities.subprocess, "run", return_value=completed
        ) as run:
            result, error = capabilities._run_json(
                capabilities._backend_command(), input_text=code
            )
        self.assertEqual(error, "")
        self.assertTrue(result["available"])
        self.assertEqual(run.call_args.args[0][-1], "-")
        self.assertEqual(run.call_args.kwargs["input"], code)
        self.assertGreater(run.call_args.kwargs["timeout"], 0)

    def test_timeout_launch_failure_and_missing_json_are_unavailable_reasons(self):
        cases = (
            (subprocess.TimeoutExpired(["probe"], 1), "exceeded"),
            (OSError("not installed"), "Cannot launch"),
            (
                SimpleNamespace(returncode=3, stdout="", stderr="missing library"),
                "missing library",
            ),
            (SimpleNamespace(returncode=0, stdout="not JSON", stderr=""), "no JSON"),
        )
        for outcome, expected in cases:
            with self.subTest(expected=expected):
                kwargs = (
                    {"side_effect": outcome}
                    if isinstance(outcome, Exception)
                    else {"return_value": outcome}
                )
                with patch.object(capabilities.subprocess, "run", **kwargs):
                    result, reason = capabilities._run_json(["probe"], timeout=1)
                self.assertIsNone(result)
                self.assertIn(expected, reason)

    def test_independent_capabilities_do_not_invent_optional_runtime_success(self):
        yes = {"available": True, "reason": "checked", "versions": {}, "details": {}}
        no = {"available": False, "reason": "no CUDA", "versions": {}, "details": {}}
        responses = [
            (yes.copy(), ""),
            ({"torch_cpu": yes.copy(), "torch_cuda": no.copy()}, ""),
            (None, "FE import failed"),
            ({"geant4": yes.copy(), "icrp110": no.copy()}, ""),
        ]
        with (
            patch.object(capabilities, "_run_json", side_effect=responses) as probes,
            patch.object(
                capabilities.importlib.metadata, "version", return_value="test-version"
            ),
            patch.dict(
                capabilities.os.environ,
                {"TORUS_GEANT4_WORKER": "/custom worker/torus-dose"},
            ),
        ):
            detected = capabilities.detect_capabilities()
        self.assertEqual(
            probes.call_args_list[3].args[0][-1], "/custom worker/torus-dose"
        )
        self.assertEqual(
            set(detected),
            {"desktop", "torch_cpu", "torch_cuda", "fenics", "geant4", "icrp110"},
        )
        self.assertTrue(detected["torch_cpu"]["available"])
        self.assertFalse(detected["torch_cuda"]["available"])
        self.assertFalse(detected["fenics"]["available"])
        self.assertIn("FE import failed", detected["fenics"]["reason"])
        self.assertTrue(detected["geant4"]["available"])
        self.assertFalse(detected["icrp110"]["available"])
        for record in detected.values():
            self.assertIsInstance(record["versions"], dict)
            self.assertIsInstance(record["details"], dict)


class MatrixAndReportTests(unittest.TestCase):
    def case(self, name="fixture", **kwargs):
        return CaseSpec(name, "fixture", "test", ("{python}", "-c", "pass"), **kwargs)

    def test_catalog_rejects_duplicate_missing_cyclic_and_unsafe_cases(self):
        for cases in (
            [self.case(), self.case()],
            [self.case(depends_on=("missing",))],
            [
                self.case("first", depends_on=("second",)),
                self.case("second", depends_on=("first",)),
            ],
            [self.case("../outside")],
        ):
            with self.subTest(cases=cases), self.assertRaises(ValueError):
                validate_cases(cases)
        for timeout in (0, -1, float("nan"), float("inf")):
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                validate_cases([self.case(timeout=timeout)])

    def test_summary_distinguishes_short_profile_and_failed_scientific_gates(self):
        passed = {
            "id": "one",
            "status": "PASS",
            "execution": "PASS",
            "qualification": "NOT_APPLICABLE",
        }
        short = reporting.summarize([passed], full_coverage=False)
        self.assertTrue(short["selection_passed"])
        self.assertFalse(short["full_coverage"])
        self.assertEqual(reporting.summarize([], full_coverage=True)["status"], "FAIL")
        for status in (
            "FAIL",
            "ERROR",
            "TIMEOUT",
            "UNAVAILABLE",
            "NOT_RUN",
            "INTERRUPTED",
        ):
            with self.subTest(status=status):
                summary = reporting.summarize(
                    [passed, {"id": "two", "status": status}], full_coverage=True
                )
                self.assertEqual(summary["status"], "FAIL")
                self.assertFalse(summary["full_coverage"])
        contradiction = {**passed, "qualification": "FAIL"}
        self.assertEqual(
            reporting.summarize([contradiction], full_coverage=True)["status"], "FAIL"
        )

    def test_reports_escape_text_and_persist_complete_evidence(self):
        report = {
            "profile": "full",
            "revision": "test",
            "summary": {"status": "FAIL"},
            "capabilities": {"reason": '<script>alert("cap")</script>'},
            "coverage": {"notes": "<matrix>"},
            "results": [
                {
                    "id": "<case>",
                    "status": "FAIL",
                    "reason": '<img src=x onerror="bad">',
                    "metrics": {"literal": "<b>value</b>"},
                    "log_path": "<log>.txt",
                }
            ],
        }
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            json_path, html_path = reporting.write_reports(directory, report)
            self.assertEqual(json.loads(json_path.read_text(encoding="utf-8")), report)
            rendered = html_path.read_text(encoding="utf-8")
            self.assertNotIn("<script>", rendered)
            self.assertNotIn("<img src=x", rendered)
            self.assertIn("&lt;script&gt;", rendered)
            self.assertIn("&lt;case&gt;", rendered)
            self.assertIn("&lt;b&gt;value&lt;/b&gt;", rendered)
            self.assertEqual(list(directory.glob("*.partial")), [])


class IsolatedRunnerTests(unittest.TestCase):
    def setUp(self):
        from simulation_suite import runner

        self.runner = runner
        self.temporary = tempfile.TemporaryDirectory(prefix="torus runner ")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.run_dir = self.root / "run"
        self.run_dir.mkdir()

    def child_case(
        self, *, result=None, exit_code=0, sleep=0.0, timeout=10.0, **kwargs
    ):
        script = self.root / "worker with spaces.py"
        script.write_text(
            "import json,os,sys,time\nfrom pathlib import Path\n"
            "out=Path(sys.argv[1]);out.parent.mkdir(parents=True,exist_ok=True)\n"
            "print('progress \\u03c6',flush=True)\n"
            f"time.sleep({sleep!r})\n"
            "artifact=out.parent/'snapshot.dat';artifact.write_text('accepted data',encoding='utf-8')\n"
            f"value={result!r}\n"
            "if value is not None:\n value.setdefault('artifacts',{'snapshot':str(artifact)})\n out.write_text(json.dumps(value),encoding='utf-8')\n"
            f"raise SystemExit({exit_code!r})\n",
            encoding="utf-8",
        )
        return CaseSpec(
            "fixture.child",
            "fixture",
            "Short subprocess fixture",
            ("{python}", "-B", str(script), "{case_dir}/result.json"),
            timeout,
            dependencies=(script.name,),
            **kwargs,
        )

    def execute(self, case, **kwargs):
        return self.runner.run_case(
            case, root=self.root, run_dir=self.run_dir, capabilities={}, **kwargs
        )

    def terminal_child(self, value, *, mode="atomic", timeout=12.0, descendant=False,
                       linger=30., return_code=1):
        """Publish a result and remain alive, like a worker stuck in teardown."""
        script = self.root / "terminal worker.py"
        script.write_text(
            "import json,os,subprocess,sys,time\nfrom pathlib import Path\n"
            "out=Path(sys.argv[1])\n"
            f"value={value!r}; mode={mode!r}\n"
            f"descendant={descendant!r}\n"
            "if descendant:\n"
            " child=subprocess.Popen([sys.executable,'-B','-c','import time;time.sleep(30)'])\n"
            " (out.parent/'descendant.pid').write_text(str(child.pid),encoding='ascii')\n"
            "if mode=='malformed': out.write_text('{',encoding='utf-8')\n"
            "elif mode=='delayed':\n"
            " out.write_text('{',encoding='utf-8');time.sleep(.3)\n"
            " temp=out.with_suffix('.partial');temp.write_text(json.dumps(value),encoding='utf-8');os.replace(temp,out)\n"
            "elif mode in ('atomic','partial'):\n"
            " temp=out.with_suffix('.partial');temp.write_text(json.dumps(value),encoding='utf-8')\n"
            " if mode=='atomic': os.replace(temp,out)\n"
            "print('published terminal result',flush=True)\n"
            f"time.sleep({linger!r})\n"
            f"raise SystemExit({return_code!r})\n",
            encoding="utf-8",
        )
        return CaseSpec(
            "fixture.terminal", "fixture", "Terminal failure with a stuck child",
            ("{python}", "-B", str(script), "{case_dir}/result.json"),
            timeout, kind="scientific", dependencies=(script.name,),
        )

    def assert_process_stopped(self, pid):
        if os.name == "nt":
            from ctypes import wintypes

            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
            kernel.OpenProcess.restype = wintypes.HANDLE
            kernel.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
            kernel.GetExitCodeProcess.restype = wintypes.BOOL
            kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
            handle = kernel.OpenProcess(0x1000, False, pid)
            if handle:
                try:
                    code = wintypes.DWORD()
                    self.assertTrue(kernel.GetExitCodeProcess(handle, ctypes.byref(code)))
                    self.assertNotEqual(code.value, 259, "Owned descendant remains active.")
                finally:
                    kernel.CloseHandle(handle)
        else:
            status = Path(f"/proc/{pid}/stat")
            if status.exists():
                self.assertEqual(status.read_text().split()[2], "Z")
            else:
                with self.assertRaises(ProcessLookupError):
                    os.kill(pid, 0)

    def test_real_child_preserves_logs_qualification_and_artifacts(self):
        case = self.child_case(
            result={
                "status": "PASS",
                "execution": "PASS",
                "qualification": "PASS",
                "metrics": {"error": 0.0},
            }
        )
        result = self.execute(case)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["qualification"], "PASS")
        self.assertEqual(result["metrics"], {"error": 0.0})
        self.assertIn(
            "progress φ", Path(result["log_path"]).read_text(encoding="utf-8")
        )
        self.assertTrue(result["artifact_hashes"])
        self.assertTrue(
            self.runner.reusable_result(result, result["fingerprint"], self.run_dir)
        )
        artifact = Path(result["artifacts"]["snapshot"])
        artifact.write_text("changed accepted data", encoding="utf-8")
        self.assertFalse(
            self.runner.reusable_result(result, result["fingerprint"], self.run_dir)
        )

    def test_scientific_failure_and_bad_exit_cannot_become_success(self):
        result = self.execute(
            self.child_case(
                result={
                    "status": "FAIL",
                    "execution": "PASS",
                    "qualification": "FAIL",
                    "reason": "accuracy gate",
                },
                exit_code=3,
                kind="scientific",
            )
        )
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(result["execution"], "PASS")
        self.assertEqual(result["qualification"], "FAIL")
        self.assertIn("accuracy gate", result["reason"])
        misleading = self.execute(
            self.child_case(
                result={"status": "PASS", "execution": "PASS", "qualification": "FAIL"},
                kind="scientific",
            )
        )
        self.assertNotEqual(misleading["status"], "PASS")
        bad_exit = self.execute(
            self.child_case(
                result={"status": "PASS", "execution": "PASS", "qualification": "PASS"},
                exit_code=1,
            )
        )
        self.assertNotEqual(bad_exit["status"], "PASS")

    def test_missing_result_and_unavailable_prerequisite_are_explicit(self):
        missing = self.execute(self.child_case(result=None))
        self.assertEqual(missing["status"], "ERROR")
        unavailable = self.execute(
            self.child_case(result={"status": "PASS"}, capabilities=("missing",))
        )
        self.assertEqual(unavailable["status"], "UNAVAILABLE")
        self.assertFalse(
            self.runner.reusable_result(
                unavailable, unavailable["fingerprint"], self.run_dir
            )
        )

    def test_result_schema_and_runner_owned_provenance_cannot_be_overwritten(self):
        malformed = self.execute(
            self.child_case(
                result={"status": "PASS", "execution": "PASS", "artifacts": []}
            )
        )
        self.assertEqual(malformed["status"], "ERROR")
        case = self.child_case(
            result={
                "status": "PASS",
                "execution": "PASS",
                "qualification": "PASS",
                "id": "another-case",
                "category": "spoof",
                "kind": "scientific",
                "fingerprint": "spoof",
                "case_dir": "elsewhere",
                "log_path": "elsewhere",
                "artifact_hashes": {"external-file": "unverified"},
            }
        )
        expected_fingerprint = self.runner.case_fingerprint(case, self.root, {}, {})
        result = self.execute(case)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["fingerprint"], expected_fingerprint)
        self.assertEqual(result["id"], case.id)
        self.assertEqual(result["category"], case.category)
        self.assertEqual(result["kind"], case.kind)
        self.assertNotIn("external-file", result["artifact_hashes"])
        self.assertTrue(Path(result["case_dir"]).is_dir())
        self.assertTrue(Path(result["log_path"]).is_file())

    def test_timeout_and_cancellation_are_bounded(self):
        started = time.monotonic()
        timed_out = self.execute(self.child_case(sleep=30.0, timeout=0.2))
        self.assertEqual(timed_out["status"], "TIMEOUT")
        self.assertLess(time.monotonic() - started, 10.0)
        event = threading.Event()
        timer = threading.Timer(0.2, event.set)
        timer.start()
        self.addCleanup(timer.cancel)
        started = time.monotonic()
        interrupted = self.execute(
            self.child_case(sleep=30.0, timeout=10.0), stop_event=event
        )
        self.assertEqual(interrupted["status"], "INTERRUPTED")
        self.assertLess(time.monotonic() - started, 10.0)

    def test_atomic_failure_gets_five_seconds_then_owned_tree_is_stopped(self):
        failure = {"case_id": "fixture.terminal", "status": "ERROR", "execution": "FAIL",
                   "qualification": "NOT_APPLICABLE", "reason": "original CUDA OOM",
                   "traceback": "original traceback", "artifacts": {}, "metrics": {}}
        unrelated = subprocess.Popen(
            [sys.executable, "-B", "-c", "import time;time.sleep(30)"],
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        try:
            started = time.monotonic()
            result = self.execute(self.terminal_child(failure, descendant=True))
            elapsed = time.monotonic() - started
            self.assertGreaterEqual(elapsed, 4.8)
            self.assertLess(elapsed, 15.0)
            self.assertEqual(result["status"], "ERROR")
            self.assertEqual(result["reason"], "original CUDA OOM")
            self.assertEqual(result["traceback"], "original traceback")
            self.assertTrue(result["terminal_failure_detected"])
            pid = int((Path(result["case_dir"]) / "descendant.pid").read_text())
            self.assert_process_stopped(pid)
            self.assertIsNone(unrelated.poll(), "An unrelated process was stopped.")
        finally:
            unrelated.terminate()
            unrelated.wait(timeout=5)

    def test_terminal_scientific_failure_is_preserved_after_cleanup(self):
        failure = {"case_id": "fixture.terminal", "status": "FAIL", "execution": "PASS",
                   "qualification": "FAIL", "reason": "pressure accuracy gate",
                   "artifacts": {}, "metrics": {"pressure_normalized_rmse": .116}}
        with patch.object(self.runner, "_FAILURE_EXIT_GRACE_SECONDS", .2):
            result = self.execute(self.terminal_child(failure))
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(result["qualification"], "FAIL")
        self.assertEqual(result["reason"], "pressure accuracy gate")
        self.assertEqual(result["metrics"]["pressure_normalized_rmse"], .116)

    def test_invalid_failure_artifact_does_not_erase_original_error(self):
        failure = {"case_id": "fixture.terminal", "status": "ERROR", "execution": "FAIL",
                   "qualification": "NOT_APPLICABLE", "reason": "original CUDA OOM",
                   "artifacts": {"bad": str(self.root / "outside")}, "metrics": {}}
        with patch.object(self.runner, "_FAILURE_EXIT_GRACE_SECONDS", .2):
            result = self.execute(self.terminal_child(failure))
        self.assertEqual(result["status"], "ERROR")
        self.assertIn("original CUDA OOM", result["reason"])
        self.assertIn("Invalid or missing case artifact", result["reason"])

    def test_nonfinite_json_is_not_an_early_terminal_result(self):
        case = self.terminal_child(None, mode="none")
        directory = self.root / "attempt"
        directory.mkdir()
        (directory / "result.json").write_text(
            '{"case_id":"fixture.terminal","status":"ERROR","execution":"FAIL",'
            '"qualification":"NOT_APPLICABLE","reason":"bad","metrics":{"x":NaN}}',
            encoding="utf-8",
        )
        self.assertIsNone(self.runner._terminal_failure(case, directory))

    def test_worker_can_exit_naturally_during_failure_grace(self):
        failure = {"case_id": "fixture.terminal", "status": "ERROR", "execution": "FAIL",
                   "qualification": "NOT_APPLICABLE", "reason": "worker failed"}
        result = self.execute(self.terminal_child(failure, linger=.4))
        self.assertEqual(result["status"], "ERROR")
        self.assertEqual(result["reason"], "worker failed")
        self.assertNotIn("terminal_failure_detected", result)
        self.assertEqual(result["returncode"], 1)

    def test_case_deadline_wins_over_failure_grace(self):
        failure = {"case_id": "fixture.terminal", "status": "ERROR", "execution": "FAIL",
                   "qualification": "NOT_APPLICABLE", "reason": "worker failed"}
        result = self.execute(self.terminal_child(failure, timeout=1.5))
        self.assertTrue((Path(result["case_dir"]) / "result.json").is_file())
        self.assertEqual(result["status"], "TIMEOUT")
        self.assertEqual(result["worker_failure"]["reason"], "worker failed")

    def test_case_deadline_wins_if_worker_exits_between_polls(self):
        success = {"case_id": "fixture.terminal", "status": "PASS", "execution": "PASS",
                   "qualification": "PASS"}
        original = self.runner._terminal_failure
        delayed = threading.Event()

        def delay_first_result(case, directory):
            value = original(case, directory)
            if (directory / "result.json").is_file() and not delayed.is_set():
                delayed.set()
                time.sleep(1.4)
            return value

        with patch.object(self.runner, "_terminal_failure", side_effect=delay_first_result):
            result = self.execute(self.terminal_child(success, timeout=1.2, linger=.6, return_code=0))
        self.assertTrue(delayed.is_set())
        self.assertEqual(result["returncode"], 0)
        self.assertEqual(result["status"], "TIMEOUT")

    def test_case_deadline_wins_during_pipe_drain_and_retains_worker_error(self):
        failure = {"case_id": "fixture.terminal", "status": "ERROR", "execution": "FAIL",
                   "qualification": "NOT_APPLICABLE", "reason": "original worker error"}
        result = self.execute(self.terminal_child(failure, timeout=1.5, descendant=True, linger=.05))
        self.assertEqual(result["status"], "TIMEOUT")
        self.assertEqual(result["worker_failure"]["reason"], "original worker error")
        pid = int((Path(result["case_dir"]) / "descendant.pid").read_text())
        self.assert_process_stopped(pid)

    def test_partial_malformed_wrong_case_and_pass_files_do_not_end_case(self):
        failure = {"case_id": "fixture.terminal", "status": "ERROR", "execution": "FAIL",
                   "qualification": "NOT_APPLICABLE", "reason": "failure"}
        cases = (
            ("partial", failure),
            ("malformed", failure),
            ("atomic", {**failure, "case_id": "different.case"}),
            ("atomic", {**failure, "execution": "PASS"}),
            ("atomic", {"case_id": "fixture.terminal", "status": "PASS",
                        "execution": "PASS", "qualification": "PASS"}),
        )
        for mode, value in cases:
            with self.subTest(mode=mode, value=value):
                with patch.object(self.runner, "_FAILURE_EXIT_GRACE_SECONDS", .2):
                    result = self.execute(self.terminal_child(value, mode=mode, timeout=1.5))
                self.assertEqual(result["status"], "TIMEOUT")
                path = Path(result["case_dir"]) / ("result.partial" if mode == "partial" else "result.json")
                self.assertTrue(path.is_file(), "The child did not publish its fixture.")

    def test_previous_attempt_failure_is_not_read_by_new_attempt(self):
        previous = self.run_dir / "cases" / "fixture.terminal" / "attempt-0001"
        previous.mkdir(parents=True)
        (previous / "result.json").write_text(json.dumps({
            "case_id": "fixture.terminal", "status": "ERROR", "execution": "FAIL",
            "qualification": "NOT_APPLICABLE", "reason": "stale error",
        }), encoding="utf-8")
        result = self.execute(self.terminal_child(None, mode="none", timeout=1.5))
        self.assertEqual(result["status"], "TIMEOUT")
        self.assertEqual(Path(result["case_dir"]).name, "attempt-0002")

    def test_incomplete_file_can_be_followed_by_an_atomic_failure(self):
        failure = {"case_id": "fixture.terminal", "status": "ERROR", "execution": "FAIL",
                   "qualification": "NOT_APPLICABLE", "reason": "eventual failure"}
        with patch.object(self.runner, "_FAILURE_EXIT_GRACE_SECONDS", .2):
            result = self.execute(self.terminal_child(failure, mode="delayed"))
        self.assertEqual(result["status"], "ERROR")
        self.assertEqual(result["reason"], "eventual failure")

    def test_user_and_budget_cancellation_override_published_failure(self):
        from simulation_suite.budget import InvocationBudget

        failure = {"case_id": "fixture.terminal", "status": "ERROR", "execution": "FAIL",
                   "qualification": "NOT_APPLICABLE", "reason": "worker failed"}
        for reason in ("user", "budget"):
            with self.subTest(reason=reason):
                event = threading.Event()
                budget = InvocationBudget(time.monotonic(), 30., 1.) if reason == "budget" else None
                found = threading.Event()
                parent = self.run_dir / "cases" / "fixture.terminal"
                attempt = 1
                while (parent / f"attempt-{attempt:04d}").exists():
                    attempt += 1
                expected_result = parent / f"attempt-{attempt:04d}" / "result.json"

                def cancel_after_result():
                    for _ in range(200):
                        if expected_result.is_file():
                            found.set()
                            if budget is not None:
                                budget.request_stop("budget")
                            else:
                                event.set()
                            return
                        time.sleep(.05)

                watcher = threading.Thread(target=cancel_after_result, daemon=True)
                watcher.start()
                result = self.execute(self.terminal_child(failure), stop_event=event, budget=budget)
                watcher.join(timeout=2)
                self.assertTrue(found.is_set(), "The failure was not published before cancellation.")
                self.assertEqual(result["status"], "INTERRUPTED")
                self.assertEqual(result["stop_reason"], reason)
                self.assertEqual(result["qualification"], "NOT_RUN")

    def test_cancel_during_pipe_cleanup_overrides_published_failure(self):
        failure = {"case_id": "fixture.terminal", "status": "ERROR", "execution": "FAIL",
                   "qualification": "NOT_APPLICABLE", "reason": "worker failed"}
        event, published = threading.Event(), threading.Event()
        path = self.run_dir / "cases" / "fixture.terminal" / "attempt-0001" / "result.json"

        def cancel_after_exit():
            for _ in range(200):
                if path.is_file():
                    published.set()
                    time.sleep(.3)
                    event.set()
                    return
                time.sleep(.05)

        watcher = threading.Thread(target=cancel_after_exit, daemon=True)
        watcher.start()
        result = self.execute(
            self.terminal_child(failure, descendant=True, linger=.05), stop_event=event
        )
        watcher.join(timeout=2)
        self.assertTrue(published.is_set())
        self.assertEqual(result["status"], "INTERRUPTED")
        self.assertEqual(result["stop_reason"], "user")
        pid = int((Path(result["case_dir"]) / "descendant.pid").read_text())
        self.assert_process_stopped(pid)

    def test_fingerprints_change_with_inputs_dependencies_and_runtime_versions(self):
        case = self.child_case(
            result={"status": "PASS"}, parameters={"seed": 42}, capabilities=("test",)
        )
        caps = {"test": {"available": True, "versions": {"worker": "1"}}}

        def fingerprint(spec=case, runtime=caps, upstream=None):
            return self.runner.case_fingerprint(
                spec, self.root, runtime, upstream or {}
            )

        original = fingerprint()
        self.assertEqual(fingerprint(), original)
        self.assertNotEqual(
            fingerprint(replace(case, parameters={"seed": 43})), original
        )
        self.assertNotEqual(
            fingerprint(
                runtime={"test": {"available": True, "versions": {"worker": "2"}}}
            ),
            original,
        )
        (self.root / "worker with spaces.py").write_text(
            "changed code", encoding="utf-8"
        )
        self.assertNotEqual(fingerprint(), original)
        dependent = replace(case, depends_on=("source",))
        before = fingerprint(
            dependent,
            upstream={
                "source": {
                    "status": "PASS",
                    "fingerprint": "one",
                    "artifact_hashes": {"source": "aaa"},
                }
            },
        )
        after = fingerprint(
            dependent,
            upstream={
                "source": {
                    "status": "PASS",
                    "fingerprint": "one",
                    "artifact_hashes": {"source": "bbb"},
                }
            },
        )
        self.assertNotEqual(before, after)
        accepted_parent = {
            "status": "PASS",
            "execution": "PASS",
            "qualification": "PASS",
            "kind": "scientific",
            "fingerprint": "one",
            "artifact_hashes": {},
        }
        before = fingerprint(dependent, upstream={"source": accepted_parent})
        for changed_gate in (
            {"execution": "FAIL"},
            {"qualification": "FAIL"},
            {"kind": "functional"},
        ):
            with self.subTest(changed_gate=changed_gate):
                self.assertNotEqual(
                    fingerprint(
                        dependent,
                        upstream={"source": {**accepted_parent, **changed_gate}},
                    ),
                    before,
                )

    def test_descendant_holding_stdout_is_stopped_without_a_pipe_hang(self):
        script = self.root / "spawn descendant.py"
        script.write_text(
            "import json,subprocess,sys\nfrom pathlib import Path\n"
            "out=Path(sys.argv[1])\n"
            "child=subprocess.Popen([sys.executable,'-B','-c','import time;time.sleep(30)'])\n"
            "(out.parent/'descendant.pid').write_text(str(child.pid),encoding='ascii')\n"
            "out.write_text(json.dumps(dict(status='PASS',execution='PASS',qualification='NOT_APPLICABLE')),encoding='utf-8')\n",
            encoding="utf-8",
        )
        case = CaseSpec(
            "fixture.descendant",
            "fixture",
            "Owned descendant pipe cleanup",
            ("{python}", "-B", str(script), "{case_dir}/result.json"),
            timeout=10.0,
            dependencies=(script.name,),
        )
        started = time.monotonic()
        result = self.execute(case)
        self.assertLess(time.monotonic() - started, 12.0)
        self.assertEqual(result["status"], "ERROR")
        self.assertIn("descendants", result["reason"])
        pid = int((Path(result["case_dir"]) / "descendant.pid").read_text())
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
            kernel.GetExitCodeProcess.restype = wintypes.BOOL
            kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
            handle = kernel.OpenProcess(0x1000, False, pid)
            if handle:
                try:
                    code = wintypes.DWORD()
                    self.assertTrue(
                        kernel.GetExitCodeProcess(handle, ctypes.byref(code))
                    )
                    self.assertNotEqual(
                        code.value, 259, "Owned descendant remains active."
                    )
                finally:
                    kernel.CloseHandle(handle)
        else:
            # A just-killed child may briefly remain a zombie until reaped.
            status = Path(f"/proc/{pid}/stat")
            if status.exists():
                self.assertEqual(status.read_text().split()[2], "Z")
            else:
                with self.assertRaises(ProcessLookupError):
                    os.kill(pid, 0)

    def test_resume_rejects_nonpassing_and_mismatched_fingerprints(self):
        result = self.execute(
            self.child_case(
                result={"status": "PASS", "execution": "PASS", "qualification": "PASS"}
            )
        )
        self.assertFalse(
            self.runner.reusable_result(result, "another fingerprint", self.run_dir)
        )
        self.assertFalse(
            self.runner.reusable_result(
                {**result, "status": "FAIL"}, result["fingerprint"], self.run_dir
            )
        )
        external = self.root / "external.dat"
        external.write_text("external accepted data", encoding="utf-8")
        self.assertFalse(
            self.runner.reusable_result(
                {**result, "artifact_hashes": {str(external): "irrelevant"}},
                result["fingerprint"],
                self.run_dir,
            )
        )
        self.assertFalse(
            self.runner.reusable_result(
                {**result, "case_dir": str(self.root)},
                result["fingerprint"],
                self.run_dir,
            )
        )
        self.assertFalse(
            self.runner.reusable_result(
                {**result, "artifact_hashes": {}},
                result["fingerprint"],
                self.run_dir,
            )
        )
        self.assertFalse(
            self.runner.reusable_result(
                {**result, "case_dir": str(self.run_dir / "missing")},
                result["fingerprint"],
                self.run_dir,
            )
        )
        for gates in ({"execution": "FAIL"}, {"qualification": "FAIL"}):
            self.assertFalse(
                self.runner.reusable_result(
                    {**result, **gates},
                    result["fingerprint"],
                    self.run_dir,
                )
            )

    def test_incomplete_report_never_claims_success_before_remaining_cases_finish(self):
        first = self.child_case(result=None)
        second = replace(first, id="fixture.second")
        completed = [
            {
                "id": case.id,
                "kind": "functional",
                "status": "PASS",
                "execution": "PASS",
                "qualification": "NOT_APPLICABLE",
                "elapsed_s": 0.0,
            }
            for case in (first, second)
        ]
        snapshots = []

        def capture(directory, report):
            snapshots.append(copy.deepcopy(report))

        with (
            patch.object(self.runner, "run_case", side_effect=completed),
            patch.object(self.runner, "write_reports", side_effect=capture),
        ):
            report = self.runner.execute(
                [first, second],
                root=self.root,
                run_dir=self.run_dir,
                capabilities={},
                coverage={"scientific_matrix": {"complete": True}},
                profile="full",
            )
        first_completed = [item for item in snapshots if len(item["results"]) == 1]
        self.assertEqual(len(first_completed), 1)
        for snapshot in snapshots[:-1]:
            summary = snapshot.get("summary", {})
            self.assertEqual(summary.get("status", "RUNNING"), "RUNNING")
            self.assertFalse(summary.get("selection_passed", False))
            self.assertFalse(summary.get("full_coverage", False))
        self.assertEqual(report["summary"]["status"], "PASS")
        self.assertTrue(report["summary"]["full_coverage"])
        self.assertEqual(snapshots[-1], report)

    def test_resume_rechecks_required_capability_availability(self):
        case = self.child_case(
            result={"status": "PASS", "execution": "PASS", "qualification": "PASS"},
            capabilities=("fixture_runtime",),
        )
        ready = {"fixture_runtime": {"available": True, "versions": {"runtime": "1"}}}
        lost = {
            "fixture_runtime": {
                "available": False,
                "versions": {"runtime": "1"},
                "reason": "runtime removed",
            }
        }
        accepted = self.runner.run_case(
            case, root=self.root, run_dir=self.run_dir, capabilities=ready
        )
        self.assertEqual(accepted["status"], "PASS")
        self.assertNotEqual(
            self.runner.case_fingerprint(case, self.root, ready, {}),
            self.runner.case_fingerprint(case, self.root, lost, {}),
        )
        report = self.runner.execute(
            [case],
            root=self.root,
            run_dir=self.run_dir,
            capabilities=lost,
            coverage={"scientific_matrix": {"complete": True}},
            profile="full",
            previous={"results": [accepted]},
        )
        self.assertEqual(report["results"][0]["status"], "UNAVAILABLE")
        self.assertNotIn("reused", report["results"][0])
        self.assertIn("runtime removed", report["results"][0]["reason"])
        self.assertEqual(report["summary"]["status"], "FAIL")
        self.assertFalse(report["summary"]["full_coverage"])

    def test_selected_resume_preserves_unselected_cache_for_next_full_resume(self):
        first = self.child_case(
            result={"status": "PASS", "execution": "PASS", "qualification": "PASS"},
        )
        second = replace(first, id="fixture.second")
        options = dict(
            root=self.root,
            run_dir=self.run_dir,
            capabilities={},
            coverage={"scientific_matrix": {"complete": True}},
            profile="full",
        )
        original = self.runner.execute([first, second], **options)
        self.assertTrue(original["summary"]["full_coverage"])
        with patch.object(
            self.runner, "run_case", wraps=self.runner.run_case
        ) as launch:
            selected = self.runner.execute(
                [first], previous=original, full_selection=False, **options
            )
            launch.assert_not_called()
        self.assertEqual([item["id"] for item in selected["results"]], [first.id])
        self.assertEqual(
            [item["id"] for item in selected["retained_results"]], [second.id]
        )
        self.assertEqual(selected["retained_results"][0], original["results"][1])
        self.assertTrue(selected["summary"]["selection_passed"])
        self.assertFalse(selected["summary"]["full_coverage"])
        # Resume from the persisted selected report, as the command-line driver
        # does, rather than relying on references retained by this test.
        persisted = json.loads(
            (self.run_dir / "report.json").read_text(encoding="utf-8")
        )
        with patch.object(
            self.runner, "run_case", wraps=self.runner.run_case
        ) as launch:
            resumed = self.runner.execute(
                [first, second], previous=persisted, **options
            )
            launch.assert_not_called()
        self.assertEqual(
            [item["id"] for item in resumed["results"]], [first.id, second.id]
        )
        self.assertTrue(all(item["reused"] for item in resumed["results"]))
        self.assertEqual(resumed["retained_results"], [])
        self.assertTrue(resumed["summary"]["full_coverage"])

    def test_crashed_resume_preserves_unvisited_cache_in_each_durable_report(self):
        first = self.child_case(
            result={"status": "PASS", "execution": "PASS", "qualification": "PASS"}
        )
        second = replace(first, id="fixture.second")
        options = dict(
            root=self.root,
            run_dir=self.run_dir,
            capabilities={},
            coverage={"scientific_matrix": {"complete": True}},
            profile="full",
        )
        original = self.runner.execute([first, second], **options)
        write_reports = self.runner.write_reports
        for crash_after_write in (1, 2):
            with self.subTest(crash_after_write=crash_after_write):
                writes = 0

                def persist_then_crash(directory, report):
                    nonlocal writes
                    write_reports(directory, report)
                    writes += 1
                    if writes == crash_after_write:
                        raise KeyboardInterrupt("simulated abrupt runner exit")

                with (
                    patch.object(
                        self.runner, "write_reports", side_effect=persist_then_crash
                    ),
                    patch.object(self.runner, "run_case") as launch,
                    self.assertRaises(KeyboardInterrupt),
                ):
                    self.runner.execute([first, second], previous=original, **options)
                launch.assert_not_called()
                persisted = json.loads(
                    (self.run_dir / "report.json").read_text(encoding="utf-8")
                )
                retained = [item["id"] for item in persisted["retained_results"]]
                self.assertEqual(
                    retained,
                    [first.id, second.id] if crash_after_write == 1 else [second.id],
                )
                self.assertFalse(
                    persisted.get("summary", {}).get("selection_passed", False)
                )
                with patch.object(self.runner, "run_case") as launch:
                    resumed = self.runner.execute(
                        [first, second], previous=persisted, **options
                    )
                launch.assert_not_called()
                self.assertTrue(all(item["reused"] for item in resumed["results"]))
                self.assertTrue(resumed["summary"]["full_coverage"])

    def test_cancelled_resume_reuses_intact_cache_but_never_reports_success(self):
        first = self.child_case(
            result={"status": "PASS", "execution": "PASS", "qualification": "PASS"}
        )
        second = replace(first, id="fixture.second")
        options = dict(
            root=self.root,
            run_dir=self.run_dir,
            capabilities={},
            coverage={"scientific_matrix": {"complete": True}},
            profile="full",
        )
        original = self.runner.execute([first, second], **options)
        stop = threading.Event()
        stop.set()
        with patch.object(self.runner, "run_case") as launch:
            cancelled = self.runner.execute(
                [first, second], previous=original, stop_event=stop, **options
            )
        launch.assert_not_called()
        self.assertTrue(cancelled["interrupted"])
        self.assertTrue(all(item["reused"] for item in cancelled["results"]))
        self.assertEqual(cancelled["summary"]["counts"], {"PASS": 2})
        self.assertEqual(cancelled["summary"]["status"], "FAIL")
        self.assertFalse(cancelled["summary"]["selection_passed"])
        self.assertFalse(cancelled["summary"]["full_coverage"])
        persisted = json.loads(
            (self.run_dir / "report.json").read_text(encoding="utf-8")
        )
        with patch.object(self.runner, "run_case") as launch:
            resumed = self.runner.execute(
                [first, second], previous=persisted, **options
            )
        launch.assert_not_called()
        self.assertTrue(resumed["summary"]["full_coverage"])

    def test_cancellation_does_not_reuse_stale_damaged_or_unavailable_cache(self):
        case = self.child_case(
            result={"status": "PASS", "execution": "PASS", "qualification": "PASS"},
            capabilities=("fixture_runtime",),
        )
        ready = {"fixture_runtime": {"available": True, "versions": {"runtime": "1"}}}
        accepted = self.runner.run_case(
            case, root=self.root, run_dir=self.run_dir, capabilities=ready
        )
        stop = threading.Event()
        stop.set()
        stale = {**accepted, "fingerprint": "outdated"}
        damaged = {
            **accepted,
            "artifact_hashes": {
                path: "changed" for path in accepted["artifact_hashes"]
            },
        }
        unavailable = {
            "fixture_runtime": {**ready["fixture_runtime"], "available": False}
        }
        for cached, runtime in (
            (stale, ready),
            (damaged, ready),
            (accepted, unavailable),
        ):
            with self.subTest(cached=cached["fingerprint"], runtime=runtime):
                with (
                    patch.object(
                        self.runner.subprocess,
                        "run",
                        return_value=SimpleNamespace(returncode=0, stdout="test"),
                    ),
                    patch.object(self.runner.subprocess, "Popen") as launch,
                ):
                    report = self.runner.execute(
                        [case],
                        root=self.root,
                        run_dir=self.run_dir,
                        capabilities=runtime,
                        coverage={"scientific_matrix": {"complete": True}},
                        profile="full",
                        previous={"results": [cached]},
                        stop_event=stop,
                    )
                launch.assert_not_called()
                self.assertEqual(report["results"][0]["status"], "INTERRUPTED")
                self.assertNotIn("reused", report["results"][0])
                self.assertFalse(report["summary"]["selection_passed"])

    def test_dependencies_require_complete_functional_or_scientific_success(self):
        case = self.child_case(
            result={
                "status": "PASS",
                "execution": "PASS",
                "qualification": "NOT_APPLICABLE",
            },
            depends_on=("upstream",),
        )
        passing = {
            "status": "PASS",
            "execution": "PASS",
            "qualification": "PASS",
            "kind": "scientific",
        }
        failures = [
            {},
            {**passing, "status": "FAIL"},
            {**passing, "execution": "FAIL"},
            {**passing, "qualification": "FAIL"},
            {**passing, "qualification": "NOT_APPLICABLE"},
            {**passing, "qualification": "NOT_RUN"},
            {**passing, "kind": "functional", "qualification": "FAIL"},
        ]
        for upstream in failures:
            with self.subTest(upstream=upstream):
                with patch.object(self.runner.subprocess, "Popen") as launch:
                    result = self.execute(
                        case, dependency_results={"upstream": upstream}
                    )
                launch.assert_not_called()
                self.assertEqual(result["status"], "BLOCKED")
                self.assertIn("upstream", result["reason"])
                for gate in ("status", "execution", "qualification"):
                    self.assertIn(
                        f"{gate}={upstream.get(gate, 'missing')}", result["reason"]
                    )
        for upstream in (
            passing,
            {**passing, "kind": "functional", "qualification": "NOT_APPLICABLE"},
        ):
            with self.subTest(upstream=upstream):
                result = self.execute(case, dependency_results={"upstream": upstream})
                self.assertEqual(result["status"], "PASS")

    def test_failed_parent_gates_cannot_be_bypassed_by_child_cache(self):
        child = self.child_case(
            result={"status": "PASS", "execution": "PASS", "qualification": "PASS"},
            depends_on=("upstream",),
        )
        parent = replace(child, id="upstream", depends_on=(), kind="scientific")
        passing = {
            "id": parent.id,
            "kind": "scientific",
            "status": "PASS",
            "execution": "PASS",
            "qualification": "PASS",
            "elapsed_s": 0.0,
            "fingerprint": "upstream-fingerprint",
        }
        cached = self.execute(child, dependency_results={parent.id: passing})
        self.assertTrue(
            self.runner.reusable_result(cached, cached["fingerprint"], self.run_dir)
        )
        failed = {**passing, "qualification": "FAIL"}
        run_case = self.runner.run_case

        def execute_case(case, **kwargs):
            return failed if case.id == parent.id else run_case(case, **kwargs)

        # Deliberately preserve the child's fingerprint to prove the dependency
        # gate is enforced independently of the fingerprint's cache invalidation.
        with (
            patch.object(
                self.runner, "case_fingerprint", return_value=cached["fingerprint"]
            ),
            patch.object(self.runner, "run_case", side_effect=execute_case),
            patch.object(
                self.runner.subprocess,
                "run",
                return_value=SimpleNamespace(returncode=0, stdout="test"),
            ),
            patch.object(self.runner.subprocess, "Popen") as launch,
        ):
            report = self.runner.execute(
                [parent, child],
                root=self.root,
                run_dir=self.run_dir,
                capabilities={},
                coverage={"scientific_matrix": {"complete": True}},
                profile="full",
                previous={"results": [cached]},
            )
        launch.assert_not_called()
        self.assertEqual(report["results"][1]["status"], "BLOCKED")
        self.assertNotIn("reused", report["results"][1])
        self.assertIn("qualification=FAIL", report["results"][1]["reason"])
        self.assertFalse(report["summary"]["full_coverage"])

    def test_full_selection_requires_complete_matrix_but_subset_remains_partial(self):
        case = self.child_case(
            result={"status": "PASS", "execution": "PASS", "qualification": "PASS"},
        )
        options = dict(
            root=self.root, run_dir=self.run_dir, capabilities={}, profile="full"
        )
        for coverage in ({}, {"scientific_matrix": {"complete": False}}):
            with self.subTest(coverage=coverage):
                full = self.runner.execute([case], coverage=coverage, **options)
                self.assertEqual(full["summary"]["status"], "FAIL")
                self.assertFalse(full["summary"]["full_coverage"])
                self.assertIn(
                    "coverage.integrity", [item["id"] for item in full["results"]]
                )
                partial = self.runner.execute(
                    [case],
                    previous=full,
                    full_selection=False,
                    coverage=coverage,
                    **options,
                )
                self.assertEqual([item["id"] for item in partial["results"]], [case.id])
                self.assertTrue(partial["summary"]["selection_passed"])
                self.assertFalse(partial["summary"]["full_coverage"])


if __name__ == "__main__":
    unittest.main()
