"""Hard deadline and owned-process cleanup for pressure research pilots."""
from __future__ import annotations

import json
import os
import signal
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import threading
import time
from hashlib import sha256
from typing import Any


def _linklike(path: Path) -> bool:
    """Recognize symlinks and Windows reparse points even on older Python."""
    info = path.lstat()
    return (path.is_symlink() or bool(getattr(info, "st_file_attributes", 0)
                                     & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)))


def _remove_owned_stages(directory: Path, pid: int) -> list[str]:
    """Remove only this reaped worker's private stages, without following links."""
    notes = []
    store = directory / "store"
    try:
        if not store.exists() and not store.is_symlink():
            return notes
        if _linklike(store) or store.resolve().parent != directory:
            return ["Stage cleanup: research store is not a direct, non-link child of the output directory."]
        if not store.is_dir():
            return ["Stage cleanup: research store is not a directory."]
        prefix = f".research-stage-{pid}-"
        for candidate in store.iterdir():
            if not candidate.name.startswith(prefix) or len(candidate.name) == len(prefix):
                continue
            if _linklike(candidate) or candidate.resolve().parent != store.resolve():
                notes.append(f"Stage cleanup: skipped linked or escaping path {candidate.name}.")
                continue
            if not candidate.is_dir():
                notes.append(f"Stage cleanup: skipped non-directory path {candidate.name}.")
                continue
            try:
                shutil.rmtree(candidate)
            except OSError as error:
                notes.append(f"Stage cleanup {candidate.name}: {type(error).__name__}: {error}")
    except OSError as error:
        notes.append(f"Stage cleanup: {type(error).__name__}: {error}")
    return notes


def _read_worker_report(path: Path, provenance: dict[str, Any]) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if (not isinstance(data, dict) or data.get("options") != provenance["options"]
            or data.get("selected_inputs") != provenance.get("selected_inputs")
            or data.get("canonical") is not False
            or data.get("status") not in {"PASS", "FAIL", "ERROR", "TIMEOUT", "INTERRUPTED"}):
        return None
    if data.get("implementation_sha256") != provenance.get("implementation_sha256"):
        return None
    original = data.get("original_sha256")
    expected = provenance.get("original_sha256", {})
    if not isinstance(original, dict) or any(expected.get(key) != value for key, value in original.items()):
        return None
    if data["status"] in {"PASS", "FAIL"} and original != expected:
        return None
    return data


def _hashes_match(values: dict[str, str], root: Path) -> bool:
    for name, expected in values.items():
        path = Path(name)
        if not path.is_absolute():
            path = root / path
        digest = sha256()
        try:
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
        except OSError:
            return False
        if digest.hexdigest() != expected:
            return False
    return True


def _verify_hashes(originals: dict[str, str], implementation: dict[str, str], root: Path,
                   deadline: float, stop: threading.Event) -> tuple[bool | None, bool | None]:
    """Do not let a stalled filesystem prevent the hard-deadline report."""
    done = threading.Event()
    values: dict[str, bool] = {}
    def verify():
        try:
            values["originals"] = _hashes_match(originals, root)
            values["implementation"] = _hashes_match(implementation, root)
        except (OSError, ValueError):
            values.update(originals=False, implementation=False)
        finally:
            done.set()
    threading.Thread(target=verify, daemon=True).start()
    while not done.wait(min(.05, max(.001, deadline - time.monotonic()))):
        if stop.is_set() or time.monotonic() >= deadline:
            return None, None
    return values.get("originals"), values.get("implementation")


def supervise(command: list[str], directory: Path, *, started: float, seconds: float,
              provenance: dict[str, Any], stop: threading.Event, cwd: Path,
              failure_grace_s: float = 5.) -> tuple[int, dict[str, Any]]:
    """Run an owned child under a wall-clock deadline; retain its separate report.

    The deadline includes startup and reserves up to five seconds for owned
    cleanup and the authoritative report. The worker report is evidence, not
    a PASS until the child exits successfully inside the deadline.
    """
    tests_root = str(Path(__file__).resolve().parents[1] / "tests")
    if tests_root not in sys.path:
        sys.path.insert(0, tests_root)
    from simulation_suite.processes import OwnedProcessGroup

    directory = directory.resolve()
    hard_deadline = started + seconds
    reserve = min(5., max(.1, seconds * .2))
    work_deadline = hard_deadline - reserve
    worker_path = directory / "worker-report.json"
    child = None
    owner = None
    reason = None
    failure_at = None
    worker_report = None
    setup_error = None
    cleanup_notes = []
    try:
        if stop.is_set():
            reason = "INTERRUPTED"
        elif time.monotonic() >= work_deadline:
            reason = "TIMEOUT"
        else:
            child = subprocess.Popen(
                command, cwd=cwd,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
                start_new_session=os.name != "nt",
            )
            owner = OwnedProcessGroup(child)
            while True:
                now = time.monotonic()
                if stop.is_set():
                    reason = "INTERRUPTED"
                    break
                if now >= work_deadline:
                    reason = "TIMEOUT"
                    break
                code = child.poll()
                worker_report = _read_worker_report(worker_path, provenance)
                if code is not None:
                    break
                if worker_report is not None and worker_report["status"] != "PASS":
                    if failure_at is None:
                        failure_at = now
                    elif now - failure_at >= failure_grace_s:
                        reason = "WORKER_FAILURE"
                        break
                time.sleep(min(.05, max(.001, work_deadline - now)))
    except (OSError, subprocess.SubprocessError) as error:
        setup_error = f"{type(error).__name__}: {error}"
        reason = "ERROR"
    finally:
        if child is not None and (child.poll() is None or reason is not None):
            try:
                (directory / ".supervisor-cancel").touch()
            except OSError:
                pass
        if child is not None and child.poll() is None and reason is not None:
            grace = max(0., min(2., hard_deadline - time.monotonic() - .4))
            if grace:
                try:
                    child.wait(timeout=grace)
                except subprocess.TimeoutExpired:
                    pass
        try:
            if owner is not None:
                owner.close()
            elif child is not None and child.poll() is None:
                if os.name == "nt":
                    subprocess.run(["taskkill", "/PID", str(child.pid), "/T", "/F"],
                                   capture_output=True, timeout=max(.001, min(2., hard_deadline - time.monotonic())),
                                   creationflags=subprocess.CREATE_NO_WINDOW)
                else:
                    os.killpg(child.pid, signal.SIGKILL)
        except (OSError, subprocess.SubprocessError) as error:
            cleanup_notes.append(f"Owned process cleanup: {error}")
            if child is not None and child.poll() is None:
                child.kill()
        if child is not None and child.poll() is None:
            try:
                child.wait(timeout=max(.001, min(2., hard_deadline - time.monotonic() - .15)))
            except subprocess.TimeoutExpired:
                pass
        if child is not None and child.poll() is not None:
            # A worker force-killed inside HDF5 cannot run TemporaryDirectory's
            # cleanup. Bound our cleanup by the same hard deadline.
            cleanup_done = threading.Event()
            owned_notes = []
            def clean_stages():
                try:
                    owned_notes.extend(_remove_owned_stages(directory, child.pid))
                finally:
                    cleanup_done.set()
            budget = max(0., hard_deadline - time.monotonic() - .1)
            if budget:
                threading.Thread(target=clean_stages, daemon=True).start()
                if not cleanup_done.wait(budget):
                    cleanup_notes.append("Stage cleanup did not finish before the deadline.")
                else:
                    cleanup_notes.extend(owned_notes)
            else:
                cleanup_notes.append("Stage cleanup could not start before the deadline.")
        elif child is not None:
            cleanup_notes.append("Stage cleanup skipped because the worker has not exited.")

    latest = _read_worker_report(worker_path, provenance)
    if latest is not None:
        worker_report = latest
    result = dict(provenance if reason in {"TIMEOUT", "INTERRUPTED"} else worker_report or provenance)
    original_hashes = provenance.get("original_sha256", {})
    implementation_hashes = provenance.get("implementation_sha256", {})
    if provenance.get("supervisor_preflight_complete", True):
        originals_match, implementation_matches = _verify_hashes(
            original_hashes, implementation_hashes, cwd, hard_deadline, stop)
    else:
        originals_match, implementation_matches = None, None
    provenance_matches = (worker_report is None or
        (worker_report.get("original_sha256") == original_hashes and
         worker_report.get("implementation_sha256") == implementation_hashes))
    if stop.is_set():
        reason = "INTERRUPTED"
    elif time.monotonic() >= hard_deadline and reason != "INTERRUPTED":
        reason = "TIMEOUT"
    if reason in {"TIMEOUT", "INTERRUPTED"}:
        result = dict(provenance)
    if reason in {"TIMEOUT", "INTERRUPTED"}:
        result.update(status=reason, completed=False,
                      error="Pressure pilot exceeded its wall-clock deadline." if reason == "TIMEOUT"
                      else "Pressure pilot interrupted by the user.")
        code = 124 if reason == "TIMEOUT" else 130
    elif setup_error is not None:
        result.update(status="ERROR", completed=False, error=setup_error)
        code = 1
    elif worker_report is None:
        result.update(status="ERROR", completed=False,
                      error="The worker exited without a valid terminal report.")
        code = 1
    else:
        status = worker_report["status"]
        returncode = child.poll() if child is not None else None
        if status in {"PASS", "FAIL"} and not (originals_match and implementation_matches and provenance_matches):
            result.update(status="ERROR", completed=False,
                          error="Pilot inputs or implementation changed during the supervised run.")
            code = 1
        elif status == "PASS" and (reason is not None or returncode != 0
                or worker_report.get("completed") is not True
                or worker_report.get("originals_unchanged") is not True
                or worker_report.get("implementation_unchanged") is not True):
            result.update(status="ERROR", completed=False,
                          error="A passing worker report requires a successful, verified worker exit.")
            code = 1
        elif status == "PASS":
            code = 0
        elif status == "FAIL":
            code = 3
        elif status == "TIMEOUT":
            result["completed"] = False
            code = 124
        elif status == "INTERRUPTED":
            result["completed"] = False
            code = 130
        else:
            result["completed"] = False
            code = 1
    result["supervisor"] = {"deadline_s": seconds, "elapsed_s": time.monotonic() - started,
                            "reason": reason, "child_returncode": child.poll() if child else None,
                            "worker_report": str(worker_path) if worker_path.exists() else None,
                            "original_hashes_match": originals_match,
                            "implementation_hashes_match": implementation_matches,
                            "worker_provenance_matches": provenance_matches,
                            "cleanup_notes": cleanup_notes}
    report_path = directory / "report.json"
    temporary = directory / "report.json.partial"
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, report_path)
    return code, result
