"""Bounded child processes, private artifacts, and conservative resume checks."""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from functools import lru_cache
import fnmatch
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import signal
import subprocess
import sys
import threading
import time
import traceback

from .reporting import mark_stopped, summarize, write_reports
from .budget import InvocationStopped
from .processes import OwnedProcessGroup, stop_owned_wsl
from .spec import CaseSpec


_FAILURE_EXIT_GRACE_SECONDS = 5.0


@lru_cache(maxsize=1)
def software_versions() -> dict:
    packages = {}
    for name in (
        "numpy",
        "scipy",
        "h5py",
        "matplotlib",
        "imageio",
        "imageio-ffmpeg",
        "Pillow",
        "torch",
        "PyQt6",
        "pyvista",
        "pyvistaqt",
        "vtk",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not installed"
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": packages,
    }


def _sha(path: Path, budget=None) -> str:
    digest = hashlib.sha256()
    if budget is not None:
        budget.poll()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            if budget is not None:
                budget.poll()
            digest.update(block)
    if budget is not None:
        budget.poll()
    return digest.hexdigest()


def _inside(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    resolved.relative_to(root.resolve())
    return resolved


def _artifact_hash(path: Path, run_dir: Path, budget=None) -> str:
    if budget is not None:
        budget.poll()
    resolved = _inside(path, run_dir)
    if not resolved.exists():
        raise FileNotFoundError(resolved)
    if resolved.is_file():
        return _sha(resolved, budget)
    digest = hashlib.sha256()
    for file in sorted(resolved.rglob("*")):
        if file.is_file():
            _inside(file, run_dir)
            digest.update(file.relative_to(resolved).as_posix().encode())
            digest.update(_sha(file, budget).encode())
    return digest.hexdigest()


def case_fingerprint(
    case: CaseSpec,
    root: Path,
    capabilities: dict,
    dependency_results: dict | None = None,
    budget=None,
) -> str:
    if budget is not None:
        budget.poll()
    files = {}
    patterns = case.dependencies or (
        "magnetic_torus/**/*.py",
        "tests/simulation_suite/**/*.py",
    )
    for name in ("runner.py", "processes.py", "spec.py", "unittest_worker.py"):
        path = root / "tests" / "simulation_suite" / name
        if path.is_file():
            files[path.relative_to(root).as_posix()] = _sha(path, budget)
    for pattern in patterns:
        for path in sorted(root.glob(pattern)):
            if path.is_file():
                files[path.relative_to(root).as_posix()] = _sha(path, budget)
    payload = {
        "case": case.to_dict(),
        "files": files,
        "software": software_versions(),
        "capabilities": {
            name: {
                key: capabilities.get(name, {}).get(key, {})
                for key in ("available", "versions", "details")
            }
            for name in case.capabilities
        },
        "parents": {
            key: {
                "fingerprint": value.get("fingerprint"),
                "status": value.get("status"),
                "execution": value.get("execution"),
                "qualification": value.get("qualification"),
                "kind": value.get("kind"),
                "artifacts": value.get("artifact_hashes", {}),
            }
            for key, value in (dependency_results or {}).items()
        },
    }
    if budget is not None:
        budget.poll()
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _passes_gates(result: dict) -> bool:
    qualification = result.get("qualification")
    return (
        result.get("status") == "PASS"
        and result.get("execution") == "PASS"
        and qualification in ("PASS", "NOT_APPLICABLE")
        and (result.get("kind") != "scientific" or qualification == "PASS")
    )


def _failed_dependencies(case: CaseSpec, parents: dict) -> list[str]:
    return [
        f"{name} (status={parent.get('status', 'missing')}, "
        f"execution={parent.get('execution', 'missing')}, "
        f"qualification={parent.get('qualification', 'missing')})"
        for name in case.depends_on
        if not _passes_gates(parent := parents.get(name, {}))
    ]


def reusable_result(
    result: dict, expected_fingerprint: str, run_dir: Path, budget=None
) -> bool:
    if budget is not None:
        budget.poll()
    if not _passes_gates(result) or result.get("fingerprint") != expected_fingerprint:
        return False
    try:
        case_dir = _inside(Path(result["case_dir"]), run_dir)
        if not case_dir.is_dir() or not (case_dir / "runner-result.json").is_file():
            return False
        artifacts = result.get("artifacts", {})
        if not isinstance(artifacts, dict) or any(
            not isinstance(value, str) for value in artifacts.values()
        ):
            return False
        if not isinstance(result.get("artifact_hashes", {}), dict):
            return False
        if set(artifacts.values()) != set(result.get("artifact_hashes", {})):
            return False
        for path, digest in result.get("artifact_hashes", {}).items():
            if _artifact_hash(Path(path), run_dir, budget) != digest:
                return False
    except (OSError, ValueError, KeyError, TypeError):
        return False
    return True


def _environment(case_dir: Path, profile: str) -> dict[str, str]:
    environment = {
        **os.environ,
        "PYTHONUTF8": "1",
        "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONOPTIMIZE": "0",
        "MAGNETIC_TORUS_DATA": str(case_dir / "history"),
        "MAGNETIC_TORUS_RUNS": str(case_dir / "worker_artifacts"),
        "MAGNETIC_TORUS_PINN_DEVICE": "auto",
        "QT_API": "pyqt6",
    }
    for name in ("TORUS_TEST_DESKTOP", "TORUS_TEST_FENICS", "TORUS_TEST_GEANT4"):
        environment[name] = "1" if profile in {"features", "full"} else "0"
    temporary = case_dir / "tmp"
    temporary.mkdir()
    for name in ("TMP", "TEMP", "TMPDIR"):
        environment[name] = str(temporary)
    # Each graphics module chooses offscreen/native itself; don't inherit a
    # platform override that could silently replace the real VTK display.
    environment.pop("QT_QPA_PLATFORM", None)
    environment.pop("KMP_DUPLICATE_LIB_OK", None)
    return environment


def _stop_owned_process(
    process: subprocess.Popen, case_dir: Path, *, external: bool, budget=None,
    grace_seconds: float | None = None,
) -> list[str]:
    notes = []

    def timeout(seconds):
        return (
            max(0.001, min(seconds, budget.remaining_seconds(hard=True)))
            if budget is not None
            else seconds
        )

    (case_dir / "cancel").touch()
    for request in case_dir.rglob("request.json"):
        try:
            _inside(request, case_dir)
            request.with_suffix(".cancel").touch()
            if request.parent.name != case_dir.name:
                (request.parent / "cancel").touch()
        except (OSError, ValueError):
            pass
    try:
        process.wait(timeout=timeout(
            (15 if external else 1) if grace_seconds is None else grace_seconds
        ))
        return notes
    except subprocess.TimeoutExpired:
        pass
    if external:
        note = stop_owned_wsl(case_dir, timeout=timeout(10))
        if note:
            notes.append(note)
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                capture_output=True,
                timeout=timeout(10),
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
        except (OSError, subprocess.SubprocessError) as error:
            notes.append(f"Owned Windows process-tree cleanup: {error}")
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=timeout(3))
    except subprocess.TimeoutExpired:
        if os.name != "nt":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            process.kill()
        try:
            process.wait(timeout=timeout(5))
        except subprocess.TimeoutExpired:
            notes.append("The owned child did not acknowledge forced termination.")
    return notes


def _validate_case_data(case: CaseSpec, data: object, returncode: int) -> dict:
    if not isinstance(data, dict) or data.get("status") not in {
        "PASS",
        "FAIL",
        "ERROR",
        "UNAVAILABLE",
    }:
        raise ValueError("Invalid child result status.")
    if not isinstance(data.get("artifacts", {}), dict) or not isinstance(
        data.get("metrics", {}), dict
    ):
        raise ValueError("Child artifacts and metrics must be mappings.")
    if data.get("execution") not in {"PASS", "FAIL"} or data.get(
        "qualification"
    ) not in {"PASS", "FAIL", "NOT_APPLICABLE"}:
        raise ValueError("Invalid child execution or qualification status.")
    if data.get("status") == "PASS" and (returncode or data.get("execution") != "PASS"):
        raise ValueError("A failing child cannot publish a passing result.")
    if (
        data.get("status") == "PASS"
        and case.kind == "scientific"
        and data.get("qualification") != "PASS"
    ):
        raise ValueError("Scientific case did not publish a passing qualification gate.")
    if data.get("qualification") == "FAIL" and data.get("status") != "ERROR":
        data["status"] = "FAIL"
    return data


def _terminal_failure(case: CaseSpec, case_dir: Path) -> dict | None:
    """Read only a complete, case-specific published failure from this attempt.

    Scientific workers publish result.json by atomic replacement. A partial,
    inaccessible, wrong-case or success result is never an early terminal event.
    """
    path = case_dir / "result.json"
    try:
        if path.is_symlink() or not path.is_file():
            return None
        _inside(path, case_dir)

        def reject_nonfinite_constant(value):
            raise ValueError(f"Non-finite JSON constant: {value}")

        data = json.loads(
            path.read_text(encoding="utf-8"), parse_constant=reject_nonfinite_constant
        )
        if (not isinstance(data, dict) or data.get("case_id") != case.id
                or data.get("status") not in {"ERROR", "FAIL"}
                or not isinstance(data.get("reason"), str)):
            return None
        if data["status"] == "ERROR" and data.get("execution") != "FAIL":
            return None
        if data["status"] == "FAIL" and data.get("qualification") == "PASS":
            return None
        return _validate_case_data(case, data, 1)
    except (OSError, UnicodeError, ValueError, TypeError):
        return None


def _decode_case(
    case: CaseSpec, case_dir: Path, returncode: int, last_lines: deque
) -> dict:
    path = case_dir / "result.json"
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
    elif case.id == "desktop.performance" and (case_dir / "benchmark.json").is_file():
        metrics = json.loads((case_dir / "benchmark.json").read_text(encoding="utf-8"))
        passed = metrics["median_fps"] >= case.parameters["minimum_fps"]
        data = {
            "status": "PASS" if passed else "FAIL",
            "execution": "PASS",
            "qualification": "NOT_APPLICABLE",
            "metrics": metrics,
            "reason": "" if passed else "Playback is below the defined 30 FPS target.",
            "artifacts": {"benchmark": str(case_dir / "benchmark.json")},
        }
    elif case.id == "desktop.journey":
        summary = None
        for line in reversed(last_lines):
            try:
                candidate = json.loads(line)
                if isinstance(candidate, dict) and "passed" in candidate:
                    summary = candidate
                    break
            except ValueError:
                pass
        if summary is None:
            raise ValueError(
                "Desktop journey published no structured completion summary."
            )
        data = {
            "status": "PASS" if summary["passed"] else "FAIL",
            "execution": "PASS",
            "qualification": "NOT_APPLICABLE",
            "reason": "",
            "metrics": summary,
            "artifacts": {"evidence": summary["artifact_directory"]},
        }
    else:
        raise ValueError("Child published no structured result; inspect its log.")
    return _validate_case_data(case, data, returncode)


def run_case(
    case: CaseSpec,
    *,
    root: Path,
    run_dir: Path,
    capabilities: dict,
    dependency_results: dict | None = None,
    profile: str = "full",
    stop_event: threading.Event | None = None,
    budget=None,
) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", case.id):
        raise ValueError("Case IDs must be safe directory names.")
    if (
        isinstance(case.timeout, bool)
        or not isinstance(case.timeout, (int, float))
        or not math.isfinite(case.timeout)
        or case.timeout <= 0
    ):
        raise ValueError("Case timeouts must be finite and positive.")
    stop = stop_event or threading.Event()
    parents = dependency_results or {}
    fingerprint = case_fingerprint(case, root, capabilities, parents, budget=budget)
    result = {
        "id": case.id,
        "category": case.category,
        "description": case.description,
        "kind": case.kind,
        "status": "NOT_RUN",
        "execution": "NOT_RUN",
        "qualification": "NOT_RUN" if case.kind == "scientific" else "NOT_APPLICABLE",
        "reason": "",
        "metrics": {},
        "artifacts": {},
        "elapsed_s": 0.0,
        "fingerprint": fingerprint,
        "artifact_hashes": {},
    }
    if stop.is_set():
        return {
            **result,
            "status": "INTERRUPTED",
            "reason": "Run interrupted before launch.",
        }
    unavailable = [
        name
        for name in case.capabilities
        if not capabilities.get(name, {}).get("available")
    ]
    if unavailable:
        return {
            **result,
            "status": "UNAVAILABLE",
            "reason": "; ".join(
                f"{name}: {capabilities.get(name, {}).get('reason', 'capability not detected')}"
                for name in unavailable
            ),
        }
    blocked = _failed_dependencies(case, parents)
    if blocked:
        return {
            **result,
            "status": "BLOCKED",
            "reason": "Dependencies did not pass required gates: " + ", ".join(blocked),
        }
    parent_dir = run_dir / "cases" / case.id
    parent_dir.mkdir(parents=True, exist_ok=True)
    attempt = 1
    while (parent_dir / f"attempt-{attempt:04d}").exists():
        attempt += 1
    case_dir = parent_dir / f"attempt-{attempt:04d}"
    case_dir.mkdir()
    log_path = case_dir / "output.log"
    result.update(case_dir=str(case_dir), log_path=str(log_path))
    request = {
        "case_id": case.id,
        "parameters": case.parameters,
        "profile": profile,
        "case_dir": str(case_dir),
        "run_dir": str(run_dir),
        "dependencies": parents,
    }
    (case_dir / "request.json").write_text(
        json.dumps(request, indent=2), encoding="utf-8"
    )
    if budget is not None:
        budget.poll()
        budget.register_owned_path(case_dir)
    substitutions = {
        "python": sys.executable,
        "root": str(root),
        "case_dir": str(case_dir),
        "run_dir": str(run_dir),
    }
    command = [argument.format(**substitutions) for argument in case.command]
    result["command"] = command
    started = time.monotonic()
    process = None
    owner = None
    reader = None
    accepted_failure = False
    capture_stop = threading.Event()
    last_lines: deque[str] = deque(maxlen=40)
    with log_path.open("w", encoding="utf-8") as log:
        try:
            process = subprocess.Popen(
                command,
                cwd=root,
                env=_environment(case_dir, profile),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
                if os.name == "nt"
                else 0,
                start_new_session=os.name != "nt",
            )
            owner = OwnedProcessGroup(process)

            def output():
                if process is None or process.stdout is None:
                    return
                try:
                    for line in process.stdout:
                        if capture_stop.is_set():
                            break
                        log.write(line)
                        log.flush()
                        last_lines.append(line.strip())
                        print(f"    {line.rstrip()}", flush=True)
                except (OSError, ValueError):
                    if not capture_stop.is_set():
                        last_lines.append("Output capture was interrupted.")

            reader = threading.Thread(target=output, daemon=True)
            reader.start()
            pending_failure = None
            failure_seen_at = None
            external = any(
                name in case.capabilities
                for name in ("fenics", "geant4", "torch_cpu", "torch_cuda")
            )

            def interrupted_reason():
                reason = "user" if stop.is_set() else None
                if budget is not None:
                    try:
                        budget.poll()
                    except InvocationStopped as stopped:
                        reason = stopped.reason
                return reason

            def preserve_failure():
                if "worker_failure" in result:
                    return
                failure = (result if accepted_failure and result.get("status") in {"ERROR", "FAIL"}
                           else _terminal_failure(case, case_dir))
                if failure is not None:
                    result["worker_failure"] = {
                        key: failure[key] for key in ("status", "reason", "traceback")
                        if key in failure
                    }

            def mark_interrupted(reason):
                preserve_failure()
                if process.poll() is None:
                    result["cleanup_notes"] = _stop_owned_process(
                        process, case_dir, external=external, budget=budget,
                    )
                result.update(
                    status="INTERRUPTED", execution="FAIL",
                    qualification="NOT_RUN" if case.kind == "scientific" else "NOT_APPLICABLE",
                    artifacts={}, metrics={},
                    reason=("Invocation time budget exhausted." if reason == "budget"
                            else "User interrupted the run."),
                    stop_reason=reason,
                )

            def mark_timeout():
                preserve_failure()
                if process.poll() is None:
                    result["cleanup_notes"] = _stop_owned_process(
                        process, case_dir, external=external, budget=budget,
                    )
                result.update(
                    status="TIMEOUT", execution="FAIL",
                    qualification="NOT_RUN" if case.kind == "scientific" else "NOT_APPLICABLE",
                    artifacts={}, metrics={},
                    reason=f"Case exceeded {case.timeout:g} seconds.",
                )

            while True:
                stop_reason = interrupted_reason()
                completed = process.poll() is not None
                if stop_reason is not None:
                    mark_interrupted(stop_reason)
                    break
                if time.monotonic() - started >= case.timeout:
                    mark_timeout()
                    break
                if completed:
                    break

                current_failure = _terminal_failure(case, case_dir)
                if current_failure != pending_failure:
                    pending_failure = current_failure
                    failure_seen_at = time.monotonic() if current_failure is not None else None
                elif (current_failure is not None and failure_seen_at is not None
                      and time.monotonic() - failure_seen_at >= _FAILURE_EXIT_GRACE_SECONDS):
                    # The failure must remain unchanged throughout the grace period.
                    # A still-running process and its descendants are ours to clean up.
                    if process.poll() is not None:
                        break
                    result.update(current_failure)
                    accepted_failure = True
                    result["terminal_failure_detected"] = True
                    try:
                        result["cleanup_notes"] = _stop_owned_process(
                            process, case_dir, external=external, budget=budget,
                            grace_seconds=0.,
                        )
                    except Exception as error:
                        result["cleanup_notes"] = [f"Owned process cleanup failed: {error}"]
                    break
                time.sleep(0.1)

            # A cancellation arriving during the grace or cleanup takes priority.
            stop_reason = interrupted_reason()
            if stop_reason is not None:
                mark_interrupted(stop_reason)
            elif result["status"] not in {"INTERRUPTED", "TIMEOUT"} and time.monotonic() - started >= case.timeout:
                mark_timeout()
            join_timeout = max(0., min(3., case.timeout - (time.monotonic() - started)))
            if budget is not None:
                join_timeout = min(join_timeout, budget.remaining_seconds())
            reader.join(timeout=join_timeout)
            stop_reason = interrupted_reason()
            if stop_reason is not None:
                mark_interrupted(stop_reason)
            elif result["status"] not in {"INTERRUPTED", "TIMEOUT"} and time.monotonic() - started >= case.timeout:
                mark_timeout()
            if reader.is_alive():
                if owner is not None:
                    owner.close()
                reader.join(timeout=3)
                if result["status"] not in {"TIMEOUT", "INTERRUPTED"}:
                    failure = pending_failure if accepted_failure else _terminal_failure(case, case_dir)
                    if failure is None:
                        raise ValueError(
                            "Child left descendants holding its output pipe; owned processes were stopped."
                        )
                    result.update(failure)
                    result.setdefault("cleanup_notes", []).append(
                        "Owned descendants holding the output pipe were stopped."
                    )
                    accepted_failure = True
            if result["status"] not in {"TIMEOUT", "INTERRUPTED"} and not accepted_failure:
                result.update(
                    _decode_case(case, case_dir, process.wait(timeout=3), last_lines)
                )
            stop_reason = interrupted_reason()
            if stop_reason is not None:
                mark_interrupted(stop_reason)
            elif result["status"] not in {"INTERRUPTED", "TIMEOUT"} and time.monotonic() - started >= case.timeout:
                mark_timeout()
            result["returncode"] = process.poll()
        except KeyboardInterrupt:
            stop.set()
            if process is not None:
                result["cleanup_notes"] = _stop_owned_process(
                    process, case_dir, external=True, budget=budget
                )
            result.update(
                status="INTERRUPTED",
                execution="FAIL",
                reason="User interrupted the run.",
            )
        except Exception as error:
            if process is not None and process.poll() is None:
                result["cleanup_notes"] = _stop_owned_process(
                    process, case_dir, external=True, budget=budget
                )
            result.update(status="ERROR", execution="FAIL", reason=str(error))
            log.write(traceback.format_exc())
        finally:
            if owner is not None:
                owner.close()
            if reader is not None:
                reader.join(timeout=2)
            capture_stop.set()
            if (
                process is not None
                and process.stdout is not None
                and (reader is None or not reader.is_alive())
            ):
                process.stdout.close()
    result["artifact_hashes"] = {}
    try:
        if not isinstance(result.get("artifacts", {}), dict):
            raise ValueError("Case artifacts must map names to file/directory paths.")
        for value in result.get("artifacts", {}).values():
            if not isinstance(value, str):
                raise ValueError("Case artifact paths must be strings.")
            result["artifact_hashes"][value] = _artifact_hash(
                Path(value), run_dir, budget
            )
    except (OSError, ValueError) as error:
        original_reason = result.get("reason", "") if accepted_failure else ""
        result.update(
            status="ERROR",
            execution="FAIL",
            reason=(f"{original_reason}; " if original_reason else "")
                   + f"Invalid or missing case artifact: {error}",
        )
    # Envelopes are owned by the runner, not trusted child-supplied metadata.
    result.update(
        id=case.id,
        category=case.category,
        description=case.description,
        kind=case.kind,
        fingerprint=fingerprint,
        elapsed_s=round(time.monotonic() - started, 3),
        case_dir=str(case_dir),
        log_path=str(log_path),
    )
    (case_dir / "runner-result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def execute(
    cases: list[CaseSpec],
    *,
    root: Path,
    run_dir: Path,
    capabilities: dict,
    coverage: dict,
    profile: str,
    previous: dict | None = None,
    full_selection: bool = True,
    stop_event: threading.Event | None = None,
    budget=None,
) -> dict:
    stop = stop_event or threading.Event()
    old = {
        item["id"]: item
        for field in ("retained_results", "results")
        for item in (previous or {}).get(field, [])
    }
    results = []
    by_id = {}
    report = {
        "schema_version": 1,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "profile": profile,
        "capabilities": capabilities,
        "coverage": coverage,
        "manifest": [case.to_dict() for case in cases],
        "results": results,
        "retained_results": list(old.values()),
        "pending_case_ids": [case.id for case in cases],
        "coverage_scope": profile,
        "software": software_versions(),
    }
    stop_reason = None
    if budget is not None:
        try:
            budget.poll()
        except InvocationStopped as stopped:
            stop_reason = stopped.reason
    try:
        revision = (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=max(0.001, min(10.0, budget.remaining_seconds()))
                if budget is not None
                else 10,
            )
            if stop_reason is None
            else None
        )
        report["revision"] = (
            revision.stdout.strip()
            if revision is not None and revision.returncode == 0
            else "unknown"
        )
    except (OSError, subprocess.SubprocessError):
        report["revision"] = "unknown"
    write_reports(run_dir, report)
    for index, case in enumerate(cases, 1):
        if stop_reason is not None:
            break
        print(f"\n[{index}/{len(cases)}] {case.id}: {case.description}", flush=True)
        parents = {key: by_id[key] for key in case.depends_on if key in by_id}
        try:
            if budget is not None:
                budget.poll()
            fingerprint = case_fingerprint(
                case, root, capabilities, parents, budget=budget
            )
            cached = old.get(case.id, {})
            available = all(
                capabilities.get(name, {}).get("available")
                for name in case.capabilities
            )
            if (
                available
                and not _failed_dependencies(case, parents)
                and reusable_result(cached, fingerprint, run_dir, budget=budget)
            ):
                result = {**cached, "reused": True}
            else:
                try:
                    result = run_case(
                        case,
                        root=root,
                        run_dir=run_dir,
                        capabilities=capabilities,
                        dependency_results=parents,
                        profile=profile,
                        stop_event=stop,
                        budget=budget,
                    )
                except InvocationStopped:
                    raise
                except Exception as error:
                    result = {
                        "id": case.id,
                        "category": case.category,
                        "kind": case.kind,
                        "status": "ERROR",
                        "execution": "FAIL",
                        "qualification": "NOT_RUN",
                        "reason": f"Runner setup failed: {error}",
                        "metrics": {},
                        "artifacts": {},
                        "elapsed_s": 0.0,
                        "fingerprint": fingerprint,
                    }
        except InvocationStopped as stopped:
            stop_reason = stopped.reason
            break
        by_id[case.id] = result
        results.append(result)
        report["retained_results"] = [
            item for item in report["retained_results"] if item.get("id") != case.id
        ]
        report["pending_case_ids"] = [item.id for item in cases if item.id not in by_id]
        print(
            f"  {result['status']} ({result['elapsed_s']:.1f}s) {result.get('reason', '')}",
            flush=True,
        )
        report["summary"] = summarize(
            results,
            full_coverage=False,
        )
        report["summary"].update(status="RUNNING", selection_passed=False)
        if budget is not None:
            report["invocation_budget"] = budget.report_fields()
            stop_reason = budget.stop_reason
        write_reports(run_dir, report)
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    report["interrupted"] = stop.is_set()
    matrix_key = "feature_matrix" if profile == "features" else "scientific_matrix"
    matrix = coverage.get(matrix_key, {})
    matrix_complete = profile == "fast" or matrix.get("complete", False)
    if (
        full_selection
        and profile in {"features", "full"}
        and not matrix_complete
        and stop_reason is None
    ):
        results.append(
            {
                "id": "coverage.integrity",
                "status": "FAIL",
                "execution": "PASS",
                "qualification": "NOT_APPLICABLE",
                "elapsed_s": 0.0,
                "reason": "Declared scenario matrix has missing coverage; see coverage report.",
            }
        )
    report["summary"] = summarize(
        results,
        full_coverage=profile == "full" and full_selection and matrix_complete,
        feature_coverage=profile == "features" and full_selection and matrix_complete,
    )
    if budget is not None:
        report["invocation_budget"] = budget.report_fields()
        stop_reason = stop_reason or budget.stop_reason
    if stop_reason is not None or stop.is_set():
        mark_stopped(report, reason=stop_reason or "user")
    write_reports(run_dir, report)
    return report


def select_cases(cases: list[CaseSpec], patterns: list[str]) -> list[CaseSpec]:
    if not patterns:
        return cases
    selected = {
        case.id
        for case in cases
        if any(fnmatch.fnmatchcase(case.id, pattern) for pattern in patterns)
    }
    if not selected:
        raise ValueError("No cases match --case; use --list to inspect case IDs.")
    by_id = {case.id: case for case in cases}

    def add_parents(identifier):
        for parent in by_id[identifier].depends_on:
            if parent not in selected:
                selected.add(parent)
                add_parents(parent)

    for identifier in tuple(selected):
        add_parents(identifier)
    return [case for case in cases if case.id in selected]
