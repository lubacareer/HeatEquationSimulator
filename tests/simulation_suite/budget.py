"""A shared invocation deadline and an owned-process hard backstop."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import threading
import time
from uuid import uuid4


class InvocationStopped(RuntimeError):
    def __init__(self, reason: str):
        self.reason = reason
        self.exit_code = 124 if reason == "budget" else 130
        super().__init__(
            "Invocation time budget exhausted."
            if reason == "budget"
            else "User interrupted the invocation."
        )


class BudgetExpired(InvocationStopped):
    def __init__(self):
        super().__init__("budget")


class InvocationInterrupted(InvocationStopped):
    def __init__(self):
        super().__init__("user")


class InvocationBudget:
    """Monotonic work deadline; cleanup may use the reserved hard-deadline time."""

    def __init__(
        self,
        started_at: float,
        total_seconds: float = 3600.0,
        cleanup_reserve_seconds: float = 300.0,
        *,
        stop_event=None,
        cancel_path: Path | None = None,
        clock=None,
    ):
        values = (started_at, total_seconds, cleanup_reserve_seconds)
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in values
        ):
            raise ValueError("Invocation budget values must be finite numbers.")
        if total_seconds <= 0 or not 0 <= cleanup_reserve_seconds < total_seconds:
            raise ValueError(
                "Cleanup reserve must be nonnegative and smaller than the positive total budget."
            )
        self.started_at = float(started_at)
        self.total_seconds = float(total_seconds)
        self.cleanup_reserve_seconds = float(cleanup_reserve_seconds)
        self.hard_deadline = self.started_at + self.total_seconds
        self.work_deadline = self.hard_deadline - self.cleanup_reserve_seconds
        self.stop_event = stop_event if stop_event is not None else threading.Event()
        self.cancel_path = Path(cancel_path) if cancel_path is not None else None
        self.clock = clock or time.monotonic
        self.stop_reason = None
        self._lock = threading.Lock()

    @classmethod
    def from_env(cls, *, stop_event=None):
        names = (
            "TORUS_SUITE_STARTED_MONOTONIC",
            "TORUS_SUITE_WORK_DEADLINE",
            "TORUS_SUITE_HARD_DEADLINE",
            "TORUS_SUITE_CANCEL_FILE",
        )
        if not any(name in os.environ for name in names):
            return None
        if not all(name in os.environ for name in names):
            raise ValueError("Incomplete invocation deadline environment.")
        started, work, hard = (float(os.environ[name]) for name in names[:3])
        return cls(
            started,
            hard - started,
            hard - work,
            stop_event=stop_event,
            cancel_path=Path(os.environ[names[3]]),
        )

    def request_stop(self, reason: str):
        if reason not in {"budget", "user"}:
            raise ValueError("Unknown invocation stop reason.")
        with self._lock:
            if self.stop_reason is None:
                self.stop_reason = reason
            self.stop_event.set()

    def poll(self):
        if self.stop_reason is None and self.stop_event.is_set():
            self.request_stop("user")
        if self.stop_reason is None and self.cancel_path is not None:
            try:
                reason = self.cancel_path.read_text(encoding="ascii").strip()
            except OSError:
                reason = ""
            if reason in {"budget", "user"}:
                self.request_stop(reason)
        if self.stop_reason is None and self.clock() >= self.work_deadline:
            self.request_stop("budget")
        if self.stop_reason == "budget":
            raise BudgetExpired()
        if self.stop_reason == "user":
            raise InvocationInterrupted()

    def remaining_seconds(self, *, hard=False):
        return max(
            0.0, (self.hard_deadline if hard else self.work_deadline) - self.clock()
        )

    def report_fields(self):
        return {
            "total_seconds": self.total_seconds,
            "cleanup_reserve_seconds": self.cleanup_reserve_seconds,
            "elapsed_seconds": max(0.0, self.clock() - self.started_at),
            "stop_reason": self.stop_reason,
        }

    @property
    def preflight_token(self):
        return (
            self.cancel_path.with_suffix("") if self.cancel_path is not None else None
        )

    def register_owned_path(self, path: Path):
        if self.cancel_path is None:
            return
        root = self.cancel_path.parent.resolve() / "cases"
        resolved = path.resolve()
        resolved.relative_to(root)
        registry = self.cancel_path.with_suffix(".owned")
        temporary = registry.with_suffix(".owned.partial")
        temporary.write_text(json.dumps({"case_dir": str(resolved)}), encoding="utf-8")
        temporary.replace(registry)

    def register_preflight(self):
        if self.preflight_token is not None:
            self.preflight_token.mkdir(parents=True, exist_ok=True)


def _owned_wsl_paths(run_dir: Path, budget: InvocationBudget) -> list[Path]:
    paths = (
        [budget.preflight_token]
        if budget.preflight_token is not None and budget.preflight_token.is_dir()
        else []
    )
    if budget.cancel_path is None:
        return paths
    try:
        registry = json.loads(
            budget.cancel_path.with_suffix(".owned").read_text(encoding="utf-8")
        )
        path = Path(registry["case_dir"]).resolve()
        relative = path.relative_to(run_dir.resolve() / "cases")
        if len(relative.parts) == 2 and relative.parts[1].startswith("attempt-"):
            paths.append(path)
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return paths


def _stop_wsl_paths(run_dir, budget, notes):
    from .processes import stop_owned_wsl

    for path in _owned_wsl_paths(run_dir, budget):
        remaining = budget.remaining_seconds(hard=True)
        if remaining <= 0.5:
            break
        note = stop_owned_wsl(path, timeout=min(5.0, remaining - 0.25))
        if note:
            notes.append(note)


def _supervisor_report(run_dir: Path, budget: InvocationBudget, notes: list[str]):
    """Best-effort finalization preserves all child-published records and hashes."""
    from .reporting import mark_stopped, write_reports

    path = run_dir / "report.json"
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        report = {
            "schema_version": 1,
            "run_dir": str(run_dir),
            "profile": "features",
            "results": [],
            "retained_results": [],
            "manifest": [],
            "coverage": {},
        }
    report["invocation_budget"] = budget.report_fields()
    report["supervisor_cleanup_notes"] = notes
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    mark_stopped(report, reason=budget.stop_reason or "budget")
    write_reports(run_dir, report)


def supervise(
    command: list[str],
    *,
    run_dir: Path,
    started_at: float,
    total_seconds: float = 3600.0,
    cleanup_reserve_seconds: float = 300.0,
    stop_event=None,
    cwd=None,
    environment=None,
) -> int:
    """Run the CLI worker with inherited live output and kill only its own tree.

    The worker gets the same monotonic deadlines. It cooperatively publishes a
    partial report at the work cutoff; this supervisor also bounds a wedged
    preflight, hashing operation, or worker cleanup independently.
    """
    from .processes import OwnedProcessGroup

    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    marker = run_dir / (".invocation-" + uuid4().hex + ".cancel")
    budget = InvocationBudget(
        started_at,
        total_seconds,
        cleanup_reserve_seconds,
        stop_event=stop_event,
        cancel_path=marker,
    )
    env = {
        **os.environ,
        **(environment or {}),
        "TORUS_SUITE_STARTED_MONOTONIC": repr(budget.started_at),
        "TORUS_SUITE_WORK_DEADLINE": repr(budget.work_deadline),
        "TORUS_SUITE_HARD_DEADLINE": repr(budget.hard_deadline),
        "TORUS_SUITE_CANCEL_FILE": str(marker),
    }
    owner = process = None
    notes = []
    force_at = budget.hard_deadline - min(15.0, cleanup_reserve_seconds / 2)
    try:
        try:
            budget.poll()
        except InvocationStopped:
            _supervisor_report(run_dir, budget, notes)
            return 124 if budget.stop_reason == "budget" else 130
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
            start_new_session=os.name != "nt",
        )
        owner = OwnedProcessGroup(process)
        while process.poll() is None:
            try:
                budget.poll()
            except InvocationStopped:
                if not marker.exists():
                    try:
                        marker.write_text(budget.stop_reason + "\n", encoding="ascii")
                    except OSError as error:
                        notes.append(
                            f"Cannot write cooperative cancellation marker: {error}"
                        )
                if budget.stop_reason == "user":
                    force_at = min(force_at, time.monotonic() + 30.0)
            if budget.stop_reason is not None and time.monotonic() >= force_at:
                _stop_wsl_paths(run_dir, budget, notes)
                owner.close()
                if process.poll() is None:
                    process.kill()
                try:
                    process.wait(
                        timeout=max(
                            0.001, min(2.0, budget.remaining_seconds(hard=True))
                        )
                    )
                except subprocess.TimeoutExpired:
                    notes.append(
                        "Owned CLI worker did not acknowledge forced termination."
                    )
                break
            time.sleep(min(0.1, max(0.001, budget.remaining_seconds(hard=True))))
        _stop_wsl_paths(run_dir, budget, notes)
        if budget.stop_reason is not None:
            # Emergency report I/O cannot hold the supervisor past its deadline.
            writer = threading.Thread(
                target=_supervisor_report, args=(run_dir, budget, notes), daemon=True
            )
            writer.start()
            writer.join(timeout=min(2.0, budget.remaining_seconds(hard=True)))
            return 124 if budget.stop_reason == "budget" else 130
        return int(process.returncode)
    except KeyboardInterrupt:
        budget.request_stop("user")
        if process is not None:
            _stop_wsl_paths(run_dir, budget, notes)
            if owner is not None:
                owner.close()
            if process.poll() is None:
                process.kill()
        _supervisor_report(run_dir, budget, notes)
        return 130
    except Exception:
        if process is not None and process.poll() is None:
            _stop_wsl_paths(run_dir, budget, notes)
            if owner is not None:
                owner.close()
            if process.poll() is None:
                process.kill()
        raise
    finally:
        if owner is not None:
            owner.close()
