"""Qt process lifecycle; workers never own widgets or overwrite active results."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import uuid

import PyQt6.QtCore as QtCore

from .models import ScenarioConfig
from .store import RunStore
from .publication import request_cancel


class JobController(QtCore.QObject):
    progress = QtCore.pyqtSignal(float, str)
    completed = QtCore.pyqtSignal(dict)
    failed = QtCore.pyqtSignal(str, str)
    cancelled = QtCore.pyqtSignal()
    busy_changed = QtCore.pyqtSignal(bool)
    log = QtCore.pyqtSignal(str)

    def __init__(self, store: RunStore, parent=None):
        super().__init__(parent)
        self.store = store
        self.process: QtCore.QProcess | None = None
        self.request_path: Path | None = None
        self._buffer = b""
        self._terminal = False
        self._requested_cancel = False

    @property
    def busy(self) -> bool:
        return self.process is not None

    def start(self, action: str, config: ScenarioConfig, source_path: Path | None = None, snapshot_index: int = 0,
              pinn_path: Path | None = None):
        if self.busy:
            raise RuntimeError("A job is already running. Cancel it or wait for completion.")
        config.validate()
        jobs = self.store.root / "jobs"
        jobs.mkdir(exist_ok=True)
        self.request_path = jobs / f"{uuid.uuid4().hex}.json"
        request = {"action": action, "config": config.to_dict(), "store": str(self.store.root),
                   "source_path": str(source_path) if source_path else None, "snapshot_index": snapshot_index,
                   "pinn_path": str(pinn_path) if pinn_path else None}
        self.request_path.write_text(json.dumps(request), encoding="utf-8")
        process = QtCore.QProcess(self)
        self.process = process
        self._buffer = b""
        self._terminal = False
        self._requested_cancel = False
        environment = QtCore.QProcessEnvironment.systemEnvironment()
        environment.insert("PYTHONUNBUFFERED", "1")
        environment.insert("PYTHONDONTWRITEBYTECODE", "1")
        process.setProcessEnvironment(environment)
        process.setWorkingDirectory(str(Path(__file__).resolve().parent.parent))
        process.readyReadStandardOutput.connect(self._read_stdout)
        process.readyReadStandardError.connect(self._read_stderr)
        process.finished.connect(self._finished)
        process.errorOccurred.connect(self._process_error)
        self.busy_changed.emit(True)
        process.start(sys.executable, ["-B", "-u", "-m", "magnetic_torus.worker", "--request", str(self.request_path)])

    def _read_stdout(self):
        if self.process is None:
            return
        self._buffer += bytes(self.process.readAllStandardOutput())
        while b"\n" in self._buffer:
            line, self._buffer = self._buffer.split(b"\n", 1)
            text = line.decode("utf-8", errors="replace").strip()
            if not text:
                continue
            try:
                event = json.loads(text)
            except ValueError:
                self.log.emit(text)
                continue
            if not isinstance(event, dict):
                self.log.emit(text)
                continue
            kind = event.get("event")
            if kind == "progress":
                self.progress.emit(float(event["fraction"]), event["message"])
            elif kind == "complete":
                self._terminal = True
                if not self._requested_cancel:
                    self.completed.emit(event)
            elif kind == "error":
                self._terminal = True
                if not self._requested_cancel:
                    self.failed.emit(event["message"], event.get("detail", ""))
            elif kind == "cancelled":
                self._terminal = True
            else:
                self.log.emit(text)

    def _read_stderr(self):
        if self.process is not None:
            self.log.emit(bytes(self.process.readAllStandardError()).decode("utf-8", errors="replace"))

    def _process_error(self, error):
        if error == QtCore.QProcess.ProcessError.FailedToStart and self.process is not None:
            self._terminal = True
            self.failed.emit("The background worker could not start.", self.process.errorString())
            self._finished(-1, QtCore.QProcess.ExitStatus.CrashExit)

    def _finished(self, exit_code, exit_status):
        self._read_stdout()
        was_cancelled = self._requested_cancel
        if not self._terminal and not was_cancelled:
            self.failed.emit(f"Worker exited before publishing a result (exit {exit_code}).", "See the diagnostic log.")
        process = self.process
        self.process = None
        if process is not None:
            process.deleteLater()
        self.busy_changed.emit(False)
        if was_cancelled:
            self.cancelled.emit()

    def cancel(self):
        if self.process is None or self.request_path is None:
            return
        if not request_cancel(self.request_path):
            self.progress.emit(1.0, "Results have been published; loading completed output…")
            return
        self._requested_cancel = True
        self.progress.emit(0.0, "Cancelling at the next safe solver boundary…")

    def shutdown(self):
        self.cancel()
        if self.process is not None:
            # WSL workers terminate their own Linux process groups before returning.
            self.process.waitForFinished(15000)
            if self.process is not None:
                self.process.terminate()
                self.process.waitForFinished(1000)
            if self.process is not None:
                self.process.kill()
                self.process.waitForFinished(1000)
