"""Isolated FE process protocol and accepted numerical artifact capture."""
from __future__ import annotations

import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import uuid
import numpy as np
from ..models import Progress, RunResult, ScenarioConfig, SimulationCancelled
from ..artifacts import FieldArtifact
from ..backend_environment import wsl_distribution, wsl_path


class PhysicsBackendUnavailable(RuntimeError):
    """The requested physics could not run; callers must not display substitute data."""


def _run_fenics(config: ScenarioConfig, progress: Progress, cancel: threading.Event) -> RunResult:
    # Artifacts remain on disk for frozen-field transport and matching FE PINNs.
    directory = Path(os.environ.get("MAGNETIC_TORUS_RUNS", Path.home() / ".magnetic_torus" / "worker_artifacts")) / uuid.uuid4().hex
    directory.mkdir(parents=True, exist_ok=True)
    request = directory / "request.json"
    request.write_text(json.dumps(config.to_dict()), encoding="utf-8")
    worker = Path(__file__).resolve().parent.parent / "workers" / "fenics_worker.py"
    interpreter = os.environ.get("MAGNETIC_TORUS_FE_PYTHON", "/usr/bin/python3")
    if sys.platform == "win32":
        command = ["wsl", "-d", wsl_distribution(), "--", interpreter, wsl_path(worker), wsl_path(request), wsl_path(directory)]
    else:
        command = [interpreter, str(worker), str(request), str(directory)]
    try:
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                encoding="utf-8", errors="replace", creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
    except OSError as exc:
        raise PhysicsBackendUnavailable("Cannot start the FEniCS worker. Install Ubuntu WSL2 and run scripts/setup_wsl_physics.sh. " + str(exc)) from exc
    lines: queue.Queue[str | None] = queue.Queue()
    def read_output() -> None:
        try:
            assert proc.stdout is not None
            with proc.stdout:
                for line in proc.stdout:
                    lines.put(line.rstrip())
        finally:
            lines.put(None)
    threading.Thread(target=read_output, daemon=True).start()
    log: list[str] = []
    try:
        # Process termination can precede the reader queuing its final traceback.
        # Only EOF establishes that every output line has been consumed.
        while True:
            if cancel.is_set():
                (directory / "cancel").touch()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.terminate()
                raise SimulationCancelled("FEniCS simulation cancelled; previous results are preserved.")
            try:
                line = lines.get(timeout=.1)
            except queue.Empty:
                continue
            if line is None:
                break
            log.append(line)
            if line.startswith("TORUS_PROGRESS "):
                event = json.loads(line[len("TORUS_PROGRESS "):])
                progress(float(event["fraction"]), event["message"])
        code = proc.wait()
    finally:
        if proc.poll() is None:
            (directory / "cancel").touch()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.terminate()
    (directory / "worker.log").write_text("\n".join(log), encoding="utf-8")
    if code != 0:
        detail = "\n".join(log[-12:])
        raise PhysicsBackendUnavailable("Full MHD worker failed. Install/check scripts/setup_wsl_physics.sh; see "
                                        + str(directory / "worker.log") + "\n" + detail)
    arrays = np.load(directory / "result.npz", allow_pickle=False)
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    for key in ("exterior_field_path", "weak_operator_path"):
        if key in metadata:
            metadata[key] = str(directory / Path(metadata[key]).name)
    artifacts = {key: FieldArtifact.capture(directory / filename, root=directory)
                 for key, filename in (("exterior_field_path", "exterior_fields.npz"),
                                       ("weak_operator_path", "weak_operators.npz"))}
    return RunResult(directory.name, config, arrays["times"], arrays["points"],
                     {key[6:]: arrays[key] for key in arrays.files if key.startswith("field_")},
                     {key[5:]: arrays[key] for key in arrays.files if key.startswith("diag_")}, metadata, artifacts=artifacts)
