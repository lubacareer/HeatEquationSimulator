"""Isolated jobs. JSON lines carry progress; scientific arrays stay in result files."""
from __future__ import annotations

import argparse
from dataclasses import dataclass, fields, replace
import json
from pathlib import Path
import sys
import threading
import traceback
import uuid
from typing import Any

import numpy as np

from .models import PINNResult, Progress, RunResult, ScenarioConfig, SimulationCancelled
from .store import RunStore, json_default
from .publication import JobPublication


def emit(kind: str, **values) -> None:
    print(json.dumps({"event": kind, **values}, default=json_default), flush=True)


@dataclass(frozen=True)
class _PreparedSource:
    """Accepted source and provenance chosen before an analysis starts."""

    result: RunResult
    path: Path
    model: PINNResult | None = None


def _prepare_source(request: dict[str, Any], config: ScenarioConfig, source: RunResult | None,
                    store: RunStore, progress: Progress) -> _PreparedSource:
    if source is None:
        raise ValueError("Run a simulation before starting an analysis.")
    if source.config.physics_hash != config.physics_hash:
        raise ValueError("Physics inputs changed. Run/Recalculate before analyzing this scenario.")
    selected_model = None
    if request["action"] in {"trace", "dose"} and request.get("pinn_path"):
        from .pinn import validate_pinn_source
        selected_model = store.load_pinn(request["pinn_path"])
        validate_pinn_source(selected_model, source)
    if source.config.config_hash != config.config_hash:
        progress(.01, "Preparing an observation configuration from accepted fields")
        origin = source
        source = replace(source, run_id=uuid.uuid4().hex, config=config,
                         metadata={**source.metadata, "reused_fields_from": source.run_id})
        source_path = store.save_run(source)
        if selected_model is not None:
            # Prior-model reuse is explicit and allowed only for unchanged physics.
            selected_model = replace(selected_model, source_run_id=source.run_id, config_hash=config.config_hash,
                                     metadata={**selected_model.metadata, "explicit_prior_model_reuse": True,
                                               "origin_run_id": origin.run_id, "origin_config_hash": origin.config.config_hash,
                                               "training_source_run_id": selected_model.metadata.get("training_source_run_id", origin.run_id),
                                               "training_config_hash": selected_model.metadata.get("training_config_hash", origin.config.config_hash),
                                               "training_epochs": selected_model.metrics.get("epochs", origin.config.pinn_epochs),
                                               "source_run_id": source.run_id, "source_config_hash": config.config_hash})
            store.save_pinn(selected_model)
    else:
        source_path = Path(request["source_path"])
    if selected_model is not None:
        from .radiation import field_source_run
        if selected_model.metadata.get("validation_status") != "passed":
            raise ValueError("The selected PINN did not pass its held-out validation. Train it successfully before radiation analysis.")
        output = store.directory(source.run_id)
        source = field_source_run(source, selected_model, output_dir=output / f"field-{selected_model.run_id}")
    return _PreparedSource(source, source_path, selected_model)


def _simulate(config: ScenarioConfig, source: RunResult | None, store: RunStore,
              progress: Progress, cancel: threading.Event) -> Path:
    if source is not None and source.config.physics_hash == config.physics_hash:
        progress(0.5, "Reusing accepted fields; only observation or training settings changed")
        result = replace(source, run_id=uuid.uuid4().hex, config=config,
                         metadata={**source.metadata, "reused_fields_from": source.run_id})
    else:
        from .physics import run_simulation
        result = run_simulation(config, progress, cancel)
    if cancel.is_set():
        raise SimulationCancelled("Job cancelled")
    return store.save_run(result)


def _save_trajectories(result: Any, output: Path) -> Path:
    path = output / f"trajectories-{uuid.uuid4().hex}.npz"
    values = {f.name: getattr(result, f.name) for f in fields(result)}
    arrays: dict[str, Any] = {k: v for k, v in values.items() if isinstance(v, np.ndarray)}
    metadata = {k: v for k, v in values.items() if not isinstance(v, np.ndarray)}
    arrays["_metadata_json"] = np.array(json.dumps(metadata, default=json_default))
    np.savez_compressed(path, **arrays)
    return path


def _analyze(action: str, source: RunResult, request: dict[str, Any], store: RunStore,
             progress: Progress, cancel: threading.Event) -> Path:
    output = store.directory(source.run_id)
    if action == "pinn":
        from .pinn import train_pinn
        result = train_pinn(source, progress, cancel, output_dir=output)
        if cancel.is_set():
            raise SimulationCancelled("Job cancelled")
        return store.save_pinn(result)
    if action == "dose":
        from .radiation import calculate_dose
        result = calculate_dose(source, int(request["snapshot_index"]), progress, cancel, output_dir=output)
        if cancel.is_set():
            raise SimulationCancelled("Job cancelled")
        return store.save_dose(result)
    if action == "trace":
        from .radiation import trace_particles
        result = trace_particles(source, int(request["snapshot_index"]), progress, cancel)
        return _save_trajectories(result, output)
    raise ValueError(f"Unknown action {action}")


def execute(request_path: Path) -> int:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    cancel = threading.Event()
    finished = threading.Event()

    def watch_cancel():
        while not finished.is_set():
            if request_path.with_suffix(".cancel").exists():
                cancel.set()
            finished.wait(0.1)

    threading.Thread(target=watch_cancel, daemon=True).start()
    store = RunStore(request["store"])
    config = ScenarioConfig.from_dict(request["config"])
    action = request["action"]

    def progress(fraction: float, message: str):
        if request_path.with_suffix(".cancel").exists():
            cancel.set()
        if cancel.is_set():
            raise SimulationCancelled("Job cancelled; previous results are preserved.")
        emit("progress", fraction=max(0.0, min(float(fraction), 1.0)), message=message)

    publication = JobPublication(store, request_path, cancel.is_set)
    store = publication.store

    try:
        source = store.load_run(request["source_path"]) if request.get("source_path") else None
        if action == "simulate":
            path = _simulate(config, source, store, progress, cancel)
            source_path = path
            selected_source = "Numerical"
        else:
            prepared = _prepare_source(request, config, source, store, progress)
            path = _analyze(action, prepared.result, request, store, progress, cancel)
            source_path = prepared.path
            selected_source = "PINN" if prepared.model is not None else "Numerical"
        if cancel.is_set():
            raise SimulationCancelled("Job cancelled")
        publication.commit()
        event = {"action": action, "path": str(store.published_path(path)),
                 "source_path": str(store.published_path(source_path))}
        if action != "simulate":
            event["selected_source"] = selected_source
        emit("complete", **event)
        return 0
    except SimulationCancelled as error:
        emit("cancelled", message=str(error))
        return 2
    except Exception as error:
        emit("error", message=str(error), detail=traceback.format_exc())
        return 1
    finally:
        finished.set()
        publication.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    sys.exit(execute(args.request))
