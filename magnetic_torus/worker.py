"""Isolated jobs. JSON lines carry progress; scientific arrays stay in result files."""
from __future__ import annotations

import argparse
from dataclasses import asdict, fields, is_dataclass, replace
import json
from pathlib import Path
import sys
import threading
import time
import traceback
import uuid
from typing import Any

import numpy as np

from .models import ScenarioConfig, SimulationCancelled
from .store import RunStore, json_default


def emit(kind: str, **values) -> None:
    print(json.dumps({"event": kind, **values}, default=json_default), flush=True)


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

    try:
        source = store.load_run(request["source_path"]) if request.get("source_path") else None
        if action == "simulate":
            if source is not None and source.config.physics_hash == config.physics_hash:
                progress(0.5, "Reusing accepted fields; only observation or training settings changed")
                result = replace(source, run_id=uuid.uuid4().hex, config=config,
                                 metadata={**source.metadata, "reused_fields_from": source.run_id})
            else:
                from .physics import run_simulation
                result = run_simulation(config, progress, cancel)
            if cancel.is_set():
                raise SimulationCancelled("Job cancelled")
            path = store.save_run(result)
            emit("complete", action=action, path=str(path), source_path=str(path))
            return 0
        if source is None:
            raise ValueError("Run a simulation before starting an analysis.")
        if source.config.physics_hash != config.physics_hash:
            raise ValueError("Physics inputs changed. Run/Recalculate before analyzing this scenario.")
        selected_model = None
        if action in {"trace", "dose"} and request.get("pinn_path"):
            from .pinn import validate_pinn_source
            selected_model = store.load_pinn(request["pinn_path"])
            validate_pinn_source(selected_model, source)
        if source.config.config_hash != config.config_hash:
            origin = source
            source = replace(source, run_id=uuid.uuid4().hex, config=config,
                             metadata={**source.metadata, "reused_fields_from": source.run_id})
            source_path = store.save_run(source)
            if selected_model is not None:
                # Explicit selection of a prior model permits reuse when only
                # observation settings change; physical fields are identical.
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
        output = store.directory(source.run_id)
        if selected_model is not None:
            from .radiation import field_source_run
            if selected_model.metadata.get("validation_status") != "passed":
                raise ValueError("The selected PINN did not pass its held-out validation. Train it successfully before radiation analysis.")
            source = field_source_run(source, selected_model, output_dir=output / f"field-{selected_model.run_id}")
        if action == "pinn":
            from .pinn import train_pinn
            result = train_pinn(source, progress, cancel, output_dir=output)
            if cancel.is_set():
                raise SimulationCancelled("Job cancelled")
            path = store.save_pinn(result)
        elif action == "dose":
            from .radiation import calculate_dose
            result = calculate_dose(source, int(request["snapshot_index"]), progress, cancel, output_dir=output)
            if cancel.is_set():
                raise SimulationCancelled("Job cancelled")
            path = store.save_dose(result)
        elif action == "trace":
            from .radiation import trace_particles
            result = trace_particles(source, int(request["snapshot_index"]), progress, cancel)
            path = output / f"trajectories-{uuid.uuid4().hex}.npz"
            values = {f.name: getattr(result, f.name) for f in fields(result)}
            arrays: dict[str, Any] = {k: v for k, v in values.items() if isinstance(v, np.ndarray)}
            metadata = {k: v for k, v in values.items() if not isinstance(v, np.ndarray)}
            arrays["_metadata_json"] = np.array(json.dumps(metadata, default=json_default))
            np.savez_compressed(path, **arrays)
        else:
            raise ValueError(f"Unknown action {action}")
        if cancel.is_set():
            raise SimulationCancelled("Job cancelled")
        emit("complete", action=action, path=str(path), source_path=str(source_path),
             selected_source="PINN" if selected_model is not None else "Numerical")
        return 0
    except SimulationCancelled as error:
        emit("cancelled", message=str(error))
        return 2
    except Exception as error:
        emit("error", message=str(error), detail=traceback.format_exc())
        return 1
    finally:
        finished.set()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    sys.exit(execute(args.request))
