"""Reproducible PINN convergence/checkpoint check against real solver artifacts.

Examples::

    python -m magnetic_torus.pinn_validation --native --device cpu --epochs 250
    python -m magnetic_torus.pinn_validation --fe-artifacts .cache/fe_coupled --device cuda

SciPy and PyTorch run in separate processes, just as they do in the desktop.
Exit code 3 means finite training completed but failed the recorded agreement
criteria; it does not mean the source numerical solution failed.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from uuid import uuid4

import numpy as np

from .models import RunResult, ScenarioConfig, preset
from .store import RunStore, json_default
from .artifacts import FieldArtifact


def load_fe_artifacts(directory: Path) -> RunResult:
    directory = directory.resolve()
    config = ScenarioConfig.from_dict(json.loads((directory / "request.json").read_text(encoding="utf-8")))
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    metadata["weak_operator_path"] = str(directory / "weak_operators.npz")
    if metadata.get("exterior_field_path"):
        metadata["exterior_field_path"] = str(directory / Path(metadata["exterior_field_path"]).name)
    with np.load(directory / "result.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    return RunResult("fe-validation-" + uuid4().hex[:12], config, arrays["times"], arrays["points"],
                     {key[6:]: value for key, value in arrays.items() if key.startswith("field_")},
                     {key[5:]: value for key, value in arrays.items() if key.startswith("diag_")},
                     {**metadata, "validation_artifact_source": str(directory)},
                     artifacts={key: FieldArtifact.capture(metadata[key], root=directory)
                                for key in RunStore.FIELD_ARTIFACTS if metadata.get(key)})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    origin = parser.add_mutually_exclusive_group(required=True)
    origin.add_argument("--native", action="store_true", help="Solve a radiating, surface-heated native benchmark first")
    origin.add_argument("--fe-artifacts", type=Path, help="FEniCS directory containing request.json, result.npz and weak_operators.npz")
    origin.add_argument("--source", type=Path, help="Existing RunStore HDF5 numerical result")
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--time-steps", type=int, default=40, help="Native solver accepted interval target")
    parser.add_argument("--output", type=Path, default=Path(".cache/pinn_validation"))
    args = parser.parse_args()
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    store = RunStore(directory / "store")
    if args.native:
        config = replace(preset("Thermal benchmark"), resolution=3, duration=20.,
                         time_steps=args.time_steps, emissivity=.4, heating_power=100000.,
                         pinn_epochs=args.epochs, pinn_batch_size=args.batch_size).validate()
        request = directory / "native-request.json"
        request.write_text(json.dumps({"action": "simulate", "store": str(store.root),
                                       "config": config.to_dict()}), encoding="utf-8")
        job = subprocess.run([sys.executable, "-B", "-u", "-m", "magnetic_torus.worker",
                              "--request", str(request)], capture_output=True, text=True,
                             encoding="utf-8", errors="replace")
        events = [json.loads(line) for line in job.stdout.splitlines() if line.startswith("{")]
        complete = next((event for event in reversed(events) if event.get("event") == "complete"), None)
        if job.returncode or complete is None:
            raise RuntimeError("Native reference worker failed:\n" + job.stdout[-6000:] + job.stderr[-2000:])
        source = store.load_run(complete["path"])
    elif args.fe_artifacts:
        source = load_fe_artifacts(args.fe_artifacts)
    else:
        source = store.load_run(args.source)
    source = replace(source, config=replace(source.config, pinn_epochs=args.epochs,
                                            pinn_batch_size=args.batch_size).validate())
    # This import must follow the isolated native solve in this process.
    os.environ["MAGNETIC_TORUS_PINN_DEVICE"] = args.device
    import torch
    from .pinn import train_pinn, validate_pinn_source
    if args.cpu_threads < 1:
        raise ValueError("CPU thread count must be positive.")
    torch.set_num_threads(args.cpu_threads)
    previous = [-1.]
    def progress(fraction, message):
        if fraction - previous[0] >= .05 or fraction == 1.:
            print(json.dumps({"progress": fraction, "message": message}), flush=True)
            previous[0] = fraction
    start = time.perf_counter()
    trained = train_pinn(source, progress, threading.Event(), output_dir=directory)
    validate_pinn_source(trained, source)
    source_path = store.save_run(source)
    pinn_path = store.save_pinn(trained)
    loaded = store.load_pinn(pinn_path)
    validate_pinn_source(loaded, source)
    report = {"source_path": str(source_path), "pinn_path": str(pinn_path),
              "source_run_id": source.run_id, "config_hash": source.config.config_hash,
              "elapsed_s": time.perf_counter() - start, "metrics": dict(trained.metrics),
              "metadata": dict(trained.metadata)}
    (directory / "report.json").write_text(json.dumps(report, default=json_default, indent=2), encoding="utf-8")
    print(json.dumps(report, default=json_default), flush=True)
    return 0 if trained.metadata["validation_status"] == "passed" else 3


if __name__ == "__main__":
    raise SystemExit(main())
