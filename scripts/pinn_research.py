"""Run one explicit bounded PINN experiment against an accepted source HDF5.

These strategies are research hypotheses. This command never enables disabled
desktop modes and always retains the original source qualification gates.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]


def file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None) -> int:
    sys.path.insert(0, str(ROOT))
    from magnetic_torus.models import SimulationCancelled
    from magnetic_torus.store import RunStore, json_default
    from magnetic_torus.learning.research import ResearchOptions, run_research

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path, help="Explicit warm start, bound to --source")
    parser.add_argument("--strategy", choices=("uniform", "front", "causal", "front-causal"), default="uniform")
    parser.add_argument("--architecture", choices=("mlp", "film", "temporal-phi"), default="mlp")
    parser.add_argument("--adam-steps", type=int, default=800)
    parser.add_argument("--lbfgs-steps", type=int, default=20)
    parser.add_argument("--seconds", type=float, default=180.)
    parser.add_argument("--balance-labels", action="store_true")
    parser.add_argument("--causal-epsilon", type=float, default=1.)
    parser.add_argument("--causal-floor", type=float, default=.05)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--cpu-threads", type=int, default=2)
    args = parser.parse_args(argv)
    options = ResearchOptions(args.strategy, args.architecture, args.adam_steps,
        args.lbfgs_steps, args.seconds, args.balance_labels, args.causal_epsilon, args.causal_floor)
    try:
        options.validate()
    except ValueError as error:
        parser.error(str(error))
    if args.cpu_threads < 1:
        parser.error("CPU thread count must be positive.")
    directory = args.output.resolve()
    if directory.exists() and any(directory.iterdir()):
        parser.error("Use an empty output directory to preserve existing evidence.")
    directory.mkdir(parents=True, exist_ok=True)
    selected = [args.source.resolve()] + ([args.checkpoint.resolve()] if args.checkpoint else [])
    originals = {}
    code_paths = [Path(__file__).resolve(), *sorted((ROOT / "magnetic_torus").rglob("*.py"))]
    implementation = {str(path.relative_to(ROOT)): file_hash(path) for path in code_paths}
    stop = threading.Event()
    previous_handler = signal.signal(signal.SIGINT, lambda *_: stop.set())
    report = {"started_at": datetime.now(timezone.utc).isoformat(), "options": asdict(options),
        "original_sha256": originals, "implementation_sha256": implementation,
        "canonical": False, "desktop_modes_enabled": False}
    code = 1
    try:
        for path in selected:
            originals[str(path)] = file_hash(path)
        try:
            revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                capture_output=True, text=True, check=True, timeout=5)
            report["git_revision"] = revision.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            report["git_revision"] = "unavailable"
        store = RunStore(directory / "store")
        source = store.load_run(args.source.resolve())
        report.update(originating_run_id=source.run_id, originating_config_hash=source.config.config_hash,
                      source_configuration=source.config.to_dict(), seed=source.config.seed)
        os.environ["MAGNETIC_TORUS_PINN_DEVICE"] = args.device
        import torch
        torch.set_num_threads(args.cpu_threads)
        def progress(fraction, message):
            print(json.dumps({"progress": fraction, "message": message}), flush=True)
        copied, model, evidence = run_research(source, options, progress, stop, directory / "training", args.checkpoint)
        source_path = store.save_run(copied)
        model_path = store.save_pinn(model)
        report.update(**evidence, metrics=dict(model.metrics), metadata=dict(model.metadata),
                      source_path=str(source_path), pinn_path=str(model_path))
        code = {"PASS": 0, "FAIL": 3, "TIMEOUT": 124, "INTERRUPTED": 130}[evidence["status"]]
        if stop.is_set() and code not in {124, 130}:
            report["status"], code = "INTERRUPTED", 130
    except SimulationCancelled as error:
        report.update(status="INTERRUPTED", error=str(error))
        code = 130
    except Exception as error:
        report.update(status="ERROR", error=f"{type(error).__name__}: {error}")
    finally:
        signal.signal(signal.SIGINT, previous_handler)
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        try:
            report["originals_unchanged"] = (all(file_hash(path) == originals[str(path)] for path in selected)
                if len(originals) == len(selected) else None)
            report["implementation_unchanged"] = all(file_hash(path) == implementation[str(path.relative_to(ROOT))]
                for path in code_paths)
        except OSError:
            report["originals_unchanged"], report["implementation_unchanged"] = False, False
        if report["originals_unchanged"] is False or not report["implementation_unchanged"]:
            report["status"], code = "ERROR", 1
            report["error"] = "An explicitly selected input or implementation changed during the experiment."
        (directory / "report.json").write_text(json.dumps(report, default=json_default, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"status": report["status"], "report": str(directory / "report.json"),
                      "originals_unchanged": report["originals_unchanged"]}), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
