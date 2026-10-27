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
import time

ROOT = Path(__file__).resolve().parents[1]


def file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stage_source(source: Path, directory: Path, expected_sha256: str) -> Path:
    """Copy verified source bytes into this experiment before artifact extraction."""
    private = directory / "input"
    private.mkdir()
    partial = private / "source.h5.partial"
    staged = private / "source.h5"
    digest = sha256()
    try:
        with source.open("rb") as original, partial.open("xb") as destination:
            for chunk in iter(lambda: original.read(1024 * 1024), b""):
                digest.update(chunk)
                destination.write(chunk)
        if digest.hexdigest() != expected_sha256:
            raise ValueError("The source changed while creating the private experiment copy.")
        os.replace(partial, staged)
    finally:
        partial.unlink(missing_ok=True)
    return staged


def main(argv=None) -> int:
    started = time.monotonic()
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    sys.path.insert(0, str(ROOT))
    from magnetic_torus.models import SimulationCancelled
    from magnetic_torus.store import RunStore, json_default
    from magnetic_torus.learning.research import ResearchOptions, ResearchPilotTimeout, run_research

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path, help="Explicit warm start, bound to --source")
    parser.add_argument("--strategy", choices=("uniform", "front", "causal", "front-causal", "temporal-pressure"), default="uniform")
    parser.add_argument("--architecture", choices=("mlp", "film", "temporal-phi", "method5", "pressure-fourier"), default="mlp")
    parser.add_argument("--adam-steps", type=int, default=800)
    parser.add_argument("--lbfgs-steps", type=int, default=20)
    parser.add_argument("--seconds", type=float, default=180.)
    parser.add_argument("--balance-labels", action="store_true")
    parser.add_argument("--causal-epsilon", type=float, default=1.)
    parser.add_argument("--causal-floor", type=float, default=.05)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--_supervised-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(raw_argv)
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
    if args.architecture in {"method5", "pressure-fourier"}:
        code_paths.append(ROOT / "scripts" / "research_supervisor.py")
        code_paths.append(ROOT / "tests" / "simulation_suite" / "processes.py")
    supervised_parent = args.architecture in {"method5", "pressure-fourier"} and not args._supervised_worker
    implementation = {} if supervised_parent else {
        str(path.relative_to(ROOT)): file_hash(path) for path in code_paths}
    stop = threading.Event()
    previous_handler = signal.signal(signal.SIGINT, lambda *_: stop.set())
    report = {"started_at": datetime.now(timezone.utc).isoformat(), "options": asdict(options),
        "original_sha256": originals, "implementation_sha256": implementation,
        "selected_inputs": [str(path) for path in selected],
        "canonical": False, "desktop_modes_enabled": False}
    if supervised_parent:
        from scripts.research_supervisor import supervise
        try:
            finished_hashing = threading.Event()
            hash_state = {}
            def hash_preflight():
                try:
                    hash_state["implementation"] = {
                        str(path.relative_to(ROOT)): file_hash(path) for path in code_paths}
                    values = {}
                    for path in selected:
                        try:
                            values[str(path)] = file_hash(path)
                        except OSError:
                            pass  # The owned worker records the unreadable input.
                    hash_state["originals"] = values
                except OSError as error:
                    hash_state["error"] = f"{type(error).__name__}: {error}"
                finally:
                    finished_hashing.set()
            threading.Thread(target=hash_preflight, daemon=True).start()
            work_deadline = started + args.seconds - min(5., max(.1, args.seconds * .2))
            while not finished_hashing.wait(.05):
                if stop.is_set() or time.monotonic() >= work_deadline:
                    break
            if finished_hashing.is_set() and "error" not in hash_state:
                implementation.update(hash_state["implementation"])
                originals.update(hash_state["originals"])
                report["supervisor_preflight_complete"] = True
            else:
                report["supervisor_preflight_complete"] = False
                if "error" in hash_state:
                    report.update(status="ERROR", completed=False, error=hash_state["error"])
                    temporary = directory / "report.json.partial"
                    temporary.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
                    os.replace(temporary, directory / "report.json")
                    print(json.dumps({"status": "ERROR", "report": str(directory / "report.json")}), flush=True)
                    return 1
            command = [sys.executable, "-B", str(Path(__file__).resolve()),
                       *raw_argv, "--_supervised-worker"] if finished_hashing.is_set() else []
            code, outcome = supervise(command, directory, started=started, seconds=args.seconds,
                                      provenance=report, stop=stop, cwd=ROOT)
        finally:
            signal.signal(signal.SIGINT, previous_handler)
        print(json.dumps({"status": outcome["status"], "report": str(directory / "report.json"),
                          "originals_unchanged": outcome.get("originals_unchanged")}), flush=True)
        return code

    watch_finished = threading.Event()
    if args._supervised_worker:
        def watch_supervisor():
            while not watch_finished.wait(.05):
                if (directory / ".supervisor-cancel").exists():
                    stop.set()
                    return
        threading.Thread(target=watch_supervisor, daemon=True).start()
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
        source_input = stage_source(selected[0], directory, originals[str(selected[0])])
        report.update(original_source_path=str(selected[0]), private_source_path=str(source_input),
                      private_source_sha256=originals[str(selected[0])])
        source = store.load_run(source_input)
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
    except ResearchPilotTimeout as error:
        report.update(status="TIMEOUT", error=str(error), actual_steps=error.steps,
                      elapsed_s=error.elapsed_s)
        code = 124
    except SimulationCancelled as error:
        report.update(status="INTERRUPTED", error=str(error))
        code = 130
    except Exception as error:
        report.update(status="ERROR", error=f"{type(error).__name__}: {error}")
    finally:
        watch_finished.set()
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
        report_path = directory / ("worker-report.json" if args._supervised_worker else "report.json")
        temporary = report_path.with_suffix(".json.partial")
        temporary.write_text(json.dumps(report, default=json_default, indent=2, allow_nan=False), encoding="utf-8")
        os.replace(temporary, report_path)
    print(json.dumps({"status": report["status"], "report": str(report_path),
                      "originals_unchanged": report["originals_unchanged"]}), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
