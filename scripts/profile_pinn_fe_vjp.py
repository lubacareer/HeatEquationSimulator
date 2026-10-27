"""Bounded, isolated CUDA profiling of a saved coupled PINN checkpoint.

Profiles the original full graph, global FE cotangents with two neural batch
sizes, and CPU-saved FE tensors. Each strategy runs in a private child with a
deadline; accepted source and checkpoint bytes are never modified.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
STRATEGIES = ("full", "vjp1024", "vjp4096", "vjp1024-cpu-saved", "full-h10", "vjp1024-h10")


def digest(path: Path) -> str:
    result = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def peak_host_bytes() -> int | None:
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = (wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD)
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        counter = Counters()
        counter.cb = ctypes.sizeof(counter)
        if psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counter), counter.cb):
            return int(counter.PeakWorkingSetSize)
        return None
    import resource
    scale = 1 if sys.platform == "darwin" else 1024
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale)


def child(args: argparse.Namespace) -> int:
    sys.path.insert(0, str(ROOT))
    import numpy as np
    import torch
    from magnetic_torus.learning import training
    from magnetic_torus.learning.corrections import restore_model_from_checkpoint
    from magnetic_torus.store import RunStore

    result = {"strategy": args.child, "status": "ERROR", "repetitions": []}
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for this profiler.")
        os.environ["MAGNETIC_TORUS_PINN_DEVICE"] = "cuda"
        torch.set_num_threads(2)
        # RunStore extracts embedded artifacts beside its input. Load a private
        # copy so profiling cannot add files to the historical source directory.
        input_dir = args.output / (args.child + "-input")
        input_dir.mkdir()
        private_source = input_dir / "source.h5"
        shutil.copy2(args.source, private_source)
        source = RunStore.load_run(private_source)
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        data = training._prepare_data(source, threading.Event())
        run = training._prepare_training(data, lambda *_: None, threading.Event(), None)
        run.field.model = restore_model_from_checkpoint(payload, source, "cuda")
        parameters = tuple(p for p in run.field.model.parameters() if p.requires_grad)
        indices = np.unique(np.linspace(1, data.snapshot_count - 1,
                                       min(4, data.snapshot_count - 1), dtype=np.int64))
        centers = np.unique(np.linspace(0, data.point_count - 1,
                                       min(run.field.maximum_batch, data.point_count), dtype=np.int64))
        result.update(source_run_id=source.run_id, source_config_hash=source.config.config_hash,
                      checkpoint_epoch=int(payload["epoch"]), intervals=indices.tolist(),
                      source_sha256=digest(args.source), checkpoint_sha256=digest(args.checkpoint),
                      device=torch.cuda.get_device_name(), torch_version=str(torch.__version__))
        history_size = 10 if args.child.endswith("-h10") else 20
        result["optimizer_history_size"] = history_size
        result["checkpoint_history_entries"] = [len(state.get("old_dirs", ()))
                                                for state in payload["optimizer_state"]["state"].values()]
        for iteration in range(args.repetitions):
            run.field.model.load_state_dict(payload["model_state"])
            optimizer = torch.optim.LBFGS(parameters, lr=1., max_iter=1, max_eval=8,
                                          history_size=history_size, line_search_fn="strong_wolfe")
            state = deepcopy(payload["optimizer_state"])
            if history_size == 10:
                for group in state["param_groups"]:
                    group["history_size"] = history_size
                for item in state["state"].values():
                    for key in ("old_dirs", "old_stps", "ro"):
                        if key in item:
                            item[key] = item[key][-history_size:]
                    if "al" in item:
                        item["al"] = [None] * history_size
            optimizer.load_state_dict(state)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            free, total = torch.cuda.mem_get_info()
            recorded = {}
            def closure():
                optimizer.zero_grad(set_to_none=True)
                terms = training._deterministic_objective(
                    run, indices, centers, backward=True,
                    coupled_vjp=not args.child.startswith("full"),
                    offload_saved_tensors=args.child == "vjp1024-cpu-saved",
                    vjp_batch_size=4096 if args.child == "vjp4096" else 1024)
                if not recorded:
                    recorded.update({key: float(value.detach()) for key, value in terms.items()})
                    gradients = [torch.zeros_like(p) if p.grad is None else p.grad for p in parameters]
                    recorded["gradients"] = torch.cat([gradient.detach().reshape(-1) for gradient in gradients]).cpu().numpy()
                return terms["loss"]
            start = time.perf_counter()
            evaluations, limited, outcomes = 0, 0, {}
            for _ in range(args.steps_per_repetition):
                step = training._recovering_lbfgs_step(optimizer, parameters, closure, threading.Event())
                evaluations += step.evaluations
                limited += int(step.limited)
                outcomes[step.outcome] = outcomes.get(step.outcome, 0) + 1
                # Production refinement also audits the objective after every step.
                training._deterministic_objective(run, indices, centers, backward=False)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            if iteration == 0:
                np.savez_compressed(args.output / (args.child + "-gradient.npz"), gradients=recorded.pop("gradients"))
                result["first_objective"] = recorded
            else:
                recorded.pop("gradients", None)
            result["repetitions"].append({"seconds": elapsed, "steps": args.steps_per_repetition,
                "evaluations": evaluations, "limited": limited, "outcomes": outcomes,
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                "free_before_bytes": int(free), "device_total_bytes": int(total),
                "host_peak_working_set_bytes": peak_host_bytes()})
            print(json.dumps({"strategy": args.child, "repetition": iteration + 1,
                              "seconds": elapsed, "steps": args.steps_per_repetition}), flush=True)
        result.update(status="PASS", median_seconds=statistics.median(item["seconds"] for item in result["repetitions"]))
    except BaseException as error:
        result.update(status="ERROR", error=f"{type(error).__name__}: {error}")
    target = args.output / (args.child + "-result.json")
    temporary = target.with_suffix(".partial")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, target)
    print(json.dumps({"strategy": args.child, "status": result["status"]}), flush=True)
    return 0 if result["status"] == "PASS" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strategy", choices=STRATEGIES, action="append")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--steps-per-repetition", type=int, default=1)
    parser.add_argument("--seconds-per-strategy", type=int, default=180)
    parser.add_argument("--child", choices=STRATEGIES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.source, args.checkpoint, args.output = (path.resolve() for path in (args.source, args.checkpoint, args.output))
    if args.repetitions < 1 or args.steps_per_repetition < 1 or args.seconds_per_strategy < 1:
        parser.error("Repetitions, steps and deadline must be positive.")
    if args.child:
        return child(args)
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use an empty output directory to preserve previous evidence.")
    args.output.mkdir(parents=True, exist_ok=True)
    source_hash, checkpoint_hash = digest(args.source), digest(args.checkpoint)
    summary = {"source": str(args.source), "checkpoint": str(args.checkpoint), "strategies": {}}
    for strategy in args.strategy or STRATEGIES:
        command = [sys.executable, "-B", __file__, "--source", str(args.source), "--checkpoint", str(args.checkpoint),
                   "--output", str(args.output), "--repetitions", str(args.repetitions),
                   "--steps-per-repetition", str(args.steps_per_repetition), "--child", strategy]
        print(f"Profiling {strategy}", flush=True)
        try:
            completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                                       timeout=args.seconds_per_strategy, check=False)
            (args.output / (strategy + "-output.log")).write_text(completed.stdout + completed.stderr, encoding="utf-8")
            result_path = args.output / (strategy + "-result.json")
            try:
                result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else {
                    "status": "ERROR", "error": f"Profiler child exited {completed.returncode} without a result."}
            except (OSError, json.JSONDecodeError) as error:
                result = {"status": "ERROR", "error": f"Invalid profiler result: {error}"}
            if completed.returncode and result.get("status") == "PASS":
                result.update(status="ERROR", error=f"Profiler child exited {completed.returncode} after a PASS.")
        except subprocess.TimeoutExpired as error:
            stdout = error.stdout or b""
            stderr = error.stderr or b""
            if isinstance(stdout, bytes):
                stdout = stdout.decode("utf-8", errors="replace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", errors="replace")
            (args.output / (strategy + "-output.log")).write_text(stdout + stderr, encoding="utf-8")
            result_path = args.output / (strategy + "-result.json")
            result = {"status": "TIMEOUT", "error": f"Profiler child exceeded {args.seconds_per_strategy} s.",
                      "partial_result": str(result_path) if result_path.is_file() else None}
        summary["strategies"][strategy] = result
        print(json.dumps({"strategy": strategy, "status": result["status"]}), flush=True)
    import numpy as np
    baseline = args.output / "full-gradient.npz"
    for strategy, result in summary["strategies"].items():
        candidate = args.output / (strategy + "-gradient.npz")
        if baseline.is_file() and candidate.is_file():
            with np.load(baseline) as first, np.load(candidate) as second:
                expected, actual = first["gradients"], second["gradients"]
                result["gradient_matches_full"] = bool(np.allclose(expected, actual, rtol=1e-6, atol=1e-9))
                result["gradient_max_abs_difference"] = float(np.max(np.abs(expected - actual)))
            reference_loss = summary["strategies"]["full"].get("first_objective", {})
            result["objective_matches_full"] = all(
                abs(result.get("first_objective", {}).get(key, float("inf")) - value)
                <= 1e-9 + 1e-6 * abs(value) for key, value in reference_loss.items())
    summary["originals_unchanged"] = digest(args.source) == source_hash and digest(args.checkpoint) == checkpoint_hash
    candidates = [(value["median_seconds"], name) for name, value in summary["strategies"].items()
                  if value.get("status") == "PASS" and value.get("gradient_matches_full") is True
                  and value.get("objective_matches_full") is True]
    summary["fastest_matching"] = min(candidates)[1] if candidates else None
    (args.output / "report.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"report": str(args.output / "report.json"), "fastest_matching": summary["fastest_matching"],
                      "originals_unchanged": summary["originals_unchanged"]}), flush=True)
    return 0 if summary["fastest_matching"] and summary["originals_unchanged"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
