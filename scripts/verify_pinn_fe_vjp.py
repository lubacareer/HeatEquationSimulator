"""Compare full-graph and recomputed global FE gradients on one saved source."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--interval", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--offload", action="store_true")
    parser.add_argument("--seconds", type=int, default=120)
    parser.add_argument("--_child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.source = args.source.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.output = args.output.resolve()
    if args.seconds < 1:
        parser.error("The time limit must be positive.")
    if args.output.exists():
        parser.error("Choose a new output file to preserve previous evidence.")
    if args.device == "cuda" and not args._child:
        # A CUDA OOM previously left Python alive after reporting its error.
        # This probe creates no descendants; subprocess.run kills its owned
        # child at the deadline and leaves any partial result for inspection.
        args.output.parent.mkdir(parents=True, exist_ok=True)
        child_result = args.output.with_name(args.output.stem + "-child.json")
        child_log = args.output.with_name(args.output.stem + "-child.log")
        if child_result.exists() or child_log.exists():
            parser.error("Choose an output name without existing child evidence.")
        command = [sys.executable, "-B", __file__, "--source", str(args.source),
                   "--checkpoint", str(args.checkpoint), "--output", str(child_result),
                   "--device", args.device, "--interval", str(args.interval),
                   "--batch-size", str(args.batch_size), "--_child"]
        if args.offload:
            command.append("--offload")
        try:
            completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                                       timeout=args.seconds, check=False)
            stdout, stderr = completed.stdout, completed.stderr
            try:
                child_data = json.loads(child_result.read_text(encoding="utf-8")) if child_result.is_file() else None
            except (OSError, json.JSONDecodeError):
                child_data = None
            if completed.returncode == 0 and child_data and child_data.get("status") == "PASS":
                result = child_data
            else:
                result = {"status": "ERROR", "returncode": completed.returncode,
                          "child_result": child_data, "error": "CUDA parity child did not pass cleanly."}
        except subprocess.TimeoutExpired as error:
            stdout, stderr = error.stdout or b"", error.stderr or b""
            if isinstance(stdout, bytes):
                stdout = stdout.decode("utf-8", errors="replace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", errors="replace")
            result = {"status": "TIMEOUT", "error": f"CUDA parity child exceeded {args.seconds} s.",
                      "partial_result": str(child_result) if child_result.is_file() else None}
        child_log.write_text(stdout + stderr, encoding="utf-8")
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
        print(json.dumps({"status": result["status"], "report": str(args.output)}), flush=True)
        return 0 if result["status"] == "PASS" else 1
    sys.path.insert(0, str(ROOT))
    import numpy as np
    import torch
    from magnetic_torus.learning import training
    from magnetic_torus.learning.corrections import restore_model_from_checkpoint
    from magnetic_torus.store import RunStore
    from profile_pinn_fe_vjp import digest, peak_host_bytes

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")
    source_hash, checkpoint_hash = digest(args.source), digest(args.checkpoint)
    os.environ["MAGNETIC_TORUS_PINN_DEVICE"] = args.device
    torch.set_num_threads(2)
    # RunStore extracts embedded artifacts beside its input; use a private copy
    # instead of changing the directory containing the historical source.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    input_dir = args.output.parent / (args.output.stem + "-input")
    input_dir.mkdir()
    private_source = input_dir / "source.h5"
    shutil.copy2(args.source.resolve(), private_source)
    source = RunStore.load_run(private_source)
    payload = torch.load(args.checkpoint.resolve(), map_location="cpu", weights_only=True)
    data = training._prepare_data(source, threading.Event())
    run = training._prepare_training(data, lambda *_: None, threading.Event(), None)
    run.field.model = restore_model_from_checkpoint(payload, source, args.device)
    parameters = tuple(p for p in run.field.model.parameters() if p.requires_grad)
    if not parameters or not 1 <= args.interval < data.snapshot_count:
        raise ValueError("The checkpoint requires trainable parameters and an accepted interval.")
    names = [name for name, p in run.field.model.named_parameters() if p.requires_grad]
    measurements = []
    for mode in ("full", "vjp"):
        for p in parameters:
            p.grad = None
        if args.device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        if mode == "full":
            physics, boundary = run.losses.coupled_physics(args.interval)
            (physics + boundary).backward()
        else:
            physics, boundary = run.losses.coupled_physics_backprop(
                args.interval, batch_size=args.batch_size, offload_saved_tensors=args.offload)
        if args.device == "cuda":
            torch.cuda.synchronize()
        arrays = [np.zeros(tuple(p.shape), dtype=np.float64) if p.grad is None
                  else p.grad.detach().cpu().numpy().copy() for p in parameters]
        measurements.append({"mode": mode, "seconds": time.perf_counter() - started,
            "physics_loss": float(physics.detach()), "boundary_loss": float(boundary.detach()),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()) if args.device == "cuda" else None,
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()) if args.device == "cuda" else None,
            "host_peak_working_set_bytes": peak_host_bytes(), "gradients": arrays})
        del physics, boundary
    first, second = measurements
    rtol, atol = ((1e-6, 1e-9) if args.device == "cuda" else (1e-8, 1e-10))
    errors = [{"parameter": name, "maximum_absolute_difference": float(np.max(np.abs(a - b))),
               "matches": bool(np.allclose(a, b, rtol=rtol, atol=atol))}
              for name, a, b in zip(names, first.pop("gradients"), second.pop("gradients"))]
    loss_matches = all(np.isclose(first[key], second[key], rtol=rtol, atol=atol)
                       for key in ("physics_loss", "boundary_loss"))
    originals_unchanged = digest(args.source) == source_hash and digest(args.checkpoint) == checkpoint_hash
    passed = bool(loss_matches and all(item["matches"] for item in errors) and originals_unchanged)
    result = {"status": "PASS" if passed else "FAIL", "source_run_id": source.run_id,
              "source_config_hash": source.config.config_hash, "checkpoint_epoch": int(payload["epoch"]),
              "architecture": payload.get("metadata", {}).get("architecture"),
              "source_sha256": source_hash, "checkpoint_sha256": checkpoint_hash,
              "originals_unchanged": originals_unchanged,
              "interval": args.interval, "accepted_times": source.times[args.interval - 1:args.interval + 1].tolist(),
              "points": data.point_count, "device": args.device, "batch_size": args.batch_size,
              "offload": args.offload, "rtol": rtol, "atol": atol,
              "loss_matches": bool(loss_matches), "parameter_errors": errors, "measurements": measurements}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".partial")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps({"status": result["status"], "report": str(args.output),
                      "maximum_absolute_gradient_difference": max(item["maximum_absolute_difference"] for item in errors)}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
