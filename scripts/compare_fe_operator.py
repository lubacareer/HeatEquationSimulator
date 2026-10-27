"""Compare the saved FE worker form and its PyTorch reconstruction on one interval."""
from __future__ import annotations

import argparse
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def read_probe_digest(directory: Path) -> str:
    """Require the digest recorded by the worker's original completed run."""
    try:
        value = (directory / "operator_probe.sha256").read_text(encoding="ascii").strip()
    except (OSError, UnicodeError) as exc:
        raise ValueError("Worker probe digest is missing or unreadable; run a fresh FE worker before reuse.") from exc
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("Worker probe digest is malformed; run a fresh FE worker before reuse.")
    return value


def verified_probe_bytes(path: Path, expected: str) -> bytes:
    """Hash the exact bytes returned for later loading, not a separately opened file."""
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError("Worker probe SHA-256 mismatch; run a fresh FE worker before reuse.")
    return data


def write_probe_digest(directory: Path, value: str) -> None:
    target = directory / "operator_probe.sha256"
    partial = directory / "operator_probe.sha256.partial"
    try:
        partial.write_text(value + "\n", encoding="ascii")
        os.replace(partial, target)
    finally:
        partial.unlink(missing_ok=True)


def comparisons(fe: np.ndarray, reconstructed: np.ndarray) -> dict[str, float | bool]:
    """Compare raw free-node vectors; norms remain in each equation's units."""
    import numpy as np
    if fe.shape != reconstructed.shape or not np.isfinite(fe).all() or not np.isfinite(reconstructed).all():
        raise ValueError("Non-finite or incompatible operator vector.")
    difference = fe - reconstructed
    max_abs = float(np.max(np.abs(difference), initial=0.))
    ref = float(np.max(np.abs(fe), initial=0.))
    return {"maximum_absolute_difference": max_abs, "reference_maximum_absolute": ref,
            "difference_l2": float(np.sqrt(np.sum(difference * difference))),
            "reference_l2": float(np.sqrt(np.sum(fe * fe))),
            "matches": bool(max_abs <= 1.e-7 + 1.e-6 * ref)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--reuse-worker", type=Path, help="Reuse a completed worker only when its recorded probe digest and all inputs match.")
    parser.add_argument("--freeze-temperature", action="store_true", help="Counterfactual diagnostic: hold temperature fixed across the pair.")
    args = parser.parse_args()
    if args.timeout < 1 or args.output.exists():
        parser.error("Choose a new output directory and a positive timeout.")
    import torch
    import numpy as np
    from magnetic_torus.backend_environment import wsl_distribution, wsl_path
    from magnetic_torus.numerics.weak_residual import _load_weak_artifact, evaluate_weak_residual_torch
    from magnetic_torus.store import RunStore

    started = time.monotonic()
    source_path = args.source.resolve()
    before_hash = digest(source_path)
    directory = args.output.resolve()
    directory.mkdir(parents=True)
    private = directory / "source.h5"
    shutil.copy2(source_path, private)
    if digest(private) != before_hash:
        raise RuntimeError("Source copy failed its hash check.")
    source = RunStore.load_run(private)
    if not 1 <= args.interval < len(source.times):
        raise ValueError("The interval must identify two accepted snapshots.")
    config = source.config
    op = _load_weak_artifact(source.artifact_bytes("weak_operator_path"))
    info = source.metadata["pinn_physics"]
    names = ("velocity", "pressure", "vector_potential", "electric_potential", "temperature")
    components = ("momentum", "continuity", "induction", "charge", "enthalpy")
    widths = (3, 1, 3, 1, 1)
    ids = np.array([args.interval - 1, args.interval])
    rng = np.random.default_rng(args.seed)
    payload = {"points": source.points, "cells": op["cells"], "cell_labels": op["cell_labels"],
               "times": source.times[ids], "imposed": op["imposed_vector_potential"][ids]}
    masks, directions = {}, {}
    for i, (name, width) in enumerate(zip(names, widths)):
        bc = np.asarray(info["bc_nodes"].get(str(i), []), dtype=np.int64)
        payload[f"bc_{i}"] = np.unique(bc)
        fluid = op["node_volumes"] > 0 if i != 2 else np.ones(len(source.points), dtype=bool)
        mask = fluid.copy()
        mask[bc] = False
        masks[name] = mask
        values = np.asarray(source.fields[name], dtype=np.float64)[ids].copy()
        if width == 1:
            values = values.reshape(2, len(source.points))
        if name == "temperature" and args.freeze_temperature:
            values[1] = values[0]
        payload[name] = values
        # Small, fixed physical perturbations, zero at every constrained DOF.
        scale = (1.e-7, .02, 1.e-8, 1.e-8, .01)[i]
        direction = rng.standard_normal(values[1].shape) * scale
        direction[~mask] = 0.
        directions[name] = direction
        payload[f"direction_{name}"] = direction
    request = directory / "request.json"
    probe = directory / "input.npz"
    output = directory / "worker"
    request.write_text(json.dumps(config.to_dict()), encoding="utf-8")
    np.savez_compressed(probe, **payload)
    worker = ROOT / "magnetic_torus" / "workers" / "fenics_worker.py"
    interpreter = os.environ.get("MAGNETIC_TORUS_FE_PYTHON", "/usr/bin/python3")
    if os.name == "nt":
        command = ["wsl", "-d", wsl_distribution(), "--", interpreter, wsl_path(worker),
                   wsl_path(request), wsl_path(output), "--operator-probe", wsl_path(probe)]
    else:
        command = [interpreter, str(worker), str(request), str(output), "--operator-probe", str(probe)]
    if args.reuse_worker:
        other = args.reuse_worker.resolve()
        if (other / "worker.sha256").read_text(encoding="ascii").strip() != digest(worker):
            raise ValueError("Worker reuse requires the identical FE worker implementation.")
        if json.loads((other / "request.json").read_text(encoding="utf-8")) != json.loads(request.read_text(encoding="utf-8")):
            raise ValueError("Worker reuse requires the identical scenario configuration.")
        with np.load(other / "input.npz", allow_pickle=False) as saved:
            if set(saved.files) != set(payload) or any(not np.array_equal(saved[k], v) for k, v in payload.items()):
                raise ValueError("Worker reuse requires identical source fields, geometry, boundaries, time and direction.")
        probe_hash = read_probe_digest(other)
        verified_probe_bytes(other / "worker" / "operator_probe.npz", probe_hash)
        output.mkdir()
        shutil.copy2(other / "worker" / "operator_probe.npz", output / "operator_probe.npz")
        probe_bytes = verified_probe_bytes(output / "operator_probe.npz", probe_hash)
        write_probe_digest(directory, probe_hash)
        shutil.copy2(other / "worker.log", directory / "worker.log")
        shutil.copy2(other / "worker.sha256", directory / "worker.sha256")
    else:
        try:
            completed = subprocess.run(command, capture_output=True, text=True, timeout=args.timeout, check=False)
        except subprocess.TimeoutExpired as exc:
            # The WSL worker watchdog owns its own process group and sees this marker.
            output.mkdir(exist_ok=True)
            (output / "cancel").touch()
            (directory / "report.json").write_text(json.dumps({"status": "TIMEOUT", "timeout_s": args.timeout,
                "source_sha256": before_hash, "source_unchanged": digest(source_path) == before_hash}, indent=2), encoding="utf-8")
            raise TimeoutError(f"FE operator worker exceeded {args.timeout} seconds.") from exc
        (directory / "worker.log").write_text(completed.stdout + completed.stderr, encoding="utf-8")
        if completed.returncode:
            raise RuntimeError(f"FE operator worker returned {completed.returncode}; see worker.log.")
        (directory / "worker.sha256").write_text(digest(worker), encoding="ascii")
        probe_bytes = (output / "operator_probe.npz").read_bytes()
        probe_hash = hashlib.sha256(probe_bytes).hexdigest()
        write_probe_digest(directory, probe_hash)
    with np.load(BytesIO(probe_bytes), allow_pickle=False) as archive:
        fenics = {key: archive[key].copy() for key in archive.files}
    torch.set_num_threads(2)
    previous = {name: torch.as_tensor(payload[name][0], dtype=torch.float64) for name in names}
    current = tuple(torch.as_tensor(payload[name][1], dtype=torch.float64) for name in names)
    tangent = tuple(torch.as_tensor(directions[name], dtype=torch.float64) for name in names)
    def evaluate(*values):
        fields = {name: torch.stack((previous[name], value)) for name, value in zip(names, values)}
        residual = evaluate_weak_residual_torch(config, source, fields, ids, include_raw_components=True)
        return tuple(residual[f"raw_{component}"] for component in components)
    print("Evaluating reconstructed FE residual and Jacobian action", flush=True)
    # The worker evaluates an exact UFL directional derivative; PyTorch's
    # JVP differentiates the independently reconstructed weak form.
    actual, jacobian = torch.autograd.functional.jvp(evaluate, current, tangent, strict=False)
    # Check a centered finite-difference sequence as a separate diagnostic.
    steps = (.01, .005, .0025)
    with torch.no_grad():
        jacobian_by_step = []
        for step in steps:
            plus = evaluate(*(value + step * vector for value, vector in zip(current, tangent)))
            minus = evaluate(*(value - step * vector for value, vector in zip(current, tangent)))
            jacobian_by_step.append(tuple((p - m) / (2 * step) for p, m in zip(plus, minus)))
            print(f"Finite difference step {step:g} complete", flush=True)
    records = {}
    for name, component, f, j in zip(names, components, actual, jacobian):
        mask = masks[name]
        records[component] = {
            "residual": comparisons(fenics[f"residual_{name}"][mask], f.detach().numpy()),
            "jacobian_action": comparisons(fenics[f"jacobian_{name}"][mask], j.detach().numpy()),
            "finite_difference_stability": comparisons(jacobian_by_step[-2][components.index(component)].numpy(), jacobian_by_step[-1][components.index(component)].numpy()),
            "free_node_count": int(mask.sum()),
        }
        if component == "enthalpy":
            scale = config.density * max(config.latent_heat, config.heat_capacity_solid * 100., 1.) / max(config.duration, float(np.diff(source.times[ids])[0]))
            denominator = np.maximum(op["node_volumes"][mask], 1.e-30) * scale
            delta = fenics[f"residual_{name}"][mask] - f.detach().numpy()
            records[component]["max_normalized_residual_difference"] = float(np.max(np.abs(delta) / denominator, initial=0.))
    report = {"status": "PASS" if all(row[key]["matches"] for row in records.values()
                                      for key in ("residual", "jacobian_action", "finite_difference_stability")) else "FAIL",
              "source_sha256": before_hash, "source_unchanged": digest(source_path) == before_hash,
              "source_run_id": source.run_id, "config_hash": config.config_hash, "seed": args.seed,
              "worker_sha256": digest(worker),
              "operator_probe_sha256": probe_hash,
              "times_s": source.times[ids].tolist(), "mesh_and_boundary_masks_match": True,
              "counterfactual_frozen_temperature": args.freeze_temperature,
              "comparison": "raw assembled free-node vectors, not solver convergence tolerance",
              "jacobian_method": "analytic UFL direction vs PyTorch autodiff JVP; finite-difference stability separately",
              "finite_difference_steps": list(steps),
              "reused_worker_from": str(args.reuse_worker.resolve()) if args.reuse_worker else None,
              "comparison_atol": 1.e-7, "comparison_rtol": 1.e-6,
              "components": records, "elapsed_s": time.monotonic() - started}
    (directory / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"status": report["status"], "report": str(directory / "report.json")}), flush=True)
    return 0 if report["status"] == "PASS" and report["source_unchanged"] else 1


if __name__ == "__main__":
    sys.exit(main())
