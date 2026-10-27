"""Create and independently check a finer-step FE source for PINN research.

The input HDF5 is copied before loading because RunStore extracts embedded
artifacts beside a loaded file. No application history is used or modified.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(path: Path) -> str:
    hasher = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify(path: Path, output: Path) -> int:
    """Use the suite's unchanged FE oracle in a separate Python process."""
    import numpy as np
    from magnetic_torus.store import RunStore

    sys.path.insert(0, str(ROOT / "tests"))
    from simulation_suite.scientific import Checks, _weak_checks

    source_hash = digest(path)
    with tempfile.TemporaryDirectory(prefix="torus-fe-verify-") as temporary:
        private_source = Path(temporary) / "source.h5"
        shutil.copy2(path, private_source)
        result = RunStore.load_run(private_source)
        checks = Checks()
        mask = np.asarray(result.metadata["conductor_mask"], dtype=bool)
        checks.add("finite_positive_temperature", bool(np.all(result.fields["temperature"][:, mask] > 0)))
        fraction = result.fields["liquid_fraction"][:, mask]
        checks.add("liquid_fraction_bounds", bool(np.all((fraction >= -1e-10) & (fraction <= 1 + 1e-10))))
        checks.add("mushy_initial_state", bool(np.all((fraction[0] > 0) & (fraction[0] < 1))))
        checks.add("accepted_timeline", bool(result.times[0] == 0 and result.times[-1] == result.config.duration
                   and np.all(np.diff(result.times) > 0) and np.any(np.isclose(result.times, .02, rtol=0, atol=1e-12))))
        energy = result.diagnostics["energy_j"]
        balance = float(np.max(np.abs(result.diagnostics["energy_balance_error_j"])))
        checks.add("assembled_energy_balance", balance <= max(.01, float(np.max(np.abs(energy))) * 1e-9), error_j=balance)
        parts = sum(result.diagnostics[key] for key in ("imposed_field_energy_j", "induced_field_energy_j", "mutual_field_energy_j"))
        checks.add("magnetic_energy_decomposition", bool(np.allclose(parts, result.diagnostics["magnetic_energy_j"], rtol=1e-10, atol=1e-8)))
        pumping = np.r_[0., np.cumsum(np.diff(result.times) * result.diagnostics["pump_power_w"][1:])]
        checks.add("pumping_work", bool(np.allclose(pumping, result.diagnostics["pump_energy_j"], rtol=1e-10, atol=1e-8)))
        checks.add("efficiency_accounting", bool(np.allclose(result.diagnostics["usable_electric_energy_j"],
                   result.config.efficiency * result.diagnostics["recoverable_energy_j"], rtol=1e-12, atol=.01)))
        imposed = result.diagnostics["imposed_field_energy_j"]
        checks.add("constant_independent_imposed_coil", bool(np.allclose(imposed, imposed[0], rtol=1e-12, atol=1e-12)))
        bc = np.asarray(result.metadata["pinn_physics"]["bc_nodes"]["0"], dtype=int)
        checks.add("no_slip", bool(len(bc) and np.max(np.abs(result.fields["velocity"][:, bc])) < 1e-9))
        _weak_checks(result, checks, lambda fraction, message: print(f"{fraction:.2f} {message}", flush=True))
        checks.add("source_unchanged", digest(path) == source_hash)
        evidence = {"status": "PASS" if not checks.failed else "FAIL", "source_path": str(path.resolve()),
                    "source_sha256": source_hash, "run_id": result.run_id, "config_hash": result.config.config_hash,
                    "times_s": result.times.tolist(), "checks": checks.records}
    output.write_text(json.dumps(evidence, indent=2, allow_nan=False), encoding="utf-8")
    return 0 if not checks.failed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    run = sub.add_parser("refine")
    run.add_argument("--source", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--timeout", type=float, default=300.)
    run.add_argument("--steps", type=int, default=6)
    check = sub.add_parser("verify")
    check.add_argument("--source", type=Path, required=True)
    check.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "verify":
        if args.output.exists():
            parser.error("Choose a new verification output file to preserve prior evidence.")
        return verify(args.source, args.output)

    from magnetic_torus.physics import run_simulation
    from magnetic_torus.store import RunStore

    if not 1 <= args.timeout <= 3600:
        parser.error("The refinement timeout must be between 1 and 3600 seconds.")
    if not 1 <= args.steps <= 1000:
        parser.error("Nominal steps must be between 1 and 1000.")
    source_path = args.source.resolve(strict=True)
    directory = args.output.resolve()
    if directory.exists() and any(directory.iterdir()):
        parser.error("Choose an empty output directory to preserve prior evidence.")
    directory.mkdir(parents=True, exist_ok=True)
    copied = directory / "input" / "source.h5"
    copied.parent.mkdir()
    original_hash = digest(source_path)
    shutil.copy2(source_path, copied)
    source = RunStore.load_run(copied)
    if source.config.solver != "fenics" or source.config.mode != "driven":
        parser.error("This follow-up requires the accepted driven FE source.")
    # The driven worker sets the seed-field multiplier to zero; seed 123 gives
    # the later PINN a new deterministic split without changing the driven field.
    if args.steps <= source.config.time_steps:
        parser.error("The refinement must use more nominal steps than its input.")
    config = replace(source.config, time_steps=args.steps, seed=123).validate()
    stop = threading.Event()
    deadline = time.monotonic() + args.timeout
    timer = threading.Timer(args.timeout, stop.set)
    timer.daemon = True
    timer.start()
    summary = {"status": "ERROR", "original_source": str(source_path), "original_sha256": original_hash,
               "nominal_steps": args.steps, "split_seed": 123, "timeout_s": args.timeout}
    try:
        result = run_simulation(config, lambda f, m: print(f"{f:.2f} {m}", flush=True), stop)
        saved = RunStore(directory / "store").save_run(result)
        verification = directory / "validation.json"
        summary.update(refined_source=str(saved.resolve()), refined_sha256=digest(saved),
                       worker_validation=str(verification.resolve()))
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Refinement expired before source validation.")
        completed = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), "verify",
                                    "--source", str(saved), "--output", str(verification)],
                                   cwd=ROOT, timeout=remaining, check=False)
        if stop.is_set() or time.monotonic() >= deadline:
            raise TimeoutError("Refinement exceeded its deadline during source validation.")
        summary.update(status="PASS" if completed.returncode == 0 else "FAIL",
                       validation_returncode=completed.returncode)
    except (TimeoutError, subprocess.TimeoutExpired) as error:
        summary.update(status="TIMEOUT", error=str(error))
    except Exception as error:
        summary.update(status="TIMEOUT" if stop.is_set() else "ERROR", error=f"{type(error).__name__}: {error}")
    finally:
        timer.cancel()
    summary["original_unchanged"] = digest(source_path) == original_hash
    if not summary["original_unchanged"]:
        summary.update(status="ERROR", error="The original source changed during refinement.")
    (directory / "report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)
    return 0 if summary["status"] == "PASS" else 124 if summary["status"] == "TIMEOUT" else 1


if __name__ == "__main__":
    raise SystemExit(main())
