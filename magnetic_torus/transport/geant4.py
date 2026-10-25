"""Geant4 binary interchange, process lifecycle, and validated dose acceptance."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from typing import Any, Mapping
import uuid
import numpy as np
from ..models import DoseResult, Progress, RunResult
from ..backend_environment import wsl_distribution, wsl_path
from .common import _check
from .fields import frozen_field
from .sampling import sample_source


def _geant4_command(request_path: Path) -> list[str]:
    binary = os.environ.get("TORUS_GEANT4_WORKER", "")
    if os.name == "nt":
        executable = binary or "/opt/magnetic-torus/bin/torus-dose-run"
        return ["wsl", "-d", wsl_distribution(), "--", executable, wsl_path(request_path)]
    executable = binary or shutil.which("torus-dose-run") or "/opt/magnetic-torus/bin/torus-dose-run"
    if not Path(executable).is_file():
        raise RuntimeError("Geant4 dose worker is not installed. Run bash scripts/setup_geant4.sh in Ubuntu/WSL, then restart the app.")
    return [executable, str(request_path)]


def parse_dose_output(data: Mapping[str, Any], result: RunResult, snapshot_index: int,
                      *, expected_field_sha256: str | None = None) -> DoseResult:
    """Validate the worker provenance before accepting scientific dose output."""
    if data.get("schema_version") != 1 or data.get("source_run_id") != result.run_id or data.get("config_hash") != result.config.config_hash:
        raise ValueError("Geant4 returned a stale or incompatible dose result.")
    if data.get("snapshot_index") != snapshot_index or data.get("histories") != result.config.radiation_histories:
        raise ValueError("Geant4 dose snapshot/history count does not match this request.")
    worker_metadata = data.get("metadata", {})
    if worker_metadata.get("physics_list") != result.config.transport_physics_list + " + EM option 4":
        raise ValueError("Geant4 dose used a different physics list than the immutable scenario.")
    cut = worker_metadata.get("production_cut_m")
    if not isinstance(cut, (int, float)) or not math.isclose(cut, result.config.production_cut_m, rel_tol=0., abs_tol=1e-15):
        raise ValueError("Geant4 dose used a different production cut than the immutable scenario.")
    if worker_metadata.get("source_model_id") != result.metadata.get("source_model_id"):
        raise ValueError("Geant4 dose came from a different selected PINN/numerical source.")
    if expected_field_sha256 is not None and worker_metadata.get("field_sha256") != expected_field_sha256:
        raise ValueError("Geant4 dose field fingerprint does not match the accepted frozen snapshot.")
    rows = data.get("organs", [])
    if not rows:
        raise ValueError("Geant4 returned no dose scoring regions.")
    dose = np.array([row["field_on_gy"] for row in rows], float)
    error = np.array([row["field_on_standard_error_gy"] for row in rows], float)
    baseline = np.array([row["field_off_gy"] for row in rows], float)
    baseline_error = np.array([row["field_off_standard_error_gy"] for row in rows], float)
    masses = np.array([row["mass_kg"] for row in rows], float)
    secondary = np.array([row["secondary_gy"] for row in rows], float)
    secondary_error = np.array([row["secondary_standard_error_gy"] for row in rows], float)
    if not np.all(np.isfinite(dose)) or not np.all(np.isfinite(error)) or np.any(dose < 0) or np.any(error < 0):
        raise ValueError("Geant4 dose or uncertainty is invalid.")
    if not np.all(np.isfinite(baseline)) or not np.all(np.isfinite(baseline_error)) or np.any(baseline < 0) or np.any(baseline_error < 0):
        raise ValueError("Geant4 field-off dose or uncertainty is invalid.")
    if (not np.all(np.isfinite(masses)) or np.any(masses <= 0) or not np.all(np.isfinite(secondary))
            or np.any(secondary < 0) or not np.all(np.isfinite(secondary_error)) or np.any(secondary_error < 0)):
        raise ValueError("Geant4 scoring mass or secondary dose is invalid.")
    return DoseResult(uuid.uuid4().hex, result.run_id, result.config.config_hash, snapshot_index,
                      result.config.target, tuple(row["name"] for row in rows), dose, error,
                      result.config.radiation_histories,
                      {**dict(data.get("metadata", {})), "field_off_dose_gy": [r["field_off_gy"] for r in rows],
                       "field_off_standard_error_gy": [r["field_off_standard_error_gy"] for r in rows],
                       "secondary_dose_gy": secondary.tolist(), "organ_masses_kg": masses.tolist(),
                       "secondary_standard_error_gy": secondary_error.tolist(),
                       "statistical_precision_met": [bool(d > 0 and e / d <= .05) for d, e in zip(dose, error)],
                       "absorbed_dose_only": True, "validation_status": "research transport; benchmark qualification required"})


def calculate_dose(result: RunResult, snapshot_index: int, progress: Progress,
                   cancel: Any, output_dir: Path | None = None) -> DoseResult:
    field_map = frozen_field(result, snapshot_index)
    _check(cancel)
    config = result.config
    if config.radiation_histories < 2:
        raise ValueError("Dose uncertainty requires at least two primary histories.")
    artifact_dir = Path(output_dir) / f"transport-{uuid.uuid4().hex}" if output_dir else Path(tempfile.mkdtemp(prefix="torus-dose-"))
    artifact_dir.mkdir(parents=True, exist_ok=True)
    progress(.01, "Preparing accepted frozen field and paired primary sample")
    source = sample_source(config, config.radiation_histories, .80 * field_map.radius)
    # Native text/binary interchange avoids a dependency on Python inside Geant4.
    map_path = artifact_dir / "field.bin"
    with map_path.open("wb") as stream:
        np.asarray([len(field_map.points), len(field_map.cells), 1 if field_map.cell_values is not None else 0], dtype="<i8").tofile(stream)
        field_map.points.astype("<f8").tofile(stream)
        field_map.cells.astype("<i8").tofile(stream)
        b_values = field_map.cell_values if field_map.cell_values is not None else field_map.values
        b_values.astype("<f8").tofile(stream)
    source_path = artifact_dir / "primaries.bin"
    # Z,A,E/n,x,y,z,dx,dy,dz; identical sample fed to both ensembles.
    np.column_stack((source.z, source.a, source.energy_mev_nucleon, source.positions, source.directions)).astype("<f8").tofile(source_path)
    output_path = artifact_dir / "dose.json"
    request = {"schema_version": 1, "source_run_id": result.run_id, "config_hash": config.config_hash,
               "snapshot_index": snapshot_index, "config": config.to_dict(),
               "field_file": wsl_path(map_path), "primary_file": wsl_path(source_path),
               "output_file": wsl_path(output_path), "field_sha256": field_map.fingerprint,
               "field_time_s": float(result.times[snapshot_index]), "source_area_m2": source.source_area_m2,
               "source": dict(source.metadata),
               "field_source": result.metadata.get("field_source", "numerical"),
               "source_model_id": result.metadata.get("source_model_id"),
               "source_model_field_sha256": result.metadata.get("source_model_field_sha256"),
               "source_model_validation_status": result.metadata.get("source_model_validation_status"),
               "world_radius_m": field_map.radius * .90,
               "imposed_coil": dict(result.metadata.get("imposed_coil", {
                   "radius_m": config.major_radius + config.minor_radius + 2 * max(config.vessel_thickness, .01 * config.minor_radius),
                   "wire_radius_m": max(config.vessel_thickness / 2, .002 * config.minor_radius), "z_m": 0.})),
               "phantom_path": wsl_path(Path(config.phantom_path)) if config.phantom_path else "/opt/magnetic-torus/data/ICRPdata",
               "physics_list": config.transport_physics_list, "production_cut_m": config.production_cut_m}
    request_path = artifact_dir / "request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    command = _geant4_command(request_path)
    log_path = artifact_dir / "geant4.log"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        try:
            while process.poll() is None:
                _check(cancel)
                progress_path = artifact_dir / "progress.json"
                if progress_path.is_file():
                    try:
                        info = json.loads(progress_path.read_text())
                        progress(.05 + .9 * info["fraction"], info["message"])
                    except (ValueError, KeyError, OSError):
                        pass
                time.sleep(.1)
        except BaseException:
            pid_path = artifact_dir / "worker.pid"
            if os.name == "nt" and pid_path.is_file():
                worker_pid = pid_path.read_text().strip()
                if worker_pid.isdigit():
                    subprocess.run(["wsl", "-d", wsl_distribution(), "--",
                                    "kill", "-TERM", worker_pid], capture_output=True, timeout=15,
                                   creationflags=subprocess.CREATE_NO_WINDOW)
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
            raise
    if process.returncode != 0 or not output_path.is_file():
        detail = log_path.read_text(encoding="utf-8", errors="replace")[-3000:]
        raise RuntimeError("Geant4 transport failed. Install its worker and datasets with bash scripts/setup_geant4.sh in WSL Ubuntu. "
                           f"For anatomical scoring verify ICRP110 data and habitat dimensions. Exit code: {process.returncode}. Log: {log_path}\n{detail}")
    data = json.loads(output_path.read_text(encoding="utf-8"))
    voxel_path = artifact_dir / "voxel_dose.bin"
    if voxel_path.is_file():
        values = np.fromfile(voxel_path, dtype="<f8").reshape(-1, 7)
        voxel_artifact = artifact_dir / "voxel_dose.npz"
        np.savez_compressed(voxel_artifact, points=values[:, :3], dose_gy=values[:, 3],
                            standard_error_gy=values[:, 4], organ_ids=values[:, 5].astype(int),
                            voxel_indices=values[:, 6].astype(int))
        data.setdefault("metadata", {})["voxel_dose_path"] = str(voxel_artifact.resolve())
        data["metadata"]["voxel_scoring_note"] = "Only voxels with observed deposition are stored. Unobserved dose is statistically unresolved."
    _check(cancel)
    output = parse_dose_output(data, result, snapshot_index, expected_field_sha256=field_map.fingerprint)
    progress(1., "Paired Geant4 dose transport complete")
    return output
