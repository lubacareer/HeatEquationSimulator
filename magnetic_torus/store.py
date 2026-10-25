"""Atomic, disk-backed results; no application state or pickle in scientific files."""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np

from .models import DoseResult, PINNResult, RunResult, ScenarioConfig


def json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (Path,)):
        return str(value)
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def default_data_dir() -> Path:
    override = os.environ.get("MAGNETIC_TORUS_DATA")
    if override:
        return Path(override).expanduser().resolve()
    return Path(os.environ.get("LOCALAPPDATA", Path.home() / ".local" / "share")) / "MagneticTorus"


class RunStore:
    FIELD_ARTIFACTS = ("weak_operator_path", "exterior_field_path")
    def __init__(self, root: str | Path | None = None):
        self.root = Path(root) if root is not None else default_data_dir()
        self.root.mkdir(parents=True, exist_ok=True)

    def directory(self, run_id: str) -> Path:
        if not run_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in run_id):
            raise ValueError("Invalid run identifier.")
        directory = self.root / "runs" / run_id
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    @staticmethod
    def _arrays(handle: h5py.Group, name: str, values: Mapping[str, np.ndarray]) -> None:
        group = handle.create_group(name)
        for key, value in values.items():
            group.create_dataset(key, data=value, compression="gzip" if np.asarray(value).size > 128 else None)

    @staticmethod
    def _text(handle: h5py.File, name: str) -> str:
        value = handle.attrs[name]
        return value.decode("utf-8") if isinstance(value, bytes) else str(value)

    @staticmethod
    def _dataset(handle: h5py.Group, name: str) -> np.ndarray:
        value = handle[name]
        if not isinstance(value, h5py.Dataset):
            raise ValueError(f"{name} must be an HDF5 dataset.")
        return np.asarray(value[...])

    @classmethod
    def _read_arrays(cls, handle: h5py.File, name: str) -> dict[str, np.ndarray]:
        group = handle[name]
        if not isinstance(group, h5py.Group):
            raise ValueError(f"{name} must be an HDF5 group.")
        return {key: cls._dataset(group, key) for key in group.keys()}

    def save_run(self, result: RunResult) -> Path:
        directory = self.directory(result.run_id)
        path = directory / "result.h5"
        temporary = path.with_suffix(".h5.partial")
        with h5py.File(temporary, "w") as file:
            file.attrs["schema_version"] = 1
            file.attrs["run_id"] = result.run_id
            file.attrs["config"] = json.dumps(result.config.to_dict())
            file.attrs["metadata"] = json.dumps(dict(result.metadata), default=json_default)
            file.create_dataset("times", data=result.times)
            file.create_dataset("points", data=result.points)
            self._arrays(file, "fields", result.fields)
            self._arrays(file, "diagnostics", result.diagnostics)
            # Keep the accepted exterior field and exact weak operator portable.
            artifacts = file.create_group("artifacts")
            for key in self.FIELD_ARTIFACTS:
                artifact = result.metadata.get(key)
                if artifact and Path(artifact).is_file():
                    data = Path(artifact).read_bytes()
                    dataset = artifacts.create_dataset(key, data=np.frombuffer(data, dtype=np.uint8))
                    dataset.attrs["sha256"] = sha256(data).hexdigest()
            file.flush()
        os.replace(temporary, path)
        manifest = {"run_id": result.run_id, "name": result.config.name,
                    "config_hash": result.config.config_hash, "physics_hash": result.config.physics_hash,
                    "created": datetime.now(timezone.utc).isoformat(),
                    "backend": result.metadata.get("backend", result.config.solver),
                    "validation": result.metadata.get("validation", result.metadata.get("validation_status", "Unvalidated")),
                    "duration": float(result.times[-1]), "path": str(path)}
        temp_manifest = directory / "manifest.json.partial"
        temp_manifest.write_text(json.dumps(manifest, indent=2, default=json_default), encoding="utf-8")
        os.replace(temp_manifest, directory / "manifest.json")
        return path

    @staticmethod
    def load_run(path: str | Path) -> RunResult:
        with h5py.File(path, "r") as file:
            if int(RunStore._text(file, "schema_version")) != 1:
                raise ValueError("This result uses an unsupported file version.")
            metadata = json.loads(RunStore._text(file, "metadata"))
            artifacts = file.get("artifacts")
            if isinstance(artifacts, h5py.Group):
                for key in RunStore.FIELD_ARTIFACTS:
                    if key not in artifacts:
                        continue
                    destination = Path(path).parent / "artifacts" / f"{key}.npz"
                    dataset = artifacts[key]
                    if not isinstance(dataset, h5py.Dataset):
                        raise ValueError("Invalid embedded scientific artifact.")
                    expected = str(dataset.attrs["sha256"])
                    existing = sha256(destination.read_bytes()).hexdigest() if destination.is_file() else None
                    if existing != expected:
                        data = np.asarray(dataset[...], dtype=np.uint8).tobytes()
                        if sha256(data).hexdigest() != expected:
                            raise ValueError("Embedded scientific artifact checksum failed.")
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        temporary = destination.with_suffix(".partial")
                        temporary.write_bytes(data)
                        os.replace(temporary, destination)
                    metadata[key] = str(destination.resolve())
            return RunResult(RunStore._text(file, "run_id"), ScenarioConfig.from_dict(json.loads(RunStore._text(file, "config"))),
                             RunStore._dataset(file, "times"), RunStore._dataset(file, "points"),
                             RunStore._read_arrays(file, "fields"), RunStore._read_arrays(file, "diagnostics"),
                             metadata)

    def list_runs(self) -> list[dict[str, Any]]:
        records = []
        for path in (self.root / "runs").glob("*/manifest.json"):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
                if Path(record["path"]).is_file():
                    records.append(record)
            except (OSError, ValueError, KeyError):
                continue
        return sorted(records, key=lambda value: value["created"], reverse=True)

    def save_pinn(self, result: PINNResult) -> Path:
        path = self.directory(result.source_run_id) / f"pinn-{result.run_id}.h5"
        temporary = path.with_suffix(".h5.partial")
        with h5py.File(temporary, "w") as file:
            for key in ("run_id", "source_run_id", "config_hash"):
                file.attrs[key] = getattr(result, key)
            file.attrs["metrics"] = json.dumps(dict(result.metrics), default=json_default)
            file.attrs["metadata"] = json.dumps(dict(result.metadata), default=json_default)
            self._arrays(file, "fields", result.fields)
            self._arrays(file, "history", result.history)
        os.replace(temporary, path)
        return path

    @staticmethod
    def load_pinn(path: str | Path) -> PINNResult:
        with h5py.File(path, "r") as file:
            return PINNResult(RunStore._text(file, "run_id"), RunStore._text(file, "source_run_id"), RunStore._text(file, "config_hash"),
                              RunStore._read_arrays(file, "fields"), RunStore._read_arrays(file, "history"),
                              json.loads(RunStore._text(file, "metrics")), json.loads(RunStore._text(file, "metadata")))

    def save_dose(self, result: DoseResult) -> Path:
        path = self.directory(result.source_run_id) / f"dose-{result.run_id}.json"
        data = {key: getattr(result, key) for key in result.__dataclass_fields__}
        temporary = path.with_suffix(".json.partial")
        temporary.write_text(json.dumps(data, default=json_default, indent=2), encoding="utf-8")
        os.replace(temporary, path)
        return path

    @staticmethod
    def load_dose(path: str | Path) -> DoseResult:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return DoseResult(**data)
