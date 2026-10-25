"""Atomic, disk-backed results; no application state or pickle in scientific files."""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np

from .models import DoseResult, PINNResult, RunResult, ScenarioConfig
from .artifacts import FIELD_ARTIFACT_KEYS, FieldArtifact


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
    FIELD_ARTIFACTS = FIELD_ARTIFACT_KEYS
    def __init__(self, root: str | Path | None = None, *, target_root: Path | None = None,
                 publication_id: str | None = None):
        self.root = Path(root) if root is not None else default_data_dir()
        self.root.mkdir(parents=True, exist_ok=True)
        self._target_root = target_root
        self._publication_id = publication_id
        self._destinations: dict[str, Path] = {}

    def directory(self, run_id: str) -> Path:
        if not run_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in run_id):
            raise ValueError("Invalid run identifier.")
        directory = self.root / "runs" / run_id
        directory.mkdir(parents=True, exist_ok=True)
        if self._target_root is not None and run_id not in self._destinations:
            destination = self._target_root / "runs" / run_id
            if destination.exists():
                destination = destination / "analyses" / str(self._publication_id)
            self._destinations[run_id] = destination
        return directory

    def published_path(self, path: str | Path) -> Path:
        value = Path(path)
        if self._target_root is not None:
            for run_id, destination in self._destinations.items():
                try:
                    return destination / value.relative_to(self.root / "runs" / run_id)
                except ValueError:
                    pass
        return value

    def _metadata(self, value: Any) -> Any:
        if isinstance(value, Mapping):
            return {key: self._metadata(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._metadata(item) for item in value]
        if isinstance(value, (str, Path)):
            # Geant4 request JSON uses Linux paths while its files live on the
            # Windows filesystem. Preserve that representation when rebasing.
            match = re.fullmatch(r"/mnt/([a-zA-Z])/(.*)", str(value)) if os.name == "nt" else None
            if match is not None:
                native = Path(f"{match[1]}:/{match[2]}")
                rebased = self.published_path(native).as_posix()
                return f"/mnt/{rebased[0].lower()}/{rebased[3:]}"
            return str(self.published_path(value)) if Path(value).is_absolute() else value
        return value

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
    def _hard_item(handle: h5py.Group, name: str):
        if not isinstance(handle.get(name, getlink=True), h5py.HardLink):
            raise ValueError(f"External or soft HDF5 links are not accepted: {name}.")
        return handle[name]

    @staticmethod
    def _dataset(handle: h5py.Group, name: str) -> np.ndarray:
        value = RunStore._hard_item(handle, name)
        if not isinstance(value, h5py.Dataset):
            raise ValueError(f"{name} must be an HDF5 dataset.")
        if value.is_virtual or value.external:
            raise ValueError(f"External HDF5 dataset storage is not accepted: {name}.")
        return np.asarray(value[...])

    @classmethod
    def _read_arrays(cls, handle: h5py.File, name: str) -> dict[str, np.ndarray]:
        group = cls._hard_item(handle, name)
        if not isinstance(group, h5py.Group):
            raise ValueError(f"{name} must be an HDF5 group.")
        return {key: cls._dataset(group, key) for key in group.keys()}

    def save_run(self, result: RunResult) -> Path:
        metadata = dict(result.metadata)
        for key in self.FIELD_ARTIFACTS:
            if metadata.get(key) and key not in result.artifacts:
                raise ValueError(f"Cannot save an uncaptured field artifact: {key}.")
        directory = self.directory(result.run_id)
        path = directory / "result.h5"
        temporary = path.with_suffix(".h5.partial")
        with h5py.File(temporary, "w") as file:
            file.attrs["schema_version"] = 1
            file.attrs["run_id"] = result.run_id
            file.attrs["config"] = json.dumps(result.config.to_dict())
            file.attrs["metadata"] = json.dumps(self._metadata(metadata), default=json_default)
            file.create_dataset("times", data=result.times)
            file.create_dataset("points", data=result.points)
            self._arrays(file, "fields", result.fields)
            self._arrays(file, "diagnostics", result.diagnostics)
            # Keep the accepted exterior field and exact weak operator portable.
            artifacts = file.create_group("artifacts")
            for key in self.FIELD_ARTIFACTS:
                artifact = result.artifacts.get(key)
                if artifact is not None:
                    data = artifact.data
                    dataset = artifacts.create_dataset(key, data=np.frombuffer(data, dtype=np.uint8))
                    dataset.attrs["sha256"] = artifact.digest
            file.flush()
        os.replace(temporary, path)
        manifest = {"run_id": result.run_id, "name": result.config.name,
                    "config_hash": result.config.config_hash, "physics_hash": result.config.physics_hash,
                    "created": datetime.now(timezone.utc).isoformat(),
                    "backend": result.metadata.get("backend", result.config.solver),
                    "validation": result.metadata.get("validation", result.metadata.get("validation_status", "Unvalidated")),
                    "duration": float(result.times[-1]), "path": str(self.published_path(path))}
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
            if not isinstance(metadata, dict):
                raise ValueError("Result metadata must be a JSON object.")
            # Paths from imported JSON never authorize filesystem reads.
            for key in RunStore.FIELD_ARTIFACTS:
                metadata.pop(key, None)
            accepted_artifacts = {}
            artifacts = RunStore._hard_item(file, "artifacts") if "artifacts" in file else None
            if isinstance(artifacts, h5py.Group):
                for key in RunStore.FIELD_ARTIFACTS:
                    if key not in artifacts:
                        continue
                    dataset = RunStore._hard_item(artifacts, key)
                    if not isinstance(dataset, h5py.Dataset):
                        raise ValueError("Invalid embedded scientific artifact.")
                    expected = str(dataset.attrs["sha256"])
                    if re.fullmatch(r"[0-9a-f]{64}", expected) is None:
                        raise ValueError("Invalid embedded artifact SHA-256.")
                    data = RunStore._dataset(artifacts, key).astype(np.uint8, copy=False).tobytes()
                    if sha256(data).hexdigest() != expected:
                        raise ValueError("Embedded scientific artifact checksum failed.")
                    parent = Path(path).resolve().parent
                    artifact_dir = parent / "artifacts"
                    artifact_dir.mkdir(exist_ok=True)
                    if artifact_dir.is_symlink() or artifact_dir.resolve().parent != parent:
                        raise ValueError("Artifact extraction directory escapes the imported run.")
                    destination = artifact_dir / f"{key}-{expected}.npz"
                    if destination.is_symlink() or destination.resolve().parent != artifact_dir.resolve():
                        raise ValueError("Artifact extraction path escapes the imported run.")
                    # Always verify embedded bytes, even if this digest is cached.
                    if not destination.is_file() or sha256(destination.read_bytes()).hexdigest() != expected:
                        with tempfile.NamedTemporaryFile(dir=artifact_dir, prefix=".artifact-", delete=False) as staged:
                            staged.write(data)
                            temporary = Path(staged.name)
                        try:
                            os.replace(temporary, destination)
                        finally:
                            temporary.unlink(missing_ok=True)
                    metadata[key] = str(destination.resolve())
                    accepted_artifacts[key] = FieldArtifact(data, destination.resolve())
            return RunResult(RunStore._text(file, "run_id"), ScenarioConfig.from_dict(json.loads(RunStore._text(file, "config"))),
                             RunStore._dataset(file, "times"), RunStore._dataset(file, "points"),
                             RunStore._read_arrays(file, "fields"), RunStore._read_arrays(file, "diagnostics"),
                             metadata, artifacts=accepted_artifacts)

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
            file.attrs["metadata"] = json.dumps(self._metadata(result.metadata), default=json_default)
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
        temporary.write_text(json.dumps(self._metadata(data), default=json_default, indent=2), encoding="utf-8")
        os.replace(temporary, path)
        return path

    @staticmethod
    def load_dose(path: str | Path) -> DoseResult:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        metadata = dict(data.get("metadata", {}))
        voxel = metadata.get("voxel_dose_path")
        if voxel and not Path(voxel).resolve().is_relative_to(Path(path).resolve().parent):
            # A result placed beside an imported run must not grant a scene
            # permission to read an unrelated local NPZ through its metadata.
            metadata.pop("voxel_dose_path", None)
        data["metadata"] = metadata
        return DoseResult(**data)
