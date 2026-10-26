"""Accepted workspace state without Qt, plotting, or mutable scientific results."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np

from ..display import DisplaySelection
from ..geometry import conductor_mask
from ..models import DoseResult, PINNResult, RunResult

if TYPE_CHECKING:
    from ..store import RunStore


def _newest_artifacts(directory: Path, pattern: str) -> list[Path]:
    """Order available sidecars without making their presence mandatory."""
    candidates = []
    for artifact in directory.rglob(pattern):
        try:
            modified = artifact.stat().st_mtime
        except OSError:
            # A concurrent cleanup or inaccessible sidecar must not prevent
            # inspection of the independently loaded numerical result.
            continue
        candidates.append((modified, artifact))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return [artifact for _, artifact in candidates]


@dataclass
class WorkspaceSession:
    """Own selection and analysis associations independently of editable inputs."""

    active: RunResult | None = None
    active_path: Path | None = None
    comparison: RunResult | None = None
    pinn: PINNResult | None = None
    pinn_path: Path | None = None
    dose: DoseResult | None = None
    trajectories: dict[str, Any] | None = None
    current_index: int = 0
    probe_index: int = 0
    view: str = "Thermal"
    source: str = "Numerical"
    previewing: bool = True
    selection_revision: int = 0
    job_revision: int = 0
    pinn_difference: dict[str, np.ndarray] | None = field(default=None, repr=False)
    comparison_difference: dict[str, np.ndarray] | None = field(
        default=None, repr=False
    )

    @property
    def current_time(self) -> float:
        return (
            float(self.active.times[self.current_index])
            if self.active is not None
            else 0.0
        )

    @property
    def job_is_current(self) -> bool:
        return self.selection_revision == self.job_revision

    def timeline_end(self) -> float:
        if self.active is None:
            return 0.0
        end = float(self.active.times[-1])
        if self.view == "Compare" and self.comparison is not None:
            end = min(end, float(self.comparison.times[-1]))
        return end

    def activate(self, result: RunResult, path: Path) -> float:
        """Retain the nearest physical time and conductor probe across meshes."""
        old_time = self.current_time
        old_probe = (
            self.active.points[self.probe_index].copy()
            if self.active is not None
            else None
        )
        mask = conductor_mask(result)
        indices = np.flatnonzero(mask)
        self.active, self.active_path = result, Path(path)
        self.selection_revision += 1
        self.current_index = int(np.argmin(np.abs(result.times - old_time)))
        self.probe_index = int(indices[0]) if len(indices) else 0
        if old_probe is not None and len(indices):
            self.probe_index = int(
                indices[
                    np.argmin(np.sum((result.points[indices] - old_probe) ** 2, axis=1))
                ]
            )
        self.pinn = None
        self.pinn_path = None
        self.pinn_difference = None
        self.comparison_difference = None
        self.dose = None
        self.trajectories = None
        self.previewing = False
        self.source = "Numerical"
        return old_time

    def load_result(self, store: RunStore, path: Path) -> float:
        """Load accepted fields and only analyses belonging to this source."""
        result = store.load_run(path)
        old_time = self.activate(result, path)
        for artifact in _newest_artifacts(Path(path).parent, "pinn-*.h5"):
            try:
                candidate = store.load_pinn(artifact)
                if candidate.config_hash == result.config.config_hash:
                    self.set_pinn(candidate, artifact)
                    break
            except (OSError, ValueError, KeyError, TypeError):
                continue
        for artifact in _newest_artifacts(Path(path).parent, "dose-*.json"):
            try:
                candidate = store.load_dose(artifact)
                if (
                    candidate.config_hash == result.config.config_hash
                    and candidate.source_run_id == result.run_id
                    and 0 <= candidate.snapshot_index < len(result.times)
                ):
                    self.dose = candidate
                    break
            except (OSError, ValueError, TypeError):
                continue
        return old_time

    def set_pinn(self, model: PINNResult, path: Path | None = None) -> None:
        if self.active is None:
            raise ValueError("The trained model has no source scenario.")
        # Import remains lazy: inspecting a saved model must not initialize Torch.
        from ..pinn import validate_pinn_source

        validate_pinn_source(model, self.active)
        self.pinn = model
        self.pinn_path = Path(path) if path is not None else None
        self.pinn_difference = None

    def set_dose(self, dose: DoseResult) -> None:
        result = self.active
        if (
            result is None
            or dose.source_run_id != result.run_id
            or dose.config_hash != result.config.config_hash
            or not 0 <= dose.snapshot_index < len(result.times)
        ):
            raise ValueError(
                "Rejected a dose result from a different source run or snapshot."
            )
        self.dose = dose

    def set_trajectories(self, data: dict[str, Any]) -> None:
        result = self.active
        if (
            result is None
            or data.get("source_run_id") != result.run_id
            or data.get("config_hash") != result.config.config_hash
            or not 0 <= int(data["snapshot_index"]) < len(result.times)
        ):
            raise ValueError(
                "Rejected trajectories from another source run or snapshot."
            )
        self.trajectories = data

    def set_comparison(self, comparison: RunResult | None) -> None:
        self.comparison = comparison
        self.comparison_difference = None
        result = self.active
        if (
            result is None
            or comparison is None
            or result.points.shape != comparison.points.shape
            or not np.allclose(result.points, comparison.points, rtol=1e-10, atol=1e-12)
        ):
            return
        left_times, right_times = result.times, comparison.times
        if len(right_times) == 1:
            upper = lower = np.zeros(len(left_times), dtype=int)
            weight = np.zeros(len(left_times))
        else:
            upper = np.clip(
                np.searchsorted(right_times, left_times, side="right"),
                1,
                len(right_times) - 1,
            )
            lower = upper - 1
            weight = np.clip(
                (left_times - right_times[lower])
                / (right_times[upper] - right_times[lower]),
                0.0,
                1.0,
            )
        differences = {}
        for key, values in result.fields.items():
            other = comparison.fields.get(key)
            if other is not None and values.shape[1:] == other.shape[1:]:
                fraction = weight.reshape((-1,) + (1,) * (values.ndim - 1))
                differences[key] = values - (
                    (1 - fraction) * other[lower] + fraction * other[upper]
                )
        self.comparison_difference = differences

    def display_selection(self) -> DisplaySelection:
        """Resolve combobox choices once for both scenes and linked charts."""
        if self.active is None:
            return DisplaySelection()
        if self.source == "Run difference" and self.comparison_difference is not None:
            return DisplaySelection(
                self.comparison_difference,
                True,
                "Run difference · interpolated display",
            )
        if self.source in {"PINN", "PINN error"} and self.pinn is not None:
            status = str(self.pinn.metadata.get("validation_status", "unvalidated"))
            if self.source == "PINN":
                return DisplaySelection(
                    self.pinn.fields, False, f"PINN · held-out {status}"
                )
            if self.pinn_difference is None:
                self.pinn_difference = {
                    key: value - self.active.fields[key]
                    for key, value in self.pinn.fields.items()
                    if key in self.active.fields
                    and value.shape == self.active.fields[key].shape
                }
            return DisplaySelection(
                self.pinn_difference, True, f"PINN − numerical · held-out {status}"
            )
        return DisplaySelection()
