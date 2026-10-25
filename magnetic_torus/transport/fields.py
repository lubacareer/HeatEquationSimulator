"""Immutable frozen FE fields and imposed-coil trajectory diagnostics."""
from __future__ import annotations

import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path
import numpy as np
from ..models import PINNResult, RunResult, ScenarioConfig
from ..artifacts import FieldArtifact


class FrozenField:
    """Exact element curl of accepted global P1 magnetic vector potential.

    Original tetrahedra are used instead of rebuilding a Delaunay mesh. No
    nearest-neighbour or zero-field extrapolation is permitted outside the map.
    """
    def __init__(self, points: np.ndarray, cells: np.ndarray, values: np.ndarray,
                 radius: float, fingerprint: str = "", vector_potential: np.ndarray | None = None):
        from scipy.spatial import cKDTree
        self.points = np.asarray(points, float)
        self.cells = np.asarray(cells, int)
        self.values = np.asarray(values, float)
        self.radius = radius
        self.fingerprint = fingerprint
        if self.points.ndim != 2 or self.points.shape[1] != 3 or self.cells.ndim != 2 or self.cells.shape[1] != 4:
            raise ValueError("Exterior magnetic field requires points(N,3) and tetrahedra(M,4).")
        if self.values.shape != self.points.shape or not np.all(np.isfinite(self.values)):
            raise ValueError("Exterior magnetic field is missing or contains invalid values.")
        vertices = self.points[self.cells]
        self.origins = vertices[:, 0]
        self.inverse = np.linalg.inv(np.moveaxis(vertices[:, 1:] - vertices[:, :1], 1, 2))
        self.cell_values = None
        self.field_mode = "nodal_b"
        if vector_potential is not None:
            potential = np.asarray(vector_potential, float)
            if potential.shape != self.points.shape or not np.all(np.isfinite(potential)):
                raise ValueError("Accepted vector potential must have finite values of shape (N,3).")
            differences = potential[self.cells][:, 1:] - potential[self.cells][:, :1]
            gradient = np.einsum("nji,njk->nik", differences, self.inverse)
            self.cell_values = np.column_stack((gradient[:, 2, 1] - gradient[:, 1, 2],
                                                gradient[:, 0, 2] - gradient[:, 2, 0],
                                                gradient[:, 1, 0] - gradient[:, 0, 1]))
            self.field_mode = "cell_curl_a"
        self.tree = cKDTree(vertices.mean(axis=1))

    def __call__(self, locations: np.ndarray) -> np.ndarray:
        locations = np.atleast_2d(locations)
        result = np.empty_like(locations, dtype=float)
        candidates = min(32, len(self.cells))
        _, nearest = self.tree.query(locations, k=candidates)
        nearest = np.asarray(nearest).reshape(len(locations), candidates)
        unresolved = np.ones(len(locations), bool)
        for k in range(candidates):
            idx = np.flatnonzero(unresolved)
            if not len(idx):
                break
            cell = nearest[idx, k]
            tail = np.einsum("nij,nj->ni", self.inverse[cell], locations[idx] - self.origins[cell])
            weights = np.column_stack((1 - tail.sum(axis=1), tail))
            inside = np.all(weights >= -1e-8, axis=1) & np.all(weights <= 1 + 1e-8, axis=1)
            valid = idx[inside]
            result[valid] = (self.cell_values[cell[inside]] if self.cell_values is not None else
                             np.einsum("ni,nij->nj", weights[inside], self.values[self.cells[cell[inside]]]))
            unresolved[valid] = False
        # Highly graded meshes may have a distant cell centroid. Exhaustive
        # fallback is slow but deterministic, and never fabricates exterior B.
        for i in np.flatnonzero(unresolved):
            tail = np.einsum("nij,nj->ni", self.inverse, locations[i] - self.origins)
            weights = np.column_stack((1 - tail.sum(axis=1), tail))
            inside = np.flatnonzero(np.all(weights >= -1e-8, axis=1) & np.all(weights <= 1 + 1e-8, axis=1))
            if not len(inside):
                raise ValueError("Particle/field probe is outside the accepted exterior FE mesh. Increase the exterior domain and rerun.")
            cell = inside[0]
            result[i] = self.cell_values[cell] if self.cell_values is not None else weights[cell] @ self.values[self.cells[cell]]
        return result


def frozen_field(result: RunResult, snapshot_index: int) -> FrozenField:
    if not 0 <= snapshot_index < len(result.times):
        raise ValueError("Select an accepted field snapshot before tracing particles.")
    try:
        payload = result.artifact_bytes("exterior_field_path")
    except ValueError as error:
        raise RuntimeError("This run has no exterior magnetic field artifact. Run the coupled FEniCS solver before tracing particles or calculating dose.") from error
    with np.load(BytesIO(payload), allow_pickle=False) as data:
        if "vector_potential" not in data:
            raise RuntimeError("This field artifact predates exact curl(A) transport. Rerun the FEniCS solver to regenerate the exterior vector potential.")
        times = data["times"]
        if len(times) != len(result.times) or not np.allclose(times, result.times):
            raise ValueError("Exterior field timeline does not match the accepted run.")
        radius = float(result.metadata.get("exterior_radius_m", result.config.exterior_radius_factor * (result.config.major_radius + result.config.minor_radius)))
        return FrozenField(data["points"], data["cells"], data["magnetic_field"][snapshot_index], radius,
                           hashlib.sha256(payload).hexdigest(), data["vector_potential"][snapshot_index])


def field_source_run(result: RunResult, pinn: PINNResult, output_dir: Path) -> RunResult:
    """Bind a selected PINN to its originating FE geometry and accepted times.

    This creates a transport-only view of the originating run, retaining its ID
    while recording the selected model's ID/hash in every derived artifact.
    No numerical exterior field is substituted for a missing PINN prediction.
    """
    if pinn.source_run_id != result.run_id or pinn.config_hash != result.config.config_hash:
        raise ValueError("PINN belongs to another scenario/run; select its originating accepted result.")
    if result.config.solver != "fenics":
        raise ValueError("Coupled PINN radiation transport requires the full global FE mesh.")
    potential = pinn.fields.get("vector_potential")
    expected = (len(result.times), len(result.points), 3)
    if potential is None or potential.shape != expected or not np.all(np.isfinite(potential)):
        raise ValueError("PINN has no finite vector-potential prediction on the complete accepted global FE mesh.")
    cells = result.metadata.get("cells")
    if cells is None:
        with np.load(BytesIO(result.artifact_bytes("exterior_field_path")), allow_pickle=False) as exterior:
            if not np.allclose(exterior["points"], result.points):
                raise ValueError("PINN/global FE point ordering is inconsistent.")
            cells = exterior["cells"]
    cells = np.asarray(cells, dtype=np.int64)
    magnetic = pinn.fields.get("magnetic_field")
    if magnetic is None or magnetic.shape != expected or not np.all(np.isfinite(magnetic)):
        raise ValueError("PINN magnetic display projection is invalid.")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    model_tag = hashlib.sha256(pinn.run_id.encode()).hexdigest()[:16]
    path = directory / f"exterior-pinn-{model_tag}.npz"
    temporary = path.with_suffix(".npz.partial")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, points=result.points, cells=cells, times=result.times,
                            vector_potential=potential, magnetic_field=magnetic)
    os.replace(temporary, path)
    potential_hash = hashlib.sha256(np.ascontiguousarray(potential).tobytes()).hexdigest()
    metadata = {**dict(result.metadata), "exterior_field_path": str(path.resolve()),
                "field_source": "PINN", "source_model_id": pinn.run_id,
                "source_model_field_sha256": potential_hash,
                "source_model_validation_status": pinn.metadata.get("validation_status", "unvalidated"),
                "source_model_metrics": dict(pinn.metrics)}
    return RunResult(result.run_id, result.config, result.times, result.points,
                     pinn.fields, result.diagnostics, metadata,
                     artifacts={**result.artifacts, "exterior_field_path": FieldArtifact.capture(path, root=directory)})


class AnalyticCoilField:
    """Actual line-integrated Biot-Savart imposed ring, diagnostic only."""
    def __init__(self, config: ScenarioConfig):
        from ..numerics.coils import coil_magnetic_field, imposed_coil
        self.config = config
        self.radius = config.exterior_radius_factor * (config.major_radius + config.minor_radius)
        self.field_mode = "analytic_biot_savart"
        self.fingerprint = hashlib.sha256(json.dumps(imposed_coil(config), sort_keys=True).encode()).hexdigest()
        self.evaluator = coil_magnetic_field
        coil = imposed_coil(config)
        theta = np.arange(256) * 2 * math.pi / 256
        probes = np.column_stack(((coil["radius_m"] - coil["wire_radius_m"]) * np.cos(theta),
                                  (coil["radius_m"] - coil["wire_radius_m"]) * np.sin(theta), np.zeros(256)))
        self.values = self(probes)
        self.cell_values = None

    def __call__(self, locations: np.ndarray) -> np.ndarray:
        points = np.atleast_2d(locations)
        if np.any(np.linalg.norm(points, axis=1) > self.radius):
            raise ValueError("Particle is outside the diagnostic source sphere.")
        return self.evaluator(points, self.config)
