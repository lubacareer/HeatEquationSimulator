"""Relativistic trajectory diagnostics and genuine Geant4 dose transport.

The Boris viewer reports geometric intersections, never absorbed dose. Dose is
published only after the separately compiled Geant4 worker completes both paired
field-on and field-off ensembles. All quantities at the Python boundary are SI.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
import uuid
from io import BytesIO

import numpy as np

from .models import DoseResult, PINNResult, Progress, RunResult, ScenarioConfig, SimulationCancelled
from .artifacts import FieldArtifact

C = 299792458.0
ELEMENTARY_CHARGE = 1.602176634e-19
ATOMIC_MASS = 1.66053906892e-27
PROTON_MASS = 1.67262192595e-27
MEV = ELEMENTARY_CHARGE * 1e6


def _cancelled(cancel: Any) -> bool:
    return bool(cancel() if callable(cancel) else cancel is not None and cancel.is_set())


def _check(cancel: Any) -> None:
    if _cancelled(cancel):
        raise SimulationCancelled("Radiation calculation cancelled; previous results retained.")


@dataclass(frozen=True)
class TrajectoryResult:
    source_run_id: str
    config_hash: str
    snapshot_index: int
    paths: np.ndarray
    baseline_paths: np.ndarray
    times: np.ndarray
    species: np.ndarray
    energies_mev_nucleon: np.ndarray
    outcomes: tuple[str, ...]
    baseline_outcomes: tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("paths", "baseline_paths", "times", "species", "energies_mev_nucleon"):
            value = np.array(getattr(self, name), copy=True)
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        if self.paths.ndim != 3 or self.paths.shape[2] != 3 or self.baseline_paths.shape != self.paths.shape:
            raise ValueError("Paired trajectories require matching (particle, frame, 3) arrays.")
        count, frames = self.paths.shape[:2]
        if self.times.shape != (count, frames) or self.species.shape != (count, 2) or self.energies_mev_nucleon.shape != (count,):
            raise ValueError("Trajectory times/species/energies do not match the particle paths.")
        if len(self.outcomes) != count or len(self.baseline_outcomes) != count:
            raise ValueError("Every primary requires field-on and field-off outcomes.")
        if any(not np.all(np.isfinite(getattr(self, key))) for key in ("paths", "baseline_paths", "times", "energies_mev_nucleon")):
            raise ValueError("Trajectory arrays must be finite.")


@dataclass(frozen=True)
class SourceParticles:
    positions: np.ndarray
    directions: np.ndarray
    z: np.ndarray
    a: np.ndarray
    energy_mev_nucleon: np.ndarray
    source_area_m2: float
    metadata: Mapping[str, Any] = field(default_factory=dict)


def load_spectrum(path: str | Path) -> dict[str, Any]:
    """Read explicit discrete spectral bins; weights are bin-integrated counts.

    JSON v1: {schema_version:1, energy_unit:'MeV/nucleon',
    angular_distribution:'beam'|'isotropic', provenance:'...', bins:[
    {z:1,a:1,energy_mev_nucleon:100,weight:1}, ...]}.
    Differential spectra must be integrated into bins by the data provider.
    """
    content = Path(path).read_bytes()
    data = json.loads(content)
    if data.get("schema_version") != 1 or data.get("energy_unit") != "MeV/nucleon":
        raise ValueError("Spectrum requires schema_version 1 and energy_unit 'MeV/nucleon'.")
    if data.get("angular_distribution") not in {"beam", "isotropic"} or not data.get("provenance"):
        raise ValueError("Spectrum requires angular_distribution and provenance.")
    bins = data.get("bins", [])
    if not bins:
        raise ValueError("Spectrum must contain at least one discrete weighted bin.")
    for item in bins:
        z, a = item.get("z"), item.get("a")
        if isinstance(z, bool) or isinstance(a, bool) or not isinstance(z, int) or not isinstance(a, int) or not 1 <= z <= 28 or a < z:
            raise ValueError("Spectrum ions require integer 1 <= Z <= 28 and A >= Z.")
        for key in ("energy_mev_nucleon", "weight"):
            if not isinstance(item.get(key), (float, int)) or not math.isfinite(item[key]) or item[key] <= 0:
                raise ValueError(f"Spectrum {key} must be finite and positive.")
    data["sha256"] = hashlib.sha256(content).hexdigest()
    return data


def _ground_top(config: ScenarioConfig) -> float:
    """The same tangent ground plane used by Geant4's finite world geometry."""
    return -max(config.minor_radius + config.vessel_thickness,
                config.habitat_radius + config.habitat_thickness) - .01


def _isotropic_geometry(rng: np.random.Generator, count: int, radius: float) -> tuple[np.ndarray, np.ndarray]:
    """Uniform sphere and cosine-law incidence for an unobstructed 4π field."""
    normal = rng.normal(size=(count, 3))
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    tangent = np.cross(normal, np.where((np.abs(normal[:, 2]) < .9)[:, None],
                                       (0., 0., 1.), (0., 1., 0.)))
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    other = np.cross(normal, tangent)
    mu = np.sqrt(rng.random(count))
    phi = rng.uniform(0, 2 * np.pi, count)
    directions = -normal * mu[:, None] + np.sqrt(1 - mu ** 2)[:, None] * (
        tangent * np.cos(phi)[:, None] + other * np.sin(phi)[:, None])
    return normal * radius, directions


def _above_ground_geometry(rng: np.random.Generator, count: int, radius: float,
                           ground_top: float, impact: float | None) -> tuple[np.ndarray, np.ndarray, float]:
    """Conditional incident vacuum samples and their exact projected measure.

    Beam fluence remains per perpendicular area. Isotropic fluence remains the
    unobstructed 4π scalar fluence, with the ground blocking the upward sky.
    For a cap n_z>=c the downward-sky irradiance is π(1+n_z)/2; integrating
    intensity Φ/(4π) over the cap gives A=πR²(3-2c-c²)/8. Conditioning directions
    without this acceptance factor would overestimate the delivered fluence.
    """
    if impact is not None:
        chord = float(np.clip(ground_top / impact, -1., 1.))
        area = impact ** 2 * (math.acos(chord) - chord * math.sqrt(max(0., 1 - chord ** 2)))
    else:
        cap = float(np.clip(ground_top / radius, -1., 1.))
        area = math.pi * radius ** 2 * (3 - 2 * cap - cap ** 2) / 8
    if area <= 0:
        raise ValueError("The incident source has no exposed vacuum region above the ground.")
    positions, directions = [], []
    remaining = count
    while remaining:
        # This ground plane is below the origin, so acceptance is at least
        # 1/2 for the beam and 3/8 for the original isotropic sphere measure.
        size = min(65536, max(32, 3 * remaining))
        if impact is not None:
            angle = rng.uniform(0, 2 * np.pi, size)
            rr = impact * np.sqrt(rng.random(size))
            yy, zz = rr * np.cos(angle), rr * np.sin(angle)
            point = np.column_stack((np.sqrt(radius ** 2 - rr ** 2), yy, zz))
            direction = np.tile((-1., 0., 0.), (size, 1))
            exposed = zz >= ground_top
        else:
            point, direction = _isotropic_geometry(rng, size, radius)
            exposed = (point[:, 2] >= ground_top) & (direction[:, 2] <= 0.)
        chosen = np.flatnonzero(exposed)[:remaining]
        positions.append(point[chosen]); directions.append(direction[chosen])
        remaining -= len(chosen)
    return np.concatenate(positions), np.concatenate(directions), float(area)


def sample_source(config: ScenarioConfig, count: int, radius: float) -> SourceParticles:
    config.validate()
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("Incident primary count must be a positive integer.")
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("Incident source radius must be finite and positive.")
    rng = np.random.default_rng(config.seed)
    z = np.full(count, config.particle_z, dtype=int)
    a = np.full(count, config.particle_a, dtype=int)
    energy = np.full(count, config.particle_energy_mev_nucleon, dtype=float)
    angular = config.source_kind
    source_metadata = {"source_kind": config.source_kind, "energy_convention": "MeV/nucleon",
                       "seed": config.seed, "provenance": "Monoenergetic benchmark primary source"}
    if angular == "spectrum":
        if not config.spectrum_path:
            raise ValueError("Select a spectrum JSON file before running radiation transport.")
        spectrum = load_spectrum(config.spectrum_path)
        bins = spectrum["bins"]
        weight = np.array([b["weight"] for b in bins])
        selected = rng.choice(len(bins), count, p=weight / weight.sum())
        z = np.array([bins[i]["z"] for i in selected])
        a = np.array([bins[i]["a"] for i in selected])
        energy = np.array([bins[i]["energy_mev_nucleon"] for i in selected])
        angular = spectrum["angular_distribution"]
        source_metadata.update(spectrum_sha256=spectrum["sha256"], provenance=spectrum["provenance"])
    if angular == "beam":
        # Uniform projected disk on the interior source sphere, incident along -x.
        impact = min(radius * .60, 1.15 * (config.major_radius + config.minor_radius))
        if config.ground_enabled:
            positions, directions, area = _above_ground_geometry(rng, count, radius, _ground_top(config), impact)
        else:
            angle = rng.uniform(0, 2 * np.pi, count)
            rr = impact * np.sqrt(rng.random(count))
            y, zz = rr * np.cos(angle), rr * np.sin(angle)
            positions = np.column_stack((np.sqrt(radius ** 2 - rr ** 2), y, zz))
            directions = np.tile((-1., 0., 0.), (count, 1))
            area = np.pi * impact ** 2
        source_metadata["fluence_convention"] = "fluence per projected area normal to beam"
    else:
        if config.ground_enabled:
            positions, directions, area = _above_ground_geometry(rng, count, radius, _ground_top(config), None)
        else:
            positions, directions = _isotropic_geometry(rng, count, radius)
            area = np.pi * radius ** 2
        source_metadata["fluence_convention"] = "unobstructed 4pi scalar fluence"
    if config.ground_enabled:
        source_metadata.update(ground_top_m=_ground_top(config), source_boundary="above-ground vacuum",
                               ground_normalization="exact exposed projected incident measure", sampling_version=2)
        if angular == "isotropic":
            source_metadata["ground_horizon"] = "downward incident directions only; no injected ground albedo"
    source_metadata["angular_distribution"] = angular
    return SourceParticles(positions, directions, z, a, energy, float(area), MappingProxyType(source_metadata))


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
        from .physics import coil_magnetic_field, imposed_coil
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


def relativistic_boris(momentum: np.ndarray, charge: np.ndarray, mass: np.ndarray,
                       magnetic: np.ndarray, dt: float) -> np.ndarray:
    """Magnetic-only Boris rotation, preserving relativistic kinetic energy."""
    p = np.atleast_2d(momentum)
    m = np.asarray(mass).reshape(-1, 1)
    q = np.asarray(charge).reshape(-1, 1)
    gamma = np.sqrt(1 + np.sum(p * p, axis=1, keepdims=True) / (m * C) ** 2)
    rotation = q * np.atleast_2d(magnetic) * dt / (2 * m * gamma)
    prime = p + np.cross(p, rotation)
    return p + np.cross(prime, 2 * rotation / (1 + np.sum(rotation ** 2, axis=1, keepdims=True)))


def _intersection(points: np.ndarray, config: ScenarioConfig) -> np.ndarray:
    radial = np.linalg.norm(points[:, :2], axis=1)
    tube = np.sqrt((radial - config.major_radius) ** 2 + points[:, 2] ** 2)
    hit = np.full(len(points), "", dtype="<U10")
    hit[tube <= config.minor_radius + config.vessel_thickness] = "hardware"
    wire = max(config.vessel_thickness / 2, .002 * config.minor_radius)
    coil_radius = config.major_radius + config.minor_radius + 2 * max(config.vessel_thickness, .01 * config.minor_radius)
    coil_distance = np.sqrt((radial - coil_radius) ** 2 + points[:, 2] ** 2)
    hit[coil_distance <= wire] = "hardware"
    # Same spherical habitat as the transport geometry.
    hit[np.linalg.norm(points, axis=1) <= config.habitat_radius + config.habitat_thickness] = "habitat"
    if config.ground_enabled:
        hit[points[:, 2] < _ground_top(config)] = "ground"
    return hit


def trace_particles(result: RunResult, snapshot_index: int, progress: Progress,
                    cancel: Any) -> TrajectoryResult:
    config = result.config
    if not 0 <= snapshot_index < len(result.times):
        raise ValueError("Select an accepted snapshot before tracing particles.")
    field_map = (AnalyticCoilField(config) if config.solver == "thermal_benchmark"
                 and "exterior_field_path" not in result.artifacts else frozen_field(result, snapshot_index))
    source = sample_source(config, config.particle_count, field_map.radius * .85)
    mass = np.where(source.a == 1, PROTON_MASS, source.a * ATOMIC_MASS)
    gamma = 1 + source.energy_mev_nucleon * source.a * MEV / (mass * C ** 2)
    speed = C * np.sqrt(1 - gamma ** -2)
    momentum = source.directions * (gamma * mass * speed)[:, None]
    # Spatial steps resolve the tube/wall; gyro-angle <= .025 rad per step.
    b_values = field_map.cell_values if field_map.cell_values is not None else field_map.values
    bmax = max(float(np.max(np.linalg.norm(b_values, axis=1))), 1e-30)
    dt = min(config.minor_radius / (12 * speed.max()), float(np.min(.025 * gamma * mass / (source.z * ELEMENTARY_CHARGE * bmax))))
    max_time = 4 * field_map.radius / speed.min()
    steps = int(np.ceil(max_time / dt))
    if steps > 100000:
        raise RuntimeError("Particle gyration requires over 100000 steps. Reduce field/current or increase particle energy for trajectory preview.")
    stride = max(1, int(np.ceil(steps / 1200)))
    paths, baselines, times = [], [], []
    sample_times = [0.]
    sample_times.extend(step * dt for step in range(stride, steps + 1, stride))
    if steps % stride:
        sample_times.append(steps * dt)
    all_outcomes = []
    for mode in (True, False):
        p = momentum.copy()
        positions = source.positions.copy()
        outcomes = np.full(config.particle_count, "", dtype="<U10")
        history = [positions.copy()]
        for step in range(steps):
            if step % 32 == 0:
                _check(cancel)
                progress(((0 if mode else .5) + .5 * step / steps), "Tracing paired relativistic particles")
            active = np.flatnonzero(outcomes == "")
            if not len(active):
                if (step + 1) % stride == 0:
                    history.append(positions.copy())
                continue
            g = np.sqrt(1 + np.sum(p[active] ** 2, axis=1) / (mass[active] * C) ** 2)
            old_v = p[active] / (mass[active] * g)[:, None]
            if mode:
                midpoint = positions[active] + .5 * old_v * dt
                p[active] = relativistic_boris(p[active], source.z[active] * ELEMENTARY_CHARGE,
                                             mass[active], field_map(midpoint), dt)
            new_v = p[active] / (mass[active] * g)[:, None]
            old = positions[active].copy()
            positions[active] += .5 * (old_v + new_v) * dt
            # Subsample each segment to avoid skipping thin habitat/vessel walls.
            detected = np.full(len(active), "", dtype="<U10")
            for fraction in (.25, .5, .75, 1.):
                hits = _intersection(old + fraction * (positions[active] - old), config)
                first = (detected == "") & (hits != "")
                detected[first] = hits[first]
            outcomes[active] = detected
            escaped = np.linalg.norm(positions[active], axis=1) >= .90 * field_map.radius
            outcomes[active[escaped & (detected == "")]] = "escaped"
            if (step + 1) % stride == 0:
                history.append(positions.copy())
        if steps % stride:
            history.append(positions.copy())
        outcomes[outcomes == ""] = "window_end"
        array = np.stack(history, axis=1)
        if mode:
            paths = array
        else:
            baselines = array
        all_outcomes.append(tuple(str(x) for x in outcomes))
    # Pad stopped ensembles so the GUI can share a physical-flight timeline.
    width = max(paths.shape[1], baselines.shape[1])
    paths = np.pad(paths, ((0, 0), (0, width - paths.shape[1]), (0, 0)), mode="edge")
    baselines = np.pad(baselines, ((0, 0), (0, width - baselines.shape[1]), (0, 0)), mode="edge")
    times = np.tile(sample_times, (config.particle_count, 1))
    progress(1., "Particle trajectories complete")
    return TrajectoryResult(result.run_id, config.config_hash, snapshot_index, paths, baselines, times,
                            np.column_stack((source.z, source.a)), source.energy_mev_nucleon,
                            all_outcomes[0], all_outcomes[1],
                            {"method": "relativistic Boris; geometric first-contact diagnostic", "electric_field": "zero",
                             "field_interpolation": field_map.field_mode,
                             "field_source": "imposed Biot-Savart coil only; no fluid induction" if isinstance(field_map, AnalyticCoilField) else "accepted FE total vector potential",
                             "field_sha256": field_map.fingerprint, "field_time_s": float(result.times[snapshot_index]),
                             "selected_source": result.metadata.get("field_source", "numerical"),
                             "source_model_id": result.metadata.get("source_model_id"),
                             "source_model_field_sha256": result.metadata.get("source_model_field_sha256"),
                             "source_model_validation_status": result.metadata.get("source_model_validation_status"),
                             "source_area_m2": source.source_area_m2, "seed": config.seed,
                             "source": dict(source.metadata),
                             "flight_window_s": max_time, "window_end_note": "No first contact or escape within the finite integration window; permanent trapping is not established.",
                             "dose_available": False, "mass_model": "proton mass; A times atomic mass unit for ions"})


def _linux_path(path: Path) -> str:
    resolved = path.resolve()
    if os.name != "nt":
        return str(resolved)
    return "/mnt/" + resolved.drive[0].lower() + str(resolved)[2:].replace("\\", "/")


def _wsl_distro() -> str:
    return os.environ.get("MAGNETIC_TORUS_WSL_DISTRO", os.environ.get("TORUS_WSL_DISTRO", "Ubuntu"))


def _geant4_command(request_path: Path) -> list[str]:
    binary = os.environ.get("TORUS_GEANT4_WORKER", "")
    if os.name == "nt":
        executable = binary or "/opt/magnetic-torus/bin/torus-dose-run"
        return ["wsl", "-d", _wsl_distro(), "--", executable, _linux_path(request_path)]
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
               "field_file": _linux_path(map_path), "primary_file": _linux_path(source_path),
               "output_file": _linux_path(output_path), "field_sha256": field_map.fingerprint,
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
               "phantom_path": _linux_path(Path(config.phantom_path)) if config.phantom_path else "/opt/magnetic-torus/data/ICRPdata",
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
                    subprocess.run(["wsl", "-d", _wsl_distro(), "--",
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
