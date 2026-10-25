"""Relativistic Boris trajectories and geometric first-contact diagnostics."""
from __future__ import annotations

from typing import Any
import numpy as np
from ..models import Progress, RunResult, ScenarioConfig
from .common import ATOMIC_MASS, C, ELEMENTARY_CHARGE, MEV, PROTON_MASS, TrajectoryResult, _check
from .fields import AnalyticCoilField, frozen_field
from .sampling import _ground_top, sample_source


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
