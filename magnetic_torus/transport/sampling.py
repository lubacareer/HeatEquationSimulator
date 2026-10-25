"""Versioned source spectra, incident geometry, and deterministic primary sampling."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any
import numpy as np
from ..models import ScenarioConfig
from .common import SourceParticles


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
