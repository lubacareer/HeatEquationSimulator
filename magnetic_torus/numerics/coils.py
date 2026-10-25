"""Prescribed filament-coil geometry and Biot-Savart calculations."""
from __future__ import annotations

import math
from typing import Any
import numpy as np
from ..models import ScenarioConfig
from .constants import MU0


def imposed_coil(config: ScenarioConfig) -> dict[str, Any]:
    return {"kind": "circular_filament_v1", "radius_m": config.major_radius + config.minor_radius
            + 2 * max(config.vessel_thickness, .01 * config.minor_radius),
            "z_m": 0.0, "ampere_turns": config.coil_current * config.coil_turns
            if config.mode == "driven" else 0.0,
            "wire_radius_m": max(config.vessel_thickness / 2, .002 * config.minor_radius),
            "segments": 256}


def coil_vector_potential(points: np.ndarray, config: ScenarioConfig) -> np.ndarray:
    """Line-integrated Biot–Savart vector potential of the explicit external ring."""
    points = np.asarray(points, dtype=float)
    coil = imposed_coil(config)
    n = coil["segments"]
    angles = (np.arange(n) + .5) * 2 * math.pi / n
    radius = coil["radius_m"]
    centers = np.column_stack((radius * np.cos(angles), radius * np.sin(angles),
                               np.full(n, coil["z_m"])))
    dl = np.column_stack((-np.sin(angles), np.cos(angles), np.zeros(n))) * (2 * math.pi * radius / n)
    result = np.zeros_like(points)
    for start in range(0, len(points), 512):
        displacement = points[start:start + 512, None] - centers[None]
        distance = np.sqrt(np.sum(displacement ** 2, axis=-1) + coil["wire_radius_m"] ** 2)
        result[start:start + 512] = MU0 * coil["ampere_turns"] / (4 * math.pi) * (1 / distance) @ dl
    return result


def coil_magnetic_field(points: np.ndarray, config: ScenarioConfig) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    coil = imposed_coil(config)
    n = coil["segments"]
    angles = (np.arange(n) + .5) * 2 * math.pi / n
    radius = coil["radius_m"]
    centers = np.column_stack((radius * np.cos(angles), radius * np.sin(angles), np.zeros(n)))
    dl = np.column_stack((-np.sin(angles), np.cos(angles), np.zeros(n))) * (2 * math.pi * radius / n)
    result = np.zeros_like(points)
    for start in range(0, len(points), 512):
        displacement = points[start:start + 512, None] - centers[None]
        distance = (np.sum(displacement ** 2, axis=-1) + coil["wire_radius_m"] ** 2) ** 1.5
        result[start:start + 512] = (MU0 * coil["ampere_turns"] / (4 * math.pi)
                                   * np.sum(np.cross(dl[None], displacement) / distance[..., None], axis=1))
    return result
