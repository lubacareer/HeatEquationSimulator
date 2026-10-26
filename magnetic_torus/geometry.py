"""Geometry queries shared by the workspace and renderer without GUI imports."""
from __future__ import annotations

import numpy as np

from .models import RunResult


def conductor_mask(result: RunResult) -> np.ndarray:
    supplied = result.metadata.get("conductor_mask")
    if supplied is not None and np.asarray(supplied).shape == (len(result.points),):
        return np.asarray(supplied, dtype=bool)
    radius = np.hypot(result.points[:, 0], result.points[:, 1])
    return (radius - result.config.major_radius) ** 2 + result.points[:, 2] ** 2 <= result.config.minor_radius ** 2 * 1.001
