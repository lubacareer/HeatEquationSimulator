"""Training-label-only weak normalization for explicit research experiments.

The default weak residual and final qualification keep their existing oracle.
No Torch runtime or worker is imported by this fitter.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..models import RunResult


def fit_training_weak_scales(result: RunResult, train_ids: Any) -> dict[str, float]:
    """Fit U/B/A amplitude scales using only the given point-time training IDs.

    Velocity labels are restricted to the conductor; magnetic field and vector
    potential labels include the exterior.  Configuration-derived lower bounds
    match the original FE weak residual normalization.
    """
    nt, nn = len(result.times), len(result.points)
    ids = np.asarray(train_ids)
    if (ids.ndim != 1 or not ids.size or ids.dtype.kind not in "iu"
            or np.any(ids < 0) or np.any(ids >= nt * nn)
            or len(np.unique(ids)) != len(ids)):
        raise ValueError("Weak scaling requires unique, nonempty, in-range integer training IDs.")
    conductor = np.asarray(result.metadata.get("conductor_mask", np.ones(nn, dtype=bool)))
    if conductor.shape != (nn,) or conductor.dtype.kind != "b" or not conductor.any():
        raise ValueError("Weak scaling requires a nonempty boolean conductor mask.")
    maxima = {}
    for name in ("velocity", "magnetic_field", "vector_potential"):
        try:
            values = np.asarray(result.fields[name], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Weak scaling requires source {name} vector fields.") from error
        if values.shape != (nt, nn, 3):
            raise ValueError(f"Weak scaling requires {name} fields on the accepted source grid.")
        maximum, count = 0., 0
        for offset in range(0, len(ids), 4096):
            ti, pi = np.divmod(ids[offset:offset + 4096], nn)
            valid = conductor[pi] if name == "velocity" else np.ones(len(pi), dtype=bool)
            labels = values[ti[valid], pi[valid]]
            if not len(labels):
                continue
            if not np.isfinite(labels).all():
                raise ValueError("Weak scaling training labels must be finite.")
            current = float(np.max(np.linalg.norm(labels, axis=-1)))
            if not math.isfinite(current):
                raise ValueError("Weak scaling training amplitudes must be finite.")
            maximum, count = max(maximum, current), count + len(labels)
        if not count:
            raise ValueError("Weak scaling has no conductor velocity training labels.")
        maxima[name] = maximum
    config = result.config
    velocity = max(maxima["velocity"], math.sqrt(abs(config.pump_force) * config.minor_radius / config.density), 1.e-4)
    magnetic = max(maxima["magnetic_field"], 1.e-6)
    vector_potential = max(maxima["vector_potential"], magnetic * config.minor_radius)
    return {"velocity": velocity, "magnetic_field": magnetic, "vector_potential": vector_potential}
