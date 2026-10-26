"""Conservative workspace training support; research training remains separate.

These checks describe supported app scenarios, not scientific qualification of
an individual trained model.  They do not import Torch or change backend gates.
"""

from __future__ import annotations

import numpy as np

from ..models import RunResult


_RESULTS_AVAILABLE = " Numerical results remain available."
_GRID_REASON = (
    "PINN training requires a finite source grid with at least two accepted times and one point."
    + _RESULTS_AVAILABLE
)
_THERMAL_REASON = (
    "Phase-changing thermal PINN training is not qualified for this workspace."
    + _RESULTS_AVAILABLE
)
_COUPLED_REASON = (
    "Evolving coupled MHD PINN training is not qualified for this workspace."
    + _RESULTS_AVAILABLE
)


def _field(
    result: RunResult, name: str, width: int, shape: tuple[int, int]
) -> np.ndarray | None:
    try:
        value = np.asarray(result.fields[name], dtype=np.float64)
    except (KeyError, TypeError, ValueError):
        return None
    if width == 1 and value.shape == (*shape, 1):
        value = value[..., 0]
    expected = shape if width == 1 else (*shape, width)
    return value if value.shape == expected and np.isfinite(value).all() else None


def pinn_training_unavailable_reason(result: RunResult) -> str | None:
    """Explain an unsupported workspace scenario, or return ``None``.

    Pure-phase thermal histories may evolve.  Coupled histories must remain
    stationary at floating-point roundoff, including exterior magnetic fields.
    The initial elliptic pressure has no prescribed dynamical initial state and
    is excluded from the stationarity comparison.
    """
    try:
        times = np.asarray(result.times, dtype=np.float64)
        points = np.asarray(result.points, dtype=np.float64)
    except (AttributeError, TypeError, ValueError):
        return _GRID_REASON
    if (
        times.ndim != 1
        or len(times) < 2
        or not np.isfinite(times).all()
        or np.any(np.diff(times) <= 0)
        or points.ndim != 2
        or points.shape[1] != 3
        or len(points) == 0
        or not np.isfinite(points).all()
    ):
        return _GRID_REASON
    shape = (len(times), len(points))
    if result.config.solver not in {"thermal_benchmark", "fenics"}:
        return (
            "PINN training is unavailable for this numerical solver."
            + _RESULTS_AVAILABLE
        )
    enthalpy = _field(result, "enthalpy", 1, shape)
    if enthalpy is None:
        return (
            "PINN training requires finite accepted enthalpy fields on the source grid."
            + _RESULTS_AVAILABLE
        )
    for name in ("temperature", "liquid_fraction"):
        if _field(result, name, 1, shape) is None:
            return (
                f"PINN training requires finite accepted {name.replace('_', ' ')} fields on the source grid."
                + _RESULTS_AVAILABLE
            )
    if result.config.solver == "thermal_benchmark":
        latent_heat = result.config.latent_heat
        if not np.isfinite(latent_heat) or latent_heat < 0:
            return (
                "PINN training requires a finite nonnegative latent heat."
                + _RESULTS_AVAILABLE
            )
        return (
            None
            if np.max(enthalpy) <= 0 or np.min(enthalpy) >= latent_heat
            else _THERMAL_REASON
        )

    try:
        conductor = np.asarray(result.metadata["conductor_mask"])
    except (KeyError, TypeError, ValueError):
        conductor = np.empty(0, dtype=bool)
    if (
        conductor.shape != (len(points),)
        or conductor.dtype.kind != "b"
        or not conductor.any()
    ):
        return (
            "PINN training requires a valid nonempty conductor mask on the source grid."
            + _RESULTS_AVAILABLE
        )
    fields = {"enthalpy": enthalpy}
    for name, width in (
        ("velocity", 3),
        ("magnetic_field", 3),
        ("vector_potential", 3),
        ("pressure", 1),
        ("electric_potential", 1),
    ):
        value = _field(result, name, width, shape)
        if value is None:
            return (
                f"PINN training requires finite accepted {name.replace('_', ' ')} fields on the source grid."
                + _RESULTS_AVAILABLE
            )
        fields[name] = value
    tolerance_factor = 64 * np.finfo(np.float64).eps
    for name, value in fields.items():
        if name not in {"magnetic_field", "vector_potential"}:
            value = value[:, conductor]
        if name == "pressure":
            value = value[1:]
        first = value[0]
        tolerance = tolerance_factor * np.maximum(1.0, np.abs(first))
        if np.any(np.abs(value - first) > tolerance):
            return _COUPLED_REASON
    return None
