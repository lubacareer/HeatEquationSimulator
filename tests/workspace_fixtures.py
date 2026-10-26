"""Small accepted-field fixtures for interface tests, never solver benchmarks."""

from dataclasses import replace

import numpy as np

from magnetic_torus.models import RunResult, ScenarioConfig


def accepted_run(
    run_id="workspace-fixture", times=(0.0, 0.2, 0.55, 1.0, 2.0), *, shift=0.0
):
    config = replace(
        ScenarioConfig(),
        solver="thermal_benchmark",
        turbulence=False,
        minor_radius=0.35,
        duration=max(float(times[-1]), 0.1),
    )
    points = np.array(
        [
            [0.8, -0.2, -0.2],
            [1.2, -0.2, -0.2],
            [1.0, 0.2, -0.2],
            [1.0, 0.0, 0.2],
            [1.5, 0.3, 0.3],
        ]
    )
    t = np.asarray(times, dtype=float)
    temperature = 1800 + t[:, None] * 2 + points[None, :, 0] + shift
    vector = np.zeros((len(t), len(points), 3))
    velocity = vector.copy()
    velocity[:, :, 0] = 0.001
    magnetic = vector.copy()
    magnetic[:, :, 2] = 0.1
    potential = vector.copy()
    potential[:, :, 0] = -0.05 * points[:, 1]
    potential[:, :, 1] = 0.05 * points[:, 0]
    fields = {
        "temperature": temperature,
        "enthalpy": config.heat_capacity_solid
        * (temperature - config.melting_temperature),
        "liquid_fraction": np.zeros_like(temperature),
        "velocity": velocity,
        "magnetic_field": magnetic,
        "vector_potential": potential,
        "pressure": np.zeros_like(temperature),
        "electric_potential": np.zeros_like(temperature),
    }
    diagnostics = {
        "recoverable_energy_j": 10000 + 20 * t,
        "energy_j": 10000 + 20 * t,
        "liquid_fraction": np.zeros_like(t),
        "net_power_w": np.full_like(t, 20.0),
        "pump_power_w": np.zeros_like(t),
        "magnetic_energy_j": 1 + t,
    }
    return RunResult(
        run_id,
        config,
        t,
        points,
        fields,
        diagnostics,
        {
            "backend": "Analytic interface fixture; not a numerical solve",
            "cells": [[0, 1, 2, 3], [1, 2, 3, 4]],
            "conductor_cells": [[0, 1, 2, 3]],
            "conductor_mask": [True, True, True, True, False],
        },
    )
