"""Conservative native thermal transport and sharp-reference material laws."""
from __future__ import annotations

import math
import os
from pathlib import Path
import sys
import threading
from typing import Any
import uuid
import numpy as np
from ..models import Progress, RunResult, ScenarioConfig, SimulationCancelled
from .constants import MU0, STEFAN_BOLTZMANN
from .coils import coil_magnetic_field, coil_vector_potential, imposed_coil


def enthalpy_from_temperature(temperature: Any, config: ScenarioConfig) -> np.ndarray:
    """Sharp-reference enthalpy in J/kg (Tm itself denotes the solid endpoint)."""
    value = np.asarray(temperature, dtype=float)
    delta = value - config.melting_temperature
    return np.where(delta <= 0, config.heat_capacity_solid * delta,
                    config.latent_heat + config.heat_capacity_liquid * delta)


def temperature_from_enthalpy(enthalpy: Any, config: ScenarioConfig) -> np.ndarray:
    value = np.asarray(enthalpy, dtype=float)
    return config.melting_temperature + np.where(
        value < 0, value / config.heat_capacity_solid,
        np.where(value > config.latent_heat,
                 (value - config.latent_heat) / config.heat_capacity_liquid, 0.0))


def liquid_fraction(enthalpy: Any, config: ScenarioConfig) -> np.ndarray:
    value = np.asarray(enthalpy, dtype=float)
    if config.latent_heat == 0:
        return (value > 0).astype(float)
    return np.clip(value / config.latent_heat, 0.0, 1.0)


def regularized_enthalpy(temperature: Any, config: ScenarioConfig) -> np.ndarray:
    """Continuous apparent-heat-capacity law used by the FE temperature solve."""
    delta = np.asarray(temperature, dtype=float) - config.melting_temperature
    width = config.transition_width
    fraction = np.clip(delta / width + .5, 0, 1)
    middle = (config.heat_capacity_solid * delta + config.latent_heat * fraction
              + .5 * (config.heat_capacity_liquid - config.heat_capacity_solid)
              * width * fraction ** 2)
    return np.where(delta < -width / 2, config.heat_capacity_solid * delta,
                    np.where(delta > width / 2,
                             config.latent_heat + config.heat_capacity_liquid * delta, middle))


def _thermal_graph(config: ScenarioConfig) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Orthogonal finite-volume mesh in toroidal coordinates, with periodic angles."""
    nr, nt, np_ = max(2, config.resolution // 2), max(8, config.resolution * 2), max(16, config.resolution * 4)
    dr, da, db = config.minor_radius / nr, 2 * math.pi / nt, 2 * math.pi / np_
    r, a, b = np.meshgrid((np.arange(nr) + .5) * dr,
                          (np.arange(nt) + .5) * da, (np.arange(np_) + .5) * db, indexing="ij")
    major = config.major_radius
    jac = major + r * np.cos(a)
    points = np.column_stack(((jac * np.cos(b)).ravel(), (jac * np.sin(b)).ravel(), (r * np.sin(a)).ravel()))
    # Exact integration of r(R+r cos(theta)) over each curvilinear cell.
    rlo, rhi = r - dr / 2, r + dr / 2
    alo, ahi = a - da / 2, a + da / 2
    volume = db * (major * da * (rhi ** 2 - rlo ** 2) / 2
                   + (np.sin(ahi) - np.sin(alo)) * (rhi ** 3 - rlo ** 3) / 3)
    area = np.zeros_like(r)
    area[-1] = db * config.minor_radius * (major * da + config.minor_radius * (np.sin(ahi[-1]) - np.sin(alo[-1])))
    edge_i, edge_j, conductance = [], [], []
    ids = np.arange(r.size).reshape(r.shape)
    k = config.thermal_conductivity
    for radial in range(nr):
        for theta in range(nt):
            for phi in range(np_):
                i = ids[radial, theta, phi]
                if radial + 1 < nr:
                    rf = (radial + 1) * dr
                    surface = rf * (major + rf * math.cos(a[radial, theta, phi])) * da * db
                    edge_i.append(i)
                    edge_j.append(ids[radial + 1, theta, phi])
                    conductance.append(k * surface / dr)
                edge_i.append(i)
                edge_j.append(ids[radial, (theta + 1) % nt, phi])
                conductance.append(k * (major + r[radial, theta, phi] * math.cos((theta + 1) * da)) * dr * db
                                   / (r[radial, theta, phi] * da))
                edge_i.append(i)
                edge_j.append(ids[radial, theta, (phi + 1) % np_])
                conductance.append(k * r[radial, theta, phi] * dr * da / (jac[radial, theta, phi] * db))
    return points, {"edge_i": np.asarray(edge_i, dtype=int), "edge_j": np.asarray(edge_j, dtype=int),
                    "conductance_w_k": np.asarray(conductance), "node_volume": volume.ravel(),
                    "boundary_area": area.ravel(), "source_weights": area.ravel() / area.sum(),
                    "shape": np.asarray([nr, nt, np_])}


def source_power(config: ScenarioConfig, physical_time: float) -> float:
    return config.heating_power if physical_time <= config.duration * config.charge_fraction else -config.discharge_power


def available_memory_bytes() -> int:
    """Physical-memory headroom without adding a runtime monitoring dependency."""
    try:
        if sys.platform == "win32":
            import ctypes
            class MemoryStatus(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                            ("total_physical", ctypes.c_ulonglong), ("available_physical", ctypes.c_ulonglong),
                            ("total_pagefile", ctypes.c_ulonglong), ("available_pagefile", ctypes.c_ulonglong),
                            ("total_virtual", ctypes.c_ulonglong), ("available_virtual", ctypes.c_ulonglong),
                            ("extended_virtual", ctypes.c_ulonglong)]
            status = MemoryStatus()
            status.length = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.available_physical)
        elif Path("/proc/meminfo").is_file():
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
        return int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (OSError, ValueError, AttributeError):
        return 8 * 1024 ** 3


def native_memory_estimate(config: ScenarioConfig) -> int:
    """Conservative peak estimate: graph, Newton work, snapshots and frozen copies."""
    nr, nt, nphi = max(2, config.resolution // 2), max(8, config.resolution * 2), max(16, config.resolution * 4)
    nodes = nr * nt * nphi
    frames = config.time_steps + 2  # accepted endpoints plus a charging transition
    return int(nodes * (22 * frames * 8 + 1800))


def _native_memory_guard(config: ScenarioConfig) -> dict[str, int]:
    available = available_memory_bytes()
    default_budget = min(8 * 1024 ** 3, max(1024 ** 2, available // 2))
    budget = int(float(os.environ.get("MAGNETIC_TORUS_NATIVE_MEMORY_MB", str(default_budget / 1024 ** 2))) * 1024 ** 2)
    if budget <= 0:
        raise ValueError("MAGNETIC_TORUS_NATIVE_MEMORY_MB must be positive.")
    estimate = native_memory_estimate(config)
    if estimate > budget:
        raise ValueError(f"The requested thermal mesh and output snapshots need about {estimate / 1024 ** 3:.2f} GiB "
                         f"of peak memory; the current budget is {budget / 1024 ** 3:.2f} GiB. "
                         "Reduce mesh resolution or output snapshots (time steps), then recalculate. Accepted results are preserved.")
    return {"estimated_peak_bytes": estimate, "memory_budget_bytes": budget}


def _implicit_enthalpy_step(old_h: np.ndarray, dt: float, mass: np.ndarray, stiffness: Any,
                            area: np.ndarray, source_w: np.ndarray,
                            config: ScenarioConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shared conservative sharp-interface step, including Dirichlet load vectors."""
    from scipy.sparse import diags
    from scipy.sparse.linalg import spsolve
    old_t = temperature_from_enthalpy(old_h, config)
    radiation_coefficient = area * config.emissivity * STEFAN_BOLTZMANN
    old_loss = radiation_coefficient * (old_t ** 4 - config.ambient_temperature ** 4)
    h = old_h + dt * (source_w - stiffness @ old_t - old_loss) / mass
    def residual_at(value):
        temp = temperature_from_enthalpy(value, config)
        loss = radiation_coefficient * (temp ** 4 - config.ambient_temperature ** 4)
        residual = mass * (value - old_h) / dt + stiffness @ temp + loss - source_w
        return residual, temp, loss
    for _iteration in range(80):
        residual, temp, radiation = residual_at(h)
        scale = max(1., float(np.sum(np.abs(source_w))), float(np.max(np.abs(mass * (h - old_h) / dt))))
        if np.max(np.abs(residual)) <= 1.e-9 * scale:
            if np.any(temp <= 0):
                raise ValueError("The requested discharge cools below absolute zero; reduce power or duration.")
            return h, temp, radiation
        derivative = np.where(h < 0, 1 / config.heat_capacity_solid,
                              np.where(h > config.latent_heat, 1 / config.heat_capacity_liquid, 0.))
        jacobian = diags(mass / dt) + stiffness @ diags(derivative) + diags(4 * radiation_coefficient * temp ** 3 * derivative)
        correction = spsolve(jacobian.tocsc(), residual)
        step, norm = 1., np.linalg.norm(residual)
        while step > 1.e-7:
            candidate = h - step * correction
            rc, tc, _ = residual_at(candidate)
            if np.all(tc > 0) and np.linalg.norm(rc) < norm:
                h = candidate
                break
            step *= .5
        else:
            raise RuntimeError("Thermal Newton iteration failed; reduce timestep or power.")
    raise RuntimeError("Thermal Newton iteration did not converge; increase time steps.")


def stefan_front_benchmark(cell_count: int = 64, time_steps: int = 500,
                            duration: float = 500.) -> dict[str, Any]:
    """One-phase planar Stefan melting test against the Neumann similarity solution.

    The exact front has not reached the remote insulated boundary. This uses the
    same sharp enthalpy Newton integrator as the toroidal FV reference, separating
    phase-front accuracy from the geometry discretization. It does not validate
    the regularized FE MHD model or establish a turbulence benchmark.
    """
    from dataclasses import replace
    from scipy.optimize import brentq
    from scipy.special import erf
    from scipy.sparse import diags
    if cell_count < 8 or time_steps < 2 or duration <= 0:
        raise ValueError("Stefan benchmark requires ≥8 cells, ≥2 steps and positive duration.")
    c = replace(ScenarioConfig(), solver="thermal_benchmark", initial_temperature=1810.,
                emissivity=0., heating_power=0., discharge_power=0.)
    length, excess = .2, 100.
    dx = length / cell_count
    stefan = c.heat_capacity_liquid * excess / c.latent_heat
    lam = brentq(lambda value: math.sqrt(math.pi) * value * math.exp(value * value) * erf(value) - stefan, 1.e-10, 3.)
    exact_front = 2 * lam * math.sqrt(c.thermal_conductivity * duration / (c.density * c.heat_capacity_liquid))
    if exact_front >= .8 * length:
        raise ValueError("Benchmark front approaches the remote boundary; reduce duration.")
    conductance = c.thermal_conductivity / dx
    diagonal = np.full(cell_count, 2 * conductance)
    diagonal[0] += conductance  # half-cell distance to imposed-temperature face
    diagonal[-1] -= conductance  # insulated remote boundary
    stiffness = diags((-conductance * np.ones(cell_count - 1), diagonal,
                       -conductance * np.ones(cell_count - 1)), (-1, 0, 1), format="csr")
    source = np.zeros(cell_count)
    source[0] = 2 * conductance * (c.melting_temperature + excess)
    mass = np.full(cell_count, c.density * dx)
    h, zero_area = np.zeros(cell_count), np.zeros(cell_count)
    for _ in range(time_steps):
        h, temp, _ = _implicit_enthalpy_step(h, duration / time_steps, mass, stiffness, zero_area, source, c)
    numerical_front = float(dx * liquid_fraction(h, c).sum())
    return {"cells": cell_count, "time_steps": time_steps, "duration_s": duration,
            "analytical_front_m": exact_front, "numerical_front_m": numerical_front,
            "relative_front_error": abs(numerical_front - exact_front) / exact_front,
            "temperature_k": temp, "enthalpy_j_kg": h,
            "reference": "Neumann one-phase Stefan similarity solution: Ste=sqrt(pi)*lambda*exp(lambda^2)*erf(lambda)"}


def _run_thermal(config: ScenarioConfig, progress: Progress, cancel: threading.Event) -> RunResult:
    from scipy.sparse import coo_matrix
    memory = _native_memory_guard(config)
    points, graph = _thermal_graph(config)
    i, j, g = graph["edge_i"], graph["edge_j"], graph["conductance_w_k"]
    n = len(points)
    K = coo_matrix((np.r_[g, g, -g, -g], (np.r_[i, j, i, j], np.r_[i, j, j, i])), shape=(n, n)).tocsr()
    volume, area = graph["node_volume"], graph["boundary_area"]
    mass = config.density * volume
    times = np.unique(np.r_[np.linspace(0, config.duration, config.time_steps + 1), config.duration * config.charge_fraction])
    h = np.full(n, float(enthalpy_from_temperature(config.initial_temperature, config)))
    hs, temperatures = [h.copy()], [temperature_from_enthalpy(h, config)]
    energy, radiated, power, residuals = [float(mass @ h)], [0.0], [0.0], [0.0]
    cumulative_input, cumulative_loss = 0., 0.
    for index, (previous_t, t) in enumerate(zip(times[:-1], times[1:]), 1):
        if cancel.is_set():
            raise SimulationCancelled("Thermal calculation cancelled.")
        dt = t - previous_t
        watts = source_power(config, (previous_t + t) / 2)
        h, T, radiation = _implicit_enthalpy_step(h, dt, mass, K, area, watts * graph["source_weights"], config)
        cumulative_input += dt * watts
        cumulative_loss += dt * float(radiation.sum())
        temperatures.append(T.copy())
        hs.append(h.copy())
        energy.append(float(mass @ h))
        radiated.append(cumulative_loss)
        power.append(watts - float(radiation.sum()))
        residuals.append(energy[-1] - energy[0] - cumulative_input + cumulative_loss)
        progress(index / (len(times) - 1), f"Conservative thermal benchmark · {t:.3g} s")
    H, T = np.asarray(hs), np.asarray(temperatures)
    fraction = liquid_fraction(H, config)
    B = coil_magnetic_field(points, config)
    A = coil_vector_potential(points, config)
    reference = float(enthalpy_from_temperature(config.discharge_temperature, config))
    magnetic_energy = float(np.sum(volume * np.sum(B * B, axis=1)) / (2 * MU0))
    physics = {"kind": "thermal_fv_lumped_v1", **{key: value.tolist() for key, value in graph.items()}}
    return RunResult(uuid.uuid4().hex, config, times, points,
                     {"temperature": T, "enthalpy": H, "liquid_fraction": fraction,
                      "velocity": np.zeros((len(times), n, 3)),
                      "magnetic_field": np.broadcast_to(B, (len(times), n, 3)),
                      "vector_potential": np.broadcast_to(A, (len(times), n, 3)),
                      "pressure": np.zeros_like(T), "electric_potential": np.zeros_like(T)},
                     {"energy_j": np.asarray(energy), "recoverable_energy_j": np.sum(mass * np.maximum(H - reference, 0), axis=1),
                      "usable_electric_energy_j": config.efficiency * np.sum(mass * np.maximum(H - reference, 0), axis=1),
                      "liquid_fraction": fraction @ volume / volume.sum(),
                      "magnetic_energy_j": np.full(len(times), magnetic_energy),
                      "imposed_field_energy_j": np.full(len(times), magnetic_energy),
                      "induced_field_energy_j": np.zeros(len(times)),
                      "pump_energy_j": np.zeros(len(times)),
                      "pump_power_w": np.zeros(len(times)), "net_power_w": np.asarray(power),
                      "radiated_energy_j": np.asarray(radiated), "energy_balance_error_j": np.asarray(residuals)},
                     {"backend": "native conservative finite-volume thermal benchmark", "validation_status": "reference checks available; not validated MHD",
                      "model": "thermal only; imposed external coil; no fluid induction, turbulence or Lorentz feedback",
                      "node_volumes": volume.tolist(), "conductor_mask": np.ones(n, dtype=bool).tolist(),
                      "structured_shape": graph["shape"].tolist(), "pinn_physics": physics,
                      "imposed_coil": imposed_coil(config), "transition_smoothing_k": 0.,
                      "magnetic_energy_domain": "Reservoir volume only; the native diagnostic has no exterior FE energy integral.",
                      "coil_energy_note": "Ideal imposed current. Stored field energy is not coil electrical consumption; conductor/coil-circuit losses are not modeled by this thermal benchmark.",
                      "memory_estimate": memory,
                      "material_provenance": config.provenance, "source_schedule": "surface charge then surface discharge", "units": "SI"})
