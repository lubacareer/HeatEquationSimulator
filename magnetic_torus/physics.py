"""Conservative thermal reference and the isolated, three-dimensional FE backend.

Reference enthalpy is zero at the solid melting point.  All quantities here are SI.
The native benchmark deliberately has no velocity or induced field: it is never
substituted for an unavailable MHD worker.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import uuid
from functools import lru_cache
from typing import Any

import numpy as np

from .models import Progress, RunResult, ScenarioConfig, SimulationCancelled

MU0 = 4e-7 * math.pi
STEFAN_BOLTZMANN = 5.670374419e-8


class PhysicsBackendUnavailable(RuntimeError):
    """The requested physics could not run; callers must not display substitute data."""


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
                    edge_i.append(i); edge_j.append(ids[radial + 1, theta, phi]); conductance.append(k * surface / dr)
                edge_i.append(i); edge_j.append(ids[radial, (theta + 1) % nt, phi])
                conductance.append(k * (major + r[radial, theta, phi] * math.cos((theta + 1) * da)) * dr * db
                                   / (r[radial, theta, phi] * da))
                edge_i.append(i); edge_j.append(ids[radial, theta, (phi + 1) % np_])
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
            status = MemoryStatus(); status.length = ctypes.sizeof(status)
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
    source = np.zeros(cell_count); source[0] = 2 * conductance * (c.melting_temperature + excess)
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
        temperatures.append(T.copy()); hs.append(h.copy()); energy.append(float(mass @ h))
        radiated.append(cumulative_loss); power.append(watts - float(radiation.sum()))
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


def _wsl_path(path: Path) -> str:
    absolute = path.resolve()
    if absolute.drive:
        return "/mnt/" + absolute.drive[0].lower() + "/" + str(absolute)[3:].replace("\\", "/")
    return str(absolute)


def _wsl_distribution() -> str:
    return os.environ.get("MAGNETIC_TORUS_WSL_DISTRO", os.environ.get("TORUS_WSL_DISTRO", "Ubuntu"))


def _run_fenics(config: ScenarioConfig, progress: Progress, cancel: threading.Event) -> RunResult:
    # Artifacts remain on disk for frozen-field transport and matching FE PINNs.
    directory = Path(os.environ.get("MAGNETIC_TORUS_RUNS", Path.home() / ".magnetic_torus" / "worker_artifacts")) / uuid.uuid4().hex
    directory.mkdir(parents=True, exist_ok=True)
    request = directory / "request.json"
    request.write_text(json.dumps(config.to_dict()), encoding="utf-8")
    worker = Path(__file__).parent / "workers" / "fenics_worker.py"
    if sys.platform == "win32":
        command = ["wsl", "-d", _wsl_distribution(), "--", "/usr/bin/python3", _wsl_path(worker), _wsl_path(request), _wsl_path(directory)]
    else:
        command = [os.environ.get("MAGNETIC_TORUS_FE_PYTHON", "/usr/bin/python3"), str(worker), str(request), str(directory)]
    try:
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                encoding="utf-8", errors="replace", creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
    except OSError as exc:
        raise PhysicsBackendUnavailable("Cannot start the FEniCS worker. Install Ubuntu WSL2 and run scripts/setup_wsl_physics.sh. " + str(exc)) from exc
    lines: queue.Queue[str | None] = queue.Queue()
    def read_output() -> None:
        assert proc.stdout is not None
        with proc.stdout:
            for line in proc.stdout:
                lines.put(line.rstrip())
        lines.put(None)
    threading.Thread(target=read_output, daemon=True).start()
    log: list[str] = []
    try:
        while proc.poll() is None or not lines.empty():
            if cancel.is_set():
                (directory / "cancel").touch()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.terminate()
                raise SimulationCancelled("FEniCS simulation cancelled; previous results are preserved.")
            try:
                line = lines.get(timeout=.1)
            except queue.Empty:
                continue
            if line is None:
                break
            log.append(line)
            if line.startswith("TORUS_PROGRESS "):
                event = json.loads(line[len("TORUS_PROGRESS "):])
                progress(float(event["fraction"]), event["message"])
        code = proc.wait()
    finally:
        if proc.poll() is None:
            (directory / "cancel").touch()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.terminate()
    (directory / "worker.log").write_text("\n".join(log), encoding="utf-8")
    if code != 0:
        detail = "\n".join(log[-12:])
        raise PhysicsBackendUnavailable("Full MHD worker failed. Install/check scripts/setup_wsl_physics.sh; see "
                                        + str(directory / "worker.log") + "\n" + detail)
    arrays = np.load(directory / "result.npz", allow_pickle=False)
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    for key in ("exterior_field_path", "weak_operator_path"):
        if key in metadata:
            metadata[key] = str(directory / Path(metadata[key]).name)
    return RunResult(directory.name, config, arrays["times"], arrays["points"],
                     {key[6:]: arrays[key] for key in arrays.files if key.startswith("field_")},
                     {key[5:]: arrays[key] for key in arrays.files if key.startswith("diag_")}, metadata)


def run_simulation(config: ScenarioConfig, progress: Progress | None = None,
                   cancel: threading.Event | None = None) -> RunResult:
    config.validate()
    report = progress or (lambda _fraction, _message: None)
    stop = cancel or threading.Event()
    if stop.is_set():
        raise SimulationCancelled("Simulation cancelled before starting.")
    if config.solver == "thermal_benchmark":
        return _run_thermal(config, report, stop)
    return _run_fenics(config, report, stop)


@lru_cache(maxsize=4)
def _load_weak_artifact(path: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def evaluate_weak_residual_torch(config: ScenarioConfig, reference_result: RunResult,
                                 predicted_fields: dict[str, Any], time_indices: Any,
                                 point_indices: Any = None) -> dict[str, Any]:
    """Differentiable, matching P1 FE weak residuals for one accepted time pair.

    This uses the worker's tetrahedral quadrature, geometry, boundary conditions,
    backward-Euler interval, material law, and mixed-VMS terms.  Spatial slicing
    would remove coupled test functions and is intentionally disallowed.
    ``vector_potential`` denotes total A; imposed A is subtracted only from the
    conductor–vacuum curlcurl/gauge terms, exactly as in the numerical solve.
    """
    import torch
    if point_indices is not None:
        raise ValueError("FE weak physics requires all mesh nodes; batch accepted time pairs instead.")
    info = reference_result.metadata.get("pinn_physics", {})
    if info.get("kind") != "fenics_a_phi_vms_v1":
        raise ValueError("Matching A–phi FE residual operators are absent from this run.")
    indices = np.asarray(time_indices, dtype=int).ravel()
    if len(indices) != 2 or indices[1] != indices[0] + 1:
        raise ValueError("Provide one consecutive accepted time pair [k-1,k].")
    path = str(reference_result.metadata.get("weak_operator_path", ""))
    if not Path(path).is_file():
        raise ValueError("The originating FE weak-operator artifact is missing: " + path)
    op = _load_weak_artifact(path)
    sample = predicted_fields["temperature"]
    device, dtype = sample.device, sample.dtype
    def tensor(value, integer=False):
        return torch.as_tensor(value, device=device, dtype=torch.long if integer else dtype)
    def pair(name):
        value = predicted_fields[name]
        return value if value.shape[0] == 2 else value[indices.tolist()]
    cells = tensor(op["cells"], True)
    gradients = tensor(op["gradients"])
    volumes = tensor(op["volumes"])
    diameter = tensor(op["cell_diameter"])
    cond = tensor(op["cell_labels"] == 1).bool()
    n = len(reference_result.points)
    dt = float(reference_result.times[indices[1]] - reference_result.times[indices[0]])
    # Basix tetrahedral degree-2 rule, weights all 1/4 of physical volume.
    qa, qb = .5854101966249685, .1381966011250105
    basis = tensor(np.full((4, 4), qb) + np.eye(4) * (qa - qb))
    weights = volumes[:, None] / 4
    def interpolate(value):
        return torch.einsum("qa,ca...->cq...", basis, value[cells])
    def gradient(value):
        return torch.einsum("ca...,caj->c...j", value[cells], gradients)
    def curl(grad):
        return torch.stack((grad[..., 2, 1] - grad[..., 1, 2], grad[..., 0, 2] - grad[..., 2, 0], grad[..., 1, 0] - grad[..., 0, 1]), dim=-1)
    U0, U = pair("velocity")
    P0, P = pair("pressure")
    A0, A = pair("vector_potential")
    Phi0, Phi = pair("electric_potential")
    T0, T = pair("temperature")
    ext = tensor(op["imposed_vector_potential"][indices])
    Ai, Ai0 = A - ext[1], A0 - ext[0]
    uq, upq = interpolate(U), interpolate(U0)
    pq, tq, tpq = interpolate(P), interpolate(T), interpolate(T0)
    gU, gp, gA, gAi, gPhi, gT = gradient(U), gradient(P), gradient(A), gradient(Ai), gradient(Phi), gradient(T)
    Bcell, Bprevious = curl(gA), curl(gradient(A0))
    B = Bcell[:, None, :].expand(-1, 4, -1)
    at = interpolate((A - A0) / dt)
    divU = gU.diagonal(dim1=-2, dim2=-1).sum(-1)
    symU = .5 * (gU + gU.transpose(-1, -2))
    def phase(temp):
        return torch.clamp((temp - config.melting_temperature) / config.transition_width + .5, 0, 1)
    def enthalpy(temp):
        delta, f = temp - config.melting_temperature, phase(temp)
        middle = config.heat_capacity_solid * delta + config.latent_heat * f + .5 * (config.heat_capacity_liquid - config.heat_capacity_solid) * config.transition_width * f * f
        return torch.where(delta < -config.transition_width / 2, config.heat_capacity_solid * delta,
                           torch.where(delta > config.transition_width / 2, config.latent_heat + config.heat_capacity_liquid * delta, middle))
    f = phase(tq)
    sigma = config.electrical_conductivity_solid * (1 - f) + config.electrical_conductivity_liquid * f
    eta = 1 / (MU0 * sigma)
    J = sigma[..., None] * (-at - gPhi[:, None, :] + torch.linalg.cross(uq, B))
    damping = 1.e6 * (1 - f) ** 2 / (f ** 3 + 1.e-3)
    xyz = interpolate(tensor(op["points"]))
    radial = torch.sqrt(xyz[..., 0] ** 2 + xyz[..., 1] ** 2 + 1.e-20)
    local_r = torch.sqrt((radial - config.major_radius) ** 2 + xyz[..., 2] ** 2 + 1.e-20)
    azimuthal = torch.stack((-xyz[..., 1] / radial, xyz[..., 0] / radial, torch.zeros_like(radial)), dim=-1)
    poloidal = torch.stack((-xyz[..., 2] * xyz[..., 0] / (local_r * radial), -xyz[..., 2] * xyz[..., 1] / (local_r * radial), (radial - config.major_radius) / local_r), dim=-1)
    force = config.pump_force * (azimuthal + config.helical_fraction * poloidal)
    force = force + torch.stack((torch.zeros_like(tq), torch.zeros_like(tq), config.density * config.gravity * config.thermal_expansion * (tq - config.initial_temperature)), dim=-1)
    rho, mu = config.density, config.dynamic_viscosity
    mom = (rho * (uq - upq) / dt + rho * torch.einsum("cij,cqj->cqi", gU, uq) + gp[:, None, :]
           - torch.linalg.cross(J, B) + damping[..., None] * uq - force)
    rb = (Bcell - Bprevious)[:, None, :] / dt - (torch.einsum("cij,cqj->cqi", gU, B) - divU[:, None, None] * B)
    invh2 = 1 / diameter[:, None] ** 2
    tauu = torch.rsqrt((2 * rho / dt) ** 2 + rho ** 2 * uq.square().sum(-1) * invh2 + 3 * mu ** 2 * invh2 ** 2
                       + rho / MU0 * B.square().sum(-1) * math.sqrt(3) * invh2 + damping ** 2)
    taub = torch.rsqrt((2 / dt) ** 2 + (uq.square().sum(-1) + B.square().sum(-1) / (MU0 * rho)) * invh2 + 3 * eta ** 2 * invh2 ** 2)
    usub, bsub = -tauu[..., None] * mom, -taub[..., None] * rb
    nut = .15 * diameter[:, None] * torch.sqrt(usub.square().sum(-1) + bsub.square().sum(-1) / (MU0 * rho) + 1.e-30) if config.turbulence else torch.zeros_like(tq)
    def outer(left, right):
        return left[..., :, None] * right[..., None, :]
    crossstress = rho * (outer(uq, usub) + outer(usub, uq)) - (outer(B, bsub) + outer(bsub, B) - (B * bsub).sum(-1)[..., None, None] * torch.eye(3, device=device, dtype=dtype)) / MU0
    emf = torch.linalg.cross(usub, B) + torch.linalg.cross(uq, bsub) if config.turbulence else torch.zeros_like(B)
    # Cell-node test-function contributions, sum over the four quadrature points.
    def nodal_mass(values):
        return torch.einsum("cq,qa,cq...->ca...", weights.expand(-1, 4), basis, values)
    def nodal_gradient(flux):
        return torch.einsum("cq,caj,cq...j->ca...", weights.expand(-1, 4), gradients, flux)
    condw = cond[:, None, None]
    momentum_local = nodal_mass(mom) + nodal_gradient(2 * (mu + rho * nut)[..., None, None] * symU[:, None])
    if config.turbulence:
        momentum_local = momentum_local - nodal_gradient(crossstress)
        graddiv = (divU[:, None] / (3 * invh2 * tauu))[..., None, None] * torch.eye(3, device=device, dtype=dtype)
        momentum_local = momentum_local + nodal_gradient(graddiv)
    continuity_local = nodal_mass(divU[:, None].expand(-1, 4)) - nodal_gradient(usub)
    # Curl of e_component*N_a, laid out (cell, basis-node, component, curl-component).
    identity = torch.eye(3, device=device, dtype=dtype)
    curltest = torch.linalg.cross(gradients[:, :, None, :], identity[None, None], dim=-1)
    curlAi = curl(gAi)
    divAi = gAi.diagonal(dim1=-2, dim2=-1).sum(-1)
    magnetic_local = (torch.einsum("caij,cj->cai", curltest, curlAi) + gradients * divAi[:, None, None]) * volumes[:, None, None] / MU0
    induction_local = nodal_mass(sigma[..., None] * (at + gPhi[:, None] - torch.linalg.cross(uq, B) - emf))
    eddy_curl = (weights * sigma * nut).sum(1)[:, None, None] * torch.einsum("caij,cj->cai", curltest, Bcell)
    magnetic_local = magnetic_local + (induction_local + eddy_curl) * condw
    charge_local = -nodal_gradient(J) + nodal_mass(1.e-10 * interpolate(Phi))
    hq, hpq = enthalpy(tq), enthalpy(tpq)
    heating_power = source_power(config, float(reference_result.times[indices].mean()))
    total_volume = float(op["volumes"][op["cell_labels"] == 1].sum())
    heat_k = config.thermal_conductivity + rho * config.heat_capacity_liquid * nut / config.thermal_turbulent_prandtl
    dissipation = 2 * mu * symU.square().sum((-1, -2))[:, None] + damping * uq.square().sum(-1)
    joule = J.square().sum(-1) / sigma
    heat_local = nodal_mass(rho * (hq - hpq) / dt - heating_power / total_volume - joule - dissipation)
    heat_local = heat_local + nodal_gradient(heat_k[..., None] * gT[:, None] - rho * hq[..., None] * uq)
    def scatter(local):
        shape = (n,) + tuple(local.shape[2:])
        out = torch.zeros(shape, device=device, dtype=dtype)
        return out.index_add(0, cells.ravel(), local.reshape((-1,) + tuple(local.shape[2:])))
    r_mom = scatter(momentum_local * condw)
    r_cont = scatter(continuity_local * cond[:, None])
    r_mag = scatter(magnetic_local)
    r_charge = scatter(charge_local * cond[:, None])
    r_heat = scatter(heat_local * cond[:, None])
    faces = tensor(op["boundary_faces"], True)
    if len(faces) and config.emissivity:
        triangle_basis = tensor(np.full((3, 3), 1 / 6) + np.eye(3) / 2)
        surface_temp = torch.einsum("qa,fa->fq", triangle_basis, T[faces])
        radiation = config.emissivity * STEFAN_BOLTZMANN * (surface_temp ** 4 - config.ambient_temperature ** 4)
        surface_local = torch.einsum("f,qa,fq->fa", tensor(op["boundary_areas"]) / 3, triangle_basis, radiation)
        r_heat = r_heat.index_add(0, faces.ravel(), surface_local.ravel())
    global_volume = torch.zeros(n, device=device, dtype=dtype).index_add(0, cells.ravel(), (volumes[:, None].expand(-1, 4) / 4).ravel())
    fluid_volume = tensor(op["node_volumes"])
    fluid = fluid_volume > 0
    bc = info.get("bc_nodes", {})
    def free_nodes(index, conductor=True):
        keep = fluid.clone() if conductor else torch.ones(n, device=device, dtype=torch.bool)
        constrained = tensor(bc.get(str(index), []), True)
        keep[constrained] = False
        return keep
    length = config.minor_radius
    velocity_scale = max(float(np.max(np.linalg.norm(reference_result.fields["velocity"], axis=-1))), math.sqrt(abs(config.pump_force) * length / rho), 1.e-4)
    bscale = max(float(np.max(np.linalg.norm(reference_result.fields["magnetic_field"], axis=-1))), 1.e-6)
    ascale = max(float(np.max(np.linalg.norm(reference_result.fields["vector_potential"], axis=-1))), bscale * length)
    tscale = max(config.duration, dt)
    hscale = max(config.latent_heat, config.heat_capacity_solid * 100., 1.)
    pscale = max(rho * velocity_scale ** 2, 1.)
    phiscale = max(ascale / tscale * length, velocity_scale * bscale * length, 1.e-8)
    momscale = rho * velocity_scale / tscale + rho * velocity_scale ** 2 / length + bscale ** 2 / (MU0 * length) + abs(config.pump_force)
    magscale = config.electrical_conductivity_liquid * (ascale / tscale + phiscale / length + velocity_scale * bscale) + ascale / (MU0 * length ** 2)
    chargescale = config.electrical_conductivity_liquid * (ascale / tscale + phiscale / length + velocity_scale * bscale) / length
    result = {}
    for name, index, residual, scale, vol in (("momentum", 0, r_mom, momscale, fluid_volume),
                                              ("continuity", 1, r_cont, velocity_scale / length, fluid_volume),
                                              ("induction", 2, r_mag, magscale, global_volume),
                                              ("charge", 3, r_charge, chargescale, fluid_volume),
                                              ("enthalpy", 4, r_heat, rho * hscale / tscale, fluid_volume)):
        mask = free_nodes(index, index != 2)
        denom = torch.clamp(vol[mask], min=1.e-30) * scale
        result[name] = residual[mask] / (denom[:, None] if residual.ndim == 2 else denom)
    # Gauge is already enforced in the induction weak equation; expose separately
    # to monitor divergence, rather than claim identically zero Coulomb error.
    result["gauge"] = divAi * length / ascale
    for index, value, scale, target in ((0, U, velocity_scale, torch.zeros_like(U)),
                                       (1, P, pscale, torch.zeros_like(P)),
                                       (2, A, ascale, ext[1]),
                                       (3, Phi, phiscale, torch.zeros_like(Phi)),
                                       (4, T, max(100., config.transition_width), torch.full_like(T, config.initial_temperature))):
        constrained = tensor(bc.get(str(index), []), True)
        if constrained.numel():
            result[f"boundary_{index}"] = (value[constrained] - target[constrained]) / scale
    # FE output B is the continuous, volume-weighted projection of cell curl(A).
    projected = scatter(Bcell[:, None, :].expand(-1, 4, -1) * volumes[:, None, None])
    normalizer = scatter(volumes[:, None].expand(-1, 4))
    projected = projected / normalizer[:, None]
    if "magnetic_field" in predicted_fields:
        result["magnetic_consistency"] = (pair("magnetic_field")[1] - projected) / bscale
    if "enthalpy" in predicted_fields:
        result["enthalpy_constitutive"] = (pair("enthalpy")[1] - enthalpy(T)) / hscale
    return result


def write_validation_report(destination: str | Path, include_fenics: bool = False,
                            progress: Progress | None = None) -> dict[str, Any]:
    """Re-run analytical checks and retain both passing and pending release gates."""
    from dataclasses import replace
    report_progress = progress or (lambda _fraction, _message: None)
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"format_version": 1, "units": "SI", "overall_status": "experimental; release validation incomplete"}
    report_progress(.02, "Checking analytical Stefan-front refinement")
    cases = []
    for count in (32, 64, 128):
        case = stefan_front_benchmark(count, count * 16)
        cases.append({key: value for key, value in case.items() if not isinstance(value, np.ndarray)})
    report["stefan_reference"] = {"status": "passed" if max(case["relative_front_error"] for case in cases) < .005 else "failed",
                                   "cases": cases, "note": "Sharp FV enthalpy integrator. Subcell front position can give nonmonotone individual refinement errors; this does not validate smoothed 3D MHD."}
    report_progress(.12, "Checking conservative surface charging and thermal losses")
    benchmark = replace(ScenarioConfig(), solver="thermal_benchmark", resolution=3, time_steps=8,
                        heating_power=20000., discharge_power=5000., charge_fraction=.37,
                        duration=10., emissivity=.1)
    native = run_simulation(benchmark)
    balance = float(np.max(np.abs(native.diagnostics["energy_balance_error_j"])))
    report["thermal_energy_balance"] = {"status": "passed" if balance < .01 else "failed", "maximum_defect_j": balance, "config": benchmark.to_dict()}
    report["full_fe_stationary"] = {"status": "not run", "reason": "Add --include-fenics to run installed Linux/WSL workers."}
    report["full_fe_driven_vms"] = dict(report["full_fe_stationary"])
    report["full_fe_seed_removal"] = dict(report["full_fe_stationary"])
    report["mhd_taylor_green"] = {"status": "pending", "reason": "Published periodic B/psi validation has not validated this conductor/exterior A/phi adaptation."}
    report["coupled_manufactured_solution"] = {"status": "pending", "reason": "Full thermal/velocity/pressure/A/phi manufactured-solution convergence study is outstanding."}
    report["exterior_refinement"] = {"status": "pending", "reason": "Conductor/exterior mesh and enclosure-distance convergence are outstanding."}
    report["subgrid_total_energy"] = {"status": "pending", "reason": "Resolved plus modeled subgrid MHD total-energy budget has not been validated."}
    if include_fenics:
        base = replace(ScenarioConfig(), resolution=3, time_steps=2, duration=.02,
                       initial_temperature=1820., heating_power=0., coil_current=0., pump_force=0., turbulence=False)
        try:
            stationary = run_simulation(base, lambda f, message: report_progress(.15 + .15 * f, message))
            mask = np.asarray(stationary.metadata["conductor_mask"], dtype=bool)
            temp_error = float(np.max(np.abs(stationary.fields["temperature"][:, mask] - base.initial_temperature)))
            velocity_error = float(np.max(np.linalg.norm(stationary.fields["velocity"], axis=-1)))
            report["full_fe_stationary"] = {"status": "passed" if temp_error < 1.e-6 and velocity_error < 1.e-9 else "failed", "temperature_error_k": temp_error, "velocity_error_m_s": velocity_error, "artifact": stationary.metadata["weak_operator_path"], "config": base.to_dict()}
            driven_config = replace(base, duration=.1, time_steps=3, initial_temperature=1810.2,
                                     heating_power=1.e6, coil_current=10000., pump_force=1., turbulence=True)
            driven = run_simulation(driven_config, lambda f, message: report_progress(.30 + .2 * f, message))
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            predictions = {key: torch.tensor(value[:2], dtype=torch.float64, device=device, requires_grad=True) for key, value in driven.fields.items()}
            residuals = evaluate_weak_residual_torch(driven_config, driven, predictions, [0, 1])
            defects = {key: float(value.detach().abs().max().cpu()) for key, value in residuals.items() if value.numel()}
            sum(value.square().mean() for value in residuals.values() if value.numel()).backward()
            gradients_finite = all(value.grad is None or bool(torch.isfinite(value.grad).all()) for value in predictions.values())
            worst = max(defects[key] for key in ("momentum", "continuity", "induction", "charge", "enthalpy", "magnetic_consistency"))
            energy_error = float(np.max(np.abs(driven.diagnostics["energy_balance_error_j"])))
            report["full_fe_driven_vms"] = {"status": "passed" if worst < 1.e-7 and energy_error < .01 and gradients_finite else "failed", "dimensionless_weak_defects": defects, "gradients_finite": gradients_finite,
                                           "thermal_energy_defect_j": energy_error, "mesh_volume_relative_error": driven.metadata["volume_relative_error"],
                                           "artifact": driven.metadata["weak_operator_path"], "config": driven_config.to_dict(), "note": "Discrete equation agreement checks implementation; it does not establish physical validation of turbulence."}
            dynamo_config = replace(driven_config, time_steps=4, initial_temperature=1820., heating_power=0.,
                                     mode="dynamo", pump_force=100., seed_duration=.02)
            dynamo = run_simulation(dynamo_config, lambda f, message: report_progress(.50 + .2 * f, message))
            energies = dynamo.diagnostics["magnetic_energy_j"]
            report["full_fe_seed_removal"] = {"status": "passed" if np.all(np.isfinite(energies)) and energies[-1] < energies[0] else "failed", "magnetic_energy_j": energies.tolist(),
                                             "times_s": dynamo.times.tolist(), "config": dynamo_config.to_dict(), "artifact": dynamo.metadata["weak_operator_path"],
                                             "note": "Low-Rm pumped scenario decays after seed removal; self-exciting dynamo operation is not asserted."}
        except Exception as exc:
            report["full_fe_execution_error"] = {"status": "failed", "reason": str(exc)}
        report_progress(.75, "Running analytical FE Hartmann and magnetic-diffusion reductions")
        worker = Path(__file__).parent / "workers" / "fenics_worker.py"
        case_file = destination.with_name(destination.stem + "_fenics.json")
        if sys.platform == "win32":
            command = ["wsl", "-d", _wsl_distribution(), "--", "/usr/bin/python3", _wsl_path(worker), "--benchmarks", _wsl_path(case_file)]
        else:
            command = [os.environ.get("MAGNETIC_TORUS_FE_PYTHON", "/usr/bin/python3"), str(worker), "--benchmarks", str(case_file)]
        try:
            completed = subprocess.run(command, check=True, capture_output=True, text=True, encoding="utf-8", errors="replace")
            cases_data = json.loads(case_file.read_text(encoding="utf-8"))
            report.update(cases_data)
        except Exception as exc:
            report["fenics_special_cases"] = {"status": "failed", "reason": str(exc)}
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report_progress(1., "Validation report saved; pending gates remain explicit")
    return report


if __name__ == "__main__":
    import argparse
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Reproducible Magnetic Torus physical checks")
    parser.add_argument("--benchmark-report", required=True, type=Path, help="Write analytical, conservation and pending-gate results as JSON.")
    parser.add_argument("--include-fenics", action="store_true", help="Run installed WSL/Linux coupled workers and analytical FE reductions.")
    args = parser.parse_args()
    write_validation_report(args.benchmark_report, args.include_fenics,
                            lambda fraction, message: print(f"{fraction:5.0%} {message}", flush=True))
    print(str(args.benchmark_report.resolve()))
