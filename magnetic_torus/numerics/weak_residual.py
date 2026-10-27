"""Differentiable residuals matching accepted P1 FE operators and closure."""
from __future__ import annotations

from functools import lru_cache
from io import BytesIO
import math
from collections.abc import Mapping
from numbers import Real
from typing import Any
import numpy as np
from ..models import RunResult, ScenarioConfig
from .constants import MU0, STEFAN_BOLTZMANN
from .thermal import source_power


@lru_cache(maxsize=4)
def _load_weak_artifact(payload: bytes) -> dict[str, np.ndarray]:
    with np.load(BytesIO(payload), allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def evaluate_weak_residual_torch(config: ScenarioConfig, reference_result: RunResult,
                                 predicted_fields: dict[str, Any], time_indices: Any,
                                 point_indices: Any = None, *,
                                 normalization_scales: Mapping[str, float] | None = None,
                                 include_raw_components: bool = False) -> dict[str, Any]:
    """Differentiable, matching P1 FE weak residuals for one accepted time pair.

    This uses the worker's tetrahedral quadrature, geometry, boundary conditions,
    backward-Euler interval, material law, and mixed-VMS terms.  Spatial slicing
    would remove coupled test functions and is intentionally disallowed.
    ``vector_potential`` denotes total A; imposed A is subtracted only from the
    conductor–vacuum curlcurl/gauge terms, exactly as in the numerical solve.
    Explicit ``normalization_scales`` are for research-training isolation only;
    omitting them retains the original full-reference qualification oracle.
    """
    import torch
    scales = None
    if normalization_scales is not None:
        names = {"velocity", "magnetic_field", "vector_potential"}
        if not isinstance(normalization_scales, Mapping) or set(normalization_scales) != names:
            raise ValueError("Weak normalization overrides require exactly velocity, magnetic_field, and vector_potential.")
        if any(isinstance(value, bool) or not isinstance(value, Real)
               or not math.isfinite(value) or value <= 0 for value in normalization_scales.values()):
            raise ValueError("Weak normalization overrides must be positive finite real values.")
        scales = {name: float(value) for name, value in normalization_scales.items()}
        velocity_floor = max(math.sqrt(abs(config.pump_force) * config.minor_radius / config.density), 1.e-4)
        if (scales["velocity"] < velocity_floor or scales["magnetic_field"] < 1.e-6
                or scales["vector_potential"] < scales["magnetic_field"] * config.minor_radius):
            raise ValueError("Weak normalization overrides must retain the configured velocity, magnetic, and potential floors.")
    if point_indices is not None:
        raise ValueError("FE weak physics requires all mesh nodes; batch accepted time pairs instead.")
    info = reference_result.metadata.get("pinn_physics", {})
    if info.get("kind") != "fenics_a_phi_vms_v1":
        raise ValueError("Matching A–phi FE residual operators are absent from this run.")
    indices = np.asarray(time_indices, dtype=int).ravel()
    if len(indices) != 2 or indices[1] != indices[0] + 1:
        raise ValueError("Provide one consecutive accepted time pair [k-1,k].")
    op = _load_weak_artifact(reference_result.artifact_bytes("weak_operator_path"))
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
    Ai, _ = A - ext[1], A0 - ext[0]
    uq, upq = interpolate(U), interpolate(U0)
    _, tq, tpq = interpolate(P), interpolate(T), interpolate(T0)
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
    if scales is None:
        velocity_scale = max(float(np.max(np.linalg.norm(reference_result.fields["velocity"], axis=-1))), math.sqrt(abs(config.pump_force) * length / rho), 1.e-4)
        bscale = max(float(np.max(np.linalg.norm(reference_result.fields["magnetic_field"], axis=-1))), 1.e-6)
        ascale = max(float(np.max(np.linalg.norm(reference_result.fields["vector_potential"], axis=-1))), bscale * length)
    else:
        velocity_scale, bscale, ascale = (scales[name] for name in ("velocity", "magnetic_field", "vector_potential"))
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
        if include_raw_components:
            result[f"raw_{name}"] = residual[mask]
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
