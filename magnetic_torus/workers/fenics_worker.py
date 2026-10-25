"""DOLFINx/PETSc conductor–vacuum A–phi MHD worker (Linux/WSL2).

P1 tetrahedra; backward Euler; fully coupled SNES/Newton.  The mixed residual
VMS terms follow Sondak et al., arXiv:1412.1055 (37–40,44–45,52,57,59),
adapted to magnetic vector potential.  This adaptation is experimental: the
published B/psi Taylor–Green validation does NOT validate this implementation.
No empirical growth or prescribed fluid velocity is inserted into a result.
"""
from __future__ import annotations
import json
import math
import os
from pathlib import Path
import sys
import traceback
import threading
import time
import signal

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from magnetic_torus.models import ScenarioConfig
from magnetic_torus.physics import (MU0, STEFAN_BOLTZMANN, coil_vector_potential,
                                    imposed_coil, regularized_enthalpy, source_power)


def report(fraction: float, message: str) -> None:
    print("TORUS_PROGRESS " + json.dumps({"fraction": fraction, "message": message}), flush=True)


class _TimestepRejections:
    """Bound one failed-step retry sequence while retaining run-wide diagnostics."""
    def __init__(self) -> None:
        self.total = 0
        self.consecutive = 0

    def rejected(self, next_dt: float, minimum_dt: float) -> bool:
        self.total += 1
        self.consecutive += 1
        return next_dt < minimum_dt or self.consecutive > 20

    def accepted(self) -> None:
        self.consecutive = 0


def run_special_case_benchmarks(output: Path) -> None:
    """Independent analytical FE reductions, not a torus turbulence validation."""
    import basix.ufl
    import dolfinx
    from dolfinx import fem, mesh
    from dolfinx.fem.petsc import LinearProblem
    from mpi4py import MPI
    from petsc4py import PETSc
    import ufl
    hartmann = []
    viscosity, conductivity, halfwidth, ha, force = .006, 7.e5, .01, 4., 1.
    b0 = ha * math.sqrt(viscosity / conductivity) / halfwidth
    scale = force / (conductivity * b0 * b0)
    for count in (16, 32, 64):
        domain = mesh.create_interval(MPI.COMM_WORLD, count, [-halfwidth, halfwidth])
        element = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
        W = fem.functionspace(domain, basix.ufl.mixed_element([element, element]))
        velocity, ay = ufl.TrialFunctions(W)
        v, a = ufl.TestFunctions(W)
        dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": 8})
        bilinear = (viscosity * velocity.dx(0) * v.dx(0) + conductivity * b0 * b0 * velocity * v
                    + ay.dx(0) * a.dx(0) / MU0 + conductivity * b0 * velocity * a) * dx
        load = force * v * dx
        boundary = mesh.locate_entities_boundary(domain, 0, lambda x: np.isclose(np.abs(x[0]), halfwidth))
        bcs = []
        for component in range(2):
            space, _ = W.sub(component).collapse()
            zero = fem.Function(space)
            dofs = fem.locate_dofs_topological((W.sub(component), space), 0, boundary)
            bcs.append(fem.dirichletbc(zero, dofs, W.sub(component)))
        problem = LinearProblem(bilinear, load, bcs=bcs, petsc_options_prefix=f"hartmann_{count}_",
                                petsc_options={"ksp_type": "preonly", "pc_type": "lu", "ksp_error_if_not_converged": True})
        solution = problem.solve()
        u_h, a_h = ufl.split(solution)
        z = ufl.SpatialCoordinate(domain)[0]
        exact_u = scale * (1 - ufl.cosh(ha * z / halfwidth) / math.cosh(ha))
        exact_bx = -MU0 * conductivity * b0 * scale * (z - halfwidth / ha * ufl.sinh(ha * z / halfwidth) / math.cosh(ha))
        error_u = math.sqrt(fem.assemble_scalar(fem.form((u_h - exact_u) ** 2 * dx)) / fem.assemble_scalar(fem.form(exact_u ** 2 * dx)))
        error_b = math.sqrt(fem.assemble_scalar(fem.form((-a_h.dx(0) - exact_bx) ** 2 * dx)) / fem.assemble_scalar(fem.form(exact_bx ** 2 * dx)))
        hartmann.append({"cells": count, "velocity_relative_l2": error_u, "induced_field_relative_l2": error_b})
    # Resistive vector-potential eigenmode, no flow: sigma*dA/dt - A''/mu0=0.
    domain = mesh.create_interval(MPI.COMM_WORLD, 64, [0., 1.])
    S = fem.functionspace(domain, ("Lagrange", 1))
    previous = fem.Function(S)
    previous.interpolate(lambda x: np.sin(math.pi * x[0]))
    test, trial = ufl.TestFunction(S), ufl.TrialFunction(S)
    eta = 1 / (MU0 * conductivity)
    duration, steps = .1 / eta, 200
    dt = duration / steps
    boundary = mesh.locate_entities_boundary(domain, 0, lambda x: np.isclose(x[0], 0) | np.isclose(x[0], 1))
    bc = fem.dirichletbc(PETSc.ScalarType(0), fem.locate_dofs_topological(S, 0, boundary), S)
    problem = LinearProblem((trial * test / dt + eta * trial.dx(0) * test.dx(0)) * ufl.dx,
                            previous * test / dt * ufl.dx, bcs=[bc], petsc_options_prefix="magnetic_diffusion_",
                            petsc_options={"ksp_type": "preonly", "pc_type": "lu", "ksp_error_if_not_converged": True})
    for _ in range(steps):
        result = problem.solve()
        previous.x.array[:] = result.x.array
    expected = math.exp(-eta * math.pi ** 2 * duration)
    reference = fem.Function(S); reference.interpolate(lambda x: expected * np.sin(math.pi * x[0]))
    relative = float(np.linalg.norm(previous.x.array - reference.x.array) / np.linalg.norm(reference.x.array))
    report_data = {"dolfinx_version": dolfinx.__version__,
                   "hartmann": {"status": "passed" if hartmann[-1]["velocity_relative_l2"] < .005 and hartmann[-1]["induced_field_relative_l2"] < .03 else "failed",
                                "hartmann_number": ha, "b0_t": b0, "cases": hartmann,
                                "assumptions": "Steady laminar infinite channel, short-circuit Ey=0, no-slip plates; transverse magnetic pressure balanced by pressure."},
                   "magnetic_diffusion": {"status": "passed" if relative < .01 else "failed", "relative_amplitude_error": relative, "analytical_decay": expected, "steps": steps, "cells": 64},
                   "mhd_taylor_green": {"status": "pending", "reason": "Published periodic B/psi benchmark has not validated the A/phi torus/exterior VMS adaptation."},
                   "coupled_manufactured_solution": {"status": "pending", "reason": "No full thermal/u/p/A/phi manufactured-solution convergence study has been completed."},
                   "exterior_refinement": {"status": "pending", "reason": "Conductor/exterior refinement and truncation studies require separate scenario ensembles."}}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report_data, indent=2), encoding="utf-8")
    print(json.dumps(report_data), flush=True)


def build_mesh(config: ScenarioConfig):
    import gmsh
    from dolfinx.io import gmsh as gmshio
    from mpi4py import MPI
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("magnetic_torus")
    radius = config.exterior_radius_factor * (config.major_radius + config.minor_radius)
    sphere = gmsh.model.occ.addSphere(0, 0, 0, radius)
    torus = gmsh.model.occ.addTorus(0, 0, 0, config.major_radius, config.minor_radius)
    _, maps = gmsh.model.occ.fragment([(3, sphere)], [(3, torus)])
    gmsh.model.occ.synchronize()
    fluid_volumes = [tag for dim, tag in maps[1] if dim == 3]
    exterior_volumes = [tag for dim, tag in gmsh.model.getEntities(3) if tag not in fluid_volumes]
    gmsh.model.addPhysicalGroup(3, fluid_volumes, 1)
    gmsh.model.addPhysicalGroup(3, exterior_volumes, 2)
    h = 2 * config.minor_radius / config.resolution
    boundary = gmsh.model.getBoundary([(3, tag) for tag in fluid_volumes], oriented=False)
    surface_ids = [tag for dim, tag in boundary if dim == 2]
    gmsh.model.mesh.field.add("Distance", 1)
    gmsh.model.mesh.field.setNumbers(1, "FacesList", surface_ids)
    gmsh.model.mesh.field.setNumber(1, "Sampling", 60)
    gmsh.model.mesh.field.add("Threshold", 2)
    for name, value in (("InField", 1), ("SizeMin", h), ("SizeMax", radius / 2),
                        ("DistMin", 2 * h), ("DistMax", config.major_radius + config.minor_radius)):
        gmsh.model.mesh.field.setNumber(2, name, value)
    gmsh.model.mesh.field.setAsBackgroundMesh(2)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.Algorithm3D", 1)
    gmsh.model.mesh.generate(3)
    data = gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=3)
    gmsh.finalize()
    return data.mesh, data.cell_tags, radius


def run(config: ScenarioConfig, directory: Path) -> None:
    import basix.ufl
    import dolfinx
    from dolfinx import fem, mesh
    from dolfinx.fem.petsc import NonlinearProblem
    from mpi4py import MPI
    from petsc4py import PETSc
    import ufl
    from scipy.spatial import cKDTree
    if MPI.COMM_WORLD.size != 1:
        raise RuntimeError("The artifact writer currently requires a serial worker; launch without mpirun.")
    report(.01, "Meshing reservoir and surrounding vacuum")
    domain, tags, exterior_radius = build_mesh(config)
    topology = domain.topology
    topology.create_connectivity(3, 0)
    topology.create_connectivity(0, 3)
    topology.create_connectivity(2, 3)
    topology.create_connectivity(2, 0)
    ncells = topology.index_map(3).size_local
    cell_labels = np.zeros(ncells, dtype=np.int32)
    cell_labels[tags.indices] = tags.values
    fluid_cells = np.flatnonzero(cell_labels == 1).astype(np.int32)
    fluid_vertices = np.unique(np.concatenate([topology.connectivity(3, 0).links(int(c)) for c in fluid_cells])).astype(np.int32)
    all_vertices = np.arange(topology.index_map(0).size_local, dtype=np.int32)
    outside_vertices = np.setdiff1d(all_vertices, fluid_vertices).astype(np.int32)
    interface_facets = []
    for facet in range(topology.index_map(2).size_local):
        adjacent = topology.connectivity(2, 3).links(facet)
        if len(adjacent) == 2 and set(cell_labels[adjacent]) == {1, 2}:
            interface_facets.append(facet)
    interface_facets = np.asarray(interface_facets, dtype=np.int32)
    interface_vertices = np.unique(np.concatenate([topology.connectivity(2, 0).links(int(f)) for f in interface_facets])).astype(np.int32)
    exterior_facets = mesh.exterior_facet_indices(topology)
    facet_tags = mesh.meshtags(domain, 2, interface_facets, np.ones(len(interface_facets), dtype=np.int32))
    scalar = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
    vector = basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(3,))
    W = fem.functionspace(domain, basix.ufl.mixed_element([vector, scalar, vector, scalar, scalar]))
    S = fem.functionspace(domain, scalar)
    V = fem.functionspace(domain, vector)
    current, previous = fem.Function(W), fem.Function(W)
    spaces, maps = zip(*(W.sub(i).collapse() for i in range(5)))
    current.x.array[maps[4]] = config.initial_temperature
    previous.x.array[:] = current.x.array
    zero_functions = [fem.Function(space) for space in spaces]
    zero_functions[4].x.array[:] = config.initial_temperature
    bcs = []
    bc_node_indices = {}
    for index in (0, 1, 3, 4):
        vertices = np.union1d(outside_vertices, interface_vertices).astype(np.int32) if index == 0 else outside_vertices
        if index == 1:
            vertices = np.union1d(vertices, fluid_vertices[:1]).astype(np.int32)
        dofs = fem.locate_dofs_topological((W.sub(index), spaces[index]), 0, vertices)
        bcs.append(fem.dirichletbc(zero_functions[index], dofs, W.sub(index)))
        bc_node_indices[index] = np.asarray(dofs[1])
    adofs = fem.locate_dofs_topological((W.sub(2), spaces[2]), 2, exterior_facets)
    bcs.append(fem.dirichletbc(zero_functions[2], adofs, W.sub(2)))
    bc_node_indices[2] = np.asarray(adofs[1])
    points = S.tabulate_dof_coordinates().copy()
    nc = len(points)
    point_tree = cKDTree(points)
    orders = [point_tree.query(space.tabulate_dof_coordinates())[1] for space in spaces]
    cells = np.asarray([S.dofmap.cell_dofs(c) for c in range(ncells)], dtype=np.int32)
    tet_points = points[cells]
    affine = np.concatenate((np.ones((ncells, 4, 1)), tet_points), axis=2)
    inverse = np.linalg.inv(affine)
    gradients = inverse[:, 1:, :].transpose(0, 2, 1)
    volumes = np.abs(np.linalg.det(tet_points[:, 1:] - tet_points[:, :1])) / 6
    conductor_mask = np.zeros(nc, dtype=bool); conductor_mask[np.unique(cells[fluid_cells])] = True
    node_volumes = np.zeros(nc)
    for a in range(4):
        np.add.at(node_volumes, cells[fluid_cells, a], volumes[fluid_cells] / 4)
    actual_volume = volumes[fluid_cells].sum()
    if actual_volume <= 0:
        raise RuntimeError("Gmsh did not produce a conducting torus subdomain.")
    external_a = fem.Function(V)
    ext_order = point_tree.query(V.tabulate_dof_coordinates())[1]
    acoil = coil_vector_potential(points, config)
    # Excite an m=1 magnetic perturbation reproducibly instead of relying on
    # mesh roundoff to break the axisymmetry of the torus and pumping force.
    seed_angle = np.random.default_rng(config.seed).uniform(0., 2 * math.pi)
    seed_vector = config.seed_field * np.array([.1 * math.cos(seed_angle), .1 * math.sin(seed_angle), math.sqrt(.99)])
    seed_a = .5 * np.cross(seed_vector[None, :], points)
    external_previous = fem.Function(V)
    seed_scale = 1.0 if config.mode == "dynamo" else 0.0
    initial_external = acoil + seed_scale * seed_a
    external_a.x.array[:] = initial_external[ext_order].ravel()
    external_previous.x.array[:] = external_a.x.array
    dt = fem.Constant(domain, PETSc.ScalarType(config.duration / config.time_steps))
    heating = fem.Constant(domain, PETSc.ScalarType(config.heating_power / actual_volume))
    u, p, A, phi, T = ufl.split(current)
    up, pp, Ap, phip, Tp = ufl.split(previous)
    v, q, a, psi, s = ufl.TestFunctions(W)
    dx = ufl.Measure("dx", domain=domain, subdomain_data=tags, metadata={"quadrature_degree": 2})
    dS = ufl.Measure("dS", domain=domain, subdomain_data=facet_tags, metadata={"quadrature_degree": 2})
    class FormParameters:
        """Scenario values are coefficients, so editing inputs reuses the JIT cache."""
        def __init__(self):
            self.constants = {}
        def __getattr__(self, name):
            value = getattr(config, name)
            if isinstance(value, (float, int)) and not isinstance(value, bool):
                if name not in self.constants:
                    self.constants[name] = fem.Constant(domain, PETSc.ScalarType(value))
                return self.constants[name]
            return value
    coefficients = FormParameters()
    rho, mu = coefficients.density, coefficients.dynamic_viscosity
    def phase(temp):
        return ufl.max_value(0., ufl.min_value(1., (temp - coefficients.melting_temperature) / coefficients.transition_width + .5))
    def enthalpy(temp):
        delta, f = temp - coefficients.melting_temperature, phase(temp)
        middle = coefficients.heat_capacity_solid * delta + coefficients.latent_heat * f + .5 * (coefficients.heat_capacity_liquid - coefficients.heat_capacity_solid) * coefficients.transition_width * f * f
        return ufl.conditional(ufl.lt(delta, -coefficients.transition_width / 2), coefficients.heat_capacity_solid * delta,
                               ufl.conditional(ufl.gt(delta, coefficients.transition_width / 2), coefficients.latent_heat + coefficients.heat_capacity_liquid * delta, middle))
    fraction = phase(T)
    sigma = coefficients.electrical_conductivity_solid * (1 - fraction) + coefficients.electrical_conductivity_liquid * fraction
    eta = 1 / (MU0 * sigma)
    B = ufl.curl(A + external_a)
    Bprevious = ufl.curl(Ap + external_previous)
    At = (A + external_a - Ap - external_previous) / dt
    electric = -At - ufl.grad(phi)
    J = sigma * (electric + ufl.cross(u, B))
    damping = 1.e6 * (1 - fraction) ** 2 / (fraction ** 3 + 1.e-3)
    x = ufl.SpatialCoordinate(domain)
    radial = ufl.sqrt(x[0] ** 2 + x[1] ** 2 + 1.e-20)
    azimuthal = ufl.as_vector((-x[1] / radial, x[0] / radial, 0.))
    local_r = ufl.sqrt((radial - coefficients.major_radius) ** 2 + x[2] ** 2 + 1.e-20)
    poloidal = ufl.as_vector((-x[2] * x[0] / (local_r * radial), -x[2] * x[1] / (local_r * radial), (radial - coefficients.major_radius) / local_r))
    force = coefficients.pump_force * (azimuthal + coefficients.helical_fraction * poloidal) + ufl.as_vector((0., 0., rho * coefficients.gravity * coefficients.thermal_expansion * (T - coefficients.initial_temperature)))
    momentum = rho * (u - up) / dt + rho * ufl.dot(ufl.grad(u), u) + ufl.grad(p) - ufl.cross(J, B) + damping * u - force
    # Cellwise P1 viscous and curlcurl second derivatives vanish in the strong subscale residual.
    induction = (B - Bprevious) / dt - (ufl.dot(ufl.grad(u), B) - ufl.div(u) * B)
    cell_h = ufl.CellDiameter(domain)
    inv_h2 = 1 / (cell_h * cell_h)
    tau_u = 1 / ufl.sqrt((2 * rho / dt) ** 2 + rho * rho * ufl.dot(u, u) * inv_h2 + 3 * mu * mu * inv_h2 ** 2 + rho / MU0 * ufl.dot(B, B) * math.sqrt(3) * inv_h2 + damping * damping)
    tau_b = 1 / ufl.sqrt((2 / dt) ** 2 + (ufl.dot(u, u) + ufl.dot(B, B) / (MU0 * rho)) * inv_h2 + 3 * eta * eta * inv_h2 ** 2)
    usub, bsub = -tau_u * momentum, -tau_b * induction
    nu_t = .15 * cell_h * ufl.sqrt(ufl.dot(usub, usub) + ufl.dot(bsub, bsub) / (MU0 * rho) + 1.e-30) if config.turbulence else 0.
    cross_stress = rho * (ufl.outer(u, usub) + ufl.outer(usub, u)) - (ufl.outer(B, bsub) + ufl.outer(bsub, B) - ufl.dot(B, bsub) * ufl.Identity(3)) / MU0
    emf = ufl.cross(usub, B) + ufl.cross(u, bsub) if config.turbulence else ufl.as_vector((0., 0., 0.))
    F = (ufl.dot(momentum, v) + 2 * (mu + rho * nu_t) * ufl.inner(ufl.sym(ufl.grad(u)), ufl.sym(ufl.grad(v)))
         + q * ufl.div(u) - ufl.dot(ufl.grad(q), usub)) * dx(1)
    if config.turbulence:
        F += (-ufl.inner(ufl.grad(v), cross_stress) + inv_h2 ** -1 * ufl.div(u) * ufl.div(v) / (3 * tau_u)) * dx(1)
    # Quasistatic vacuum curlcurl with Coulomb gauge.  Coil/seed are analytic imposed A.
    F += ((ufl.inner(ufl.curl(A), ufl.curl(a)) + ufl.div(A) * ufl.div(a)) / MU0) * dx
    F += (sigma * ufl.dot(At + ufl.grad(phi) - ufl.cross(u, B) - emf, a)
          + sigma * nu_t * ufl.inner(ufl.curl(A + external_a), ufl.curl(a))
          - ufl.dot(J, ufl.grad(psi)) + 1.e-10 * phi * psi) * dx(1)
    H, Hp = enthalpy(T), enthalpy(Tp)
    heat_k = coefficients.thermal_conductivity + rho * coefficients.heat_capacity_liquid * nu_t / coefficients.thermal_turbulent_prandtl
    F += (rho * (H - Hp) * s / dt - rho * H * ufl.dot(u, ufl.grad(s)) + heat_k * ufl.dot(ufl.grad(T), ufl.grad(s))
          - (heating + ufl.dot(J, J) / sigma + 2 * mu * ufl.inner(ufl.sym(ufl.grad(u)), ufl.sym(ufl.grad(u))) + damping * ufl.dot(u, u)) * s) * dx(1)
    F += coefficients.emissivity * STEFAN_BOLTZMANN * (ufl.avg(T) ** 4 - coefficients.ambient_temperature ** 4) * ufl.avg(s) * dS(1)
    report(.04, f"Compiling coupled forms · {nc} nodes / {ncells} tetrahedra")
    # Direct factors can exceed the mesh storage by orders of magnitude. Bound
    # MUMPS itself instead of letting the OS kill a desktop/transport worker.
    meminfo = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    available_mb = int(meminfo["MemAvailable"].strip().split()[0]) // 1024
    memory_mb = int(os.environ.get("MAGNETIC_TORUS_MUMPS_MB", str(min(4096, max(256, available_mb // 2)))))
    if memory_mb < 128:
        raise ValueError("MAGNETIC_TORUS_MUMPS_MB must allow at least 128 MB for the direct solver.")
    if nc > 8000:
        report(.04, f"Large coupled mesh: {nc} nodes; direct-factor ceiling {memory_mb} MB. Reduce resolution if factorization fails.")
    problem = NonlinearProblem(F, current, bcs=bcs, petsc_options_prefix="torus_",
                               petsc_options={"snes_type": "newtonls", "snes_linesearch_type": "bt", "snes_rtol": 1.e-8,
                                              "snes_atol": 1.e-7, "snes_max_it": 30, "ksp_type": "preonly", "pc_type": "lu",
                                              "pc_factor_mat_solver_type": "mumps", "ksp_error_if_not_converged": True,
                                              "mat_mumps_icntl_23": memory_mb},
                               jit_options={"cffi_extra_compile_args": ["-O1"]})
    interface_faces = []
    for facet in interface_facets:
        vertices = topology.connectivity(2, 0).links(int(facet))
        vertex_coords = mesh.entities_to_geometry(domain, 0, vertices, False).ravel()
        face_points = domain.geometry.x[vertex_coords]
        interface_faces.append(point_tree.query(face_points)[1])
    faces = np.asarray(interface_faces, dtype=np.int32)
    face_areas = np.linalg.norm(np.cross(points[faces[:, 1]] - points[faces[:, 0]], points[faces[:, 2]] - points[faces[:, 0]]), axis=1) / 2
    fields = {name: [] for name in ("temperature", "enthalpy", "liquid_fraction", "velocity", "pressure", "vector_potential", "electric_potential", "magnetic_field")}
    diagnostics = {name: [] for name in ("energy_j", "recoverable_energy_j", "usable_electric_energy_j", "liquid_fraction", "magnetic_energy_j", "imposed_field_energy_j", "induced_field_energy_j", "mutual_field_energy_j", "pump_power_w", "pump_energy_j", "net_power_w", "joule_power_w", "velocity_divergence_l2", "energy_balance_error_j", "newton_iterations")}
    external_history = []
    def extract(index):
        arr = current.x.array[maps[index]].reshape(-1, 3) if index in (0, 2) else current.x.array[maps[index]]
        out = np.empty((nc, 3)) if index in (0, 2) else np.empty(nc)
        out[orders[index]] = arr
        return out
    def sample(t, iterations=0):
        U, P, Ai, Phi, Temp = [extract(i) for i in range(5)]
        ext = external_a.x.array.reshape(-1, 3)
        ext_points = np.empty_like(ext); ext_points[ext_order] = ext
        Atotal = Ai + ext_points
        gradA = np.einsum("cai,caj->cij", Atotal[cells], gradients)
        Bcell = np.column_stack((gradA[:, 2, 1] - gradA[:, 1, 2], gradA[:, 0, 2] - gradA[:, 2, 0], gradA[:, 1, 0] - gradA[:, 0, 1]))
        Bnode, global_weights = np.zeros((nc, 3)), np.zeros(nc)
        for node in range(4):
            np.add.at(Bnode, cells[:, node], Bcell * volumes[:, None]); np.add.at(global_weights, cells[:, node], volumes)
        Bnode /= global_weights[:, None]
        Hnodal = regularized_enthalpy(Temp, config)
        fnodal = np.clip((Temp - config.melting_temperature) / config.transition_width + .5, 0, 1)
        for key, value in (("temperature", Temp), ("enthalpy", Hnodal), ("liquid_fraction", fnodal), ("velocity", U), ("pressure", P), ("vector_potential", Atotal), ("electric_potential", Phi), ("magnetic_field", Bnode)):
            fields[key].append(value.copy())
        external_history.append(ext_points.copy())
        energy = fem.assemble_scalar(fem.form(rho * H * dx(1)))
        recoverable = fem.assemble_scalar(fem.form(rho * ufl.max_value(H - float(regularized_enthalpy(config.discharge_temperature, config)), 0) * dx(1)))
        fl = fem.assemble_scalar(fem.form(fraction * dx(1))) / actual_volume
        bm = fem.assemble_scalar(fem.form(ufl.dot(B, B) / (2 * MU0) * dx))
        imposed_b = ufl.curl(external_a)
        imposed_energy = fem.assemble_scalar(fem.form(ufl.dot(imposed_b, imposed_b) / (2 * MU0) * dx))
        induced_energy = fem.assemble_scalar(fem.form(ufl.dot(ufl.curl(A), ufl.curl(A)) / (2 * MU0) * dx))
        ppump = fem.assemble_scalar(fem.form(config.pump_force * ufl.dot(azimuthal + config.helical_fraction * poloidal, u) * dx(1)))
        pj = fem.assemble_scalar(fem.form(ufl.dot(J, J) / sigma * dx(1)))
        pr = fem.assemble_scalar(fem.form(config.emissivity * STEFAN_BOLTZMANN * (ufl.avg(T) ** 4 - config.ambient_temperature ** 4) * dS(1)))
        divu = math.sqrt(max(0, fem.assemble_scalar(fem.form(ufl.div(u) ** 2 * dx(1)))))
        dissipation = fem.assemble_scalar(fem.form((2 * mu * ufl.inner(ufl.sym(ufl.grad(u)), ufl.sym(ufl.grad(u))) + damping * ufl.dot(u, u)) * dx(1)))
        net_heat = float(heating.value) * actual_volume + pj + dissipation - pr
        cumulative = 0. if not diagnostics["energy_j"] else (diagnostics["energy_j"][-1] - diagnostics["energy_j"][0] - diagnostics["energy_balance_error_j"][-1] + float(dt.value) * net_heat)
        balance = 0. if not diagnostics["energy_j"] else energy - diagnostics["energy_j"][0] - cumulative
        pump_energy = 0. if not diagnostics["pump_energy_j"] else diagnostics["pump_energy_j"][-1] + float(dt.value) * ppump
        for key, value in (("energy_j", energy), ("recoverable_energy_j", recoverable), ("usable_electric_energy_j", config.efficiency * recoverable), ("liquid_fraction", fl), ("magnetic_energy_j", bm), ("imposed_field_energy_j", imposed_energy), ("induced_field_energy_j", induced_energy), ("mutual_field_energy_j", bm - imposed_energy - induced_energy), ("pump_power_w", ppump), ("pump_energy_j", pump_energy), ("net_power_w", net_heat), ("joule_power_w", pj), ("velocity_divergence_l2", divu), ("energy_balance_error_j", balance), ("newton_iterations", iterations)):
            diagnostics[key].append(value)
    times = [0.]
    sample(0.)
    nominal = config.duration / config.time_steps
    next_dt = nominal
    t = 0.
    rejections = _TimestepRejections()
    while t < config.duration - 1.e-12:
        if (directory / "cancel").exists():
            raise RuntimeError("Simulation cancelled at accepted timestep boundary.")
        step = min(next_dt, config.duration - t)
        for event in (config.duration * config.charge_fraction, config.seed_duration):
            if t + 1.e-12 < event < t + step - 1.e-12:
                step = event - t
        dt.value = step
        heating.value = source_power(config, t + .5 * step) / actual_volume
        scale = max(0., 1 - (t + step) / config.seed_duration) if config.mode == "dynamo" and config.seed_duration > 0 else 0.
        external_a.x.array[:] = (acoil + scale * seed_a)[ext_order].ravel()
        current.x.array[:] = previous.x.array
        try:
            problem.solve()
            reason = int(problem.solver.getConvergedReason())
            if reason <= 0 or not np.all(np.isfinite(current.x.array)):
                raise RuntimeError(f"SNES convergence reason {reason}")
            if np.min(extract(4)[conductor_mask]) <= 0:
                raise RuntimeError("Temperature crossed absolute zero.")
        except Exception as exc:
            next_dt = step / 2
            current.x.array[:] = previous.x.array
            if rejections.rejected(next_dt, nominal / 256):
                raise RuntimeError(f"Coupled solve failed after timestep refinement at t={t:g}: {exc}. "
                                   f"For a factorization/memory failure, reduce mesh resolution (start at 3); "
                                   f"MUMPS factor memory is limited to {memory_mb} MB by MAGNETIC_TORUS_MUMPS_MB.") from exc
            report(.05 + .9 * t / config.duration, f"Rejected timestep at {t:.4g} s; retrying Δt={next_dt:.3g}")
            continue
        t += step; times.append(t)
        sample(t, problem.solver.getIterationNumber())
        previous.x.array[:] = current.x.array
        external_previous.x.array[:] = external_a.x.array
        rejections.accepted()
        next_dt = min(nominal, step * 1.5)
        report(.05 + .9 * t / config.duration, f"Accepted full MHD snapshot · t={t:.4g} s · Newton {problem.solver.getIterationNumber()}")
    # The identical local operators are consumed by differentiable PINN residuals.
    bc_nodes = {}
    for index, indices in bc_node_indices.items():
        if index in (0, 2):
            bc_nodes[str(index)] = np.unique(orders[index][indices // 3]).tolist()
        else:
            bc_nodes[str(index)] = orders[index][indices].tolist()
    np.savez_compressed(directory / "weak_operators.npz", points=points, cells=cells,
                        gradients=gradients, volumes=volumes, cell_labels=cell_labels,
                        cell_diameter=np.max(np.linalg.norm(tet_points[:, :, None] - tet_points[:, None, :], axis=-1), axis=(1, 2)),
                        boundary_faces=faces, boundary_areas=face_areas,
                        imposed_vector_potential=np.asarray(external_history), node_volumes=node_volumes)
    np.savez_compressed(directory / "exterior_fields.npz", points=points, cells=cells, times=np.asarray(times),
                        magnetic_field=np.asarray(fields["magnetic_field"]), vector_potential=np.asarray(fields["vector_potential"]))
    np.savez_compressed(directory / "result.npz", times=np.asarray(times), points=points,
                        **{"field_" + k: np.asarray(v) for k, v in fields.items()},
                        **{"diag_" + k: np.asarray(v) for k, v in diagnostics.items()})
    metadata = {"backend": "DOLFINx/PETSc full 3D conductor–vacuum A–phi MHD", "dolfinx_version": dolfinx.__version__,
                "validation_status": "experimental; benchmark checks do not establish torus/turbulence validation",
                "model": "incompressible resistive MHD, Boussinesq buoyancy, enthalpy-porosity, Coulomb gauge",
                "closure": "Sondak mixed residual VMS A–phi adaptation" if config.turbulence else "laminar with pressure residual stabilization",
                "closure_reference": "https://arxiv.org/abs/1412.1055", "closure_coefficient": .15,
                "thermal_turbulent_prandtl": config.thermal_turbulent_prandtl, "transition_smoothing_k": config.transition_width,
                "cells": cells.tolist(), "conductor_cells": cells[fluid_cells].tolist(), "conductor_mask": conductor_mask.tolist(),
                "node_volumes": node_volumes.tolist(), "mesh_volume_m3": actual_volume, "analytic_volume_m3": config.volume,
                "volume_relative_error": abs(actual_volume - config.volume) / config.volume,
                "mumps_memory_limit_mb": memory_mb,
                "exterior_radius_m": exterior_radius, "imposed_coil": imposed_coil(config),
                "exterior_field_path": "exterior_fields.npz", "weak_operator_path": "weak_operators.npz",
                "pinn_physics": {"kind": "fenics_a_phi_vms_v1", "bc_nodes": bc_nodes, "quadrature_degree": 2},
                "time_rejections": rejections.total, "material_provenance": config.provenance, "units": "SI"}
    metadata["seed_field_vector_t"] = seed_vector.tolist()
    metadata["gauge_treatment"] = "Coulomb div-div penalty for induced A; pressure point anchor; phi mass gauge 1e-10"
    metadata["energy_budget_note"] = "Thermal balance includes molecular Joule/viscous/Darcy dissipation; full resolved+subgrid MHD energy closure is not validated."
    metadata["magnetic_energy_domain"] = "Full truncated conductor plus vacuum FE enclosure."
    metadata["coil_energy_note"] = "Ideal imposed current source. Imposed/induced/mutual stored magnetic energy is reported separately; coil circuit resistance and electrical consumption are not modeled."
    try:
        metadata["mumps_memory_limit_verified_mb"] = int(problem.solver.getKSP().getPC().getFactorMatrix().getMumpsIcntl(23))
    except Exception:
        # A stationary zero-residual case can converge without constructing LU.
        metadata["mumps_memory_limit_verified_mb"] = None
    (directory / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    report(1., "Full MHD run saved with FE operators and global field map")


if __name__ == "__main__":
    try:
        if len(sys.argv) > 1 and sys.argv[1] == "--benchmarks":
            run_special_case_benchmarks(Path(sys.argv[2]))
            sys.exit(0)
        request, output = Path(sys.argv[1]), Path(sys.argv[2])
        output.mkdir(parents=True, exist_ok=True)
        try:
            os.setsid()
        except PermissionError:
            pass
        def cancellation_watchdog():
            while True:
                if (output / "cancel").exists():
                    # Terminate this isolated worker, including a long JIT/SNES
                    # call. No accepted desktop result is modified by this job.
                    if os.getpgrp() == os.getpid():
                        os.killpg(os.getpgrp(), signal.SIGTERM)
                    else:
                        os.kill(os.getpid(), signal.SIGTERM)
                time.sleep(.1)
        threading.Thread(target=cancellation_watchdog, daemon=True).start()
        run(ScenarioConfig.from_dict(json.loads(request.read_text(encoding="utf-8"))), output)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
