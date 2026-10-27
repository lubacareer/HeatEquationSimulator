"""Complete, deterministic case catalog and dependency validation."""

from __future__ import annotations

import ast
from dataclasses import replace
import math
from pathlib import Path
import re

from .spec import CaseSpec

_ISOLATED_FE_METHODS = (
    "test_torus_physics.ThermalPhysicsTests.test_full_fe_stationary_closed_system_and_weak_operator",
    "test_torus_physics.ThermalPhysicsTests.test_full_fe_driven_vms_matches_differentiable_equations",
)
_GEANT4_INTEGRATION_METHODS = (
    "test_torus_radiation.Geant4IntegrationTests.test_water_paired_zero_field_transport",
    "test_torus_radiation.Geant4IntegrationTests.test_ground_water_transport_uses_above_ground_incident_source",
    "test_torus_radiation.Geant4IntegrationTests.test_male_icrp110_transport",
    "test_torus_radiation.Geant4IntegrationTests.test_female_icrp110_transport",
)
_EXTRA_UNIT_DEPENDENCIES = {
    "test_pressure_diagnostics.py": ("scripts/pinn_pressure_diagnostics.py",),
    "test_pinn_pressure_pilots.py": ("scripts/pinn_research.py", "scripts/research_supervisor.py"),
}


def _module_capabilities(path: Path) -> tuple[str, ...]:
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "SUITE_CAPABILITIES"
            for t in node.targets
        ):
            return tuple(ast.literal_eval(node.value))
    if path.name in {
        "test_scene_updates.py",
        "test_torus_ui.py",
        "test_torus_charts.py",
    }:
        return ("desktop",)
    if path.name == "test_torus_physics.py":
        return ("desktop",)
    if path.name == "test_torus_radiation.py":
        return ("geant4", "icrp110")
    if path.name == "test_torus_pinn.py":
        return ("torch_cpu", "torch_cuda")
    return ()


def unit_cases(
    root: Path, *, full: bool, include_integrations: bool = True
) -> list[CaseSpec]:
    cases = []
    for path in sorted((root / "tests").glob("test_*.py")):
        caps = _module_capabilities(path)
        if not full and "desktop" in caps and path.name != "test_torus_physics.py":
            continue
        cases.append(
            CaseSpec(
                id="unit." + path.stem,
                category="regression",
                description=f"Isolated {path.name}",
                command=(
                    "{python}",
                    "-B",
                    "{root}/tests/simulation_suite/unittest_worker.py",
                    "--request",
                    "{case_dir}/request.json",
                    "--result",
                    "{case_dir}/result.json",
                ),
                timeout=900 if full and caps else 300,
                capabilities=caps if full else (),
                parameters={
                    "pattern": path.name,
                    "exclude_ids": list(_ISOLATED_FE_METHODS)
                    if path.name == "test_torus_physics.py"
                    else [],
                },
                dependencies=(
                    "magnetic_torus/**/*.py",
                    "backend.py",
                    "interface3.py",
                    "np_compat.py",
                    "tests/*.py",
                    "tests/simulation_suite/**/*.py",
                    "scripts/setup_*.sh",
                ) + _EXTRA_UNIT_DEPENDENCIES.get(path.name, ()),
            )
        )
    if full and include_integrations:
        for identifier in _ISOLATED_FE_METHODS:
            cases.append(
                CaseSpec(
                    "integration." + identifier.rsplit(".", 1)[-1],
                    "integration",
                    "Isolated FE/PyTorch regression (kept separate from native SciPy tests)",
                    (
                        "{python}",
                        "-B",
                        "{root}/tests/simulation_suite/unittest_worker.py",
                        "--request",
                        "{case_dir}/request.json",
                        "--result",
                        "{case_dir}/result.json",
                    ),
                    900,
                    ("fenics", "torch_cpu"),
                    parameters={"test_id": identifier},
                    dependencies=(
                        "magnetic_torus/**/*.py",
                        "tests/test_torus_physics.py",
                    ),
                )
            )
    return cases


def _desktop_cases(*, representative: bool = False) -> list[CaseSpec]:
    return [
        CaseSpec(
            "desktop.journey",
            "desktop",
            "Real repeated-run, source, probe, export and cancellation journey",
            ("{python}", "-B", "{root}/scripts/desktop_smoke.py"),
            180 if representative else 900,
            ("desktop",),
            dependencies=(
                "magnetic_torus/**/*.py",
                "interface3.py",
                "scripts/desktop_smoke.py",
            ),
        ),
        CaseSpec(
            "desktop.performance",
            "performance",
            "Five-repeat resolution-8 playback including Qt events",
            (
                "{python}",
                "-B",
                "{root}/scripts/benchmark_desktop.py",
                "--output",
                "{case_dir}/benchmark.json",
            ),
            120 if representative else 600,
            ("desktop",),
            parameters={"minimum_fps": 30.0},
            dependencies=(
                "magnetic_torus/**/*.py",
                "interface3.py",
                "scripts/benchmark_desktop.py",
            ),
        ),
    ]


def all_cases(root: Path, *, profile: str = "features") -> tuple[list[CaseSpec], dict]:
    if profile not in {"features", "full", "fast"}:
        raise ValueError(f"Unknown test profile: {profile}")
    full = profile == "full"
    cases = unit_cases(root, full=profile != "fast", include_integrations=full)
    if profile == "features":
        cases = [
            replace(
                case,
                capabilities=(),
                parameters={
                    **case.parameters,
                    "exclude_ids": list(_GEANT4_INTEGRATION_METHODS),
                },
            )
            if case.id == "unit.test_torus_radiation"
            else case
            for case in cases
        ]
    coverage = {
        "profile": profile,
        "modern_workspace": True,
        "legacy": "Existing enthalpy regression tests retained; full legacy workflow excluded.",
        "unittest_modules": [
            case.parameters["pattern"] for case in cases if "pattern" in case.parameters
        ],
    }
    if full:
        from .scientific_catalog import matrix_coverage, scientific_cases

        scientific = scientific_cases()
        cases.extend(scientific)
        coverage["scientific_matrix"] = matrix_coverage(scientific)
        cases.extend(_desktop_cases())
    elif profile == "features":
        from .feature_catalog import feature_cases, feature_coverage

        # Keep the long precision ensemble after the visual checks.
        representatives = feature_cases()
        cases.extend(
            case for case in representatives if case.id != "feature.dose_water"
        )
        cases.extend(_desktop_cases(representative=True))
        cases.extend(
            case for case in representatives if case.id == "feature.dose_water"
        )
        coverage["feature_matrix"] = feature_coverage(cases, root=root)
    validate_cases(cases)
    return cases, coverage


def validate_cases(cases: list[CaseSpec]) -> None:
    for case in cases:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", case.id):
            raise ValueError(f"Unsafe test case ID: {case.id}")
        if (
            isinstance(case.timeout, bool)
            or not isinstance(case.timeout, (int, float))
            or not math.isfinite(case.timeout)
            or case.timeout <= 0
        ):
            raise ValueError(f"Case timeout must be finite and positive: {case.id}")
    by_id = {case.id: case for case in cases}
    if len(by_id) != len(cases):
        raise ValueError("Duplicate test case IDs.")
    visited, active = set(), set()

    def visit(identifier):
        if identifier not in by_id:
            raise ValueError(f"Missing dependency: {identifier}")
        if identifier in active:
            raise ValueError(f"Cyclic test dependency: {identifier}")
        if identifier in visited:
            return
        active.add(identifier)
        for parent in by_id[identifier].depends_on:
            visit(parent)
        active.remove(identifier)
        visited.add(identifier)

    for identifier in by_id:
        visit(identifier)


def dependency_order(cases: list[CaseSpec]) -> list[CaseSpec]:
    validate_cases(cases)
    by_id, ordered, visited = {case.id: case for case in cases}, [], set()

    def add(case):
        if case.id not in visited:
            for dependency in case.depends_on:
                add(by_id[dependency])
            visited.add(case.id)
            ordered.append(case)

    for case in cases:
        add(case)
    return ordered
