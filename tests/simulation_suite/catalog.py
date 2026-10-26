"""Complete, deterministic case catalog and dependency validation."""

from __future__ import annotations

import ast
import math
from pathlib import Path
import re

from .spec import CaseSpec

_ISOLATED_FE_METHODS = (
    "test_torus_physics.ThermalPhysicsTests.test_full_fe_stationary_closed_system_and_weak_operator",
    "test_torus_physics.ThermalPhysicsTests.test_full_fe_driven_vms_matches_differentiable_equations",
)


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


def unit_cases(root: Path, *, full: bool) -> list[CaseSpec]:
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
                ),
            )
        )
    if full:
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


def all_cases(root: Path, *, profile: str = "full") -> tuple[list[CaseSpec], dict]:
    full = profile == "full"
    cases = unit_cases(root, full=full)
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
        cases.extend(
            [
                CaseSpec(
                    "desktop.journey",
                    "desktop",
                    "Real repeated-run, source, probe, export and cancellation journey",
                    ("{python}", "-B", "{root}/scripts/desktop_smoke.py"),
                    900,
                    ("desktop",),
                    dependencies=("magnetic_torus/**/*.py", "scripts/desktop_smoke.py"),
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
                    600,
                    ("desktop",),
                    parameters={"minimum_fps": 30.0},
                    dependencies=(
                        "magnetic_torus/**/*.py",
                        "scripts/benchmark_desktop.py",
                    ),
                ),
            ]
        )
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
