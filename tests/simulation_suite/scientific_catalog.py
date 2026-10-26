"""Finite, reproducible simulation coverage; no scientific runtimes at import."""

from __future__ import annotations

from itertools import combinations, product
from typing import Any

from .spec import CaseSpec

COMMAND = (
    "{python}",
    "-B",
    "{root}/tests/simulation_suite/scientific.py",
    "--request",
    "{case_dir}/request.json",
    "--result",
    "{case_dir}/result.json",
)
PRESETS = ("Iron", "Lunar habitat", "Regolith composite", "Thermal benchmark")
BORIS_FACTORS = {
    "species": ("proton", "alpha", "iron"),
    "energy": (10.0, 100.0, 1000.0),
    "source": ("beam", "isotropic", "spectrum"),
    "ground": (False, True),
    "field": ("zero", "imposed", "driven", "dynamo"),
}
DOSE_FACTORS = {
    "species": ("proton", "alpha", "iron"),
    "energy": (10.0, 100.0, 1000.0),
    "source": ("beam", "isotropic", "spectrum"),
    "ground": (False, True),
    "field": ("zero", "imposed", "driven", "dynamo", "pinn"),
    "target": ("water", "male", "female"),
    "physics_list": ("Shielding", "QGSP_BIC_HP"),
}


def _pairs(row: dict[str, Any]) -> set[tuple[str, Any, str, Any]]:
    return {(a, row[a], b, row[b]) for a, b in combinations(row, 2)}


def pairwise_rows(factors: dict[str, tuple]) -> list[dict[str, Any]]:
    """Greedy feasible-pair coverage with stable product-order tie breaking."""
    rows = [dict(zip(factors, values)) for values in product(*factors.values())]
    row_pairs = [_pairs(row) for row in rows]
    uncovered = set().union(*row_pairs)
    selected = []
    while uncovered:
        best = max(
            range(len(rows)), key=lambda index: len(row_pairs[index] & uncovered)
        )
        gain = row_pairs[best] & uncovered
        if not gain:
            raise AssertionError("Unreachable factor-pair coverage")
        selected.append(rows[best])
        uncovered.difference_update(gain)
    return selected


def pair_coverage(
    factors: dict[str, tuple], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    required = set().union(
        *(_pairs(dict(zip(factors, values))) for values in product(*factors.values()))
    )
    covered = set().union(*(_pairs(row) for row in rows)) if rows else set()
    missing = required - covered
    levels = {
        key: [
            value for value in values if not any(row.get(key) == value for row in rows)
        ]
        for key, values in factors.items()
    }
    return {
        "required_pairs": len(required),
        "covered_pairs": len(required & covered),
        "missing_pairs": sorted(missing, key=repr),
        "missing_levels": levels,
        "complete": not missing and not any(levels.values()),
        "cases": len(rows),
    }


def _case(
    case_id,
    action,
    description,
    *,
    depends=(),
    capabilities=(),
    timeout=300.0,
    scientific=True,
    **parameters,
):
    return CaseSpec(
        case_id,
        "scientific" if scientific else "simulation",
        description,
        COMMAND,
        timeout,
        tuple(capabilities),
        tuple(depends),
        {"action": action, **parameters},
        "scientific" if scientific else "functional",
        (
            "tests/simulation_suite/scientific.py",
            "tests/simulation_suite/scientific_catalog.py",
            "magnetic_torus/**/*.py",
            "magnetic_torus/geant4/**/*",
        ),
    )


def transport_source(field: str, ground: bool) -> str:
    return f"fe.transport.{field}.{'ground' if ground else 'vacuum'}"


def scientific_cases() -> list[CaseSpec]:
    cases = []
    for name in PRESETS:
        token = name.lower().replace(" ", "-")
        backend = "native" if name == "Thermal benchmark" else "fe"
        caps = () if backend == "native" else ("fenics", "torch_cpu")
        cases.append(
            _case(
                f"preset.{token}",
                "solve",
                f"As-shipped {name} simulation",
                capabilities=caps,
                timeout=1800.0,
                preset=name,
                recipe="preset",
            )
        )
    for recipe in (
        "solid_closed",
        "solid_heating",
        "liquid_cooling",
        "melting",
        "freezing",
        "radiating",
        "zero_latent",
        "charge_zero",
        "charge_one",
        "charge_roundoff",
        "efficiency_zero",
        "efficiency_one",
    ):
        cases.append(
            _case(
                "native." + recipe,
                "solve",
                "Native invariant: " + recipe.replace("_", " "),
                recipe=recipe,
            )
        )
    for material in ("Iron", "Lunar habitat", "Regolith composite"):
        cases.append(
            _case(
                "native.material." + material.lower().replace(" ", "-"),
                "solve",
                f"Native transport with {material} material",
                recipe="material",
                preset=material,
            )
        )
    cases.append(
        _case(
            "native.reference",
            "references",
            "Analytical Stefan, enthalpy and coil reference checks",
            timeout=300.0,
        )
    )
    for phase, mode, closure in product(
        ("solid", "mushy", "liquid"), ("driven", "dynamo"), (False, True)
    ):
        case_id = f"fe.{phase}.{mode}.{'vms' if closure else 'laminar'}"
        cases.append(
            _case(
                case_id,
                "solve",
                f"FE {phase}, {mode}, {'VMS' if closure else 'laminar'}",
                capabilities=("fenics", "torch_cpu"),
                timeout=600.0,
                recipe="fe_matrix",
                phase=phase,
                mode=mode,
                closure=closure,
            )
        )
    for recipe in ("fe_melting", "fe_freezing", "fe_stationary"):
        cases.append(
            _case(
                "fe." + recipe.removeprefix("fe_"),
                "solve",
                "FE invariant: " + recipe,
                capabilities=("fenics", "torch_cpu"),
                timeout=600.0,
                recipe=recipe,
            )
        )
    cases.append(
        _case(
            "fe.references",
            "fe_references",
            "Actual FE Hartmann and magnetic-diffusion reference reductions",
            capabilities=("fenics",),
            timeout=360.0,
        )
    )
    for ground, field in product(
        (False, True), ("zero", "imposed", "driven", "dynamo")
    ):
        cases.append(
            _case(
                transport_source(field, ground),
                "solve",
                f"Accepted {field} field for {'ground' if ground else 'vacuum'} transport",
                capabilities=("fenics", "torch_cpu"),
                timeout=600.0,
                recipe="transport",
                field=field,
                ground=ground,
            )
        )
    for device in ("cpu", "cuda"):
        caps = ("torch_cpu",) if device == "cpu" else ("torch_cuda",)
        cases.append(
            _case(
                f"pinn.thermal.{device}",
                "pinn",
                f"Native PINN accuracy/conservation on {device}",
                depends=("preset.thermal-benchmark",),
                capabilities=caps,
                timeout=1800.0,
                source="preset.thermal-benchmark",
                device=device,
                epochs=1000,
                batch=256,
            )
        )
        cases.append(
            _case(
                f"pinn.stationary.{device}",
                "pinn",
                f"Stationary coupled PINN on {device}",
                depends=("fe.stationary",),
                capabilities=caps,
                timeout=1200.0,
                source="fe.stationary",
                device=device,
                epochs=250,
                batch=256,
            )
        )
    for source in (
        "preset.iron",
        "preset.lunar-habitat",
        "preset.regolith-composite",
        "fe.mushy.driven.vms",
        "fe.liquid.dynamo.vms",
        "fe.freezing",
    ):
        cases.append(
            _case(
                "pinn." + source,
                "pinn",
                "Coupled held-out qualification: " + source,
                depends=(source,),
                capabilities=("torch_cuda",),
                timeout=2400.0,
                source=source,
                device="cuda",
                epochs=1000,
                batch=256,
            )
        )
    for ground in (False, True):
        source = transport_source("zero", ground)
        model_id = f"pinn.transport.zero.{'ground' if ground else 'vacuum'}"
        cases.append(
            _case(
                model_id,
                "pinn",
                "Qualified stationary PINN field for transport",
                depends=(source,),
                capabilities=("torch_cuda",),
                timeout=1200.0,
                source=source,
                device="cuda",
                epochs=250,
                batch=256,
            )
        )
    for index, row in enumerate(pairwise_rows(BORIS_FACTORS), 1):
        deps = (
            (transport_source(row["field"], row["ground"]),)
            if row["field"] in {"driven", "dynamo"}
            else ()
        )
        cases.append(
            _case(
                f"boris.pair.{index:02d}",
                "boris",
                "Boris interaction: " + repr(row),
                depends=deps,
                timeout=300.0,
                scientific=False,
                **row,
            )
        )
    for index, row in enumerate(pairwise_rows(DOSE_FACTORS), 1):
        field = row["field"]
        source = transport_source("zero" if field == "pinn" else field, row["ground"])
        deps = [source]
        if field == "pinn":
            deps.append(
                f"pinn.transport.zero.{'ground' if row['ground'] else 'vacuum'}"
            )
        caps = ("geant4", "icrp110") if row["target"] != "water" else ("geant4",)
        cases.append(
            _case(
                f"dose.pair.{index:02d}",
                "dose",
                "Geant4 interaction: " + repr(row),
                depends=deps,
                capabilities=caps,
                timeout=900.0,
                scientific=False,
                histories=128,
                **row,
            )
        )
    cases.append(
        _case(
            "dose.mixed-spectrum",
            "dose",
            "Mixed proton/alpha/iron spectrum with isotropic incidence",
            depends=(transport_source("zero", False),),
            capabilities=("geant4",),
            timeout=900.0,
            scientific=False,
            field="zero",
            ground=False,
            species="mixed",
            energy=100.0,
            source="spectrum",
            spectrum_angular="isotropic",
            target="water",
            physics_list="Shielding",
            histories=256,
        )
    )
    cases.append(
        _case(
            "dose.fluence-linearity",
            "dose_linearity",
            "Paired fluence normalization and event uncertainty",
            depends=(transport_source("zero", False),),
            capabilities=("geant4",),
            timeout=1800.0,
            histories=2048,
        )
    )
    cases.append(
        _case(
            "dose.water-precision",
            "dose_precision",
            "Water reference must reach at most 5% relative standard error",
            depends=(transport_source("zero", False),),
            capabilities=("geant4",),
            timeout=3600.0,
            maximum_histories=262144,
        )
    )
    for invalid in (
        "native_dose",
        "failed_pinn",
        "small_anatomy",
        "stale_pinn",
        "out_of_map",
    ):
        depends = (
            (transport_source("zero", False),)
            if invalid in {"failed_pinn", "stale_pinn", "out_of_map"}
            else ()
        )
        caps = ("geant4", "icrp110") if invalid == "small_anatomy" else ()
        cases.append(
            _case(
                "invalid." + invalid,
                "invalid",
                "Explicit rejection: " + invalid.replace("_", " "),
                scientific=False,
                recipe=invalid,
                depends=depends,
                capabilities=caps,
            )
        )
    return cases


def matrix_coverage(cases: list[CaseSpec] | None = None) -> dict[str, Any]:
    """Prove coverage from the executable catalog, independently of generation."""
    cases = scientific_cases() if cases is None else cases
    actual_presets = {
        case.parameters["preset"]
        for case in cases
        if case.parameters.get("recipe") == "preset"
    }
    missing_presets = sorted(set(PRESETS) - actual_presets)
    expected_fe = set(
        product(("solid", "mushy", "liquid"), ("driven", "dynamo"), (False, True))
    )
    actual_fe = {
        (
            case.parameters.get("phase"),
            case.parameters.get("mode"),
            case.parameters.get("closure"),
        )
        for case in cases
        if case.parameters.get("recipe") == "fe_matrix"
    }
    missing_fe = sorted(expected_fe - actual_fe)
    boris = pair_coverage(
        BORIS_FACTORS,
        [
            {key: case.parameters[key] for key in BORIS_FACTORS}
            for case in cases
            if case.id.startswith("boris.pair.")
        ],
    )
    geant4 = pair_coverage(
        DOSE_FACTORS,
        [
            {key: case.parameters[key] for key in DOSE_FACTORS}
            for case in cases
            if case.id.startswith("dose.pair.")
        ],
    )
    pinn_expected = {
        f"pinn.{problem}.{device}"
        for problem, device in product(("thermal", "stationary"), ("cpu", "cuda"))
    }
    missing_pinn = sorted(pinn_expected - {case.id for case in cases})
    return {
        "finite_scope": True,
        "complete": not missing_presets
        and not missing_fe
        and not missing_pinn
        and boris["complete"]
        and geant4["complete"],
        "case_count": len(cases),
        "presets": [name for name in PRESETS if name in actual_presets],
        "missing_presets": missing_presets,
        "fe_phase_mode_closure": {
            "required": 12,
            "covered": len(actual_fe & expected_fe),
            "missing": missing_fe,
        },
        "missing_pinn_device_problem_cases": missing_pinn,
        "boris": boris,
        "geant4": geant4,
        "sum_declared_timeouts_s": sum(case.timeout for case in cases),
        "not_qualified_by_this_matrix": [
            "published turbulent MHD Taylor–Green adaptation",
            "full coupled manufactured-solution convergence",
            "certified material datasets",
            "independent anatomical dose reference qualification",
        ],
    }
