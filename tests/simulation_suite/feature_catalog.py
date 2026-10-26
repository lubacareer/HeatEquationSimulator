"""One deterministic representative per major feature; not the full scenario matrix."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from .scientific_catalog import _case
from .spec import CaseSpec


REQUIRED_FEATURES = {
    "configuration": ("unit.test_configuration_matrix", "unit.test_editor_features"),
    "session_and_publication": (
        "unit.test_torus_session",
        "unit.test_torus_publication",
        "unit.test_full_suite_runner",
    ),
    "persistence_and_security": (
        "unit.test_torus_store",
        "unit.test_artifact_security",
    ),
    "graphics_and_exports": (
        "unit.test_workspace_features",
        "unit.test_export_features",
        "desktop.journey",
    ),
    "playback_and_performance": ("unit.test_playback_features", "desktop.performance"),
    "thermal_phase_cycle": ("feature.thermal_cycle",),
    "driven_laminar_mhd": ("feature.mhd_driven",),
    "dynamo_vms_mhd": ("feature.mhd_dynamo",),
    "thermal_cpu_pinn": ("feature.native_source", "feature.native_pinn"),
    "coupled_cuda_pinn": ("feature.zero_field", "feature.coupled_pinn"),
    "relativistic_trajectories": ("feature.boris",),
    "water_dose_precision": ("feature.zero_field", "feature.dose_water"),
    "anatomical_dose": ("feature.mhd_driven", "feature.dose_anatomy"),
    "analytical_references": ("native.reference", "fe.references"),
    "analysis_rejection_guards": (
        "invalid.native_dose",
        "invalid.failed_pinn",
        "invalid.small_anatomy",
        "invalid.stale_pinn",
        "invalid.out_of_map",
    ),
}

# Independent manifest contracts prevent a mislabeled provider from claiming
# coverage after its scenario, gate, device, or source association changes.
_PROVIDER_PARAMETERS = {
    "feature.thermal_cycle": {"action": "solve", "recipe": "feature_thermal_cycle"},
    "feature.native_source": {"action": "solve", "recipe": "feature_native_source"},
    "feature.mhd_driven": {"action": "solve", "recipe": "feature_fe_driven"},
    "feature.mhd_dynamo": {"action": "solve", "recipe": "feature_fe_dynamo"},
    "feature.zero_field": {"action": "solve", "recipe": "feature_fe_zero"},
    "feature.native_pinn": {
        "action": "pinn",
        "source": "feature.native_source",
        "device": "cpu",
        "epochs": 1000,
        "batch": 256,
    },
    "feature.coupled_pinn": {
        "action": "pinn",
        "source": "feature.zero_field",
        "device": "cuda",
        "epochs": 250,
        "batch": 256,
    },
    "feature.boris": {
        "action": "boris",
        "source_case": "feature.mhd_driven",
        "species": "proton",
        "energy": 100.0,
        "field": "driven",
        "source": "beam",
        "ground": True,
    },
    "feature.dose_anatomy": {
        "action": "dose",
        "source_case": "feature.mhd_driven",
        "species": "mixed",
        "source": "spectrum",
        "spectrum_angular": "isotropic",
        "ground": True,
        "field": "driven",
        "target": "male",
        "physics_list": "Shielding",
        "histories": 128,
    },
    "feature.dose_water": {
        "action": "dose_fixed_precision",
        "source_case": "feature.zero_field",
        "histories": 524288,
    },
    "native.reference": {"action": "references"},
    "fe.references": {"action": "fe_references"},
    **{
        "invalid." + recipe: {"action": "invalid", "recipe": recipe}
        for recipe in (
            "native_dose",
            "failed_pinn",
            "small_anatomy",
            "stale_pinn",
            "out_of_map",
        )
    },
}
_PROVIDER_DEPENDENCIES = {
    "feature.native_pinn": ("feature.native_source",),
    "feature.coupled_pinn": ("feature.zero_field",),
    "feature.boris": ("feature.mhd_driven",),
    "feature.dose_anatomy": ("feature.mhd_driven",),
    "feature.dose_water": ("feature.zero_field",),
    "invalid.failed_pinn": ("feature.zero_field",),
    "invalid.stale_pinn": ("feature.zero_field",),
    "invalid.out_of_map": ("feature.zero_field",),
}
_PROVIDER_CAPABILITIES = {
    "feature.mhd_driven": ("fenics", "torch_cpu"),
    "feature.mhd_dynamo": ("fenics", "torch_cpu"),
    "feature.zero_field": ("fenics", "torch_cpu"),
    "feature.native_pinn": ("torch_cpu",),
    "feature.coupled_pinn": ("torch_cuda",),
    "feature.dose_anatomy": ("geant4", "icrp110"),
    "feature.dose_water": ("geant4",),
    "fe.references": ("fenics",),
    "invalid.small_anatomy": ("geant4", "icrp110"),
}
_EXPECTED_UNIT_EXCLUSIONS = {
    "unit.test_torus_physics": {
        "test_torus_physics.ThermalPhysicsTests.test_full_fe_stationary_closed_system_and_weak_operator",
        "test_torus_physics.ThermalPhysicsTests.test_full_fe_driven_vms_matches_differentiable_equations",
    },
    "unit.test_torus_radiation": {
        "test_torus_radiation.Geant4IntegrationTests.test_water_paired_zero_field_transport",
        "test_torus_radiation.Geant4IntegrationTests.test_ground_water_transport_uses_above_ground_incident_source",
        "test_torus_radiation.Geant4IntegrationTests.test_male_icrp110_transport",
        "test_torus_radiation.Geant4IntegrationTests.test_female_icrp110_transport",
    },
}


def feature_cases() -> list[CaseSpec]:
    cases = [
        _case(
            "feature.thermal_cycle",
            "solve",
            "Iron charging, melting, discharging, freezing and radiation",
            recipe="feature_thermal_cycle",
            timeout=120,
        ),
        _case(
            "feature.native_source",
            "solve",
            "Short solid-heating source for CPU PINN qualification",
            recipe="feature_native_source",
            timeout=120,
        ),
        _case(
            "feature.mhd_driven",
            "solve",
            "Liquid driven laminar MHD with Lorentz feedback",
            recipe="feature_fe_driven",
            capabilities=("fenics", "torch_cpu"),
            timeout=240,
        ),
        _case(
            "feature.mhd_dynamo",
            "solve",
            "Mushy dynamo VMS with removable seed",
            recipe="feature_fe_dynamo",
            capabilities=("fenics", "torch_cpu"),
            timeout=240,
        ),
        _case(
            "feature.zero_field",
            "solve",
            "Shared accepted stationary FE map for CUDA PINN and water dose",
            recipe="feature_fe_zero",
            capabilities=("fenics", "torch_cpu"),
            timeout=180,
        ),
        _case(
            "feature.native_pinn",
            "pinn",
            "Strict short-source thermal PINN qualification on CPU",
            source="feature.native_source",
            device="cpu",
            epochs=1000,
            batch=256,
            depends=("feature.native_source",),
            capabilities=("torch_cpu",),
            timeout=600,
        ),
        _case(
            "feature.coupled_pinn",
            "pinn",
            "Strict stationary coupled PINN qualification on CUDA",
            source="feature.zero_field",
            device="cuda",
            epochs=250,
            batch=256,
            depends=("feature.zero_field",),
            capabilities=("torch_cuda",),
            timeout=600,
        ),
        _case(
            "feature.boris",
            "boris",
            "Paired proton trajectory and relativistic gyro oracle",
            source_case="feature.mhd_driven",
            species="proton",
            energy=100.0,
            source="beam",
            ground=True,
            field="driven",
            depends=("feature.mhd_driven",),
            scientific=False,
            timeout=120,
        ),
        _case(
            "feature.dose_anatomy",
            "dose",
            "Mixed spectrum, nonzero field, ground and male anatomical scoring",
            source_case="feature.mhd_driven",
            species="mixed",
            energy=100.0,
            source="spectrum",
            spectrum_angular="isotropic",
            ground=True,
            field="driven",
            target="male",
            physics_list="Shielding",
            histories=128,
            depends=("feature.mhd_driven",),
            capabilities=("geant4", "icrp110"),
            scientific=False,
            timeout=300,
        ),
        _case(
            "native.reference",
            "references",
            "Analytical Stefan, enthalpy and coil checks",
            timeout=120,
        ),
        _case(
            "fe.references",
            "fe_references",
            "FE Hartmann and magnetic-diffusion references",
            capabilities=("fenics",),
            timeout=120,
        ),
    ]
    for recipe in (
        "native_dose",
        "failed_pinn",
        "small_anatomy",
        "stale_pinn",
        "out_of_map",
    ):
        cases.append(
            _case(
                "invalid." + recipe,
                "invalid",
                "Explicit rejection: " + recipe.replace("_", " "),
                recipe=recipe,
                depends=("feature.zero_field",)
                if recipe in {"failed_pinn", "stale_pinn", "out_of_map"}
                else (),
                capabilities=("geant4", "icrp110") if recipe == "small_anatomy" else (),
                scientific=False,
                timeout=120,
            )
        )
    # The longest independent job comes last, after each other feature has run.
    cases.append(
        _case(
            "feature.dose_water",
            "dose_fixed_precision",
            "One fixed water ensemble; measured relative standard error must be at most 5%",
            source_case="feature.zero_field",
            histories=524288,
            depends=("feature.zero_field",),
            capabilities=("geant4",),
            timeout=1800,
        )
    )
    return cases


def feature_coverage(cases: list[CaseSpec], *, root: Path | None = None) -> dict:
    """Validate actual executable providers, all unit modules and their entire DAG."""
    from .catalog import validate_cases

    counts = Counter(case.id for case in cases)
    by_id = {case.id: case for case in cases}
    duplicates = sorted(name for name, count in counts.items() if count > 1)
    invalid = []
    for name, expected in _PROVIDER_PARAMETERS.items():
        case = by_id.get(name)
        if case is None:
            continue
        functional = expected["action"] in {"boris", "dose", "invalid"}
        if (
            any(case.parameters.get(key) != value for key, value in expected.items())
            or case.kind != ("functional" if functional else "scientific")
            or case.depends_on != _PROVIDER_DEPENDENCIES.get(name, ())
            or case.capabilities != _PROVIDER_CAPABILITIES.get(name, ())
        ):
            invalid.append(name)
    for name, script in (
        ("desktop.journey", "desktop_smoke.py"),
        ("desktop.performance", "benchmark_desktop.py"),
    ):
        case = by_id.get(name)
        if case is not None and (
            case.capabilities != ("desktop",)
            or case.kind != "functional"
            or len(case.command) < 3
            or not case.command[2].endswith("/scripts/" + script)
            or (
                name == "desktop.performance"
                and case.parameters.get("minimum_fps") != 30.0
            )
        ):
            invalid.append(name)
    graph_errors = []
    try:
        validate_cases(cases)
    except ValueError as error:
        graph_errors.append(str(error))
    project = root or Path(__file__).resolve().parents[2]
    expected_units = {
        "unit." + path.stem for path in (project / "tests").glob("test_*.py")
    }
    missing_units = sorted(expected_units - set(by_id))
    for name in expected_units & set(by_id):
        case = by_id[name]
        if (
            case.category != "regression"
            or case.kind != "functional"
            or case.parameters.get("pattern") != name.removeprefix("unit.") + ".py"
            or set(case.parameters.get("exclude_ids", ()))
            != _EXPECTED_UNIT_EXCLUSIONS.get(name, set())
            or len(case.command) < 3
            or not case.command[2].endswith(
                "/tests/simulation_suite/unittest_worker.py"
            )
        ):
            invalid.append(name)
    covered = [
        feature
        for feature, providers in REQUIRED_FEATURES.items()
        if all(counts[name] == 1 and name not in invalid for name in providers)
    ]
    missing = sorted(set(REQUIRED_FEATURES) - set(covered))
    return {
        "complete": not missing
        and not missing_units
        and not duplicates
        and not graph_errors
        and not invalid,
        "required_features": list(REQUIRED_FEATURES),
        "covered_features": covered,
        "missing_features": missing,
        "missing_regression_modules": missing_units,
        "duplicate_case_ids": duplicates,
        "invalid_providers": invalid,
        "graph_errors": graph_errors,
        "case_ids": [case.id for case in cases],
        "provider_case_ids": {
            key: list(value) for key, value in REQUIRED_FEATURES.items()
        },
        "scope": "One representative per major feature; independent of the complete scenario/pairwise qualification matrix.",
        "unselected_qualification": [
            "As-shipped Lunar duration and geometry solve",
            "phase-changing and dynamic coupled PINN qualifications",
            "all phase/mode/closure combinations",
            "female anatomy and the alternate Geant4 physics list",
            "complete radiation option-pair coverage",
        ],
    }
