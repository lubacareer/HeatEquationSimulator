"""SI-valued, immutable contracts shared by the desktop and scientific workers."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from hashlib import sha256
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping

import numpy as np

from .artifacts import FIELD_ARTIFACT_KEYS, FieldArtifact

Progress = Callable[[float, str], None]


class SimulationCancelled(RuntimeError):
    """Raised at a safe cancellation boundary without publishing a result."""


@dataclass(frozen=True)
class ScenarioConfig:
    name: str = "Iron torus · benchmark"
    preset: str = "Iron"
    major_radius: float = 1.0
    minor_radius: float = 0.2
    habitat_radius: float = 0.35
    vessel_thickness: float = 0.01
    habitat_thickness: float = 0.01
    ground_enabled: bool = False
    density: float = 7000.0
    heat_capacity_solid: float = 449.0
    heat_capacity_liquid: float = 820.0
    thermal_conductivity: float = 35.0
    latent_heat: float = 247000.0
    melting_temperature: float = 1810.0
    electrical_conductivity_solid: float = 1.0e6
    electrical_conductivity_liquid: float = 7.0e5
    dynamic_viscosity: float = 0.006
    thermal_expansion: float = 1.0e-4
    initial_temperature: float = 1800.0
    ambient_temperature: float = 293.0
    discharge_temperature: float = 1700.0
    heating_power: float = 20000.0
    discharge_power: float = 0.0
    charge_fraction: float = 0.65
    emissivity: float = 0.0
    efficiency: float = 0.3
    gravity: float = 1.62
    pump_force: float = 1.0
    helical_fraction: float = 0.25
    coil_current: float = 10000.0
    coil_turns: int = 10
    seed_field: float = 1.0e-5
    seed_duration: float = 0.2
    mode: str = "driven"
    turbulence: bool = True
    thermal_turbulent_prandtl: float = 0.85
    duration: float = 5.0
    time_steps: int = 40
    resolution: int = 3
    exterior_radius_factor: float = 4.0
    transition_width: float = 2.0
    solver: str = "fenics"
    particle_z: int = 1
    particle_a: int = 1
    particle_energy_mev_nucleon: float = 100.0
    particle_count: int = 64
    radiation_histories: int = 10000
    transport_physics_list: str = "Shielding"
    production_cut_m: float = .0001
    source_kind: str = "beam"
    spectrum_path: str = ""
    incident_fluence_m2: float = 1.0
    target: str = "water"
    phantom_path: str = ""
    seed: int = 42
    pinn_epochs: int = 250
    pinn_batch_size: int = 256
    provenance: str = "Illustrative constant-property iron benchmark; not a certified material dataset."
    composition: tuple[tuple[int, float], ...] = ((26, 1.0),)
    schema_version: int = 2

    def validate(self) -> "ScenarioConfig":
        if isinstance(self.schema_version, bool) or not isinstance(self.schema_version, int) or self.schema_version not in {1, 2}:
            raise ValueError("Unsupported scenario schema version.")
        for key, value in asdict(self).items():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"{key.replace('_', ' ').capitalize()} must be finite.")
        positive = ("major_radius", "minor_radius", "habitat_radius", "density",
                    "heat_capacity_solid", "heat_capacity_liquid", "thermal_conductivity",
                    "melting_temperature", "electrical_conductivity_solid",
                    "electrical_conductivity_liquid", "dynamic_viscosity", "duration",
                    "initial_temperature", "ambient_temperature", "discharge_temperature",
                    "transition_width", "thermal_turbulent_prandtl", "production_cut_m")
        for key in positive:
            if getattr(self, key) <= 0:
                raise ValueError(f"{key.replace('_', ' ').capitalize()} must be greater than zero.")
        for key in ("latent_heat", "vessel_thickness", "habitat_thickness", "heating_power",
                    "discharge_power", "gravity", "thermal_expansion", "seed_field",
                    "seed_duration", "incident_fluence_m2"):
            if getattr(self, key) < 0:
                raise ValueError(f"{key.replace('_', ' ').capitalize()} must not be negative.")
        if self.major_radius <= self.minor_radius + self.vessel_thickness:
            raise ValueError("Major radius must exceed the reservoir tube's outer radius.")
        if self.habitat_radius + self.habitat_thickness >= self.major_radius - self.minor_radius - self.vessel_thickness:
            raise ValueError("The habitat and wall must fit inside the torus opening.")
        for key in ("efficiency", "emissivity", "charge_fraction"):
            if not 0 <= getattr(self, key) <= 1:
                raise ValueError(f"{key.replace('_', ' ').capitalize()} must be between 0 and 1.")
        if not 0 <= self.helical_fraction <= 1:
            raise ValueError("Helical fraction must be between 0 and 1.")
        if self.mode not in {"driven", "dynamo"}:
            raise ValueError("Magnetic mode must be driven or dynamo.")
        if self.solver not in {"fenics", "thermal_benchmark"}:
            raise ValueError("Select the full FEniCS solver or the thermal benchmark.")
        if self.target not in {"water", "male", "female"}:
            raise ValueError("Dose target must be water, male, or female.")
        if self.transport_physics_list not in {"Shielding", "QGSP_BIC_HP"}:
            raise ValueError("Select Shielding or QGSP_BIC_HP for Geant4 transport.")
        if self.source_kind not in {"beam", "isotropic", "spectrum"}:
            raise ValueError("Source must be beam, isotropic, or spectrum.")
        for key, minimum, maximum in (("time_steps", 2, 100000), ("resolution", 3, 128),
                                     ("particle_count", 1, 10000), ("radiation_histories", 1, 100000000),
                                     ("pinn_epochs", 1, 1000000), ("pinn_batch_size", 1, 65536),
                                     ("coil_turns", 1, 100000)):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
                raise ValueError(f"{key.replace('_', ' ').capitalize()} must be an integer in [{minimum}, {maximum}].")
        if not 1 <= self.particle_z <= 28 or self.particle_a < self.particle_z:
            raise ValueError("Particle species requires 1 ≤ Z ≤ 28 and A ≥ Z.")
        for key in ("particle_z", "particle_a", "seed"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{key.replace('_', ' ').capitalize()} must be an integer.")
        if not 0 <= self.seed <= 2147483647:
            raise ValueError("Random seed must be between 0 and 2147483647.")
        if self.particle_energy_mev_nucleon <= 0 or self.exterior_radius_factor < 2:
            raise ValueError("Particle energy must be positive; exterior radius factor must be at least 2.")
        if not self.composition or any(isinstance(z, bool) or not isinstance(z, int) or z < 1
                                       or not math.isfinite(fraction) or fraction <= 0
                                       for z, fraction in self.composition):
            raise ValueError("Elemental composition requires positive integer atomic numbers and finite positive mass fractions.")
        if not math.isclose(sum(f for _, f in self.composition), 1.0, abs_tol=1e-6):
            raise ValueError("Elemental mass fractions must sum to one.")
        return self

    @property
    def config_hash(self) -> str:
        values = asdict(self)
        values.pop("name")
        values.pop("preset")
        # Existing v1 runs predate configurable Geant4 list/cut. Preserve their
        # identity for the original defaults, including their saved PINNs/dose.
        if self.schema_version == 1 and self.transport_physics_list == "Shielding" and self.production_cut_m == .0001:
            values.pop("transport_physics_list")
            values.pop("production_cut_m")
        return sha256(json.dumps(values, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    @property
    def physics_hash(self) -> str:
        """Reuse accepted fields when only observation/training settings change."""
        values = asdict(self)
        for key in ("name", "preset", "particle_z", "particle_a", "particle_energy_mev_nucleon",
                    "particle_count", "radiation_histories", "source_kind", "spectrum_path",
                    "incident_fluence_m2", "target", "phantom_path", "pinn_epochs", "pinn_batch_size",
                    "transport_physics_list", "production_cut_m"):
            values.pop(key, None)
        return sha256(json.dumps(values, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    @property
    def volume(self) -> float:
        return 2 * math.pi ** 2 * self.major_radius * self.minor_radius ** 2

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ScenarioConfig":
        values = dict(data)
        if "composition" in values:
            values["composition"] = tuple((z, float(f)) for z, f in values["composition"])
        return cls(**values).validate()


def _immutable_arrays(values: Mapping[str, np.ndarray]) -> Mapping[str, np.ndarray]:
    output = {}
    for key, value in values.items():
        array = np.array(value, copy=True)
        array.setflags(write=False)
        output[key] = array
    return MappingProxyType(output)


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, np.ndarray):
        array = np.array(value, copy=True)
        array.setflags(write=False)
        return array
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


@dataclass(frozen=True)
class RunResult:
    run_id: str
    config: ScenarioConfig
    times: np.ndarray
    points: np.ndarray
    fields: Mapping[str, np.ndarray]
    diagnostics: Mapping[str, np.ndarray]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    artifacts: Mapping[str, FieldArtifact] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.config.validate()
        arrays = _immutable_arrays({"times": self.times, "points": self.points})
        object.__setattr__(self, "times", arrays["times"])
        object.__setattr__(self, "points", arrays["points"])
        object.__setattr__(self, "fields", _immutable_arrays(self.fields))
        object.__setattr__(self, "diagnostics", _immutable_arrays(self.diagnostics))
        object.__setattr__(self, "metadata", _freeze(self.metadata))
        if any(key not in FIELD_ARTIFACT_KEYS or not isinstance(value, FieldArtifact)
               for key, value in self.artifacts.items()):
            raise ValueError("Result artifacts require explicitly captured field data.")
        object.__setattr__(self, "artifacts", MappingProxyType(dict(self.artifacts)))
        if self.times.ndim != 1 or self.times.size < 1 or not np.all(np.isfinite(self.times)):
            raise ValueError("Result requires a finite one-dimensional timeline.")
        if np.any(np.diff(self.times) <= 0):
            raise ValueError("Result times must increase strictly.")
        if self.points.ndim != 2 or self.points.shape[1] != 3 or not np.all(np.isfinite(self.points)):
            raise ValueError("Result points must have finite coordinates of shape (N, 3).")
        for key, value in self.fields.items():
            if value.shape[:2] != (len(self.times), len(self.points)):
                raise ValueError(f"Field {key} must start with dimensions (time, point).")
            if not np.all(np.isfinite(value)):
                raise ValueError(f"Field {key} contains non-finite values; no result can be published.")

    def artifact_bytes(self, key: str) -> bytes:
        artifact = self.artifacts.get(key)
        if artifact is None:
            raise ValueError(f"The accepted run has no embedded/captured {key}; recalculate or import a self-contained result.")
        return artifact.data


@dataclass(frozen=True)
class PINNResult:
    run_id: str
    source_run_id: str
    config_hash: str
    fields: Mapping[str, np.ndarray]
    history: Mapping[str, np.ndarray]
    metrics: Mapping[str, float]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "fields", _immutable_arrays(self.fields))
        object.__setattr__(self, "history", _immutable_arrays(self.history))
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))
        object.__setattr__(self, "metadata", _freeze(self.metadata))


@dataclass(frozen=True)
class DoseResult:
    run_id: str
    source_run_id: str
    config_hash: str
    snapshot_index: int
    target: str
    organ_names: tuple[str, ...]
    dose_gy: np.ndarray
    standard_error_gy: np.ndarray
    histories: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        arrays = _immutable_arrays({"dose": self.dose_gy, "error": self.standard_error_gy})
        object.__setattr__(self, "dose_gy", arrays["dose"])
        object.__setattr__(self, "standard_error_gy", arrays["error"])
        object.__setattr__(self, "organ_names", tuple(self.organ_names))
        object.__setattr__(self, "metadata", _freeze(self.metadata))
        if self.dose_gy.shape != self.standard_error_gy.shape or self.dose_gy.shape != (len(self.organ_names),):
            raise ValueError("Dose, uncertainty, and organ names must have matching dimensions.")
        if not np.all(np.isfinite(self.dose_gy)) or not np.all(np.isfinite(self.standard_error_gy)):
            raise ValueError("Dose results must be finite.")
        if np.any(self.dose_gy < 0) or np.any(self.standard_error_gy < 0) or self.histories < 1:
            raise ValueError("Dose and uncertainty cannot be negative; at least one history is required.")


def preset(name: str) -> ScenarioConfig:
    base = ScenarioConfig()
    if name == "Lunar habitat":
        return replace(base, name="Lunar habitat · illustrative", preset=name,
                       major_radius=10.0, minor_radius=1.0, habitat_radius=3.0,
                       duration=60.0, heating_power=1.0e6, initial_temperature=1820.0, ground_enabled=True)
    if name == "Regolith composite":
        return replace(base, name="Regolith composite · illustrative", preset=name,
                       density=1800.0, heat_capacity_solid=800.0, heat_capacity_liquid=1200.0,
                       thermal_conductivity=1.5, latent_heat=1429.0, melting_temperature=1373.0,
                       initial_temperature=1363.0, discharge_temperature=1200.0,
                       dynamic_viscosity=1.0, electrical_conductivity_solid=1.0,
                       electrical_conductivity_liquid=10.0,
                       composition=((8, .45), (14, .21), (13, .08), (20, .11), (26, .10), (12, .05)),
                       provenance="Illustrative synthetic composite; transport coefficients require experimental calibration.")
    if name == "Thermal benchmark":
        return replace(base, name="Thermal battery benchmark", preset=name,
                       solver="thermal_benchmark", turbulence=False,
                       major_radius=.3, minor_radius=.06, habitat_radius=.12,
                       duration=3000.0, time_steps=80, discharge_power=10000.0, resolution=8)
    return base
