"""Finite configuration choices, numeric edges, and repeatable custom edits.

The validator is the declaration of the supported input domain. Its literal
choices and bounds are enumerated for coverage; each case exercises the public
immutable configuration interface rather than its private control flow.
No solver, graphics runtime, or training dependency is imported here.
"""

from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError, dataclass, fields, replace
import inspect
import json
import math
import random
import textwrap
import unittest

import numpy as np

from magnetic_torus.models import ScenarioConfig, preset


@dataclass(frozen=True)
class Bounds:
    low: float | int | None = None
    high: float | int | None = None
    low_inclusive: bool = True
    high_inclusive: bool = True
    integer: bool = False


def declared_domain():
    """Enumerate literal public choices/bounds, failing on uncovered numerics."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(ScenarioConfig.validate)))
    literals, choices, bounds = {}, {}, {}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            try:
                literals[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
        if (
            isinstance(node, ast.Compare)
            and len(node.ops) == 1
            and isinstance(node.ops[0], ast.NotIn)
        ):
            if (
                isinstance(node.left, ast.Attribute)
                and isinstance(node.left.value, ast.Name)
                and node.left.value.id == "self"
            ):
                try:
                    choices[node.left.attr] = tuple(
                        sorted(ast.literal_eval(node.comparators[0]))
                    )
                except (ValueError, TypeError):
                    pass

    def literal(node):
        try:
            return ast.literal_eval(node)
        except (ValueError, TypeError):
            return literals.get(node.id) if isinstance(node, ast.Name) else None

    def field_name(node, loop_field=None):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
        ):
            return node.attr
        if (
            loop_field
            and isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
        ):
            if (
                len(node.args) == 2
                and isinstance(node.args[0], ast.Name)
                and node.args[0].id == "self"
            ):
                return loop_field
        return None

    def add_relations(test, *, valid=False, loop_field=None):
        if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
            add_relations(test.operand, valid=not valid, loop_field=loop_field)
        elif isinstance(test, ast.BoolOp) and isinstance(test.op, ast.Or):
            for item in test.values:
                add_relations(item, valid=valid, loop_field=loop_field)
        elif isinstance(test, ast.Compare):
            terms = [test.left, *test.comparators]
            for left, op, right in zip(terms, test.ops, terms[1:]):
                name = field_name(left, loop_field)
                endpoint = literal(right)
                reverse = False
                if name is None:
                    name, endpoint, reverse = (
                        field_name(right, loop_field),
                        literal(left),
                        True,
                    )
                if (
                    name is None
                    or isinstance(endpoint, bool)
                    or not isinstance(endpoint, (int, float))
                ):
                    continue
                if not isinstance(op, (ast.Lt, ast.LtE, ast.Gt, ast.GtE)):
                    continue
                lower = isinstance(op, (ast.Gt, ast.GtE))
                inclusive = isinstance(op, (ast.LtE, ast.GtE))
                if reverse:
                    lower = not lower
                if not valid:
                    lower, inclusive = not lower, not inclusive
                old = bounds.get(name, Bounds())
                bounds[name] = replace(
                    old,
                    **(
                        {"low": endpoint, "low_inclusive": inclusive}
                        if lower
                        else {"high": endpoint, "high_inclusive": inclusive}
                    ),
                )

    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            add_relations(node.test)
        elif isinstance(node, ast.For):
            values = literal(node.iter)
            if not isinstance(values, tuple):
                continue
            if isinstance(node.target, ast.Tuple) and [
                item.id for item in node.target.elts
            ] == ["key", "minimum", "maximum"]:
                for name, low, high in values:
                    bounds[name] = Bounds(low, high, integer=True)
            elif (
                isinstance(node.target, ast.Name)
                and node.target.id == "key"
                and all(isinstance(item, str) for item in values)
            ):
                for name in values:
                    for condition in (
                        item for item in ast.walk(node) if isinstance(item, ast.If)
                    ):
                        add_relations(condition.test, loop_field=name)
    # The mass number's lower bound depends on Z, rather than a scalar literal.
    bounds["particle_a"] = Bounds(1, integer=True)
    for name in ("particle_z", "seed"):
        bounds[name] = replace(bounds[name], integer=True)
    return choices, bounds


class ConfigurationMatrixTests(unittest.TestCase):
    def setUp(self):
        self.base = ScenarioConfig().validate()
        self.choices, self.bounds = declared_domain()

    def test_every_declared_choice_boolean_and_numeric_field_has_coverage(self):
        declared = {
            item.name: getattr(self.base, item.name) for item in fields(self.base)
        }
        numeric = {
            name
            for name, value in declared.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        self.assertEqual(
            numeric, set(self.bounds) | {"schema_version", "pump_force", "coil_current"}
        )
        self.assertEqual(
            set(self.choices),
            {
                "schema_version",
                "mode",
                "solver",
                "target",
                "transport_physics_list",
                "source_kind",
            },
        )
        for name, values in self.choices.items():
            for value in values:
                with self.subTest(field=name, value=value):
                    self.assertEqual(
                        getattr(replace(self.base, **{name: value}).validate(), name),
                        value,
                    )
            with self.subTest(field=name, invalid=True), self.assertRaises(ValueError):
                replace(
                    self.base,
                    **{name: -1 if name == "schema_version" else "unsupported choice"},
                ).validate()
        for value in (True, False, 1.0, 0, 3):
            with self.subTest(schema_version=value), self.assertRaises(ValueError):
                replace(self.base, schema_version=value).validate()
        booleans = {name for name, value in declared.items() if isinstance(value, bool)}
        self.assertEqual(booleans, {"ground_enabled", "turbulence"})
        for name in booleans:
            for value in (False, True):
                with self.subTest(field=name, value=value):
                    self.assertIs(
                        getattr(replace(self.base, **{name: value}).validate(), name),
                        value,
                    )

    def test_each_scalar_boundary_and_adjacent_invalid_value(self):
        for name, limit in sorted(self.bounds.items()):
            for edge, inclusive, direction in (
                (limit.low, limit.low_inclusive, -1),
                (limit.high, limit.high_inclusive, 1),
            ):
                if edge is None:
                    continue
                outside = (
                    edge + direction
                    if limit.integer
                    else np.nextafter(float(edge), math.copysign(math.inf, direction))
                )
                with (
                    self.subTest(field=name, invalid=outside),
                    self.assertRaises(ValueError),
                ):
                    replace(
                        self.base,
                        **{name: int(outside) if limit.integer else float(outside)},
                    ).validate()
                if not inclusive:
                    with (
                        self.subTest(field=name, excluded=edge),
                        self.assertRaises(ValueError),
                    ):
                        replace(self.base, **{name: edge}).validate()
                accepted = (
                    edge
                    if inclusive
                    else (
                        edge + 1
                        if limit.integer
                        else max(float(np.nextafter(float(edge), math.inf)), 1e-12)
                    )
                )
                changes = {name: accepted}
                if name == "major_radius":
                    changes.update(
                        minor_radius=accepted / 10,
                        habitat_radius=accepted / 10,
                        vessel_thickness=0.0,
                        habitat_thickness=0.0,
                    )
                elif name == "particle_z":
                    changes["particle_a"] = int(accepted)
                with self.subTest(field=name, accepted=accepted):
                    self.assertEqual(
                        getattr(replace(self.base, **changes).validate(), name),
                        accepted,
                    )
            if limit.integer:
                for value in (True, False, 3.5):
                    with (
                        self.subTest(field=name, invalid_type=value),
                        self.assertRaises(ValueError),
                    ):
                        replace(self.base, **{name: value}).validate()

    def test_every_float_rejects_nonfinite_inputs_and_signed_controls_are_valid(self):
        for item in fields(self.base):
            if not isinstance(getattr(self.base, item.name), float):
                continue
            for value in (math.nan, math.inf, -math.inf):
                with (
                    self.subTest(field=item.name, value=value),
                    self.assertRaises(ValueError),
                ):
                    replace(self.base, **{item.name: value}).validate()
        for name in ("pump_force", "coil_current"):
            for value in (-100.0, 0.0, 100.0):
                with self.subTest(field=name, value=value):
                    replace(self.base, **{name: value}).validate()

    def test_geometry_touching_walls_species_and_composition_constraints(self):
        invalid = (
            {"major_radius": self.base.minor_radius + self.base.vessel_thickness},
            {
                "habitat_radius": self.base.major_radius
                - self.base.minor_radius
                - self.base.vessel_thickness
                - self.base.habitat_thickness
            },
            {"vessel_thickness": self.base.major_radius},
            {"habitat_thickness": self.base.major_radius},
            {"particle_z": 2, "particle_a": 1},
            {"composition": ()},
            {"composition": ((0, 1.0),)},
            {"composition": ((True, 1.0),)},
            {"composition": ((1.5, 1.0),)},
            {"composition": ((26, 0.0),)},
            {"composition": ((26, -1.0),)},
            {"composition": ((26, math.nan),)},
            {"composition": ((26, math.inf),)},
            {"composition": ((26, 0.8),)},
        )
        for changes in invalid:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(self.base, **changes).validate()
        for species in ((1, 1), (2, 4), (26, 56), (28, 64)):
            replace(self.base, particle_z=species[0], particle_a=species[1]).validate()
        replace(self.base, composition=((8, 0.4), (26, 0.6))).validate()
        with self.assertRaises(FrozenInstanceError):
            self.base.duration = 10.0

    def test_shipped_presets_and_seeded_custom_edits_roundtrip_without_drift(self):
        tree = ast.parse(textwrap.dedent(inspect.getsource(preset)))
        names = {"Iron"}
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Compare)
                and isinstance(node.left, ast.Name)
                and node.left.id == "name"
            ):
                names.update(
                    item.value
                    for item in node.comparators
                    if isinstance(item, ast.Constant) and isinstance(item.value, str)
                )
        self.assertEqual(
            names, {"Iron", "Lunar habitat", "Regolith composite", "Thermal benchmark"}
        )
        for name in sorted(names):
            config = preset(name).validate()
            self.assertEqual(config.preset, name)
            self.assertEqual(
                ScenarioConfig.from_dict(json.loads(json.dumps(config.to_dict()))),
                config,
            )
            self.assertAlmostEqual(
                config.volume,
                math.tau * config.major_radius * math.pi * config.minor_radius**2,
            )
        rng = random.Random(1729)
        hashes = set()
        original = self.base.to_dict()
        for index in range(40):
            radius = rng.uniform(0.3, 10.0)
            config = replace(
                self.base,
                name=f"Custom {index}",
                preset="Custom",
                major_radius=radius,
                minor_radius=radius * rng.uniform(0.05, 0.2),
                habitat_radius=radius * rng.uniform(0.1, 0.3),
                vessel_thickness=radius * 0.005,
                habitat_thickness=radius * 0.005,
                density=rng.uniform(1000.0, 9000.0),
                heat_capacity_solid=rng.uniform(300.0, 1500.0),
                heat_capacity_liquid=rng.uniform(400.0, 2000.0),
                latent_heat=rng.uniform(0.0, 400000.0),
                duration=rng.uniform(0.1, 10000.0),
                time_steps=rng.randint(2, 200),
                charge_fraction=rng.random(),
                efficiency=rng.random(),
                emissivity=rng.random(),
                pump_force=rng.uniform(-2.0, 2.0),
                coil_current=rng.uniform(-10000.0, 10000.0),
                ground_enabled=bool(index % 2),
                turbulence=bool(index % 3),
                seed=rng.randrange(2147483648),
                **{name: rng.choice(values) for name, values in self.choices.items()},
            ).validate()
            loaded = ScenarioConfig.from_dict(json.loads(json.dumps(config.to_dict())))
            self.assertEqual(loaded.config_hash, config.config_hash)
            self.assertEqual(loaded.physics_hash, config.physics_hash)
            self.assertNotIn(config.config_hash, hashes)
            hashes.add(config.config_hash)
        self.assertEqual(self.base.to_dict(), original)

    def test_field_reuse_hash_ignores_only_observation_and_training_changes(self):
        descriptive = replace(self.base, name="Renamed", preset="Custom")
        self.assertEqual(descriptive.config_hash, self.base.config_hash)
        self.assertEqual(descriptive.physics_hash, self.base.physics_hash)
        observation_edits = {
            "particle_z": 2,
            "particle_a": 4,
            "particle_energy_mev_nucleon": 250.0,
            "particle_count": 100,
            "radiation_histories": 20000,
            "source_kind": "isotropic",
            "spectrum_path": "versioned-spectrum.csv",
            "incident_fluence_m2": 2.0,
            "target": "female",
            "phantom_path": "phantoms",
            "pinn_epochs": 500,
            "pinn_batch_size": 128,
            "transport_physics_list": "QGSP_BIC_HP",
            "production_cut_m": 0.0002,
        }
        for name, value in observation_edits.items():
            extra = {"particle_a": 4} if name == "particle_z" else {}
            changed = replace(self.base, **extra, **{name: value}).validate()
            with self.subTest(observation=name):
                self.assertNotEqual(changed.config_hash, self.base.config_hash)
                self.assertEqual(changed.physics_hash, self.base.physics_hash)
        excluded = {"name", "preset", *observation_edits}
        for item in fields(self.base):
            if item.name in excluded:
                continue
            value = getattr(self.base, item.name)
            if isinstance(value, bool):
                new = not value
            elif item.name in self.choices:
                new = next(
                    option for option in self.choices[item.name] if option != value
                )
            elif isinstance(value, int):
                new = value + 1
            elif isinstance(value, float):
                new = value + (0.001 if item.name != "efficiency" else 0.01)
            elif item.name == "composition":
                new = ((8, 0.1), (26, 0.9))
            else:
                new = value + " updated"
            changed = replace(self.base, **{item.name: new}).validate()
            with self.subTest(physical=item.name):
                self.assertNotEqual(changed.physics_hash, self.base.physics_hash)
                self.assertNotEqual(changed.config_hash, self.base.config_hash)


if __name__ == "__main__":
    unittest.main()
