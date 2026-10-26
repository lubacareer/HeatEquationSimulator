"""Workspace restrictions leave supported PINNs and research APIs available."""

from dataclasses import replace
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from magnetic_torus.learning.support import pinn_training_unavailable_reason
from magnetic_torus.store import RunStore
from workspace_fixtures import accepted_run


def thermal_run(enthalpy):
    base = accepted_run(times=(0.0, 0.1, 1.0))
    values = np.broadcast_to(
        np.asarray(enthalpy, dtype=float), (len(base.times), len(base.points))
    ).copy()
    return replace(base, fields={**base.fields, "enthalpy": values})


def stationary_coupled_run():
    base = accepted_run(times=(0.0, 0.1, 1.0))
    fields = {
        name: np.broadcast_to(value[:1], value.shape).copy()
        for name, value in base.fields.items()
    }
    return replace(base, config=replace(base.config, solver="fenics"), fields=fields)


def malformed(source, **changes):
    return SimpleNamespace(
        **{
            **{
                name: getattr(source, name)
                for name in ("times", "points", "fields", "metadata", "config")
            },
            **changes,
        }
    )


class PINNSupportTests(unittest.TestCase):
    def test_evolving_solid_and_liquid_histories_and_sharp_endpoints_remain_supported(
        self,
    ):
        source = thermal_run(0.0)
        latent = source.config.latent_heat
        for values in ([-10.0, -5.0, 0.0], [latent, latent + 1.0, latent + 10.0]):
            with self.subTest(values=values):
                run = thermal_run(np.asarray(values)[:, None])
                self.assertIsNone(pinn_training_unavailable_reason(run))
        scalar = thermal_run(-1.0)
        scalar = replace(
            scalar,
            fields={**scalar.fields, "enthalpy": scalar.fields["enthalpy"][..., None]},
        )
        self.assertIsNone(pinn_training_unavailable_reason(scalar))

    def test_latent_plateau_cycle_and_spatially_mixed_phase_are_unavailable(self):
        latent = thermal_run(0.0).config.latent_heat
        for values in (
            latent / 2,
            np.array([-1.0, latent / 2, latent + 1.0])[:, None],
            np.array([-1.0, latent + 1.0, -1.0, -1.0, -1.0])[None],
        ):
            with self.subTest(values=values):
                reason = pinn_training_unavailable_reason(thermal_run(values))
                self.assertIn(
                    "Phase-changing thermal PINN training is not qualified", reason
                )
                self.assertTrue(reason.endswith("Numerical results remain available."))

    def test_saved_stationary_accepted_fields_are_supported_after_independent_roundtrip(
        self,
    ):
        source = stationary_coupled_run()
        with tempfile.TemporaryDirectory() as directory:
            store = RunStore(Path(directory) / "store")
            loaded = store.load_run(store.save_run(source))
            self.assertIsNone(pinn_training_unavailable_reason(loaded))
            self.assertEqual(loaded.run_id, source.run_id)
            self.assertEqual(loaded.config.config_hash, source.config.config_hash)

    def test_each_evolving_coupled_component_is_rejected(self):
        source = stationary_coupled_run()
        for name in (
            "enthalpy",
            "velocity",
            "magnetic_field",
            "vector_potential",
            "pressure",
            "electric_potential",
        ):
            width = (
                3 if name in {"velocity", "magnetic_field", "vector_potential"} else 1
            )
            for component in range(width):
                with self.subTest(field=name, component=component):
                    value = np.array(source.fields[name], copy=True)
                    if width == 1:
                        value[-1, 0] += 1.0
                    else:
                        value[-1, 0, component] += 1.0
                    changed = replace(source, fields={**source.fields, name: value})
                    reason = pinn_training_unavailable_reason(changed)
                    self.assertIn(
                        "Evolving coupled MHD PINN training is not qualified", reason
                    )
                    self.assertTrue(
                        reason.endswith("Numerical results remain available.")
                    )

    def test_conductor_restriction_ignores_only_nonmagnetic_vacuum_histories(self):
        source = stationary_coupled_run()
        outside = np.flatnonzero(~np.asarray(source.metadata["conductor_mask"]))[0]
        fields = dict(source.fields)
        for name in ("enthalpy", "velocity", "pressure", "electric_potential"):
            value = np.array(fields[name], copy=True)
            value[-1, outside] += 100.0
            fields[name] = value
        self.assertIsNone(
            pinn_training_unavailable_reason(replace(source, fields=fields))
        )
        for name in ("magnetic_field", "vector_potential"):
            value = np.array(source.fields[name], copy=True)
            value[-1, outside, 0] += 1.0
            self.assertIn(
                "Evolving coupled",
                pinn_training_unavailable_reason(
                    replace(source, fields={**source.fields, name: value})
                ),
            )

    def test_initial_pressure_is_excluded_but_all_subsequent_pressures_are_compared(
        self,
    ):
        source = stationary_coupled_run()
        pressure = np.array(source.fields["pressure"], copy=True)
        pressure[0] = np.arange(len(source.points)) * 1e6
        pressure[1:] = 12.0
        self.assertIsNone(
            pinn_training_unavailable_reason(
                replace(source, fields={**source.fields, "pressure": pressure})
            )
        )
        pressure[-1, 0] += 1.0
        self.assertIn(
            "Evolving coupled",
            pinn_training_unavailable_reason(
                replace(source, fields={**source.fields, "pressure": pressure})
            ),
        )

    def test_stationarity_uses_the_prescribed_roundoff_floor_and_componentwise_relative_scale(
        self,
    ):
        source = stationary_coupled_run()
        tolerance = 64 * np.finfo(np.float64).eps
        velocity = np.zeros_like(source.fields["velocity"])
        velocity[:, :, 0] = 1e9
        velocity[-1, 0, 0] += tolerance * 1e9 / 2
        velocity[-1, 0, 1] = tolerance / 2
        supported = replace(source, fields={**source.fields, "velocity": velocity})
        self.assertIsNone(pinn_training_unavailable_reason(supported))
        for component, change in ((0, 2 * tolerance * 1e9), (1, 2 * tolerance)):
            value = velocity.copy()
            value[-1, 0, component] += change
            self.assertIn(
                "Evolving coupled",
                pinn_training_unavailable_reason(
                    replace(source, fields={**source.fields, "velocity": value})
                ),
            )

    def test_missing_malformed_and_nonfinite_required_fields_are_explicitly_unavailable(
        self,
    ):
        for source in (thermal_run(-1.0), stationary_coupled_run()):
            for fields in (
                {},
                {**source.fields, "enthalpy": np.ones((3, 5, 2))},
                {**source.fields, "enthalpy": np.full((3, 5), np.nan)},
            ):
                with self.subTest(solver=source.config.solver, fields=list(fields)):
                    self.assertIn(
                        "finite accepted enthalpy",
                        pinn_training_unavailable_reason(
                            malformed(source, fields=fields)
                        ),
                    )
        source = stationary_coupled_run()
        for name in (
            "velocity",
            "magnetic_field",
            "vector_potential",
            "pressure",
            "electric_potential",
        ):
            fields = {key: value for key, value in source.fields.items() if key != name}
            self.assertIn(
                name.replace("_", " "),
                pinn_training_unavailable_reason(malformed(source, fields=fields)),
            )
        invalid = dict(source.fields)
        invalid["velocity"] = np.ones((3, 5, 2))
        self.assertIn(
            "velocity",
            pinn_training_unavailable_reason(malformed(source, fields=invalid)),
        )

    def test_missing_invalid_or_empty_conductor_masks_are_rejected(self):
        source = stationary_coupled_run()
        for raw in (None, [], [False] * 5, [True] * 4, [1, 1, 1, 1, 0], ["yes"] * 5):
            metadata = {} if raw is None else {"conductor_mask": raw}
            with self.subTest(mask=raw):
                self.assertIn(
                    "conductor mask",
                    pinn_training_unavailable_reason(
                        replace(source, metadata=metadata)
                    ),
                )

    def test_every_training_source_requires_finite_aligned_temperature_and_liquid_fraction(
        self,
    ):
        for source in (thermal_run(-1.0), stationary_coupled_run()):
            for name in ("temperature", "liquid_fraction"):
                absent = {
                    key: value for key, value in source.fields.items() if key != name
                }
                for fields in (
                    absent,
                    {**source.fields, name: np.zeros((3, 5, 2))},
                    {**source.fields, name: np.full((3, 5), np.nan)},
                ):
                    with self.subTest(solver=source.config.solver, field=name):
                        self.assertIn(
                            "finite accepted " + name.replace("_", " "),
                            pinn_training_unavailable_reason(
                                malformed(source, fields=fields)
                            ),
                        )
                singleton = {**source.fields, name: source.fields[name][..., None]}
                self.assertIsNone(
                    pinn_training_unavailable_reason(replace(source, fields=singleton))
                )

    def test_invalid_or_insufficient_source_grids_cannot_claim_workspace_support(self):
        source = thermal_run(-1.0)
        for changes in (
            {"times": np.array([0.0])},
            {"times": np.array([0.0, 0.0, 1.0])},
            {"times": np.array([0.0, np.inf, 1.0])},
            {"points": np.empty((0, 3))},
            {"points": np.ones((5, 2))},
            {"points": np.full((5, 3), np.nan)},
        ):
            with self.subTest(changes=changes):
                self.assertIn(
                    "finite source grid",
                    pinn_training_unavailable_reason(malformed(source, **changes)),
                )

    def test_import_does_not_load_torch_or_scipy(self):
        program = "import sys; import magnetic_torus.learning.support; assert 'torch' not in sys.modules; assert 'scipy' not in sys.modules"
        completed = subprocess.run(
            [sys.executable, "-B", "-c", program],
            capture_output=True,
            text=True,
            timeout=15,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
