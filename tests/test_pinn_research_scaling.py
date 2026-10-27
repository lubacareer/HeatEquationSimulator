"""Training-only normalization does not change the default FE qualification oracle."""

from dataclasses import replace
from pathlib import Path
import math
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from magnetic_torus.artifacts import FieldArtifact
from magnetic_torus.learning.research_scaling import fit_training_weak_scales
from magnetic_torus.models import RunResult, ScenarioConfig
from workspace_fixtures import accepted_run

try:
    import torch
except ImportError:
    torch = None


SUITE_CAPABILITIES = ("torch_cpu",)


def tiny_fe_source(directory):
    """Analytic one-cell fixture; no FE installation or native solve needed."""
    config = replace(
        ScenarioConfig(), initial_temperature=1820., pump_force=0., gravity=0.,
        heating_power=0., discharge_power=0., coil_current=0., seed_field=0.,
        duration=.02, time_steps=2, resolution=3, turbulence=True,
    )
    points = np.array([[.9, 0., 0.], [1.1, 0., 0.], [1., .1, 0.], [1., 0., .1]])
    gradients = np.array([[-5., -5., -5.], [5., -5., -5.], [0., 10., 0.], [0., 0., 10.]])
    volume = .2 * .1 * .1 / 6
    times = np.array([0., .01, .02])
    scalar = np.zeros((3, 4))
    vector = np.zeros((3, 4, 3))
    artifact = Path(directory) / "weak.npz"
    np.savez(
        artifact, points=points, cells=np.array([[0, 1, 2, 3]]),
        gradients=gradients[None], volumes=np.array([volume]), cell_labels=np.array([1]),
        cell_diameter=np.array([.2]), boundary_faces=np.empty((0, 3), dtype=np.int64),
        boundary_areas=np.empty(0), imposed_vector_potential=vector,
        node_volumes=np.full(4, volume / 4),
    )
    return RunResult(
        "research-scaling-cell", config, times, points,
        {"temperature": np.full((3, 4), 1820.),
         "enthalpy": np.full((3, 4), config.latent_heat + config.heat_capacity_liquid * 10.),
         "liquid_fraction": np.ones_like(scalar), "velocity": vector,
         "pressure": scalar, "magnetic_field": vector,
         "vector_potential": vector, "electric_potential": scalar}, {},
        {"conductor_mask": [True] * 4,
         "pinn_physics": {"kind": "fenics_a_phi_vms_v1", "bc_nodes": {}}},
        artifacts={"weak_operator_path": FieldArtifact.capture(artifact, root=directory)},
    )


class TrainingWeakScaleTests(unittest.TestCase):
    def source(self):
        source = accepted_run(times=(0., .2, .6))
        return replace(source, config=replace(source.config, solver="fenics", pump_force=0.))

    def test_heldout_vector_labels_do_not_fit_scales_and_source_stays_immutable(self):
        source = self.source()
        train_ids = np.array([0, 1, 4, 5, 6, 9, 10, 11, 14], dtype=np.int64)
        before = {name: values.copy() for name, values in source.fields.items()}
        first = fit_training_weak_scales(source, train_ids)
        heldout = np.setdiff1d(np.arange(15), train_ids)
        ti, pi = np.divmod(heldout, len(source.points))
        changed = dict(source.fields)
        for name in ("velocity", "magnetic_field", "vector_potential"):
            changed[name] = source.fields[name].copy()
            changed[name][ti, pi] += 1.e8
        second = fit_training_weak_scales(replace(source, fields=changed), train_ids)
        self.assertEqual(first, second)
        for name, values in source.fields.items():
            np.testing.assert_array_equal(values, before[name])
            self.assertFalse(values.flags.writeable)

    def test_velocity_uses_conductor_but_magnetic_vectors_include_exterior(self):
        source = self.source()
        changed = dict(source.fields)
        for name, amplitude in (("velocity", 100.), ("magnetic_field", 2.), ("vector_potential", 3.)):
            changed[name] = source.fields[name].copy()
            changed[name][:, -1] = [amplitude, 0., 0.]
        source = replace(source, fields=changed)
        scales = fit_training_weak_scales(source, np.arange(15))
        self.assertEqual(scales["velocity"], .001)
        self.assertEqual(scales["magnetic_field"], 2.)
        self.assertEqual(scales["vector_potential"], 3.)

    def test_configuration_floors_match_the_original_weak_operator(self):
        source = self.source()
        vectors = {name: np.zeros_like(source.fields[name]) for name in (
            "velocity", "magnetic_field", "vector_potential")}
        source = replace(source, fields={**source.fields, **vectors})
        scales = fit_training_weak_scales(source, np.arange(15))
        self.assertEqual(scales, {"velocity": 1.e-4, "magnetic_field": 1.e-6,
                                 "vector_potential": 1.e-6 * source.config.minor_radius})
        source = replace(source, config=replace(source.config, pump_force=100.))
        self.assertEqual(fit_training_weak_scales(source, np.arange(15))["velocity"],
                         math.sqrt(100. * source.config.minor_radius / source.config.density))

    def test_invalid_training_ids_masks_and_missing_conductor_samples_are_rejected(self):
        source = self.source()
        for ids in ([], [0., 1.], [True, False], [-1], [15], [0, 0], [[0, 1]]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                fit_training_weak_scales(source, ids)
        for mask in ([], [False] * 5, [1] * 5):
            invalid = replace(source, metadata={**source.metadata, "conductor_mask": mask})
            with self.subTest(mask=mask), self.assertRaises(ValueError):
                fit_training_weak_scales(invalid, [0, 1])
        with self.assertRaisesRegex(ValueError, "no conductor"):
            fit_training_weak_scales(source, [4, 9])

    def test_scale_fitter_does_not_import_torch_or_workers(self):
        code = (
            "import sys; import magnetic_torus.learning.research_scaling; "
            "assert 'torch' not in sys.modules; "
            "assert 'scipy' not in sys.modules; "
            "assert 'magnetic_torus.physics' not in sys.modules"
        )
        result = subprocess.run([sys.executable, "-B", "-c", code],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


@unittest.skipIf(torch is None, "PyTorch is required")
class WeakOverrideTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def prediction(self, source):
        fields = {name: torch.tensor(values[:2], dtype=torch.float64, requires_grad=True)
                  for name, values in source.fields.items()}
        with torch.no_grad():
            fields["temperature"][:, 1] += .1
            fields["velocity"][:, 1, 0] += 1.e-4
            fields["vector_potential"][:, 2, 1] += 1.e-5
        return fields

    def test_all_reference_overrides_match_default_residuals_and_gradients_exactly(self):
        from magnetic_torus.numerics.weak_residual import evaluate_weak_residual_torch

        with tempfile.TemporaryDirectory() as directory:
            source = tiny_fe_source(directory)
            overrides = fit_training_weak_scales(source, np.arange(12))
            predictions = [self.prediction(source), self.prediction(source)]
            residuals = [evaluate_weak_residual_torch(source.config, source, predictions[0], [0, 1]),
                         evaluate_weak_residual_torch(source.config, source, predictions[1], [0, 1],
                                                      normalization_scales=overrides)]
            for name in residuals[0]:
                torch.testing.assert_close(residuals[0][name], residuals[1][name], rtol=0, atol=0)
            gradients = []
            for prediction, components in zip(predictions, residuals):
                loss = sum(value.square().mean() for value in components.values())
                gradients.append(torch.autograd.grad(loss, tuple(prediction.values()), allow_unused=True))
            for first, second in zip(*gradients):
                if first is None:
                    self.assertIsNone(second)
                else:
                    torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_fixed_override_never_reads_heldout_reference_amplitudes(self):
        from magnetic_torus.numerics.weak_residual import evaluate_weak_residual_torch

        with tempfile.TemporaryDirectory() as directory:
            source = tiny_fe_source(directory)
            ids = np.arange(8)
            overrides = fit_training_weak_scales(source, ids)
            changed = dict(source.fields)
            for name in ("velocity", "magnetic_field", "vector_potential"):
                changed[name] = source.fields[name].copy()
                changed[name][2] = 1.e6
            altered = replace(source, fields=changed)
            fields = self.prediction(source)
            values = [evaluate_weak_residual_torch(source.config, reference, fields, [0, 1],
                                                   normalization_scales=overrides)
                      for reference in (source, altered)]
            for name in values[0]:
                torch.testing.assert_close(values[0][name], values[1][name], rtol=0, atol=0)
            original = evaluate_weak_residual_torch(source.config, source, fields, [0, 1])
            altered_default = evaluate_weak_residual_torch(source.config, altered, fields, [0, 1])
            self.assertFalse(torch.equal(original["momentum"], altered_default["momentum"]))

    def test_invalid_overrides_are_rejected_without_changing_the_default(self):
        from magnetic_torus.numerics.weak_residual import evaluate_weak_residual_torch

        with tempfile.TemporaryDirectory() as directory:
            source = tiny_fe_source(directory)
            fields = self.prediction(source)
            valid = fit_training_weak_scales(source, np.arange(12))
            invalid = [[], {}, {**valid, "pressure": 1.}, {"velocity": 1.},
                       {**valid, "velocity": float("nan")}, {**valid, "velocity": float("inf")},
                       {**valid, "velocity": 0.}, {**valid, "velocity": -1.},
                       {**valid, "velocity": True}, {**valid, "velocity": "1"},
                       {**valid, "velocity": 1.e-8}, {**valid, "magnetic_field": 1.e-8},
                       {**valid, "vector_potential": 1.e-12}]
            before = evaluate_weak_residual_torch(source.config, source, fields, [0, 1])
            for override in invalid:
                with self.subTest(override=override), self.assertRaises(ValueError):
                    evaluate_weak_residual_torch(source.config, source, fields, [0, 1],
                                                normalization_scales=override)
            after = evaluate_weak_residual_torch(source.config, source, fields, [0, 1])
            for name in before:
                torch.testing.assert_close(before[name], after[name], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
