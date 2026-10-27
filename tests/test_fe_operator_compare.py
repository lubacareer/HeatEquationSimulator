"""Checks for raw FE operator comparison and its tolerance accounting."""
from dataclasses import replace
import tempfile
import unittest

import numpy as np
import torch

from magnetic_torus.numerics.weak_residual import evaluate_weak_residual_torch
from scripts.compare_fe_operator import comparisons
from test_pinn_research_scaling import tiny_fe_source

SUITE_CAPABILITIES = ("torch_cpu",)


class OperatorComparisonTests(unittest.TestCase):
    def test_comparison_reports_discrepancies_without_reusing_solver_tolerance(self):
        matched = comparisons(np.array([1., 2.]), np.array([1., 2.]))
        self.assertTrue(matched["matches"])
        self.assertEqual(matched["maximum_absolute_difference"], 0.)
        mismatch = comparisons(np.array([1., 2.]), np.array([1., 2.01]))
        self.assertFalse(mismatch["matches"])
        self.assertAlmostEqual(mismatch["difference_l2"], .01)
        for invalid in (np.array([float("nan"), 0.]), np.ones(3)):
            with self.assertRaises(ValueError):
                comparisons(np.array([1., 2.]), invalid)

    def test_raw_components_are_opt_in_and_preserve_boundary_masks_and_gradients(self):
        with tempfile.TemporaryDirectory() as directory:
            original = tiny_fe_source(directory)
            source = replace(original, metadata={**original.metadata, "pinn_physics": {
                "kind": "fenics_a_phi_vms_v1", "bc_nodes": {"0": [0], "1": [1], "2": [2], "3": [3], "4": [0]}}})
            fields = {name: torch.tensor(values[:2], dtype=torch.float64, requires_grad=True)
                      for name, values in source.fields.items()}
            default = evaluate_weak_residual_torch(source.config, source, fields, [0, 1])
            raw = evaluate_weak_residual_torch(source.config, source, fields, [0, 1], include_raw_components=True)
            for key, value in default.items():
                torch.testing.assert_close(raw[key], value, rtol=0., atol=0.)
            for key in ("momentum", "continuity", "induction", "charge", "enthalpy"):
                self.assertEqual(raw[f"raw_{key}"].shape[0], 3)
                self.assertTrue(torch.isfinite(raw[f"raw_{key}"]).all())
            gradient = torch.autograd.grad(raw["raw_momentum"].square().sum(), fields["pressure"])[0]
            self.assertTrue(torch.isfinite(gradient).all())

    def test_raw_pressure_jacobian_action_matches_centered_difference(self):
        with tempfile.TemporaryDirectory() as directory:
            source = tiny_fe_source(directory)
            fields = {name: torch.tensor(values[:2], dtype=torch.float64)
                      for name, values in source.fields.items()}
            pressure = fields["pressure"][1]
            tangent = torch.tensor([0., .1, -.05, .08], dtype=torch.float64)
            def momentum(value):
                selected = dict(fields)
                selected["pressure"] = torch.stack((fields["pressure"][0], value))
                return evaluate_weak_residual_torch(source.config, source, selected, [0, 1],
                                                    include_raw_components=True)["raw_momentum"]
            _, jvp = torch.autograd.functional.jvp(momentum, pressure, tangent)
            epsilon = 1.e-4
            finite = (momentum(pressure + epsilon * tangent) - momentum(pressure - epsilon * tangent)) / (2 * epsilon)
            torch.testing.assert_close(jvp, finite, rtol=1.e-6, atol=1.e-10)


if __name__ == "__main__":
    unittest.main()
