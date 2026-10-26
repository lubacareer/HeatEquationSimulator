"""Distinguish discrete equation agreement from the Coulomb divergence diagnostic."""

import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from magnetic_torus.learning.diagnostics import RESIDUAL_RMS_LIMIT
from magnetic_torus.models import ScenarioConfig
from simulation_suite.scientific import Checks, _weak_checks

SUITE_CAPABILITIES = ("torch_cpu",)


class ScientificGaugeTests(unittest.TestCase):
    def evaluate(self, gauge, induction=0.0):
        result = SimpleNamespace(
            times=np.array([0.0, 1.0]),
            config=ScenarioConfig(),
            fields={"vector_potential": np.zeros((2, 2, 3))},
        )

        def residual(_config, _source, values, _indices):
            unknown = values["vector_potential"][-1, :, 0]
            components = {"induction": unknown + induction}
            if gauge is not None:
                components["gauge"] = unknown + torch.as_tensor(
                    gauge, dtype=torch.float64
                )
            return components

        checks = Checks()
        with patch(
            "magnetic_torus.physics.evaluate_weak_residual_torch", side_effect=residual
        ):
            _weak_checks(result, checks, lambda *_: None)
        return {check["name"]: check for check in checks.records}

    def test_penalty_diagnostic_uses_existing_rms_limit_not_equation_tolerance(self):
        checks = self.evaluate([0.0, 0.12])
        gauge = checks["weak_gauge"]
        self.assertTrue(gauge["passed"])
        self.assertAlmostEqual(gauge["normalized_rms"], 0.12 / math.sqrt(2))
        self.assertEqual(gauge["normalized_max"], 0.12)
        self.assertEqual(gauge["limit"], RESIDUAL_RMS_LIMIT)
        self.assertTrue(checks["weak_induction"]["passed"])
        self.assertEqual(checks["weak_induction"]["limit"], 1e-7)
        self.assertTrue(checks["weak_gradient_1"]["passed"])

    def test_excessive_gauge_still_fails(self):
        self.assertFalse(self.evaluate([0.10001, 0.10001])["weak_gauge"]["passed"])

    def test_discrete_equation_defects_still_fail_at_original_tolerance(self):
        checks = self.evaluate([0.0, 0.0], induction=1.1e-7)
        self.assertFalse(checks["weak_induction"]["passed"])
        self.assertTrue(checks["weak_gauge"]["passed"])

    def test_nonfinite_gauge_evidence_is_not_hidden_by_maximum_accumulation(self):
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value):
                gauge = self.evaluate([0.0, value])["weak_gauge"]
                self.assertFalse(gauge["passed"])
                self.assertFalse(math.isfinite(gauge["normalized_rms"]))

    def test_nonfinite_assembled_residual_still_fails(self):
        self.assertFalse(
            self.evaluate([0.0, 0.0], induction=math.nan)["weak_induction"]["passed"]
        )

    def test_missing_gauge_diagnostic_is_unsuccessful(self):
        self.assertFalse(self.evaluate(None)["weak_gauge"]["passed"])


if __name__ == "__main__":
    unittest.main()
