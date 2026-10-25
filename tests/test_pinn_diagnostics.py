"""Validation failures remain explicit, including old saved result metadata."""

import unittest

from magnetic_torus.learning.diagnostics import (
    failed_checks,
    training_epochs,
    validation_message,
)
from magnetic_torus.models import PINNResult


def model(metrics, **metadata):
    return PINNResult(
        "diagnostic",
        "source",
        "hash",
        {},
        {},
        metrics,
        {"validation_status": "failed", **metadata},
    )


class PINNDiagnosticsTests(unittest.TestCase):
    def test_failed_coupled_fit_names_only_failed_checks_and_actual_epochs(self):
        fit = model(
            {
                "epochs": 250.0,
                "validation_loss": 0.137,
                "physics_loss": 1e-5,
                "boundary_loss": 0.0,
                "temperature_normalized_rmse": 5.4e-5,
                "velocity_normalized_rmse": 0.286,
                "pressure_normalized_rmse": 0.463,
                "electric_potential_normalized_rmse": 0.726,
                "charge_residual_rms": 0.019,
            },
            thermal_physics_only=False,
        )
        self.assertEqual(
            {check.metric for check in failed_checks(fit)},
            {
                "validation_loss",
                "velocity_normalized_rmse",
                "pressure_normalized_rmse",
                "electric_potential_normalized_rmse",
            },
        )
        message = validation_message(fit)
        for fragment in (
            "250 epochs",
            "held-out loss 0.137 > 0.01",
            "velocity",
            "pressure",
            "electric potential",
            "0.726 > 0.1",
            "numerical results remain available",
        ):
            self.assertIn(fragment, message)
        self.assertEqual(fit.metadata["validation_status"], "failed")

    def test_exact_gate_boundaries_do_not_report_failures(self):
        fit = model(
            {
                "validation_loss": 0.01,
                "physics_loss": 0.01,
                "boundary_loss": 0.01,
                "temperature_normalized_rmse": 0.1,
                "enthalpy_residual_rms": 0.1,
                "energy_defect_relative": 0.05,
            },
            thermal_physics_only=True,
        )
        self.assertEqual(failed_checks(fit), ())
        self.assertIn(
            "failed", validation_message(fit)
        )  # Do not reinterpret saved status.

    def test_missing_labels_and_old_thermal_metadata_are_explained(self):
        fit = model(
            {"energy_defect_relative": 0.06},
            fields_without_heldout_labels=("velocity",),
        )
        self.assertEqual(failed_checks(fit)[0].metric, "energy_defect_relative")
        self.assertIn("missing held-out labels: velocity", validation_message(fit))

    def test_unknown_legacy_criteria_are_not_invented(self):
        self.assertIn("saved validation criteria", validation_message(model({})))
        self.assertIn(
            "unvalidated",
            validation_message(model({}, validation_status="unvalidated")),
        )

    def test_nonfinite_legacy_errors_are_reported_without_a_false_comparison(self):
        message = validation_message(model({"validation_loss": float("nan")}))
        self.assertIn("held-out loss is non-finite", message)
        self.assertNotIn("nan >", message)

    def test_training_epochs_use_the_trained_model_and_support_legacy_metadata(self):
        self.assertEqual(
            training_epochs(model({"epochs": 250.0}, training_epochs=1000)), 250
        )
        self.assertEqual(training_epochs(model({}, training_epochs=750)), 750)
        self.assertIsNone(training_epochs(model({}, training_epochs="unknown")))


if __name__ == "__main__":
    unittest.main()
