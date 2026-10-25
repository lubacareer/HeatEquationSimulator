"""Held-out temperature validation must not fit its scale to validation labels."""

from types import SimpleNamespace
import unittest

import numpy as np

from magnetic_torus.learning.fields import FieldScaling, UnsupportedPINNConfiguration
from magnetic_torus.learning.training import TrainingData
from magnetic_torus.learning.validation import _field_metrics
from magnetic_torus.models import RunResult, ScenarioConfig


class TemperatureValidationScaleTests(unittest.TestCase):
    def evaluate(self, temperature, *, train_ids=(0, 3, 2), error=2.0):
        temperature = np.asarray(temperature, dtype=float)
        result = RunResult(
            "temperature-scale-fixture",
            ScenarioConfig(),
            np.array([0.0, 1.0]),
            np.zeros((3, 3)),
            {"temperature": temperature},
            {},
            {"conductor_mask": [True, True, False]},
        )
        # Include a vacuum label in the training split: magnetic/electrical
        # models can use exterior labels, but temperature scales cannot.
        scaling = FieldScaling(("enthalpy",), (1,), np.zeros(1), np.ones(1))
        data = TrainingData(
            result,
            True,
            None,
            None,
            np.random.default_rng(0),
            np.asarray(train_ids, dtype=np.int64),
            np.array([1, 4]),
            scaling,
            np.zeros((2, 3, 1)),
        )
        run = SimpleNamespace(
            field=SimpleNamespace(data=data),
            metadata={},
            history={
                key: [0.0]
                for key in ("validation_loss", "loss", "physics_loss", "boundary_loss")
            },
        )
        metrics = _field_metrics(run, {"temperature": temperature + error}, 0.0)
        return metrics, run.metadata["validation_field_scales"]["temperature"]

    def test_extreme_heldout_temperatures_do_not_weaken_normalized_error(self):
        ordinary, ordinary_scale = self.evaluate(
            [[300.0, 304.0, 300.0], [310.0, 314.0, 300.0]]
        )
        extreme, extreme_scale = self.evaluate(
            [[300.0, 1e9, 300.0], [310.0, 2e9, 300.0]]
        )
        self.assertEqual(ordinary_scale, 10.0)
        self.assertEqual(extreme_scale, ordinary_scale)
        self.assertEqual(ordinary["temperature_rmse"], 2.0)
        self.assertEqual(extreme["temperature_rmse"], 2.0)
        self.assertEqual(ordinary["temperature_normalized_rmse"], 0.2)
        self.assertEqual(extreme["temperature_normalized_rmse"], 0.2)

    def test_exterior_training_temperatures_cannot_change_material_scale(self):
        ordinary, ordinary_scale = self.evaluate(
            [[300.0, 304.0, 300.0], [310.0, 314.0, 300.0]]
        )
        extreme, extreme_scale = self.evaluate(
            [[300.0, 304.0, 1e12], [310.0, 314.0, 2e12]]
        )
        self.assertEqual(extreme_scale, ordinary_scale)
        self.assertEqual(
            extreme["temperature_normalized_rmse"],
            ordinary["temperature_normalized_rmse"],
        )

    def test_small_training_range_keeps_one_kelvin_floor(self):
        metrics, scale = self.evaluate(
            [[300.0, 304.0, 1e12], [300.25, 314.0, 2e12]], error=0.2
        )
        self.assertEqual(scale, 1.0)
        self.assertAlmostEqual(metrics["temperature_normalized_rmse"], 0.2)

    def test_missing_training_conductor_labels_raise_explicit_error(self):
        for train_ids in ((), (2, 5)):
            with self.subTest(train_ids=train_ids):
                with self.assertRaisesRegex(
                    UnsupportedPINNConfiguration, "no training labels for temperature"
                ):
                    self.evaluate(
                        [[300.0, 304.0, 300.0], [310.0, 314.0, 300.0]],
                        train_ids=train_ids,
                    )


if __name__ == "__main__":
    unittest.main()
