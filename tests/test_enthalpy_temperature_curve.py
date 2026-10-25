"""Regression tests for the dashboard's material T(E) curve, without a Tk window."""

from types import SimpleNamespace
import unittest

import numpy as np

import backend
from interface3 import App


class EnthalpyTemperatureCurveTests(unittest.TestCase):
    @staticmethod
    def curve(pcm, temperatures):
        app = App.__new__(App)
        app.pcm = pcm
        return app.build_enthalpy_temperature_curve(np.asarray(temperatures, dtype=float))

    @staticmethod
    def material(**overrides):
        properties = dict(T_m=500.0, LH=200_000.0, c_solid=800.0,
                          c_liquid=1200.0, rho=2500.0)
        properties.update(overrides)
        return SimpleNamespace(**properties)

    def assert_three_branches(self, energy, temperature, pcm, c_solid, c_liquid):
        latent_kj = pcm.LH / 1000.0
        self.assertTrue(np.all(np.isfinite(energy)))
        self.assertTrue(np.all(np.isfinite(temperature)))
        self.assertTrue(np.all(np.diff(energy) > 0))

        solid = energy < 0.0
        plateau = (energy >= 0.0) & (energy <= latent_kj)
        liquid = energy > latent_kj
        self.assertGreater(np.count_nonzero(solid), 1)
        self.assertGreater(np.count_nonzero(liquid), 1)
        self.assertTrue(np.any(energy == 0.0), "The plateau must start exactly at E=0")
        self.assertTrue(np.any(energy == latent_kj), "The plateau must end exactly at E=LH")
        np.testing.assert_allclose(temperature[solid],
                                   pcm.T_m + energy[solid] / (c_solid / 1000.0))
        np.testing.assert_array_equal(temperature[plateau],
                                      np.full(np.count_nonzero(plateau), pcm.T_m))
        np.testing.assert_allclose(temperature[liquid], pcm.T_m +
                                   (energy[liquid] - latent_kj) / (c_liquid / 1000.0))
        self.assertTrue(np.all(np.diff(temperature[solid]) > 0))
        self.assertTrue(np.all(np.diff(temperature[liquid]) > 0))

    def test_custom_material_has_signed_energy_plateau_and_rising_branches(self):
        pcm = backend.customPCM(k=30.0, c=800.0, rho=2500.0,
                                T_m=500.0, LH=200_000.0)
        pcm.c_liquid = 1200.0
        energy, temperature = self.curve(pcm, [[300.0, 500.0, 750.0]])

        self.assert_three_branches(energy, temperature, pcm, 800.0, 1200.0)
        self.assertAlmostEqual(energy[0], -160.0)
        self.assertAlmostEqual(energy[-1], 500.0)

    def test_specific_energy_does_not_depend_on_density(self):
        low_density = self.material(rho=1000.0)
        high_density = self.material(rho=8000.0)
        low_energy, low_temperature = self.curve(low_density, [200.0, 900.0])
        high_energy, high_temperature = self.curve(high_density, [200.0, 900.0])

        np.testing.assert_array_equal(low_energy, high_energy)
        np.testing.assert_array_equal(low_temperature, high_temperature)

    def test_iron_heat_capacities_are_converted_from_joules_per_gram(self):
        pcm = backend.Iron()
        original_capacities = (pcm.c_solid, pcm.c_liquid)
        energy, temperature = self.curve(pcm, [pcm.T_m - 100.0, pcm.T_m + 100.0])

        self.assert_three_branches(energy, temperature, pcm, 449.0, 820.0)
        self.assertAlmostEqual(energy[0], -44.9)
        self.assertAlmostEqual(energy[-1], 329.0)
        self.assertEqual((pcm.c_solid, pcm.c_liquid), original_capacities)

    def test_single_phase_surfaces_still_show_both_rising_branches(self):
        pcm = self.material()
        for values in ([200.0, 250.0], [600.0, 1000.0]):
            with self.subTest(temperatures=values):
                energy, temperature = self.curve(pcm, values)
                self.assert_three_branches(energy, temperature, pcm, 800.0, 1200.0)
                minimum_span = max(0.25 * (max(values) - min(values)), 25.0)
                self.assertLessEqual(temperature[0], pcm.T_m - minimum_span)
                self.assertGreaterEqual(temperature[-1], pcm.T_m + minimum_span)
                self.assertLessEqual(temperature[0], min(values))
                self.assertGreaterEqual(temperature[-1], max(values))

    def test_empty_and_nonfinite_surfaces_produce_finite_complete_curves(self):
        pcm = self.material()
        surfaces = ([], [[np.nan, np.inf, -np.inf]],
                    [[300.0, np.nan, np.inf, -np.inf, 1100.0]])
        for values in surfaces:
            with self.subTest(temperatures=values):
                energy, temperature = self.curve(pcm, values)
                self.assert_three_branches(energy, temperature, pcm, 800.0, 1200.0)
                self.assertLessEqual(temperature[0], pcm.T_m - 25.0)
                self.assertGreaterEqual(temperature[-1], pcm.T_m + 25.0)
        self.assertEqual(temperature[0], 300.0)
        self.assertEqual(temperature[-1], 1100.0)

    def test_zero_latent_heat_has_one_join_and_no_duplicate_energy(self):
        pcm = self.material(LH=0.0)
        energy, temperature = self.curve(pcm, [300.0, 750.0])

        self.assert_three_branches(energy, temperature, pcm, 800.0, 1200.0)
        self.assertEqual(np.count_nonzero(energy == 0.0), 1)


if __name__ == "__main__":
    unittest.main()
