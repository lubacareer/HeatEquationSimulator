"""Independent checks for exploratory pressure evidence."""
from io import BytesIO
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from scripts.pinn_pressure_diagnostics import (check_pressure_anchors, compare_sources, coordinate_mapping,
                                               error_statistics, exact_time, heldout_mask, lorentz_statistics)


class PressureDiagnosticsTests(unittest.TestCase):
    def test_pressure_error_statistics_distinguish_node_and_volume_weights(self):
        stats = error_statistics(np.array([1., 3.]), np.array([3., 1.]))
        self.assertAlmostEqual(stats["node_rms_pa"], np.sqrt(5.))
        self.assertAlmostEqual(stats["volume_weighted_rms_pa"], np.sqrt(3.))
        self.assertEqual(stats["absolute_quantiles_pa"]["0.5"], 2.)
        self.assertIsNone(error_statistics(np.array([]), np.array([]))["node_rms_pa"])
        with self.assertRaises(ValueError):
            error_statistics(np.array([1.]), np.array([-1.]))

    def test_seeded_holdout_matches_recorded_split(self):
        mask = heldout_mask(7, 1961, 42)
        self.assertEqual(mask.shape, (7, 1961))
        self.assertEqual(int(mask.sum()), 2745)
        for identifier in (10735, 4501, 10358, 6398, 9292):
            self.assertTrue(mask.ravel()[identifier])
        np.testing.assert_array_equal(mask, heldout_mask(7, 1961, 42))

    def test_comparison_uses_accepted_times_only(self):
        times = np.array([0., .016666666666666666, .02, .025])
        self.assertEqual(exact_time(times, .02), 2)
        with self.assertRaisesRegex(ValueError, "no interpolation"):
            exact_time(times, .019)

    def test_mesh_matching_reorders_and_rejects_duplicates(self):
        points = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
        np.testing.assert_array_equal(coordinate_mapping(points, points[[2, 0, 1]], 1.), [1, 2, 0])
        with self.assertRaisesRegex(ValueError, "one-to-one"):
            coordinate_mapping(points, points[[0, 0, 1]], 1.)

    def test_pressure_comparison_requires_matching_anchor(self):
        original = SimpleNamespace(metadata={"pinn_physics": {"bc_nodes": {"1": [0]}}})
        same = SimpleNamespace(metadata={"pinn_physics": {"bc_nodes": {"1": [2]}}})
        changed = SimpleNamespace(metadata={"pinn_physics": {"bc_nodes": {"1": [1]}}})
        check_pressure_anchors(original, same, np.array([2, 0, 1]))
        with self.assertRaisesRegex(ValueError, "anchor differs"):
            check_pressure_anchors(original, changed, np.array([2, 0, 1]))

    def test_source_comparison_uses_only_common_accepted_times(self):
        stream = BytesIO()
        np.savez_compressed(stream, node_volumes=np.array([1., 3.]))
        config = SimpleNamespace(seed=42, mode="driven", seed_field=0., major_radius=1.,
                                 to_dict=lambda: {"time_steps": 2, "seed": 42})
        def source(times, pressures, name):
            n = len(times)
            return SimpleNamespace(config=config, points=np.array([[0., 0., 0.], [1., 0., 0.]]),
                                   times=np.array(times), run_id=name,
                                   metadata={"conductor_mask": [True, True],
                                             "pinn_physics": {"bc_nodes": {"1": [0]}}},
                                   fields={"pressure": np.array(pressures), "velocity": np.zeros((n, 2, 3)),
                                           "liquid_fraction": np.zeros((n, 2))},
                                   artifact_bytes=lambda _: stream.getvalue())
        config.config_hash = "hash"
        first = source([0., .5, 1.], [[0., 0.], [0., 1.], [0., 1.]], "first")
        second = source([0., .25, 1.], [[0., 0.], [0., 2.], [0., 3.]], "second")
        with patch("scripts.pinn_pressure_diagnostics.accepted_residuals", return_value={}):
            comparison = compare_sources(first, second)
        self.assertEqual([row["time_s"] for row in comparison["common_accepted_times"]], [0., 1.])
        self.assertTrue(all(not row["interpolated"] for row in comparison["common_accepted_times"]))
        self.assertAlmostEqual(comparison["common_accepted_times"][-1]["pressure_difference"]["volume_weighted_rms_pa"], np.sqrt(3.))

    def test_lorentz_reconstruction_on_uniform_analytic_field(self):
        points = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        gradients = np.array([[[-1., -1., -1.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]])
        stream = BytesIO()
        np.savez_compressed(stream, cells=np.array([[0, 1, 2, 3]]), gradients=gradients,
                            volumes=np.array([1 / 6]), cell_labels=np.array([1]))
        potential = np.column_stack((-2 * points[:, 1], 2 * points[:, 0], np.zeros(4)))
        velocity = np.tile(np.array([3., 0., 0.]), (4, 1))
        source = SimpleNamespace(
            points=points, times=np.array([0., 1.]),
            config=SimpleNamespace(melting_temperature=100., transition_width=1.,
                                   electrical_conductivity_solid=2., electrical_conductivity_liquid=2.),
            fields={"vector_potential": np.stack((potential, potential)),
                    "velocity": np.stack((velocity, velocity)),
                    "temperature": np.full((2, 4), 100.),
                    "electric_potential": np.zeros((2, 4))},
            artifact_bytes=lambda _key: stream.getvalue(),
        )
        stats, nodal = lorentz_statistics(source, 1)
        self.assertAlmostEqual(stats["volume_weighted_mean_n_per_m3"], 96.)
        self.assertAlmostEqual(stats["maximum_quadrature_n_per_m3"], 96.)
        np.testing.assert_allclose(nodal, 96.)


if __name__ == "__main__":
    unittest.main()
