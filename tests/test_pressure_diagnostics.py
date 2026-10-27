"""Independent checks for exploratory pressure evidence."""
from io import BytesIO
from types import SimpleNamespace
import unittest

import numpy as np

from scripts.pinn_pressure_diagnostics import (check_pressure_anchors, coordinate_mapping,
                                               exact_time, heldout_mask, lorentz_statistics)


class PressureDiagnosticsTests(unittest.TestCase):
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
