"""Frame updates reuse geometry and coalesce rendering without changing fields."""
from dataclasses import replace
import os
import unittest
from unittest.mock import patch

import numpy as np

from magnetic_torus.display import DisplaySelection, SceneOptions
from magnetic_torus.models import RunResult, ScenarioConfig


@unittest.skipUnless(os.environ.get("TORUS_TEST_DESKTOP") == "1", "Requires a real Qt/VTK session")
class SceneUpdateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PyQt6 import QtWidgets
        cls.application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        from magnetic_torus.scene import SceneWidget
        self.scene = SceneWidget()
        self.scene.show()
        self.addCleanup(self.scene.close_scene)
        config = replace(ScenarioConfig(), minor_radius=.35)
        points = np.array([[.8, -.2, -.2], [1.2, -.2, -.2], [1., .2, -.2], [1., 0., .2]])
        linear = 1700 + points @ np.array([20., 30., 40.])
        self.result = RunResult("scene-update", config, np.array([0., 1.]), points,
                                {"temperature": np.stack((linear, linear + 10)),
                                 "velocity": np.zeros((2, 4, 3))}, {},
                                {"cells": [[0, 1, 2, 3]], "conductor_cells": [[0, 1, 2, 3]],
                                 "conductor_mask": [True] * 4})
        self.scene.set_result(self.result)
        self.application.processEvents()
        self.application.processEvents()

    def test_render_requests_coalesce_and_export_flush_consumes_pending_frame(self):
        with patch.object(self.scene.plotter, "render") as render:
            for _ in range(3):
                self.scene.request_render()
            self.application.processEvents()
            render.assert_called_once()
            render.reset_mock()
            self.scene.request_render()
            self.scene.flush_render()
            self.application.processEvents()
            render.assert_called_once()

    def test_section_reuses_topology_and_preserves_linear_interpolation(self):
        self.scene.set_options(SceneOptions(slice_enabled=True, slice_position=1.))
        surface = self.scene._slice_surface
        actor = self.scene._slice_actor
        operator = self.scene._slice_operator
        for index in (0, 1, 0):
            self.scene.update_index(index)
            self.assertIs(self.scene._slice_surface, surface)
            self.assertIs(self.scene._slice_actor, actor)
            self.assertIs(self.scene._slice_operator, operator)
            expected = 1700 + surface.points @ np.array([20., 30., 40.]) + index * 10
            np.testing.assert_allclose(surface["slice_value"], expected, atol=1e-10)
        shifted = {"temperature": self.result.fields["temperature"] + 100}
        self.scene.set_display(DisplaySelection(shifted, source_label="PINN"))
        self.scene.update_index(1)
        self.assertIs(self.scene._slice_surface, surface)
        self.assertIs(self.scene.display_selection.fields, shifted)
        np.testing.assert_allclose(surface["slice_value"], 1810 + surface.points @ np.array([20., 30., 40.]), atol=1e-10)
        self.scene.set_options(replace(self.scene.options, slice_position=1.05))
        self.assertIsNot(self.scene._slice_surface, surface)

    def test_particle_updates_queue_one_render_and_same_run_reload_clears_analysis(self):
        paths = np.array([[[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]]])
        data = {"paths": paths, "baseline_paths": paths, "times": np.array([[0., 1.e-8, 2.e-8]]),
                "snapshot_index": 0, "metadata": {"selected_source": "numerical"}}
        with patch.object(self.scene.plotter, "render") as render:
            self.scene.set_trajectories(data)
            self.scene.draw_particles(.5)
            render.assert_not_called()
            self.application.processEvents()
            render.assert_called_once()
        self.scene.set_result(self.result)
        self.scene.clear_analysis()
        self.assertIsNone(self.scene.trajectory_data)
        self.assertNotIn("particle_paths", self.scene.plotter.actors)
        self.assertNotIn("frozen_note", self.scene.plotter.actors)

    def test_difference_bounds_and_flow_availability_invalidate_for_new_source(self):
        difference = {"temperature": np.full((2, 4), 10.)}
        self.scene.set_display(DisplaySelection(difference, difference=True))
        self.scene.update_index(0)
        limits = self.scene._difference_ranges["Temperature"]
        self.scene.update_index(1)
        self.assertIs(self.scene._difference_ranges["Temperature"], limits)
        self.scene.set_display(DisplaySelection({"temperature": np.full((2, 4), 20.)}, True))
        self.scene.update_index(0)
        self.assertEqual(tuple(self.scene.plotter.actors["reservoir"].mapper.scalar_range), (-20., 20.))
        self.scene.set_display(DisplaySelection())
        self.scene.set_options(replace(self.scene.options, show_tracers=True))
        self.assertFalse(self.scene._has_flow)
        self.scene.set_display(DisplaySelection({**self.result.fields, "velocity": np.ones((2, 4, 3)) * .001}))
        self.scene.update_index(0)
        self.assertTrue(self.scene._has_flow)


if __name__ == "__main__":
    unittest.main()
