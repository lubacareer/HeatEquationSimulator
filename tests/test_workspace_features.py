"""Real Qt/VTK journeys with explicit accepted-array and physical-state oracles."""

from dataclasses import replace
import os
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PyQt6 import QtWidgets

from magnetic_torus.models import DoseResult, PINNResult
from workspace_fixtures import accepted_run

SUITE_CAPABILITIES = ("desktop",)


@unittest.skipUnless(
    os.environ.get("TORUS_TEST_DESKTOP") == "1",
    "Requires a real Qt/VTK graphical session",
)
class WorkspaceFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from magnetic_torus.app import configure_application

        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        configure_application(cls.app)

    def setUp(self):
        from magnetic_torus.app import TorusWindow

        self.temporary = tempfile.TemporaryDirectory(prefix="torus-workspace-")
        self.addCleanup(self.temporary.cleanup)
        self.window = TorusWindow(self.temporary.name)
        self.window.resize(1282, 780)
        self.window.show()
        self.app.processEvents()
        self.addCleanup(self.close_window)
        self.result = accepted_run()
        self.path = self.activate(self.result)
        self.result = self.window.session.active

    def close_window(self):
        self.window.close()
        self.app.processEvents()

    def activate(self, result):
        path = self.window.store.save_run(result)
        self.window.activate_result(path)
        self.window.editor.set_config(result.config)
        return path

    def select_comparison(self, result):
        path = self.window.store.save_run(result)
        self.window.refresh_history()
        index = self.window.editor.compare_history.findData(str(path))
        self.assertGreaterEqual(index, 0)
        self.window._compare_selected(index)
        return path

    def attach_model(self):
        fields = dict(self.result.fields)
        fields["temperature"] = fields["temperature"] + 1.0
        fields["enthalpy"] = fields["enthalpy"] + self.result.config.heat_capacity_solid
        model = PINNResult(
            "workspace-pinn-fixture",
            self.result.run_id,
            self.result.config.config_hash,
            fields,
            {},
            {},
            {"validation_status": "failed", "note": "UI fixture; never trained"},
        )
        self.window.session.set_pinn(model)
        self.window._update_actions()
        return model

    def test_actual_world_pick_uses_nearest_conductor_and_links_probe_histories(self):
        point = self.result.points[4]
        mask = np.asarray(self.result.metadata["conductor_mask"])
        ids = np.flatnonzero(mask)
        expected = int(
            ids[np.argmin(np.linalg.norm(self.result.points[ids] - point, axis=1))]
        )
        self.window.scene._pick(point)
        self.assertEqual(self.window.session.probe_index, expected)
        self.assertNotEqual(expected, 4)
        self.window.playback.seek_time(0.55)
        card = self.window.chart_panel.cards[0]
        np.testing.assert_allclose(
            card.marker.get_ydata(), [self.result.fields["temperature"][2, expected]]
        )
        self.window.thermal_plot.setCurrentText("Probe temperature")
        np.testing.assert_array_equal(
            self.window.chart_panel.cards[2].ax.lines[0].get_ydata(),
            self.result.fields["temperature"][:, expected],
        )

    def test_views_and_permitted_sources_keep_exact_field_identity(self):
        model = self.attach_model()
        self.select_comparison(accepted_run("comparison-for-sources"))
        for view in ("Thermal", "Dynamo", "Shielding", "PINN", "Compare"):
            self.window.set_view(view)
            for source in ("Numerical", "PINN", "PINN error"):
                with self.subTest(view=view, source=source):
                    self.window.source.setCurrentText(source)
                    selection = self.window.scene.display_selection
                    self.assertEqual(self.window.session.source, source)
                    if source == "Numerical":
                        self.assertIsNone(selection.fields)
                    elif source == "PINN":
                        self.assertIs(selection.fields, model.fields)
                        self.assertFalse(selection.difference)
                    else:
                        self.assertTrue(selection.difference)
                        np.testing.assert_allclose(selection.fields["temperature"], 1.0)
        np.testing.assert_array_equal(
            self.result.fields["temperature"], accepted_run().fields["temperature"]
        )

    def test_run_difference_clears_explicitly_for_incompatible_geometry(self):
        self.select_comparison(
            accepted_run("matching", (0.0, 0.4, 0.9, 2.0), shift=2.0)
        )
        self.window.source.setCurrentText("Run difference")
        self.assertTrue(self.window.scene.display_selection.difference)
        np.testing.assert_allclose(
            self.window.scene.display_selection.fields["temperature"], -2.0
        )
        self.window.playback.seek_time(0.55)
        self.assertEqual(self.window.compare_scene.index, 1)
        self.window.scene.plotter.camera.azimuth += 15
        np.testing.assert_allclose(
            list(self.window.compare_scene.plotter.camera_position),
            list(self.window.scene.plotter.camera_position),
        )
        incompatible = replace(
            accepted_run("different"),
            points=self.result.points + np.array([0.0, 0.0, 0.01]),
        )
        self.select_comparison(incompatible)
        self.assertIsNone(self.window.session.comparison_difference)
        self.assertEqual(self.window.source.currentText(), "Numerical")
        self.assertEqual(self.window.session.source, "Numerical")
        self.assertIsNone(self.window.scene.display_selection.fields)
        self.assertFalse(self.window.scene.display_selection.difference)

    def test_nonoverlapping_comparison_is_rejected_without_replacing_selected_runs(
        self,
    ):
        self.select_comparison(accepted_run("selected-comparison"))
        previous = self.window.session.comparison
        self.select_comparison(accepted_run("no-overlap", (3.0, 4.0, 5.0)))
        self.assertIs(self.window.session.comparison, previous)
        self.assertIs(self.window.session.active, self.result)
        self.assertIn("no overlapping physical timeline", self.window.message.text())

    def test_selecting_comparison_clamps_time_before_render_and_updates_linked_controls(
        self,
    ):
        emissions = []
        self.window.playback.render_requested.connect(lambda: emissions.append(True))
        for name, times, current in (
            ("late-start", (0.55, 1.0, 2.0), 0.0),
            ("early-end", (0.0, 0.2, 0.55), 2.0),
            ("current-valid", (0.2, 0.55, 1.0), 0.55),
            ("one-snapshot", (0.55,), 2.0),
        ):
            with self.subTest(comparison=name):
                self.window.set_view("Thermal")
                self.window.playback.seek_time(current)
                self.window.playback._play_position = -99.0
                emissions.clear()
                with patch.object(
                    self.window, "_render", wraps=self.window._render
                ) as render:
                    self.select_comparison(accepted_run(name, times))
                self.assertEqual(render.call_count, 1)
                self.assertEqual(emissions, [])
                self.assertEqual(self.window.session.current_time, 0.55)
                self.assertEqual(self.window.playback._play_position, 0.55)
                self.assertEqual(self.window.scene.index, 2)
                comparison = self.window.session.comparison
                self.assertEqual(
                    comparison.times[self.window.compare_scene.index], 0.55
                )
                self.assertEqual(self.window.playback.time_label.text(), "t = 0.55 s")
                start, end = max(0.0, times[0]), min(2.0, times[-1])
                expected_slider = (
                    int(1000 * (0.55 - start) / (end - start)) if end > start else 0
                )
                self.assertEqual(self.window.playback.timeline.value(), expected_slider)
                for card in self.window.chart_panel.cards:
                    if card.cursor is not None:
                        np.testing.assert_array_equal(
                            card.cursor.get_xdata(), [0.55, 0.55]
                        )

    def test_reentering_compare_clamps_before_render_without_emitting_seek(self):
        self.select_comparison(accepted_run("later-window", (0.55, 1.0, 2.0)))
        self.window.set_view("Thermal")
        self.window.playback.seek_time(0.0)
        emissions = []
        self.window.playback.render_requested.connect(lambda: emissions.append(True))
        with patch.object(self.window, "_render", wraps=self.window._render) as render:
            self.window.set_view("Compare")
        self.assertEqual(self.window.session.current_time, 0.55)
        self.assertEqual(self.window.playback._play_position, 0.55)
        self.assertEqual(render.call_count, 1)
        self.assertEqual(emissions, [])

    def test_overlap_without_active_snapshot_is_rejected_without_replacing_comparison(
        self,
    ):
        self.activate(accepted_run("coarse", (0.0, 2.0)))
        self.select_comparison(accepted_run("previous-window", (0.0, 1.0, 2.0)))
        previous = self.window.session.comparison
        active = self.window.session.active
        self.select_comparison(accepted_run("between-snapshots", (0.5, 1.5)))
        self.assertIs(self.window.session.comparison, previous)
        self.assertIs(self.window.session.active, active)
        self.assertIn("no accepted active snapshot", self.window.message.text())

    def test_reactivation_preserves_camera_physical_time_probe_overlays_and_chart_zoom(
        self,
    ):
        self.window.scene.camera_preset("Side")
        self.window.scene._pick(self.result.points[3])
        self.window.playback.seek_time(0.55)
        self.window.slice.setChecked(True)
        self.window.section_slider.setValue(200)
        self.window.opacity.setValue(75)
        self.window.detail.setCurrentText("Low")
        card = self.window.chart_panel.cards[1]
        card.ax.set_xlim(0.1, 1.5)
        card.ax.set_ylim(-1.0, 1.0)
        card._remember_view(SimpleNamespace(inaxes=card.ax))
        camera = np.asarray(list(self.window.scene.plotter.camera_position))
        old_point = self.result.points[self.window.session.probe_index]
        new = replace(
            accepted_run("recalculated", (0.0, 0.6, 1.8, 3.0)),
            points=self.result.points + np.array([0.0, 0.0, 0.001]),
        )
        self.activate(new)
        np.testing.assert_allclose(
            list(self.window.scene.plotter.camera_position), camera
        )
        self.assertEqual(self.window.session.current_time, 0.6)
        ids = np.flatnonzero(new.metadata["conductor_mask"])
        nearest = int(
            ids[np.argmin(np.linalg.norm(new.points[ids] - old_point, axis=1))]
        )
        self.assertEqual(self.window.session.probe_index, nearest)
        self.assertTrue(self.window.scene.options.slice_enabled)
        self.assertEqual(self.window.scene.options.opacity, 0.75)
        self.assertEqual(self.window.scene.options.visual_detail, "Low")
        np.testing.assert_allclose(
            self.window.chart_panel.cards[1].ax.get_xlim(), (0.1, 1.5)
        )
        np.testing.assert_allclose(
            self.window.chart_panel.cards[1].ax.get_ylim(), (-1.0, 1.0)
        )

    def test_dose_layer_uses_accepted_snapshot_and_error_maps_hide_it(self):
        dose = DoseResult(
            "dose-fixture",
            self.result.run_id,
            self.result.config.config_hash,
            1,
            "water",
            ("Water",),
            np.array([0.001]),
            np.array([0.0001]),
            100,
            {"field_source": "numerical", "statistical_precision_met": [False]},
        )
        self.window.session.set_dose(dose)
        self.window.scene.set_dose(dose)
        self.window.playback.seek_time(0.2)
        actor = self.window.scene.plotter.actors["dose_caption"]
        self.assertTrue(actor.GetVisibility())
        self.assertIn("Gy", " ".join(actor.GetText(index) or "" for index in range(4)))
        self.window.playback.seek_time(0.55)
        self.assertFalse(actor.GetVisibility())
        self.attach_model()
        self.window.playback.seek_time(0.2)
        self.window.source.setCurrentText("PINN error")
        self.assertFalse(actor.GetVisibility())
        self.activate(accepted_run("new-source"))
        self.assertIsNone(self.window.session.dose)
        self.assertNotIn("dose_caption", self.window.scene.plotter.actors)

    def test_missing_quantity_is_disabled_instead_of_mislabeling_a_fallback_field(self):
        self.window.quantity.setCurrentText("Flow speed")
        minimal = replace(
            accepted_run("temperature-only"),
            fields={
                "temperature": self.result.fields["temperature"],
                "enthalpy": self.result.fields["enthalpy"],
            },
        )
        self.activate(minimal)
        self.assertEqual(self.window.quantity.currentText(), "Temperature")
        item = self.window.quantity.model().item(
            self.window.quantity.findText("Flow speed")
        )
        self.assertFalse(item.isEnabled())
        self.assertEqual(self.window.scene.quantity, "Temperature")


if __name__ == "__main__":
    unittest.main()
