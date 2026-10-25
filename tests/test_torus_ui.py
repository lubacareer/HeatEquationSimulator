"""Worker-protocol checks and opt-in real Qt/VTK source/probe regressions.

The small analytic field fixtures exercise display state only. They are not
numerical solutions or trained PINNs and make no physics-validation claim.
"""
from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import PyQt6.QtCore as QtCore

from magnetic_torus.jobs import JobController
from magnetic_torus.models import PINNResult, RunResult, ScenarioConfig
from magnetic_torus.store import RunStore


class WorkerProtocolTests(unittest.TestCase):
    def test_nonobject_json_is_logged_and_later_events_are_processed(self):
        with tempfile.TemporaryDirectory() as directory:
            controller = JobController(RunStore(directory))
            process = Mock(spec=QtCore.QProcess)
            process.readAllStandardOutput.return_value = QtCore.QByteArray(
                b'0.5\n[1, 2]\nnull\n"library status"\ntrue\n'
                b'{"event":"progress","fraction":0.75,"message":"Accepted step"}\n'
                b'{"event":"complete","path":"result.h5"}\n')
            controller.process = process
            logs, progress, completed = [], [], []
            controller.log.connect(logs.append)
            controller.progress.connect(lambda fraction, message: progress.append((fraction, message)))
            controller.completed.connect(completed.append)
            controller._read_stdout()
            self.assertEqual(logs, ['0.5', '[1, 2]', 'null', '"library status"', 'true'])
            self.assertEqual(progress, [(0.75, "Accepted step")])
            self.assertEqual(completed, [{"event": "complete", "path": "result.h5"}])


@unittest.skipUnless(os.environ.get("TORUS_TEST_DESKTOP") == "1", "Set TORUS_TEST_DESKTOP=1 for real Qt/VTK regressions")
class DesktopSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import PyQt6.QtWidgets as QtWidgets
        from magnetic_torus.app import configure_application
        cls.application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        configure_application(cls.application)

    def setUp(self):
        from magnetic_torus.app import TorusWindow
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.window = TorusWindow(self.temporary.name)
        self.window.setWindowTitle("Magnetic Torus - UI regression verification")
        self.window.show()
        self.application.processEvents()
        self.addCleanup(self.close_window)

    def close_window(self):
        self.window.jobs.process = None
        self.window.close()
        self.application.processEvents()

    def ring_result(self, run_id, count):
        from magnetic_torus.physics import enthalpy_from_temperature
        config = replace(ScenarioConfig(), solver="thermal_benchmark", turbulence=False,
                         major_radius=.3, minor_radius=.06, habitat_radius=.12)
        angle = np.arange(count) * 2 * np.pi / count
        points = np.column_stack((config.major_radius * np.cos(angle),
                                  config.major_radius * np.sin(angle), np.zeros(count)))
        temperature = np.stack((np.full(count, 1800.), np.full(count, 1820.)))
        fields = {"temperature": temperature, "enthalpy": enthalpy_from_temperature(temperature, config)}
        return RunResult(run_id, config, np.array([0., 1.]), points, fields, {}, {"backend": "UI fixture"})

    def tetra_result(self, run_id="ui-tetra", regularized=False):
        from magnetic_torus.physics import enthalpy_from_temperature, regularized_enthalpy
        config = replace(ScenarioConfig(), minor_radius=.35,
                         solver="fenics" if regularized else "thermal_benchmark")
        points = np.array([[.8, -.2, -.2], [1.2, -.2, -.2], [1., .2, -.2],
                           [1., 0., .2], [1.4, .2, .2]])
        temperature = np.tile(config.melting_temperature + np.array([-.25, .25])[:, None], (1, len(points)))
        enthalpy = regularized_enthalpy(temperature, config) if regularized else enthalpy_from_temperature(temperature, config)
        fields = {"temperature": temperature, "enthalpy": enthalpy,
                  "liquid_fraction": np.clip((temperature - config.melting_temperature) / config.transition_width + .5, 0, 1),
                  "velocity": np.zeros((2, len(points), 3))}
        return RunResult(run_id, config, np.array([0., 1.]), points, fields, {},
                         {"backend": "UI constitutive fixture", "cells": [[0, 1, 2, 3], [1, 2, 3, 4]],
                          "conductor_cells": [[0, 1, 2, 3]], "conductor_mask": [True, True, True, True, False]})

    def activate(self, result):
        path = self.window.store.save_run(result)
        self.window.activate_result(path)
        self.window.editor.set_config(result.config)
        return path

    def attach_ui_pinn(self, result, shift=100.):
        fields = {key: value.copy() for key, value in result.fields.items()}
        fields["temperature"] += shift
        fields["enthalpy"] += 100000.
        fields["velocity"][:] = np.array([3., 4., 0.])
        fields["velocity"][1] *= 2
        model = PINNResult("ui-pinn", result.run_id, result.config.config_hash, fields, {}, {},
                           {"validation_status": "unvalidated UI fixture"})
        self.window.session.set_pinn(model)
        self.window._update_actions()
        return model

    def test_temperature_profile_rebuilt_when_switching_to_smaller_run(self):
        self.activate(self.ring_result("large-profile", 32))
        self.window.thermal_plot.setCurrentText("Temperature profile")
        self.assertGreater(int(np.max(self.window.chart_panel.profile_samples[1])), 3)
        self.activate(self.ring_result("small-profile", 4))
        self.assertLess(int(np.max(self.window.chart_panel.profile_samples[1])), 4)
        self.assertEqual(self.window.session.active.run_id, "small-profile")

    def test_display_controls_do_not_repeat_full_scene_updates(self):
        self.activate(self.tetra_result())
        with patch.object(self.window.scene, "update_index", wraps=self.window.scene.update_index) as update:
            self.window.slice.setChecked(True)
            self.assertEqual(update.call_count, 1)
        with patch.object(self.window.scene, "update_index", wraps=self.window.scene.update_index) as update:
            self.window.quantity.setCurrentText("Specific enthalpy")
            self.assertEqual(update.call_count, 1)

    def test_section_and_cutaway_keep_pinn_and_error_sources(self):
        result = self.tetra_result()
        self.activate(result)
        model = self.attach_ui_pinn(result)
        self.window.source.setCurrentText("PINN")
        self.window.slice.setChecked(True)
        self.window.section_slider.setValue(740)
        self.assertIs(self.window.scene.display_selection.fields, model.fields)
        self.assertFalse(self.window.scene.display_selection.difference)
        self.window.cutaway.setChecked(False)
        self.assertIs(self.window.scene.display_selection.fields, model.fields)
        self.window.source.setCurrentText("PINN error")
        self.window.section_slider.setValue(740)
        self.window.cutaway.setChecked(True)
        self.assertTrue(self.window.scene.display_selection.difference)
        self.assertIs(self.window.scene.display_selection.fields, self.window.session.pinn_difference)
        section = self.window.scene.plotter.actors["slice"]
        np.testing.assert_allclose(section.mapper.scalar_range, (-100., 100.))

    def test_pinn_color_limits_are_stable_shared_and_invalidate_for_new_model(self):
        result = self.tetra_result()
        self.activate(result)
        model = self.attach_ui_pinn(result)
        self.window._render()
        expected = (float(result.fields["temperature"][:, :4].min()), float(model.fields["temperature"][:, :4].max()))
        np.testing.assert_allclose(self.window.scene.color_range(), expected)
        self.window.source.setCurrentText("PINN")
        np.testing.assert_allclose(self.window.scene._actor.mapper.scalar_range, expected)
        self.window.playback.seek_time(1.)
        np.testing.assert_allclose(self.window.scene._actor.mapper.scalar_range, expected)
        self.window.source.setCurrentText("Numerical")
        np.testing.assert_allclose(self.window.scene._actor.mapper.scalar_range, expected)
        replacement = self.attach_ui_pinn(result, shift=200.)
        self.window._render()
        self.assertAlmostEqual(self.window.scene.color_range()[1], float(replacement.fields["temperature"][:, :4].max()))
        self.window.quantity.setCurrentText("Flow speed")
        np.testing.assert_allclose(self.window.scene.color_range(), (0., 10.))
        self.window.quantity.setCurrentText("Specific enthalpy")
        expected_enthalpy = (float(result.fields["enthalpy"][:, :4].min() / 1000),
                             float(replacement.fields["enthalpy"][:, :4].max() / 1000))
        np.testing.assert_allclose(self.window.scene.color_range(), expected_enthalpy)

    def test_section_and_overlay_keep_run_difference_and_shared_compare_scale(self):
        result = self.tetra_result()
        self.activate(result)
        fields = {key: value.copy() for key, value in result.fields.items()}
        fields["temperature"] -= 10.
        alternative = replace(result, run_id="comparison-ui", fields=fields)
        path = self.window.store.save_run(alternative)
        self.window.refresh_history()
        self.window._compare_selected(self.window.editor.compare_history.findData(str(path)))
        self.window.source.setCurrentText("Run difference")
        self.window.slice.setChecked(True)
        self.window.section_slider.setValue(740)
        self.window.cutaway.setChecked(False)
        self.assertTrue(self.window.scene.display_selection.difference)
        self.assertIs(self.window.scene.display_selection.fields, self.window.session.comparison_difference)
        self.assertFalse(self.window.compare_scene.display_selection.difference)
        np.testing.assert_allclose(self.window.scene.plotter.actors["slice"].mapper.scalar_range, (-10., 10.))
        self.window.source.setCurrentText("Numerical")
        np.testing.assert_allclose(self.window.scene._actor.mapper.scalar_range,
                                   self.window.compare_scene._actor.mapper.scalar_range)

    def test_fe_plot_keeps_sharp_plateau_and_matches_computational_probe(self):
        from magnetic_torus.physics import regularized_enthalpy
        result = self.tetra_result(regularized=True)
        self.activate(result)
        chart = self.window.chart_panel.cards[0]
        sharp, computational = chart.ax.lines[:2]
        np.testing.assert_allclose(sharp.get_xdata()[1:3], [0., result.config.latent_heat / 1000])
        np.testing.assert_allclose(sharp.get_ydata()[1:3], result.config.melting_temperature)
        np.testing.assert_allclose(computational.get_xdata(), regularized_enthalpy(computational.get_ydata(), result.config) / 1000)
        self.window.playback.seek_time(1.)
        np.testing.assert_allclose(chart.marker.get_xdata(), [result.fields["enthalpy"][1, self.window.session.probe_index] / 1000])
        np.testing.assert_allclose(chart.marker.get_ydata(), [result.fields["temperature"][1, self.window.session.probe_index]])

    def test_radiation_completion_restores_numerical_source_and_busy_controls(self):
        result = self.tetra_result()
        source_path = self.activate(result)
        self.attach_ui_pinn(result)
        self.window.jobs.process = Mock(spec=QtCore.QProcess)
        self.window._busy_changed(True)
        self.assertFalse(self.window.source.isEnabled())
        # Programmatically simulate a queued selector change during job delivery.
        self.window.source.setCurrentText("PINN")
        artifact = Path(self.temporary.name) / "trajectory.npz"
        data = {"source_run_id": result.run_id, "config_hash": result.config.config_hash,
                "snapshot_index": 0, "outcomes": ["escaped"], "baseline_outcomes": ["escaped"],
                "metadata": {"selected_source": "numerical"}}
        np.savez(artifact, paths=np.array([[[0., 0., 0.], [1., 0., 0.]]]),
                 baseline_paths=np.array([[[0., 0., 0.], [1., 0., 0.]]]), times=np.array([[0., 1.e-8]]),
                 species=np.array([[1, 1]]), energies_mev_nucleon=np.array([100.]),
                 _metadata_json=np.array(json.dumps(data)))
        self.window.session.job_revision = self.window.session.selection_revision
        self.window._job_completed({"action": "trace", "source_path": str(source_path), "path": str(artifact), "selected_source": "Numerical"})
        self.assertEqual(self.window.source.currentText(), "Numerical")
        self.assertIsNone(self.window.scene.display_selection.fields)
        self.window.jobs.process = None
        self.window._busy_changed(False)
        self.assertTrue(self.window.source.isEnabled())
        self.window._job_failed("Fixture failure", "No active process")
        self.assertTrue(self.window.source.isEnabled())


if __name__ == "__main__":
    unittest.main()
