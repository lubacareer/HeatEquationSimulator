"""Worker-protocol checks and opt-in real Qt/VTK source/probe regressions.

The small analytic field fixtures exercise display state only. They are not
numerical solutions or trained PINNs and make no physics-validation claim.
"""
from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock, PropertyMock, patch

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
        from magnetic_torus.physics import enthalpy_from_temperature, liquid_fraction
        config = replace(ScenarioConfig(), solver="thermal_benchmark", turbulence=False,
                         major_radius=.3, minor_radius=.06, habitat_radius=.12)
        angle = np.arange(count) * 2 * np.pi / count
        points = np.column_stack((config.major_radius * np.cos(angle),
                                  config.major_radius * np.sin(angle), np.zeros(count)))
        temperature = np.stack((np.full(count, 1800.), np.full(count, 1820.)))
        enthalpy = enthalpy_from_temperature(temperature, config)
        fields = {"temperature": temperature, "enthalpy": enthalpy,
                  "liquid_fraction": liquid_fraction(enthalpy, config)}
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

    def test_unqualified_training_is_removed_but_guard_and_playback_remain(self):
        from magnetic_torus.learning.support import pinn_training_unavailable_reason

        with patch("magnetic_torus.app.pinn_training_unavailable_reason",
                   wraps=pinn_training_unavailable_reason) as classify:
            self.activate(self.ring_result("phase-changing-training", 8))
            reason = pinn_training_unavailable_reason(self.window.session.active)
            self.assertIn("Phase-changing thermal PINN training is not qualified", reason)
            self.assertFalse(self.window.editor.train_button.isVisible())
            self.assertEqual(self.window.editor.train_button.toolTip(), reason)
            self.assertFalse(self.window.editor.train_button.isEnabled())
            self.window.set_view("PINN")
            self.window.chart_panel._refresh()
            self.assertFalse(self.window.chart_panel.training_settings_button.isVisible())
            self.assertFalse(self.window.chart_panel.training_settings_button.isEnabled())
            self.assertEqual(self.window.chart_panel.cards[0].ax.texts[0].get_text(),
                             "No PINN available for this saved scenario")
            for _ in range(3):
                self.window._update_actions()
            with patch.object(self.window.jobs, "start") as start, patch.object(self.window.playback, "stop_all") as stop:
                self.window.start_job("pinn")
            start.assert_not_called()
            stop.assert_not_called()
            self.assertIn(reason, self.window.message.text())
            self.assertEqual(classify.call_count, 1)
        for button in (self.window.editor.run_button, self.window.editor.trace_button,
                       self.window.editor.dose_button):
            self.assertTrue(button.isEnabled())
        accepted = self.window.session.active
        for status in ("passed", "failed"):
            model = PINNResult("saved-" + status, accepted.run_id,
                               accepted.config.config_hash, accepted.fields, {}, {},
                               {"validation_status": status})
            path = self.window.store.save_pinn(model)
            self.window.session.set_pinn(self.window.store.load_pinn(path), path)
            self.window._update_actions()
            self.assertFalse(self.window.editor.train_button.isEnabled())
            self.assertFalse(self.window.editor.train_button.isVisible())
            self.window.source.setCurrentText("PINN")
            self.assertIs(self.window.scene.display_selection.fields, self.window.session.pinn.fields)
            self.window.source.setCurrentText("PINN error")
            self.assertIs(self.window.scene.display_selection.fields, self.window.session.pinn_difference)

    def test_supported_training_returns_after_switching_sources_and_respects_busy(self):
        from magnetic_torus.physics import enthalpy_from_temperature, liquid_fraction

        self.assertFalse(self.window.editor.train_button.isEnabled())
        self.assertTrue(self.window.editor.train_button.isVisible())
        self.activate(self.ring_result("unsupported-source", 8))
        native = self.ring_result("single-phase-source", 8)
        temperature = np.stack((np.full(8, 1800.), np.full(8, 1801.)))
        enthalpy = enthalpy_from_temperature(temperature, native.config)
        native = replace(native, fields={"temperature": temperature,
                         "enthalpy": enthalpy,
                         "liquid_fraction": liquid_fraction(enthalpy, native.config)})
        coupled = self.tetra_result("stationary-coupled-source", regularized=True)
        fields = {key: np.repeat(values[:1], len(coupled.times), axis=0)
                  for key, values in coupled.fields.items()}
        for key in ("pressure", "electric_potential"):
            fields[key] = np.zeros((2, len(coupled.points)))
        for key in ("vector_potential", "magnetic_field"):
            fields[key] = np.zeros((2, len(coupled.points), 3))
        coupled = replace(coupled, fields=fields)
        evolving = replace(coupled, run_id="evolving-coupled-source",
                           fields={**fields, "electric_potential": np.stack(
                               (np.zeros(len(coupled.points)), np.ones(len(coupled.points))))})
        self.activate(evolving)
        self.assertFalse(self.window.editor.train_button.isEnabled())
        self.assertFalse(self.window.editor.train_button.isVisible())
        for result in (native, coupled):
            with self.subTest(solver=result.config.solver):
                self.activate(result)
                self.assertTrue(self.window.editor.train_button.isVisible())
                self.assertTrue(self.window.editor.train_button.isEnabled())
                self.window.set_view("PINN")
                self.window.chart_panel._refresh()
                self.assertTrue(self.window.chart_panel.training_settings_button.isVisible())
                self.assertTrue(self.window.chart_panel.training_settings_button.isEnabled())
                for controller in (self.window.jobs, self.window.exports):
                    with patch.object(type(controller), "busy", new_callable=PropertyMock, return_value=True):
                        self.window._update_actions()
                        self.assertFalse(self.window.editor.train_button.isEnabled())
                        self.assertTrue(self.window.editor.train_button.isVisible())
                        with patch.object(self.window.jobs, "start") as start:
                            self.window.start_job("pinn")
                        start.assert_not_called()
                    self.window._update_actions()
                    self.assertTrue(self.window.editor.train_button.isEnabled())
                with patch.object(self.window.jobs, "start") as start, patch.object(self.window.playback, "stop_all") as stop:
                    self.window.start_job("pinn")
                start.assert_called_once()
                self.assertEqual(start.call_args.args[0], "pinn")
                stop.assert_called_once()

    def test_temperature_profile_rebuilt_when_switching_to_smaller_run(self):
        self.activate(self.ring_result("large-profile", 32))
        self.window.thermal_plot.setCurrentText("Temperature profile")
        self.assertGreater(int(np.max(self.window.chart_panel.profile_samples[1])), 3)
        self.activate(self.ring_result("small-profile", 4))
        self.assertLess(int(np.max(self.window.chart_panel.profile_samples[1])), 4)
        self.assertEqual(self.window.session.active.run_id, "small-profile")

    def test_duration_is_visible_before_run_and_recalculation_uses_edited_duration(self):
        from magnetic_torus.models import preset

        editor = self.window.editor
        duration = editor.editors["duration"]
        self.assertIsNone(self.window.session.active)
        self.assertTrue(duration.isVisible())
        self.assertLess(duration.mapTo(editor, QtCore.QPoint()).y(), editor.run_button.mapTo(editor, QtCore.QPoint()).y())
        config = replace(preset("Thermal benchmark"), resolution=3, time_steps=4)
        editor.set_config(config)
        duration.setText("invalid")
        editor.validate()
        self.assertFalse(editor.run_button.isEnabled())
        editor.set_config(config)
        self.assertTrue(editor.run_button.isEnabled())
        errors = []
        self.window.jobs.failed.connect(lambda message, detail: errors.append(message))

        def wait_for_result():
            deadline = time.monotonic() + 30
            while self.window.jobs.busy and time.monotonic() < deadline:
                self.application.processEvents()
                time.sleep(.01)
            self.application.processEvents()
            self.assertFalse(self.window.jobs.busy, "Duration regression worker timed out")
            self.assertEqual(errors, [])

        duration.setText("7.5")
        self.assertEqual(editor.read_config().duration, 7.5)
        self.window.start_job("simulate")
        wait_for_result()
        first = self.window.session.active
        self.assertEqual(first.config.duration, 7.5)
        self.assertAlmostEqual(float(first.times[-1]), 7.5)
        duration.setText("12.5")
        self.window.start_job("simulate")
        self.assertIs(self.window.session.active, first)
        wait_for_result()
        self.assertNotEqual(self.window.session.active.run_id, first.run_id)
        self.assertEqual(self.window.session.active.config.duration, 12.5)
        self.assertAlmostEqual(float(self.window.session.active.times[-1]), 12.5)

    def test_switching_to_shorter_run_cancels_movie_before_restoring_frame(self):
        self.activate(self.ring_result("long-export", 8))
        short = self.ring_result("short-export", 4)
        short = replace(short, times=short.times[:1], fields={key: values[:1] for key, values in short.fields.items()})
        short_path = self.window.store.save_run(short)
        writer = Mock()
        self.window.exports.writer = writer
        self.window.exports._movie_restore = 1
        self.window.exports._movie_frames = [0, 1]
        self.window.activate_result(short_path)
        writer.close.assert_called_once()
        self.assertFalse(self.window.exports.busy)
        self.assertEqual(self.window.session.active.run_id, short.run_id)
        self.assertEqual(self.window.session.current_index, 0)
        self.window.exports.writer = Mock()
        self.window.exports._movie_restore = 999
        self.window.exports.finish_movie(cancelled=True)
        self.assertEqual(self.window.session.current_index, 0)

    def test_movie_startup_is_busy_before_first_capture_and_cleans_up_failures(self):
        self.activate(self.ring_result("startup-export", 4))
        export = self.window.exports
        observed = []
        export.busy_changed.connect(lambda busy: observed.append((busy, export.busy, self.window.job_widget.isVisible())))

        def screenshot(**kwargs):
            self.assertTrue(export.busy)
            self.assertTrue(self.window.job_widget.isVisible())
            self.assertFalse(self.window.editor.run_button.isEnabled())
            return np.zeros((64, 64, 3), dtype=np.uint8)

        writer = Mock()
        with patch.object(self.window.scene.plotter, "screenshot", side_effect=screenshot), patch("imageio_ffmpeg.write_frames", return_value=writer):
            export.begin_movie_export("fixture.mp4")
        self.assertEqual(observed[0], (True, True, True))
        self.assertTrue(export.busy)
        export.finish_movie(cancelled=True)
        writer.close.assert_called_once()
        self.assertFalse(export.busy)
        for failure in ("capture", "encoder"):
            with self.subTest(failure=failure):
                with patch.object(self.window.scene.plotter, "screenshot", side_effect=RuntimeError("capture failed") if failure == "capture" else screenshot), patch("imageio_ffmpeg.write_frames", side_effect=RuntimeError("encoder failed")):
                    export.begin_movie_export("fixture-failed.mp4")
                self.assertFalse(export.busy)
                self.assertFalse(export.movie_timer.isActive())
                self.assertTrue(self.window.editor.run_button.isEnabled())

    def test_history_selection_during_movie_startup_cancels_superseded_request(self):
        self.activate(self.ring_result("startup-old", 8))
        replacement = self.ring_result("startup-new", 4)
        path = self.window.store.save_run(replacement)
        QtCore.QTimer.singleShot(0, lambda: self.window.activate_result(path))
        with patch("imageio_ffmpeg.write_frames") as encoder:
            self.window.exports.begin_movie_export("cancelled-startup.mp4")
        encoder.assert_not_called()
        self.assertFalse(self.window.exports.busy)
        self.assertEqual(self.window.session.active.run_id, replacement.run_id)

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


    def test_failed_pinn_explains_gates_and_adjust_training_reveals_existing_inputs(self):
        from magnetic_torus.physics import enthalpy_from_temperature, liquid_fraction
        source = self.tetra_result("failed-fit-source")
        temperature = np.full_like(source.fields["temperature"], 1800.)
        enthalpy = enthalpy_from_temperature(temperature, source.config)
        source = replace(source, fields={**source.fields, "temperature": temperature,
                                         "enthalpy": enthalpy,
                                         "liquid_fraction": liquid_fraction(enthalpy, source.config)})
        source_path = self.activate(source)
        accepted = self.window.session.active
        model = PINNResult(
            "failed-fit", accepted.run_id, accepted.config.config_hash, accepted.fields,
            {"loss": np.array([.5, .137])},
            {"validation_loss": .137, "physics_loss": 1.e-6, "boundary_loss": 1.e-6,
             "velocity_normalized_rmse": .286, "pressure_normalized_rmse": .463,
             "electric_potential_normalized_rmse": .726, "charge_residual_rms": .2, "epochs": 250.},
            {"validation_status": "failed", "training_epochs": 5000},
        )
        model_path = self.window.store.save_pinn(model)
        self.window.editor.set_config(replace(accepted.config, pinn_epochs=2500))
        self.window.session.job_revision = self.window.session.selection_revision
        self.window._job_completed({"action": "pinn", "source_path": str(source_path), "path": str(model_path)})
        self.assertTrue(self.window.editor.train_button.isEnabled())
        self.assertTrue(self.window.source.model().item(1).isEnabled())
        self.assertTrue(self.window.source.model().item(2).isEnabled())
        self.assertEqual(self.window.session.pinn.run_id, model.run_id)
        message = self.window.message.text()
        self.assertIn("250 epochs", message)
        for check in ("held-out loss 0.137 > 0.01", "velocity normalized RMSE 0.286 > 0.1",
                      "pressure normalized RMSE 0.463 > 0.1", "electric potential normalized RMSE 0.726 > 0.1"):
            self.assertIn(check, message)
        self.assertIn("Accepted numerical results remain available", message)
        self.assertIn("1 more failed checks", message)
        self.assertIn("charge residual: 0.2 (limit 0.1)", self.window.log_text.toPlainText())
        self.assertIs(self.window.session.active, accepted)
        before = self.window.editor.read_config().to_dict()
        self.window.sidebar_button.setChecked(False)
        self.window.chart_panel.training_settings_button.click()
        self.application.processEvents()
        self.application.processEvents()
        epoch = self.window.editor.editors["pinn_epochs"]
        self.assertTrue(self.window.sidebar.isVisible())
        self.assertTrue(epoch.isVisible())
        self.assertIs(self.window.focusWidget(), epoch)
        scroll = self.window.editor._scroll_area
        self.assertTrue(scroll.viewport().rect().intersects(QtCore.QRect(epoch.mapTo(scroll.viewport(), QtCore.QPoint()), epoch.size())))
        self.assertEqual(self.window.editor.read_config().to_dict(), before)
        self.assertIs(self.window.session.active, accepted)


if __name__ == "__main__":
    unittest.main()
