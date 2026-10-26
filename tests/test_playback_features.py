"""Clock-driven playback checks without a renderer or numerical worker."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from unittest.mock import Mock, patch
import unittest

import numpy as np
from PyQt6 import QtWidgets

from magnetic_torus.ui.playback import PlaybackController
from magnetic_torus.ui.session import WorkspaceSession
from workspace_fixtures import accepted_run

SUITE_CAPABILITIES = ("desktop",)


class PlaybackFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.session = WorkspaceSession()
        self.session.activate(accepted_run(), "fixture.h5")
        self.scene = Mock()
        self.controller = PlaybackController(self.session, self.scene)
        self.frames = []
        self.controller.render_requested.connect(
            lambda: self.frames.append(self.session.current_index)
        )
        self.addCleanup(self.controller.close)
        self.addCleanup(self.controller.stop_all)

    def test_every_speed_uses_physical_time_and_nonuniform_nearest_snapshot(self):
        for i in range(self.controller.speed.count()):
            with self.subTest(speed=self.controller.speed.itemText(i)):
                self.controller.stop_all()
                self.controller.seek_time(0)
                self.controller.loop.setChecked(False)
                self.controller.speed.setCurrentIndex(i)
                with patch(
                    "magnetic_torus.ui.playback.time.monotonic", return_value=10.0
                ):
                    self.controller.toggle_play()
                speed = float(self.controller.speed.currentText().replace("×", ""))
                with patch(
                    "magnetic_torus.ui.playback.time.monotonic", return_value=10.1
                ):
                    self.controller._play_tick()
                expected_time = min(0.1 * speed, self.session.timeline_end())
                expected = int(
                    np.argmin(np.abs(self.session.active.times - expected_time))
                )
                self.assertEqual(self.session.current_index, expected)

    def test_elapsed_time_clamps_long_frame_delays(self):
        with patch("magnetic_torus.ui.playback.time.monotonic", return_value=1.0):
            self.controller.toggle_play()
        with patch("magnetic_torus.ui.playback.time.monotonic", return_value=11.0):
            self.controller._play_tick()
        self.assertAlmostEqual(self.controller._play_position, 0.2)
        self.assertEqual(self.session.current_index, 1)

    def test_nonloop_overshoot_renders_last_frame_before_stopping(self):
        self.controller.loop.setChecked(False)
        self.controller.speed.setCurrentText("1000×")
        with patch("magnetic_torus.ui.playback.time.monotonic", return_value=0.0):
            self.controller.toggle_play()
        with patch("magnetic_torus.ui.playback.time.monotonic", return_value=0.1):
            self.controller._play_tick()
        self.assertEqual(self.session.current_index, len(self.session.active.times) - 1)
        self.assertFalse(self.controller.play_timer.isActive())
        self.assertEqual(self.frames[-1], self.session.current_index)

    def test_loop_wraps_to_first_time_and_pause_resume_does_not_accumulate_delay(self):
        self.controller.seek_time(2)
        self.controller.loop.setChecked(True)
        with patch("magnetic_torus.ui.playback.time.monotonic", return_value=1.0):
            self.controller.toggle_play()
        with patch("magnetic_torus.ui.playback.time.monotonic", return_value=1.1):
            self.controller._play_tick()
        self.assertEqual(self.session.current_index, 0)
        self.assertEqual(self.controller._play_position, 0.0)
        self.controller.stop_play()
        with patch("magnetic_torus.ui.playback.time.monotonic", return_value=100.0):
            self.controller.toggle_play()
        with patch("magnetic_torus.ui.playback.time.monotonic", return_value=100.1):
            self.controller._play_tick()
        self.assertAlmostEqual(self.controller._play_position, 0.1)

    def test_scrub_step_and_signal_update_keep_control_and_physical_time_consistent(
        self,
    ):
        self.controller.scrub(1000)
        self.assertEqual(self.session.current_index, 4)
        self.controller.step(-100)
        self.assertEqual(self.session.current_index, 0)
        self.controller.seek_time(0.5)
        self.assertEqual(self.session.current_index, 2)
        before = len(self.frames)
        self.controller.update_position()
        self.assertEqual(len(self.frames), before)
        self.assertEqual(self.controller.timeline.value(), 275)
        self.assertIn("0.55", self.controller.time_label.text())

    def test_compare_playback_never_selects_a_frame_after_overlap_end(self):
        self.session.set_comparison(accepted_run("other", (0.0, 0.3, 0.6)))
        self.session.view = "Compare"
        self.controller.scrub(1000)
        self.assertEqual(self.session.current_index, 2)
        self.assertLessEqual(self.session.current_time, 0.6)
        self.controller.step(100)
        self.assertLessEqual(self.session.current_time, 0.6)

    def test_timeline_clamp_handles_overlap_bounds_without_requesting_a_render(self):
        for times, current, expected in (
            ((0.3, 0.8, 1.6), 0, 2),
            ((0.0, 0.3, 0.6), 4, 2),
            ((0.55,), 4, 2),
            ((0.3, 0.8, 1.6), 3, 3),
        ):
            with self.subTest(times=times, current=current):
                self.session.set_comparison(accepted_run("clamp", times))
                self.session.view = "Compare"
                self.session.current_index = current
                self.controller._play_position = -99.0
                self.frames.clear()
                self.controller.clamp_to_timeline()
                self.assertEqual(self.session.current_index, expected)
                self.assertEqual(
                    self.controller._play_position, self.session.active.times[expected]
                )
                self.assertEqual(self.frames, [])
                self.controller.update_position()
                self.assertIn(
                    f"{self.session.current_time:.4g}",
                    self.controller.time_label.text(),
                )
                self.assertEqual(self.frames, [])

    def test_timeline_clamp_is_safe_without_a_valid_overlap_or_active_result(self):
        self.session.activate(accepted_run("coarse", (0.0, 2.0)), "coarse.h5")
        self.session.set_comparison(accepted_run("between", (0.5, 1.5)))
        self.session.view = "Compare"
        self.controller._play_position = 99.0
        self.controller.clamp_to_timeline()
        self.assertEqual(self.session.current_index, 0)
        self.assertEqual(self.controller._play_position, 99.0)
        self.session.active = None
        self.controller.clamp_to_timeline()
        self.assertEqual(self.frames, [])

    def test_comparison_indices_require_accepted_snapshots_from_both_runs(self):
        self.session.activate(accepted_run("middle", (0.4, 0.5, 0.6)), "middle.h5")
        active, other = self.controller.comparison_indices(
            accepted_run("valid", (0.39, 0.59, 1.0))
        )
        np.testing.assert_array_equal(active, [0, 1, 2])
        np.testing.assert_array_equal(other, [1])
        active, other = self.controller.comparison_indices(
            accepted_run("endpoint", (0.6, 0.8))
        )
        np.testing.assert_array_equal(active, [2])
        np.testing.assert_array_equal(other, [0])
        for times, message in (
            ((0.3, 0.7), "no accepted comparison snapshot"),
            ((0.41, 0.49), "no accepted active snapshot"),
            ((0.7, 0.8), "no overlapping physical timeline"),
        ):
            with self.subTest(times=times), self.assertRaisesRegex(ValueError, message):
                self.controller.comparison_indices(accepted_run("invalid", times))
        self.session.active = None
        with self.assertRaisesRegex(ValueError, "Load an active run"):
            self.controller.comparison_indices(accepted_run("no-active"))

    def test_particle_and_thermal_playback_pause_each_other(self):
        self.session.trajectories = {"snapshot_index": 2}
        self.controller.toggle_play()
        self.assertTrue(self.controller.play_timer.isActive())
        self.controller.toggle_particle_replay()
        self.assertFalse(self.controller.play_timer.isActive())
        self.assertTrue(self.controller.particle_timer.isActive())
        self.assertEqual(self.session.current_index, 2)
        self.controller._particle_tick()
        self.scene.draw_particles.assert_called_once_with(0.008)
        self.controller.toggle_play()
        self.assertFalse(self.controller.particle_timer.isActive())
        self.assertTrue(self.controller.play_timer.isActive())

    def test_comparison_scrub_and_loop_share_the_overlapping_start(self):
        self.session.set_comparison(accepted_run("delayed", (0.3, 0.8, 1.6)))
        self.session.view = "Compare"
        self.controller.scrub(0)
        self.assertEqual(self.session.current_index, 2)
        self.controller.update_position()
        self.assertEqual(
            self.controller.timeline.value(), int(1000 * (0.55 - 0.3) / (1.6 - 0.3))
        )
        self.controller.seek_time(1.0)
        self.controller.speed.setCurrentText("1000×")
        with patch("magnetic_torus.ui.playback.time.monotonic", return_value=0.0):
            self.controller.toggle_play()
        with patch("magnetic_torus.ui.playback.time.monotonic", return_value=0.1):
            self.controller._play_tick()
        self.assertEqual(self.session.current_index, 2)
        self.assertEqual(self.controller._play_position, 0.3)

    def test_unavailable_results_disable_controls_and_ignore_seek(self):
        self.controller.set_available(False, True, False)
        self.assertFalse(self.controller.timeline.isEnabled())
        self.assertFalse(self.controller.replay.isEnabled())
        self.session.active = None
        self.controller.seek_time(1.0)
        self.controller.toggle_play()
        self.assertFalse(self.controller.play_timer.isActive())


if __name__ == "__main__":
    unittest.main()
