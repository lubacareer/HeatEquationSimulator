"""Export roundtrips and failure cleanup using real files and the real encoder."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import xml.etree.ElementTree as ET

import imageio_ffmpeg
import numpy as np
from PyQt6 import QtGui, QtWidgets
from matplotlib.figure import Figure

from magnetic_torus.models import ScenarioConfig
from magnetic_torus.store import RunStore
from magnetic_torus.ui.editor import ScenarioEditor
from magnetic_torus.ui.exports import ExportController
from magnetic_torus.ui.session import WorkspaceSession
from workspace_fixtures import accepted_run

SUITE_CAPABILITIES = ("desktop",)


class FixturePlotter:
    shape = (11, 13)

    def screenshot(self, path=None, **kwargs):
        height, width = self.shape
        image = np.zeros((height, width, 3), dtype=np.uint8)
        image[:, :, 0] = np.arange(width) * 7
        image[:, :, 1] = np.arange(height)[:, None] * 11
        image[:, :, 2] = 128
        if path:
            qimage = QtGui.QImage(
                image.data,
                width,
                height,
                image.strides[0],
                QtGui.QImage.Format.Format_RGB888,
            )
            if not qimage.save(str(path)):
                raise OSError("Cannot write PNG")
        return image


class ExportFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="torus-exports-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.store = RunStore(self.root / "source-store")
        self.result = accepted_run()
        self.source = self.store.save_run(self.result)
        self.session = WorkspaceSession()
        self.session.activate(self.result, self.source)
        self.session.current_index = 2
        self.scene = SimpleNamespace(plotter=FixturePlotter(), flush_render=Mock())
        self.editor = ScenarioEditor()
        self.editor.set_config(replace(self.result.config, name="Thermal Δ scenario"))
        self.addCleanup(self.editor.close)
        self.cards = []
        for _ in range(3):
            figure = Figure()
            ax = figure.add_subplot(111)
            ax.plot([0, 1], [1800, 1810], label="Numerical")
            ax.set(xlabel="Time [s]", ylabel="Temperature [K]")
            ax.legend()
            self.cards.append(SimpleNamespace(figure=figure))
        self.playback = SimpleNamespace(stop_all=Mock())
        self.frames = []
        self.exports = ExportController(
            self.session,
            self.scene,
            SimpleNamespace(cards=self.cards),
            self.editor,
            self.playback,
            lambda: self.frames.append(self.session.current_index),
            lambda: False,
        )
        self.notices = []
        self.exports.notified.connect(
            lambda message, error, sticky: self.notices.append((message, error))
        )
        self.addCleanup(self.exports.movie_timer.stop)

    def save_dialog(self, path):
        return patch(
            "magnetic_torus.ui.exports.QtWidgets.QFileDialog.getSaveFileName",
            return_value=(str(path), ""),
        )

    def test_png_pixels_and_svg_structure_include_real_units(self):
        path = self.root / "scene Δ with spaces.png"
        with self.save_dialog(path):
            self.exports.export_image()
        image = QtGui.QImage(str(path))
        self.assertFalse(image.isNull())
        self.assertEqual((image.width(), image.height()), (13, 11))
        self.assertEqual(image.pixelColor(5, 4).getRgb()[:3], (35, 44, 128))
        self.scene.flush_render.assert_called_once()
        directory = self.root / "SVG Δ with spaces"
        directory.mkdir()
        with patch(
            "magnetic_torus.ui.exports.QtWidgets.QFileDialog.getExistingDirectory",
            return_value=str(directory),
        ):
            self.exports.export_charts()
        for path in directory.glob("*.svg"):
            self.assertTrue(ET.parse(path).getroot().tag.endswith("svg"))
            text = path.read_text(encoding="utf-8")
            for label in ("Time [s]", "Temperature [K]", "Numerical"):
                self.assertIn(label, text)
        self.assertEqual(len(list(directory.glob("*.svg"))), 3)

    def test_config_and_hdf_exports_roundtrip_without_mutating_source(self):
        config_path = self.root / "scenario Δ with spaces.json"
        before = self.result.fields["temperature"].copy()
        with self.save_dialog(config_path):
            self.exports.export_config()
        expected_config = self.editor.read_config()
        self.assertEqual(
            ScenarioConfig.from_dict(
                json.loads(config_path.read_text(encoding="utf-8"))
            ),
            expected_config,
        )
        self.editor.set_config(replace(expected_config, duration=99.0))
        with patch(
            "magnetic_torus.ui.exports.QtWidgets.QFileDialog.getOpenFileName",
            return_value=(str(config_path), ""),
        ):
            self.exports.import_config()
        self.assertEqual(self.editor.read_config(), expected_config)
        output = self.root / "results Δ with spaces.h5"
        with self.save_dialog(output):
            self.exports.export_data()
        loaded = self.store.load_run(output)
        self.assertEqual(loaded.config, self.result.config)
        self.assertEqual(loaded.run_id, self.result.run_id)
        for key in self.result.fields:
            np.testing.assert_array_equal(loaded.fields[key], self.result.fields[key])
        np.testing.assert_array_equal(before, self.result.fields["temperature"])
        self.assertFalse(self.result.fields["temperature"].flags.writeable)

    def test_cancelled_dialogs_do_not_write_or_start_encoding(self):
        files = set(self.root.rglob("*"))
        with self.save_dialog(""):
            self.exports.export_image()
            self.exports.export_data()
            self.exports.export_config()
            self.exports.export_movie()
        with patch(
            "magnetic_torus.ui.exports.QtWidgets.QFileDialog.getExistingDirectory",
            return_value="",
        ):
            self.exports.export_charts()
        self.assertEqual(set(self.root.rglob("*")), files)
        self.assertFalse(self.exports.busy)

    def test_filesystem_failures_are_reported_and_preserve_numerical_result(self):
        output = self.root / "out.png"
        with (
            self.save_dialog(output),
            patch.object(
                self.scene.plotter, "screenshot", side_effect=PermissionError("denied")
            ),
        ):
            self.exports.export_image()
        with self.save_dialog(self.source):
            self.exports.export_data()
        with self.save_dialog(self.root / "no-parent" / "scenario.json"):
            self.exports.export_config()
        with (
            patch(
                "magnetic_torus.ui.exports.QtWidgets.QFileDialog.getExistingDirectory",
                return_value=str(self.root),
            ),
            patch.object(
                self.cards[0].figure, "savefig", side_effect=PermissionError("denied")
            ),
        ):
            self.exports.export_charts()
        self.assertEqual(len(self.notices), 4)
        self.assertTrue(all(error for _, error in self.notices))
        self.assertIs(self.session.active, self.result)
        self.assertFalse(self.exports.busy)
        self.assertTrue(self.source.is_file())

    def test_malformed_config_does_not_replace_editable_values(self):
        expected = self.editor.read_config()
        path = self.root / "bad.json"
        path.write_text('{"duration": "bad"}', encoding="utf-8")
        with patch(
            "magnetic_torus.ui.exports.QtWidgets.QFileDialog.getOpenFileName",
            return_value=(str(path), ""),
        ):
            self.exports.import_config()
        self.assertEqual(self.editor.read_config(), expected)
        self.assertTrue(self.notices[-1][1])

    def test_real_mp4_has_fixed_even_frames_through_resize_and_restores_time(self):
        path = self.root / "odd-sized resized.mp4"
        self.exports.begin_movie_export(str(path))
        self.exports.movie_timer.stop()
        expected_frames = len(self.exports._movie_frames)
        for index in range(expected_frames + 1):
            self.scene.plotter.shape = ((11, 13), (15, 21), (10, 8))[index % 3]
            self.exports._movie_tick()
        self.assertFalse(self.exports.busy)
        self.assertEqual(self.session.current_index, 2)
        self.assertEqual(self.frames[-1], 2)
        self.playback.stop_all.assert_called_once()
        reader = imageio_ffmpeg.read_frames(str(path))
        metadata = next(reader)
        self.assertEqual(metadata["size"], (12, 10))
        decoded = list(reader)
        self.assertEqual(len(decoded), expected_frames)
        self.assertTrue(all(len(frame) == 12 * 10 * 3 for frame in decoded))

    def test_cancelled_real_mp4_closes_encoder_and_retains_decodable_partial_frames(
        self,
    ):
        path = self.root / "cancelled.mp4"
        busy = []
        self.exports.busy_changed.connect(busy.append)
        self.exports.begin_movie_export(str(path))
        self.exports.movie_timer.stop()
        for _ in range(5):
            self.exports._movie_tick()
        self.exports.finish_movie(cancelled=True)
        self.assertFalse(self.exports.busy)
        self.assertFalse(self.exports.movie_timer.isActive())
        self.assertIsNone(self.exports.writer)
        self.assertEqual(self.session.current_index, 2)
        self.assertEqual(busy, [True, False])
        reader = imageio_ffmpeg.read_frames(str(path))
        self.assertEqual(next(reader)["size"], (12, 10))
        self.assertEqual(sum(1 for _ in reader), 5)
        self.assertIn("cancelled", self.notices[-1][0])

    def test_encoder_startup_and_frame_failures_release_state_and_restore_time(self):
        with patch(
            "imageio_ffmpeg.write_frames", side_effect=PermissionError("denied")
        ):
            self.exports.begin_movie_export(str(self.root / "startup.mp4"))
        self.assertFalse(self.exports.busy)
        self.assertFalse(self.exports.movie_timer.isActive())
        self.assertEqual(self.session.current_index, 2)
        self.assertTrue(self.notices[-1][1])
        writer = Mock()
        writer.send.side_effect = [None, OSError("encoder pipe closed")]
        with patch("imageio_ffmpeg.write_frames", return_value=writer):
            self.exports.begin_movie_export(str(self.root / "frame-failure.mp4"))
            self.exports.movie_timer.stop()
            self.exports._movie_tick()
        writer.close.assert_called_once()
        self.assertFalse(self.exports.busy)
        self.assertFalse(self.exports.movie_timer.isActive())
        self.assertEqual(self.session.current_index, 2)
        self.assertIn("encoder pipe closed", self.notices[-1][0])
        self.assertTrue(self.notices[-1][1])


if __name__ == "__main__":
    unittest.main()
