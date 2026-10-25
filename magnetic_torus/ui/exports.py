"""Explicit image/data exports and cancellable annotated movie encoding."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from PyQt6 import QtCore, QtWidgets
from ..models import ScenarioConfig


class ExportController(QtCore.QObject):
    notified = QtCore.pyqtSignal(str, bool, bool)
    progress = QtCore.pyqtSignal(float, str)
    busy_changed = QtCore.pyqtSignal(bool)

    def __init__(
        self,
        session,
        scene,
        chart_panel,
        editor,
        playback,
        render_frame,
        job_is_busy,
        parent=None,
    ):
        super().__init__(parent)
        self.session = session
        self.scene = scene
        self.chart_panel = chart_panel
        self.editor = editor
        self.playback = playback
        self.render_frame = render_frame
        self.job_is_busy = job_is_busy
        self.writer = None
        self._starting = False
        self._movie_generation = 0
        self._movie_frames = []
        self._movie_position = 0
        self._movie_restore = 0
        self.movie_timer = QtCore.QTimer(self)
        self.movie_timer.setInterval(0)
        self.movie_timer.timeout.connect(self._movie_tick)

    @property
    def busy(self):
        return self._starting or self.writer is not None

    def _notify(self, message, error=False, sticky=False):
        self.notified.emit(message, error, sticky)

    def export_image(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self.parent(), "Export scene", "torus-scene.png", "PNG image (*.png)"
        )
        if path:
            self.scene.flush_render()
            self.scene.plotter.screenshot(path, transparent_background=False)
            self._notify(f"Exported scene: {path}")

    def export_charts(self):
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self.parent(), "Export chart SVGs"
        )
        if directory:
            for index, card in enumerate(self.chart_panel.cards, 1):
                card.figure.savefig(
                    Path(directory) / f"torus-chart-{index}.svg", bbox_inches="tight"
                )
            self._notify(f"Exported three charts to {directory}")

    def export_data(self):
        if self.session.active_path is None or self.session.active is None:
            self._notify("Run a simulation before exporting scientific results.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self.parent(),
            "Export result",
            f"{self.session.active.run_id}.h5",
            "HDF5 scientific data (*.h5)",
        )
        if path:
            import shutil

            shutil.copy2(self.session.active_path, path)
            self._notify(f"Exported SI-valued fields and configuration to {path}")

    def export_config(self):
        try:
            config = self.editor.read_config()
        except ValueError as error:
            self._notify(str(error), error=True)
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self.parent(),
            "Export scenario",
            "torus-scenario.json",
            "JSON scenario (*.json)",
        )
        if path:
            Path(path).write_text(
                json.dumps(config.to_dict(), indent=2), encoding="utf-8"
            )

    def import_config(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.parent(), "Import scenario", "", "JSON scenario (*.json)"
        )
        if path:
            try:
                self.editor.set_config(
                    ScenarioConfig.from_dict(
                        json.loads(Path(path).read_text(encoding="utf-8"))
                    )
                )
            except (OSError, ValueError, TypeError) as error:
                self._notify(f"Cannot import scenario: {error}", error=True)

    def export_movie(self):
        if self.session.active is None or self.job_is_busy():
            self._notify(
                "Select a saved result and wait for the current job before exporting an animation."
            )
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self.parent(),
            "Export scene animation",
            "torus-animation.mp4",
            "MP4 animation (*.mp4)",
        )
        if path:
            self.begin_movie_export(path)

    def begin_movie_export(self, path):
        if self.session.active is None or self.busy or self.job_is_busy():
            return
        self.playback.stop_all()
        self._movie_restore = self.session.current_index
        self._movie_frames = (
            np.linspace(
                0,
                len(self.session.active.times) - 1,
                max(60, min(300, len(self.session.active.times) * 3)),
            )
            .astype(int)
            .tolist()
        )
        self._movie_position = 0
        self._starting = True
        self._movie_generation += 1
        generation = self._movie_generation
        self.busy_changed.emit(True)
        try:
            import imageio_ffmpeg

            QtWidgets.QApplication.processEvents()
            # Cancellation or history selection can run while Qt applies the
            # export layout. Never resume startup for that superseded request.
            if not self._starting or generation != self._movie_generation:
                return
            self.scene.flush_render()
            image = self.scene.plotter.screenshot(return_img=True)
            height, width = image.shape[:2]
            # Even dimensions are required by the H.264 YUV420 format.
            self._movie_size = (width - width % 2, height - height % 2)
            self.writer = imageio_ffmpeg.write_frames(
                path,
                self._movie_size,
                fps=30,
                codec="libx264",
                macro_block_size=2,
                pix_fmt_in="rgb24",
                pix_fmt_out="yuv420p",
                ffmpeg_log_level="error",
            )
            self.writer.send(None)
            self._starting = False
            self._movie_path = path
            self.movie_timer.start()
        except Exception as error:
            if generation == self._movie_generation:
                self.finish_movie(cancelled=True)
                self._notify(f"Animation export failed: {error}", error=True)

    def _movie_tick(self):
        if self.writer is None:
            return
        if self._movie_position >= len(self._movie_frames):
            self.finish_movie()
            return
        try:
            self.session.current_index = self._movie_frames[self._movie_position]
            self.render_frame()
            self.scene.flush_render()
            image = self.scene.plotter.screenshot(return_img=True)
            width, height = self._movie_size
            # Dock changes and window resizing can change the render size during export.
            # The encoder requires exactly the dimensions supplied at initialization.
            image = image[:, :, :3]
            if image.shape[:2] != (height, width):
                from PIL import Image

                image = np.asarray(
                    Image.fromarray(image).resize(
                        (width, height), Image.Resampling.LANCZOS
                    )
                )
            self.writer.send(np.ascontiguousarray(image))
            self._movie_position += 1
            self.progress.emit(
                self._movie_position / len(self._movie_frames),
                "Exporting annotated scene animation…",
            )
        except Exception as error:
            self.finish_movie(cancelled=True)
            self._notify(f"Animation export failed: {error}", error=True)

    def finish_movie(self, cancelled=False):
        self.movie_timer.stop()
        self._starting = False
        self._movie_generation += 1
        writer = self.writer
        self.writer = None
        close_error = None
        if writer is not None:
            try:
                writer.close()
            except Exception as error:
                close_error = error
        count = len(self.session.active.times) if self.session.active is not None else 0
        self.session.current_index = min(max(0, self._movie_restore), max(0, count - 1))
        self.render_frame()
        self.scene.flush_render()
        self.busy_changed.emit(False)
        if close_error is not None:
            self._notify(f"Animation export failed: {close_error}", error=True)
            return
        self._notify(
            "Animation export cancelled; any encoded frames remain in the selected file."
            if cancelled
            else f"Exported animation: {self._movie_path}"
        )
