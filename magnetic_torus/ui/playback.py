"""Independent thermal/MHD and particle-flight playback clocks."""

from __future__ import annotations

import time
import numpy as np
from PyQt6 import QtCore, QtWidgets
from .session import WorkspaceSession


class PlaybackController(QtWidgets.QFrame):
    render_requested = QtCore.pyqtSignal()

    def __init__(self, session: WorkspaceSession, scene, parent=None):
        super().__init__(parent)
        self.session = session
        self.scene = scene
        self.setObjectName("timeline")
        self._play_position = 0.0
        self._last_tick = time.monotonic()
        self._particle_fraction = 0.0
        self.play_timer = QtCore.QTimer(self)
        self.play_timer.setInterval(33)
        self.play_timer.timeout.connect(self._play_tick)
        self.particle_timer = QtCore.QTimer(self)
        self.particle_timer.setInterval(33)
        self.particle_timer.timeout.connect(self._particle_tick)
        time_row = QtWidgets.QHBoxLayout(self)
        time_row.setContentsMargins(8, 5, 8, 5)
        self.back = QtWidgets.QToolButton()
        self.back.setText("⏮")
        self.play = QtWidgets.QToolButton()
        self.play.setText("▶ Play")
        self.forward = QtWidgets.QToolButton()
        self.forward.setText("⏭")
        for button in (self.back, self.play, self.forward):
            time_row.addWidget(button)
        self.timeline = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.timeline.setRange(0, 1000)
        time_row.addWidget(self.timeline, 1)
        self.time_label = QtWidgets.QLabel("t = — s")
        self.time_label.setMinimumWidth(110)
        time_row.addWidget(self.time_label)
        self.speed = QtWidgets.QComboBox()
        self.speed.addItems(
            ["0.25×", "0.5×", "1×", "2×", "5×", "10×", "50×", "100×", "500×", "1000×"]
        )
        self.speed.setCurrentText("1×")
        self.speed.setToolTip("Physical seconds per second of playback")
        time_row.addWidget(self.speed)
        self.loop = QtWidgets.QCheckBox("Loop")
        self.loop.setChecked(True)
        time_row.addWidget(self.loop)
        self.replay = QtWidgets.QPushButton("Particle replay")
        time_row.addWidget(self.replay)
        self.back.clicked.connect(lambda: self.step(-1))
        self.forward.clicked.connect(lambda: self.step(1))
        self.play.clicked.connect(self.toggle_play)
        self.timeline.valueChanged.connect(self.scrub)
        self.timeline.sliderPressed.connect(self.stop_play)
        self.replay.clicked.connect(self.toggle_particle_replay)

    def set_available(self, has_result, previewing, has_trajectories):
        for control in (self.play, self.back, self.forward, self.timeline):
            control.setEnabled(has_result and not previewing)
        self.replay.setEnabled(has_trajectories and not previewing)

    def _timeline_start(self):
        result = self.session.active
        if result is None:
            return 0.0
        start = float(result.times[0])
        if self.session.view == "Compare" and self.session.comparison is not None:
            start = max(start, float(self.session.comparison.times[0]))
        return start

    def _timeline_indices(self):
        result = self.session.active
        if result is None:
            return np.empty(0, dtype=int)
        return np.flatnonzero(
            (result.times >= self._timeline_start())
            & (result.times <= self.session.timeline_end())
        )

    def clamp_to_timeline(self):
        """Align state and playback time without requesting a recursive render."""
        indices = self._timeline_indices()
        if not len(indices):
            return
        self.session.current_index = int(
            indices[np.argmin(np.abs(indices - self.session.current_index))]
        )
        self._play_position = float(
            self.session.active.times[self.session.current_index]
        )

    def stop_all(self):
        self.stop_play()
        self.particle_timer.stop()
        self.replay.setText("Particle replay")

    def update_position(self):
        result = self.session.active
        if result is None:
            return
        current_time = float(result.times[self.session.current_index])
        self.time_label.setText(f"t = {current_time:.4g} s")
        blocker = QtCore.QSignalBlocker(self.timeline)
        start = self._timeline_start()
        span = self.session.timeline_end() - start
        self.timeline.setValue(
            int(1000 * (current_time - start) / span) if span > 0 else 0
        )
        del blocker

    def scrub(self, value):
        if self.session.active is None:
            return
        end = self.session.timeline_end()
        start = self._timeline_start()
        target = start + value / 1000 * (end - start)
        self.seek_time(target)

    def seek_time(self, value):
        if self.session.active is None or self.session.previewing:
            return
        indices = self._timeline_indices()
        if not len(indices):
            return
        self.session.current_index = int(
            indices[np.argmin(np.abs(self.session.active.times[indices] - value))]
        )
        self._play_position = float(
            self.session.active.times[self.session.current_index]
        )
        self.render_requested.emit()

    def step(self, delta):
        self.stop_play()
        if self.session.active is not None:
            indices = self._timeline_indices()
            if not len(indices):
                return
            position = int(np.argmin(np.abs(indices - self.session.current_index)))
            self.session.current_index = int(
                indices[np.clip(position + delta, 0, len(indices) - 1)]
            )
            self._play_position = float(
                self.session.active.times[self.session.current_index]
            )
            self.render_requested.emit()

    def toggle_play(self):
        if self.session.active is None or self.session.previewing:
            return
        if self.play_timer.isActive():
            self.stop_play()
            return
        self.particle_timer.stop()
        self.replay.setText("Particle replay")
        self._play_position = float(
            self.session.active.times[self.session.current_index]
        )
        self._last_tick = time.monotonic()
        self.play.setText("Ⅱ Pause")
        self.play_timer.start()

    def stop_play(self):
        self.play_timer.stop()
        if hasattr(self, "play"):
            self.play.setText("▶ Play")

    def _play_tick(self):
        if self.session.active is None:
            self.stop_play()
            return
        now = time.monotonic()
        speed = float(self.speed.currentText().replace("×", ""))
        self._play_position += min(now - self._last_tick, 0.2) * speed
        self._last_tick = now
        if self._play_position > self.session.timeline_end():
            if self.loop.isChecked():
                self._play_position = self._timeline_start()
            else:
                self._play_position = self.session.timeline_end()
                self.stop_play()
        indices = self._timeline_indices()
        if not len(indices):
            self.stop_play()
            return
        index = int(
            indices[
                np.argmin(
                    np.abs(self.session.active.times[indices] - self._play_position)
                )
            ]
        )
        if index != self.session.current_index:
            self.session.current_index = index
            self.render_requested.emit()

    def toggle_particle_replay(self):
        if self.session.trajectories is None or self.session.active is None:
            return
        if self.particle_timer.isActive():
            self.particle_timer.stop()
            self.replay.setText("Particle replay")
        else:
            self.stop_play()
            self.seek_time(
                float(
                    self.session.active.times[
                        int(self.session.trajectories["snapshot_index"])
                    ]
                )
            )
            self._particle_fraction = 0.0
            self.particle_timer.start()
            self.replay.setText("Pause particles")

    def _particle_tick(self):
        self._particle_fraction += 0.008
        if self._particle_fraction > 1:
            self._particle_fraction = 0.0
        self.scene.draw_particles(self._particle_fraction)
