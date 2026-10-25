"""Visualization-first Qt workspace. Physics and training run in isolated processes."""
from __future__ import annotations

from dataclasses import fields, replace
from datetime import datetime
import json
import math
from itertools import cycle
import os
from pathlib import Path
import sys
import time
from typing import Any

os.environ.setdefault("QT_API", "pyqt6")

import numpy as np
import PyQt6.QtCore as QtCore
import PyQt6.QtGui as QtGui
import PyQt6.QtWidgets as QtWidgets
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt import NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from .jobs import JobController
from .models import DoseResult, PINNResult, RunResult, ScenarioConfig, preset
from .scene import QUANTITIES, SceneWidget, conductor_mask
from .store import RunStore, json_default


STYLE = """
QMainWindow, QWidget#workspace { background: #edf2f5; color: #203547; }
QWidget { font-family: 'Segoe UI'; font-size: 12px; color: #203547; }
QFrame#header { background: #ffffff; border-bottom: 1px solid #d5e0e7; }
QLabel#brand { font-size: 20px; font-weight: 650; color: #163c50; }
QLabel#subtitle { color: #647e8d; font-size: 11px; }
QLabel#badge { color: #317b80; background: #e3f1ee; padding: 5px 10px; border-radius: 8px; }
QLabel#message { padding: 6px 10px; background: #fff7e5; color: #806026; border-radius: 5px; }
QPushButton, QToolButton { padding: 6px 10px; border: 1px solid #d0dce4; border-radius: 5px; background: white; }
QPushButton:hover, QToolButton:hover { background: #e9f3f5; border-color: #82acb7; }
QPushButton:disabled, QToolButton:disabled { color: #a1afb7; background: #eef1f3; }
QPushButton#run { color: white; background: #187e86; border-color: #187e86; font-weight: 600; }
QPushButton#run:hover { background: #126670; }
QToolButton:checked { background: #d5eaeb; color: #14656d; border-color: #8bb9bb; }
QLineEdit, QComboBox { background: white; border: 1px solid #cedae3; padding: 5px 6px; border-radius: 4px; min-height: 17px; }
QLineEdit:focus, QComboBox:focus { border: 1px solid #258994; }
QScrollArea { border: none; background: #f7f9fb; }
QFrame#sidebar { background: #f7f9fb; border-right: 1px solid #d6e0e7; }
QFrame#section { border-bottom: 1px solid #dce5eb; }
QToolButton#sectionTitle { border: none; background: transparent; text-align: left; font-weight: 600; padding: 9px 0; }
QFrame#chart { background: white; border: 1px solid #d7e2e8; border-radius: 6px; }
QFrame#timeline { background: white; border: 1px solid #d7e2e8; border-radius: 5px; }
QProgressBar { border: none; background: #dfe8ec; border-radius: 3px; height: 6px; max-height: 6px; }
QProgressBar::chunk { background: #2a9c9c; border-radius: 3px; }
QSplitter::handle { background: #dce5ea; }
QPlainTextEdit { background: #142332; color: #d2e5ef; border: none; font-family: Consolas; font-size: 11px; }
QStatusBar { color: #627b8b; }
"""


class ChartCard(QtWidgets.QFrame):
    time_clicked = QtCore.pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("chart")
        self.layout_ = QtWidgets.QVBoxLayout(self)
        self.layout_.setContentsMargins(3, 2, 3, 0)
        toolbar = QtWidgets.QHBoxLayout()
        self.title = QtWidgets.QLabel()
        self.title.setStyleSheet("font-weight: 600; padding: 3px 7px;")
        toolbar.addWidget(self.title, 1)
        detach = QtWidgets.QToolButton()
        detach.setText("↗")
        detach.setToolTip("Expand this chart")
        detach.clicked.connect(self.detach)
        toolbar.addWidget(detach)
        self.layout_.addLayout(toolbar)
        self.figure = Figure(figsize=(4.0, 2.4), dpi=100, facecolor="white")
        self.figure.subplots_adjust(left=.21, right=.96, bottom=.24, top=.96)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumHeight(105)
        self.layout_.addWidget(self.canvas)
        self.ax = self.figure.add_subplot(111)
        self.time_axis = False
        self.cursor: Line2D | None = None
        self.marker: Line2D | None = None
        self._dialog = None
        self._background = None
        self._saved_limits = {}
        self._default_limits = None
        self.canvas.mpl_connect("draw_event", self._cache_background)
        self.canvas.mpl_connect("button_release_event", self._remember_view)
        self.canvas.mpl_connect("button_press_event", self._click)

    def reset(self, title: str, time_axis=False):
        self._background = None
        self.title.setText(title)
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        self.ax.tick_params(labelsize=8, colors="#526a79")
        self.ax.grid(True, color="#e7edf1", linewidth=.6)
        for spine in self.ax.spines.values():
            spine.set_color("#d2dfe6")
        self.time_axis = time_axis
        self.cursor = None
        self.marker = None
        return self.ax

    def empty(self, title, message):
        ax = self.reset(title)
        ax.text(.5, .5, message, ha="center", va="center", transform=ax.transAxes,
                color="#8596a1", fontsize=9, wrap=True)
        ax.set_axis_off()
        self.canvas.draw_idle()

    def finish(self, current_time: float = 0.0):
        if self.time_axis:
            self.cursor = self.ax.axvline(current_time, color="#187e86", linewidth=1, alpha=.8)
            self.cursor.set_animated(True)
        if self.marker is not None:
            self.marker.set_animated(True)
        self.ax.xaxis.label.set_fontsize(8)
        self.ax.yaxis.label.set_fontsize(8)
        self._default_limits = (self.ax.get_xlim(), self.ax.get_ylim())
        limits = self._saved_limits.get(self.title.text())
        if limits is not None:
            self.ax.set_xlim(limits[0])
            self.ax.set_ylim(limits[1])
        self.canvas.draw_idle()

    def _remember_view(self, event):
        if event.inaxes != self.ax or self._default_limits is None:
            return
        limits = (self.ax.get_xlim(), self.ax.get_ylim())
        if limits != self._default_limits:
            self._saved_limits[self.title.text()] = limits
        else:
            self._saved_limits.pop(self.title.text(), None)

    def _cache_background(self, event):
        self._background = self.canvas.copy_from_bbox(self.ax.bbox)
        self.update_artists()

    def update_artists(self):
        if self._background is None:
            self.canvas.draw_idle()
            return
        self.canvas.restore_region(self._background)
        for artist in (self.cursor, self.marker):
            if artist is not None:
                self.ax.draw_artist(artist)
        # Qt's Matplotlib blit calls synchronous repaint(). Queue a paint of the
        # already-updated Agg buffer so all charts and VTK share one UI frame.
        self.canvas.update()

    def _click(self, event):
        toolbar = getattr(self.canvas, "toolbar", None)
        navigating = bool(toolbar is not None and toolbar.mode)
        if not navigating and event.button == 1 and self.time_axis and event.inaxes == self.ax and event.xdata is not None:
            self.time_clicked.emit(float(event.xdata))

    def detach(self):
        if self._dialog is not None:
            self._dialog.raise_()
            return
        dialog = QtWidgets.QDialog(self.window())
        dialog.setWindowTitle(self.title.text())
        dialog.resize(950, 620)
        layout = QtWidgets.QVBoxLayout(dialog)
        layout.addWidget(NavigationToolbar2QT(self.canvas, dialog))
        layout.addWidget(self.canvas)
        self._dialog = dialog

        def restore():
            layout.removeWidget(self.canvas)
            self.layout_.addWidget(self.canvas)
            self._dialog = None
            self.canvas.draw_idle()

        dialog.finished.connect(restore)
        dialog.show()


class TorusWindow(QtWidgets.QMainWindow):
    def __init__(self, data_dir=None):
        super().__init__()
        self.setWindowTitle("Magnetic Torus Simulator")
        self.setMinimumSize(980, 640)
        screen = QtWidgets.QApplication.primaryScreen().availableGeometry()
        self.resize(min(1580, screen.width() - 35), min(1020, screen.height() - 60))
        self.setStyleSheet(STYLE)
        self.store = RunStore(data_dir)
        self.jobs = JobController(self.store, self)
        self.active: RunResult | None = None
        self.active_path: Path | None = None
        self.comparison: RunResult | None = None
        self.pinn: PINNResult | None = None
        self._pinn_difference = None
        self.dose: DoseResult | None = None
        self.trajectories: dict[str, Any] | None = None
        self.current_index = 0
        self._selection_revision = 0
        self._job_revision = 0
        self.pinn_path: Path | None = None
        self.probe_index = 0
        self.view = "Thermal"
        self.previewing = True
        self._loading = False
        self._movie_writer = None
        self._movie_frames = []
        self._movie_position = 0
        self._movie_restore = 0
        self._play_position = 0.0
        self._last_tick = time.monotonic()
        self._draft_timer = QtCore.QTimer(self)
        self._draft_timer.setSingleShot(True)
        self._draft_timer.setInterval(160)
        self._draft_timer.timeout.connect(self._draft_changed)
        self.play_timer = QtCore.QTimer(self)
        self.play_timer.setInterval(33)
        self.play_timer.timeout.connect(self._play_tick)
        self.particle_timer = QtCore.QTimer(self)
        self.particle_timer.setInterval(33)
        self.particle_timer.timeout.connect(self._particle_tick)
        self.movie_timer = QtCore.QTimer(self)
        self.movie_timer.setInterval(0)
        self.movie_timer.timeout.connect(self._movie_tick)
        self._particle_fraction = 0.0
        self._profile_samples = None
        self._comparison_difference = None
        self._build()
        self._connect()
        self.set_config(preset("Thermal benchmark"))
        self.scene.preview(self._base_config, reset_camera=True)
        self.refresh_history()
        self._plot_empty()
        self._update_actions()
        self.statusBar().showMessage("Geometry preview · choose a benchmark or configure a full MHD experiment")

    def _build(self):
        workspace = QtWidgets.QWidget()
        workspace.setObjectName("workspace")
        self.setCentralWidget(workspace)
        layout = QtWidgets.QVBoxLayout(workspace)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        header = QtWidgets.QFrame()
        header.setObjectName("header")
        row = QtWidgets.QHBoxLayout(header)
        row.setContentsMargins(18, 11, 18, 11)
        brand = QtWidgets.QVBoxLayout()
        title = QtWidgets.QLabel("MAGNETIC TORUS")
        title.setObjectName("brand")
        subtitle = QtWidgets.QLabel("Thermal storage  /  magnetic dynamics  /  radiation transport")
        subtitle.setObjectName("subtitle")
        brand.addWidget(title)
        brand.addWidget(subtitle)
        row.addLayout(brand)
        row.addStretch()
        self.badge = QtWidgets.QLabel("GEOMETRY PREVIEW")
        self.badge.setObjectName("badge")
        row.addWidget(self.badge)
        self.sidebar_button = QtWidgets.QToolButton()
        self.sidebar_button.setText("Parameters")
        self.sidebar_button.setCheckable(True)
        self.sidebar_button.setChecked(True)
        row.addWidget(self.sidebar_button)
        export = QtWidgets.QToolButton()
        export.setText("Export ▾")
        export.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QtWidgets.QMenu(export)
        for label, callback in (("Scene image…", self.export_image), ("Charts (SVG)…", self.export_charts),
                                ("Scene animation (MP4)…", self.export_movie),
                                ("Scientific results (HDF5)…", self.export_data),
                                ("Scenario settings (JSON)…", self.export_config),
                                ("Import scenario settings…", self.import_config)):
            menu.addAction(label, callback)
        export.setMenu(menu)
        row.addWidget(export)
        layout.addWidget(header)
        self.body = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.sidebar = self._build_sidebar()
        self.body.addWidget(self.sidebar)
        self.body.setCollapsible(0, True)
        center = QtWidgets.QWidget()
        center_layout = QtWidgets.QVBoxLayout(center)
        center_layout.setContentsMargins(10, 8, 12, 8)
        center_layout.setSpacing(7)
        self.message = QtWidgets.QLabel("Preview is geometry only. Run a simulation to display computed fields.")
        self.message.setObjectName("message")
        self.message.setWordWrap(True)
        center_layout.addWidget(self.message)
        toolbar = QtWidgets.QHBoxLayout()
        self.view_group = QtWidgets.QButtonGroup(self)
        for name in ("Thermal", "Dynamo", "Shielding", "PINN", "Compare"):
            button = QtWidgets.QToolButton()
            button.setText(name)
            button.setCheckable(True)
            button.setChecked(name == "Thermal")
            button.clicked.connect(lambda checked, name=name: self.set_view(name))
            self.view_group.addButton(button)
            toolbar.addWidget(button)
        toolbar.addStretch()
        self.plots_button = QtWidgets.QToolButton()
        self.plots_button.setText("Plots")
        self.plots_button.setCheckable(True)
        self.plots_button.setChecked(True)
        self.plots_button.setToolTip("Hide plots to give the 3D scene more room")
        toolbar.addWidget(self.plots_button)
        self.camera = QtWidgets.QComboBox()
        self.camera.addItems(["Isometric", "Top", "Side"])
        toolbar.addWidget(self.camera)
        center_layout.addLayout(toolbar)
        quantity_row = QtWidgets.QHBoxLayout()
        quantity_row.addWidget(QtWidgets.QLabel("Display"))
        self.quantity = QtWidgets.QComboBox()
        self.quantity.addItems(QUANTITIES)
        self.quantity.setToolTip("Color the reservoir by a computed physical quantity")
        quantity_row.addWidget(self.quantity)
        self.source = QtWidgets.QComboBox()
        self.source.addItems(["Numerical", "PINN", "PINN error", "Run difference"])
        quantity_row.addWidget(self.source)
        quantity_row.addStretch()
        self.thermal_plot = QtWidgets.QComboBox()
        self.thermal_plot.addItems(["Probe temperature", "Temperature profile", "Liquid fraction history", "Thermal power history"])
        self.thermal_plot.setToolTip("Choose the third thermal chart")
        quantity_row.addWidget(self.thermal_plot)
        self.shield_plot = QtWidgets.QComboBox()
        self.shield_plot.addItems(["First-contact outcomes", "Transmission vs energy", "Secondary dose"])
        self.shield_plot.setVisible(False)
        quantity_row.addWidget(self.shield_plot)
        center_layout.addLayout(quantity_row)
        self.vertical = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.scenes = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.scene = SceneWidget()
        self.compare_scene = SceneWidget()
        self.scenes.addWidget(self.scene)
        self.scenes.addWidget(self.compare_scene)
        self.compare_scene.hide()
        self.vertical.addWidget(self.scenes)
        chart_panel = QtWidgets.QWidget()
        chart_layout = QtWidgets.QHBoxLayout(chart_panel)
        chart_layout.setContentsMargins(0, 0, 0, 0)
        chart_layout.setSpacing(7)
        self.charts = [ChartCard() for _ in range(3)]
        for chart in self.charts:
            chart_layout.addWidget(chart)
        self.vertical.addWidget(chart_panel)
        self.plots_button.toggled.connect(chart_panel.setVisible)
        self.vertical.setSizes([540, 245])
        center_layout.addWidget(self.vertical, 1)
        overlay = QtWidgets.QHBoxLayout()
        self.cutaway = QtWidgets.QCheckBox("Cutaway")
        self.cutaway.setChecked(True)
        self.phase = QtWidgets.QCheckBox("Phase boundary")
        self.flow = QtWidgets.QCheckBox("Flow vectors")
        self.tracers = QtWidgets.QCheckBox("Tracers")
        self.field_lines = QtWidgets.QCheckBox("Field lines")
        self.slice = QtWidgets.QCheckBox("Section")
        for control in (self.cutaway, self.phase, self.flow, self.tracers, self.field_lines, self.slice):
            overlay.addWidget(control)
        self.section_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.section_slider.setRange(-1000, 1000)
        self.section_slider.setMaximumWidth(110)
        self.section_slider.setToolTip("Move the x-normal section through the torus")
        overlay.addWidget(self.section_slider)
        overlay.addStretch()
        self.detail = QtWidgets.QComboBox()
        self.detail.addItems(["Low", "Medium", "High"])
        self.detail.setCurrentText("Medium")
        self.detail.setToolTip("Visual detail; does not change the computed solution")
        overlay.addWidget(self.detail)
        overlay.addWidget(QtWidgets.QLabel("Opacity"))
        self.opacity = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.opacity.setRange(10, 100)
        self.opacity.setValue(100)
        self.opacity.setMaximumWidth(85)
        overlay.addWidget(self.opacity)
        self.probe_label = QtWidgets.QLabel("Right-click a reservoir point to probe")
        self.probe_label.setObjectName("subtitle")
        overlay.addWidget(self.probe_label)
        center_layout.addLayout(overlay)
        timeline = QtWidgets.QFrame()
        timeline.setObjectName("timeline")
        time_row = QtWidgets.QHBoxLayout(timeline)
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
        self.speed.addItems(["0.25×", "0.5×", "1×", "2×", "5×", "10×", "50×", "100×", "500×", "1000×"])
        self.speed.setCurrentText("1×")
        self.speed.setToolTip("Physical seconds per second of playback")
        time_row.addWidget(self.speed)
        self.loop = QtWidgets.QCheckBox("Loop")
        self.loop.setChecked(True)
        time_row.addWidget(self.loop)
        self.replay = QtWidgets.QPushButton("Particle replay")
        time_row.addWidget(self.replay)
        center_layout.addWidget(timeline)
        self.job_widget = QtWidgets.QWidget()
        job_row = QtWidgets.QHBoxLayout(self.job_widget)
        job_row.setContentsMargins(0, 0, 0, 0)
        self.job_text = QtWidgets.QLabel("No background job")
        self.job_text.setObjectName("subtitle")
        job_row.addWidget(self.job_text, 1)
        self.cancel_button = QtWidgets.QPushButton("Cancel")
        job_row.addWidget(self.cancel_button)
        center_layout.addWidget(self.job_widget)
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 1000)
        self.progress_bar.setTextVisible(False)
        center_layout.addWidget(self.progress_bar)
        self.body.addWidget(center)
        self.body.setSizes([285, 1250])
        layout.addWidget(self.body, 1)
        self.logs = QtWidgets.QDockWidget("Scientific diagnostics", self)
        self.log_text = QtWidgets.QPlainTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumBlockCount(1500)
        self.logs.setWidget(self.log_text)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, self.logs)
        self.logs.hide()
        diagnostics_button = QtWidgets.QToolButton()
        diagnostics_button.setText("Diagnostics")
        diagnostics_button.clicked.connect(lambda: self.logs.setVisible(not self.logs.isVisible()))
        self.statusBar().addPermanentWidget(diagnostics_button)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+R"), self, activated=lambda: self.start_job("simulate"))
        QtGui.QShortcut(QtGui.QKeySequence("Space"), self, activated=self.toggle_play)

    def _build_sidebar(self):
        frame = QtWidgets.QFrame()
        frame.setObjectName("sidebar")
        frame.setMinimumWidth(250)
        frame.setMaximumWidth(390)
        layout = QtWidgets.QVBoxLayout(frame)
        layout.setContentsMargins(12, 10, 12, 8)
        self.preset_combo = QtWidgets.QComboBox()
        self.preset_combo.addItems(["Thermal benchmark", "Iron", "Lunar habitat", "Regolith composite", "Custom"])
        layout.addWidget(QtWidgets.QLabel("SCENARIO"))
        layout.addWidget(self.preset_combo)
        self.name_edit = QtWidgets.QLineEdit()
        self.name_edit.setPlaceholderText("Scenario name")
        layout.addWidget(self.name_edit)
        self.run_button = QtWidgets.QPushButton("Run simulation")
        self.run_button.setObjectName("run")
        layout.addWidget(self.run_button)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QtWidgets.QWidget()
        self.sections_layout = QtWidgets.QVBoxLayout(scroll_content)
        self.sections_layout.setContentsMargins(0, 0, 0, 0)
        self.sections_layout.setSpacing(0)
        self.editors: dict[str, QtWidgets.QWidget] = {}
        self._section("Geometry", [("major_radius", "Major radius", "m"), ("minor_radius", "Tube radius", "m"),
                                    ("habitat_radius", "Habitat radius", "m"), ("vessel_thickness", "Vessel wall", "m"),
                                    ("habitat_thickness", "Habitat wall", "m"), ("ground_enabled", "Include ground", "")], True)
        self._section("Thermal battery", [("initial_temperature", "Initial temperature", "K"),
                                          ("melting_temperature", "Melting point", "K"),
                                          ("heating_power", "Charge power", "W"), ("discharge_power", "Discharge power", "W"),
                                          ("discharge_temperature", "Discharge cutoff", "K"), ("efficiency", "Electrical efficiency", "0–1"),
                                          ("duration", "Simulated duration", "s")], True)
        self._section("Material properties", [("density", "Density", "kg/m³"), ("heat_capacity_solid", "Solid heat capacity", "J/(kg K)"),
                                              ("heat_capacity_liquid", "Liquid heat capacity", "J/(kg K)"),
                                              ("latent_heat", "Latent heat", "J/kg"), ("thermal_conductivity", "Thermal conductivity", "W/(m K)"),
                                              ("electrical_conductivity_solid", "Solid conductivity", "S/m"),
                                              ("electrical_conductivity_liquid", "Liquid conductivity", "S/m"),
                                              ("dynamic_viscosity", "Dynamic viscosity", "Pa s"), ("thermal_expansion", "Thermal expansion", "1/K")])
        self._section("Flow & magnetic field", [("mode", "Experiment", ("driven", "dynamo")),
                                               ("coil_current", "Coil current", "A"), ("coil_turns", "Coil turns", ""),
                                               ("pump_force", "Pump body force", "N/m³"), ("helical_fraction", "Helical forcing", "0–1"),
                                               ("seed_field", "Seed field", "T"), ("seed_duration", "Seed pulse", "s"),
                                               ("gravity", "Gravity", "m/s²"), ("turbulence", "MHD turbulence closure", "")])
        self._section("Particles & dose", [("particle_z", "Atomic number Z", ""), ("particle_a", "Mass number A", ""),
                                          ("particle_energy_mev_nucleon", "Primary energy", "MeV/nucleon"),
                                          ("source_kind", "Source", ("beam", "isotropic", "spectrum")),
                                          ("spectrum_path", "Spectrum JSON", "file"),
                                          ("particle_count", "Preview particles", ""), ("radiation_histories", "Transport histories", ""),
                                          ("transport_physics_list", "Geant4 physics", ("Shielding", "QGSP_BIC_HP")),
                                          ("production_cut_m", "Production cut", "m"),
                                          ("incident_fluence_m2", "Incident fluence", "1/m²"),
                                          ("target", "Dose target", ("water", "male", "female")),
                                          ("phantom_path", "ICRP110 data", "directory")])
        self._section("Advanced solver & training", [("solver", "Numerical backend", ("thermal_benchmark", "fenics")),
                                                    ("resolution", "Mesh resolution", ""), ("time_steps", "Output/time steps", ""),
                                                    ("transition_width", "Phase regularization", "K"),
                                                    ("exterior_radius_factor", "Exterior radius", "× R"),
                                                    ("thermal_turbulent_prandtl", "Turbulent Prandtl", ""),
                                                    ("ambient_temperature", "Radiation background", "K"), ("emissivity", "Emissivity", "0–1"),
                                                    ("charge_fraction", "Charging fraction", "0–1"),
                                                    ("pinn_epochs", "Training epochs", ""), ("pinn_batch_size", "Training batch", ""),
                                                    ("seed", "Random seed", "")])
        self.sections_layout.addStretch()
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll, 1)
        analysis = QtWidgets.QHBoxLayout()
        self.train_button = QtWidgets.QPushButton("Train PINN")
        self.trace_button = QtWidgets.QPushButton("Trace rays")
        analysis.addWidget(self.train_button)
        analysis.addWidget(self.trace_button)
        layout.addLayout(analysis)
        self.dose_button = QtWidgets.QPushButton("Calculate radiation dose")
        layout.addWidget(self.dose_button)
        layout.addWidget(QtWidgets.QLabel("SAVED RUNS"))
        self.history = QtWidgets.QComboBox()
        self.history.setMinimumContentsLength(12)
        layout.addWidget(self.history)
        self.compare_history = QtWidgets.QComboBox()
        layout.addWidget(self.compare_history)
        return frame

    def _section(self, title, specifications, opened=False):
        section = QtWidgets.QFrame()
        section.setObjectName("section")
        layout = QtWidgets.QVBoxLayout(section)
        layout.setContentsMargins(0, 0, 0, 3)
        button = QtWidgets.QToolButton()
        button.setObjectName("sectionTitle")
        button.setText(title)
        button.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        button.setArrowType(QtCore.Qt.ArrowType.DownArrow if opened else QtCore.Qt.ArrowType.RightArrow)
        button.setCheckable(True)
        button.setChecked(opened)
        layout.addWidget(button)
        content = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(content)
        form.setContentsMargins(0, 0, 0, 7)
        form.setHorizontalSpacing(6)
        form.setVerticalSpacing(5)
        defaults = ScenarioConfig().to_dict()
        for key, label, units in specifications:
            if isinstance(units, tuple):
                editor = QtWidgets.QComboBox()
                editor.addItems(units)
                editor.currentIndexChanged.connect(self._queue_draft)
            elif isinstance(defaults[key], bool):
                editor = QtWidgets.QCheckBox()
                editor.toggled.connect(self._queue_draft)
            else:
                editor = QtWidgets.QLineEdit()
                editor.setMinimumWidth(72)
                editor.textEdited.connect(self._queue_draft)
                if units not in {"", "file", "directory"}:
                    editor.setToolTip(f"{label} [{units}]")
            self.editors[key] = editor
            field_widget = editor
            if units in {"file", "directory"}:
                field_widget = QtWidgets.QWidget()
                field_layout = QtWidgets.QHBoxLayout(field_widget)
                field_layout.setContentsMargins(0, 0, 0, 0)
                field_layout.setSpacing(2)
                field_layout.addWidget(editor, 1)
                browse = QtWidgets.QToolButton()
                browse.setText("…")
                browse.clicked.connect(lambda checked, key=key, directory=units == "directory": self._browse_input(key, directory))
                field_layout.addWidget(browse)
            text = label + (f"\n[{units}]" if isinstance(units, str) and units not in {"", "0–1", "file", "directory"} else "")
            caption = QtWidgets.QLabel(text)
            caption.setWordWrap(True)
            caption.setMaximumWidth(140)
            form.addRow(caption, field_widget)
        content.setVisible(opened)
        button.toggled.connect(content.setVisible)
        button.toggled.connect(lambda checked: button.setArrowType(QtCore.Qt.ArrowType.DownArrow if checked else QtCore.Qt.ArrowType.RightArrow))
        layout.addWidget(content)
        self.sections_layout.addWidget(section)

    def _connect(self):
        self.sidebar_button.toggled.connect(self.sidebar.setVisible)
        self.preset_combo.currentTextChanged.connect(self._preset_changed)
        self.name_edit.textEdited.connect(self._queue_draft)
        self.run_button.clicked.connect(lambda: self.start_job("simulate"))
        self.train_button.clicked.connect(lambda: self.start_job("pinn"))
        self.trace_button.clicked.connect(lambda: self.start_job("trace"))
        self.dose_button.clicked.connect(lambda: self.start_job("dose"))
        self.cancel_button.clicked.connect(self.cancel_job)
        self.jobs.progress.connect(self._job_progress)
        self.jobs.busy_changed.connect(self._busy_changed)
        self.jobs.completed.connect(self._job_completed)
        self.jobs.failed.connect(self._job_failed)
        self.jobs.cancelled.connect(lambda: self._notify("Job cancelled. Previous accepted results are preserved."))
        self.jobs.log.connect(self._log)
        self.scene.point_selected.connect(self._probe_selected)
        self.quantity.currentTextChanged.connect(self._quantity_changed)
        self.source.currentIndexChanged.connect(self._source_changed)
        self.thermal_plot.currentTextChanged.connect(self.refresh_plots)
        self.shield_plot.currentTextChanged.connect(self.refresh_plots)
        self.camera.currentTextChanged.connect(self._camera_changed)
        for control in (self.cutaway, self.phase, self.flow, self.tracers, self.field_lines, self.slice):
            control.toggled.connect(self._overlays_changed)
        self.detail.currentTextChanged.connect(self._detail_changed)
        self.opacity.valueChanged.connect(self._opacity_changed)
        self.section_slider.valueChanged.connect(self._section_changed)
        self.back.clicked.connect(lambda: self._step(-1))
        self.forward.clicked.connect(lambda: self._step(1))
        self.play.clicked.connect(self.toggle_play)
        self.timeline.valueChanged.connect(self._scrub)
        self.timeline.sliderPressed.connect(self.stop_play)
        self.replay.clicked.connect(self.toggle_particle_replay)
        for chart in self.charts:
            chart.time_clicked.connect(self.seek_time)
        self.history.activated.connect(self._history_selected)
        self.compare_history.activated.connect(self._compare_selected)

    def read_config(self):
        base = getattr(self, "_base_config", preset("Thermal benchmark")).to_dict()
        for key, editor in self.editors.items():
            if isinstance(editor, QtWidgets.QComboBox):
                base[key] = editor.currentText()
            elif isinstance(editor, QtWidgets.QCheckBox):
                base[key] = editor.isChecked()
            elif isinstance(editor, QtWidgets.QLineEdit):
                value = editor.text().strip()
                original = base[key]
                try:
                    base[key] = int(value) if isinstance(original, int) else float(value) if isinstance(original, float) else value
                except ValueError:
                    raise ValueError(f"{key.replace('_', ' ').capitalize()}: enter a valid {'integer' if isinstance(original, int) else 'number'}.") from None
        base["name"] = self.name_edit.text().strip() or "Custom scenario"
        base["preset"] = self.preset_combo.currentText()
        return ScenarioConfig.from_dict(base)

    def set_config(self, config):
        self._loading = True
        self._base_config = config
        self.name_edit.setText(config.name)
        self.preset_combo.setCurrentText(config.preset if config.preset in [self.preset_combo.itemText(i) for i in range(self.preset_combo.count())] else "Custom")
        values = config.to_dict()
        for key, editor in self.editors.items():
            value = values[key]
            if isinstance(editor, QtWidgets.QComboBox):
                editor.setCurrentText(str(value))
            elif isinstance(editor, QtWidgets.QCheckBox):
                editor.setChecked(bool(value))
            elif isinstance(editor, QtWidgets.QLineEdit):
                editor.setText(f"{value:g}" if isinstance(value, float) else str(value))
        self._loading = False
        self._draft_changed()

    def _preset_changed(self, name):
        if self._loading or name == "Custom":
            return
        self.set_config(preset(name))

    def _queue_draft(self, *args):
        if self._loading:
            return
        self.preset_combo.blockSignals(True)
        self.preset_combo.setCurrentText("Custom")
        self.preset_combo.blockSignals(False)
        self._draft_timer.start()

    def _draft_changed(self):
        try:
            config = self.read_config()
        except ValueError as error:
            self.message.setText(str(error))
            self.message.show()
            self.run_button.setEnabled(False)
            return
        self.run_button.setEnabled(not self.jobs.busy)
        if self.active is None or self.previewing:
            self.scene.preview(config)
            self.badge.setText("GEOMETRY PREVIEW")
        else:
            self.scene.preview_geometry(config)
        if self.active is not None and self.active.config.config_hash != config.config_hash:
            self.message.show()
            self.message.setText("Changes pending · displayed results belong to the saved scenario. Run/Recalculate to apply physics changes.")
            self.run_button.setText("Recalculate scenario")
        elif self.active is not None:
            self.message.hide()
            self.message.setText(f"{self.active.config.name} · {self.active.metadata.get('validation', self.active.metadata.get('validation_status', 'Unvalidated'))} · right-click to probe")
            self.run_button.setText("Run again")
        else:
            self.message.show()
            self.message.setText("Geometry preview · run the thermal benchmark or select FEniCS for a full coupled MHD experiment.")
        self._update_actions()

    def _browse_input(self, key, directory):
        if directory:
            value = QtWidgets.QFileDialog.getExistingDirectory(self, "Select phantom data directory")
        else:
            value, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select spectrum", "", "Spectrum (*.json *.csv);;All files (*)")
        if value:
            self.editors[key].setText(value)
            self._queue_draft()

    def start_job(self, action):
        if self.jobs.busy or self._movie_writer is not None:
            return
        try:
            config = self.read_config()
            if action != "simulate" and self.active is None:
                raise ValueError("Run a simulation first.")
            self.stop_play()
            self.particle_timer.stop()
            pinn_path = None
            if action in {"dose", "trace"} and self.source.currentIndex():
                if self.source.currentIndex() in {2, 3}:
                    raise ValueError("Select Numerical or PINN fields for radiation analysis; an error map is not a physical field.")
                if self.pinn_path is None:
                    raise ValueError("Train a coupled PINN for this saved scenario first.")
                pinn_path = self.pinn_path
            self._job_revision = self._selection_revision
            self.jobs.start(action, config, self.active_path, self.current_index, pinn_path)
            self._notify({"simulate": "Starting numerical solve…", "pinn": "Starting per-scenario PINN training…",
                          "trace": "Integrating paired relativistic trajectories…", "dose": "Starting Geant4 transport…"}[action])
        except (ValueError, RuntimeError) as error:
            self._notify(str(error), error=True)

    def _job_progress(self, fraction, message):
        self.progress_bar.setValue(int(fraction * 1000))
        self.job_text.setText(message)
        self.statusBar().showMessage(message)

    def _busy_changed(self, busy):
        if not busy:
            self.progress_bar.setValue(0)
            self.job_text.setText("No background job")
        self._update_actions()

    def _update_actions(self):
        busy = self.jobs.busy or self._movie_writer is not None
        self.job_widget.setVisible(busy)
        self.progress_bar.setVisible(busy)
        self.cancel_button.setEnabled(busy)
        try:
            self.read_config()
            valid_draft = True
        except (ValueError, TypeError):
            valid_draft = False
        self.run_button.setEnabled(not busy and valid_draft)
        has_result = self.active is not None
        for control in (self.train_button, self.trace_button, self.dose_button):
            control.setEnabled(has_result and not busy)
        for control in (self.play, self.back, self.forward, self.timeline):
            control.setEnabled(has_result and not self.previewing)
        self.replay.setEnabled(self.trajectories is not None and not self.previewing)
        self.source.model().item(1).setEnabled(self.pinn is not None)
        self.source.model().item(2).setEnabled(self.pinn is not None)
        self.source.model().item(3).setEnabled(self._comparison_difference is not None and self.view == "Compare")

    def _job_completed(self, event):
        try:
            self.refresh_history()
            if self._job_revision != self._selection_revision:
                self._notify("Saved the completed job to run history. The scenario selected while it ran remains displayed.")
                return
            source_path = Path(event["source_path"])
            action = event["action"]
            if action == "simulate" or self.active_path != source_path:
                self.activate_result(source_path)
            if event.get("selected_source") == "PINN" and self.pinn is not None:
                self.source.setCurrentIndex(1)
            if action == "pinn":
                if self.active is None:
                    raise ValueError("The trained model has no source scenario.")
                self.pinn = self.store.load_pinn(event["path"])
                self._pinn_difference = None
                self.pinn_path = Path(event["path"])
                from .pinn import validate_pinn_source
                validate_pinn_source(self.pinn, self.active)
                self.set_view("PINN")
                self._notify(f"PINN validation: {self.pinn.metadata.get('validation_status', 'unvalidated')}. Inspect held-out field errors and conservation in this view.",
                             sticky=self.pinn.metadata.get("validation_status") != "passed")
            elif action == "dose":
                if self.active is None:
                    raise ValueError("The dose result has no source scenario.")
                dose = self.store.load_dose(event["path"])
                if (dose.source_run_id != self.active.run_id or dose.config_hash != self.active.config.config_hash
                        or not 0 <= dose.snapshot_index < len(self.active.times)):
                    raise ValueError("Rejected a dose result from a different source run or snapshot.")
                self.dose = dose
                self.seek_time(float(self.active.times[self.dose.snapshot_index]))
                self.scene.set_dose(self.dose)
                self.set_view("Shielding")
                precision = self.dose.metadata.get("statistical_precision_met", ())
                self._notify(f"Dose transport complete · {self.dose.histories:,} primaries · {sum(precision)}/{len(precision)} regions meet ≤5% statistical uncertainty. Zero-hit regions remain insufficient.",
                             sticky=not all(precision))
            elif action == "trace":
                with np.load(event["path"], allow_pickle=False) as archive:
                    data = {key: archive[key] for key in archive.files if key != "_metadata_json"}
                    data.update(json.loads(str(archive["_metadata_json"])))
                if (self.active is None or data.get("source_run_id") != self.active.run_id
                        or data.get("config_hash") != self.active.config.config_hash
                        or not 0 <= int(data["snapshot_index"]) < len(self.active.times)):
                    raise ValueError("Rejected trajectories from another source run or snapshot.")
                self.trajectories = data
                self.seek_time(float(self.active.times[int(data["snapshot_index"])]))
                self.scene.set_trajectories(data)
                self.set_view("Shielding")
                self._notify("Paired particle trajectories ready · particle replay uses its own flight-time clock")
            else:
                self._notify("Simulation complete · edit parameters and recalculate at any time")
            self._update_actions()
            self.refresh_plots()
        except Exception as error:
            self._job_failed(str(error), "Loading completed worker artifacts failed.")

    def _job_failed(self, message, detail):
        self._notify(message, error=True)
        self._log(detail)
        # No modal interruption and no replacement of accepted result state.
        self.logs.show()

    def _notify(self, message, error=False, sticky=False):
        self.message.setText(message)
        self.message.setVisible(error or sticky)
        self.statusBar().showMessage(message)
        if error:
            self._log(message)

    def _log(self, message):
        if message.strip():
            self.log_text.appendPlainText(message.rstrip())

    def cancel_job(self):
        if self._movie_writer is not None:
            self._finish_movie(cancelled=True)
        else:
            self.jobs.cancel()

    def refresh_history(self):
        records = self.store.list_runs()
        self._records = records
        active_path = str(self.active_path) if self.active_path else None
        for combo in (self.history, self.compare_history):
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("Select saved run…", None)
            for record in records:
                label = f"{record['name']} · {record['created'][11:19]}"
                combo.addItem(label, record["path"])
            index = combo.findData(active_path) if combo == self.history else 0
            combo.setCurrentIndex(max(0, index))
            combo.blockSignals(False)

    def _history_selected(self, index):
        path = self.history.itemData(index)
        if path:
            try:
                self.activate_result(Path(path))
                if self.active is not None:
                    self.set_config(self.active.config)
            except Exception as error:
                self._notify(str(error), error=True)

    def activate_result(self, path):
        old_time = float(self.active.times[self.current_index]) if self.active is not None else 0.0
        old_probe = self.active.points[self.probe_index].copy() if self.active is not None else None
        self.stop_play()
        self.active = self.store.load_run(path)
        self._selection_revision += 1
        self._comparison_difference = None
        self.active_path = Path(path)
        self.previewing = False
        self.current_index = int(np.argmin(np.abs(self.active.times - old_time)))
        mask = conductor_mask(self.active)
        self.probe_index = int(np.flatnonzero(mask)[0]) if mask.any() else 0
        if old_probe is not None and mask.any():
            indices = np.flatnonzero(mask)
            self.probe_index = int(indices[np.argmin(np.sum((self.active.points[indices] - old_probe) ** 2, axis=1))])
        self.pinn = None
        self._pinn_difference = None
        self.pinn_path = None
        self.dose = None
        self.trajectories = None
        self.particle_timer.stop()
        self.scene.trajectory_data = None
        self.source.setCurrentIndex(0)
        self._set_quantity_availability()
        self.scene.set_result(self.active, self.quantity.currentText(), self.current_index)
        self.badge.setText("COUPLED MHD" if self.active.config.solver == "fenics" else "THERMAL BENCHMARK")
        self.badge.setToolTip(str(self.active.metadata.get("backend", "Numerical")))
        if old_time == 0.0 and self.active.config.duration > 60:
            candidates = [float(self.speed.itemText(i).replace("×", "")) for i in range(self.speed.count())]
            nearest = int(np.argmin(np.abs(np.asarray(candidates) - self.active.config.duration / 20)))
            self.speed.setCurrentIndex(nearest)
        for artifact in sorted(Path(path).parent.glob("pinn-*.h5"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                candidate = self.store.load_pinn(artifact)
                if candidate.config_hash == self.active.config.config_hash:
                    from .pinn import validate_pinn_source
                    validate_pinn_source(candidate, self.active)
                    self.pinn = candidate
                    self.pinn_path = artifact
                    break
            except (OSError, ValueError):
                pass
        for artifact in sorted(Path(path).parent.glob("dose-*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                candidate = self.store.load_dose(artifact)
                if (candidate.config_hash == self.active.config.config_hash and candidate.source_run_id == self.active.run_id
                        and 0 <= candidate.snapshot_index < len(self.active.times)):
                    self.dose = candidate
                    break
            except (OSError, ValueError, TypeError):
                pass
        self.scene.set_dose(self.dose)
        self._render()
        self.refresh_plots()
        self.refresh_history()
        self._draft_changed()
        self._update_actions()
        self.statusBar().showMessage(f"Loaded {self.active.config.name} · accepted numerical snapshots")

    def _compare_selected(self, index):
        path = self.compare_history.itemData(index)
        if not path:
            self.comparison = None
            self._comparison_difference = None
            if self.source.currentIndex() == 3:
                self.source.setCurrentIndex(0)
            self.compare_scene.hide()
            self._update_actions()
            return
        try:
            self.comparison = self.store.load_run(path)
            self._comparison_difference = None
            if (self.active is not None and self.active.points.shape == self.comparison.points.shape
                    and np.allclose(self.active.points, self.comparison.points, rtol=1e-10, atol=1e-12)):
                left_times, right_times = self.active.times, self.comparison.times
                if len(right_times) == 1:
                    upper = lower = np.zeros(len(left_times), dtype=int)
                    weight = np.zeros(len(left_times))
                else:
                    upper = np.clip(np.searchsorted(right_times, left_times, side="right"), 1, len(right_times) - 1)
                    lower = upper - 1
                    weight = np.clip((left_times - right_times[lower]) / (right_times[upper] - right_times[lower]), 0., 1.)
                self._comparison_difference = {}
                for key, values in self.active.fields.items():
                    other = self.comparison.fields.get(key)
                    if other is not None and values.shape[1:] == other.shape[1:]:
                        fraction = weight.reshape((-1,) + (1,) * (values.ndim - 1))
                        self._comparison_difference[key] = values - ((1 - fraction) * other[lower] + fraction * other[upper])
            self.set_view("Compare")
            self.compare_scene.set_result(self.comparison, self.quantity.currentText())
            self.compare_scene.plotter.camera = self.scene.plotter.camera
            self._share_color_range()
            self._render()
            self.refresh_plots()
        except Exception as error:
            self._notify(str(error), error=True)

    def set_view(self, view):
        self.view = view
        self.thermal_plot.setVisible(view == "Thermal")
        self.shield_plot.setVisible(view == "Shielding")
        for button in self.view_group.buttons():
            button.setChecked(button.text() == view)
        self.compare_scene.setVisible(view == "Compare" and self.comparison is not None)
        if view != "Compare" and self.source.currentIndex() == 3:
            self.source.setCurrentIndex(0)
        self._update_actions()
        if view == "Compare" and self.comparison is None:
            self._notify("Choose a second saved run below the parameter sidebar to compare scenarios.")
        self.refresh_plots()

    def _set_quantity_availability(self):
        if self.active is None:
            return
        available = self.active.fields
        if self.source.currentIndex() in {1, 2} and self.pinn is not None:
            available = self.pinn.fields
        elif self.source.currentIndex() == 3 and self._comparison_difference is not None:
            available = self._comparison_difference
        for index, name in enumerate(QUANTITIES):
            self.quantity.model().item(index).setEnabled(QUANTITIES[name][0] in available)
        if QUANTITIES[self.quantity.currentText()][0] not in available:
            for name, (key, _, _) in QUANTITIES.items():
                if key in available:
                    self.quantity.setCurrentText(name)
                    break

    def _quantity_changed(self):
        if self.active is None or self.previewing:
            return
        self.scene.quantity = self.quantity.currentText()
        self.scene._fixed_clim = None
        self.scene.rebuild_surface()
        if self.comparison is not None:
            self.compare_scene.quantity = self.quantity.currentText()
            self.compare_scene._fixed_clim = None
            self.compare_scene.rebuild_surface()
            self._share_color_range()
        self._render()
        self.refresh_plots()

    def _share_color_range(self):
        if self.comparison is None:
            return
        self.scene._fixed_clim = None
        self.compare_scene._fixed_clim = None
        left = self.scene.color_range()
        right = self.compare_scene.color_range()
        shared = (min(left[0], right[0]), max(left[1], right[1]))
        self.scene._fixed_clim = shared
        self.compare_scene._fixed_clim = shared

    def _source_changed(self):
        self._set_quantity_availability()
        self._render()
        self.refresh_plots()
        if self.source.currentIndex() == 1 and self.pinn is not None and self.pinn.metadata.get("explicit_prior_model_reuse"):
            self._notify(f"Selected prior PINN · {self.pinn.metadata.get('training_epochs', self.pinn.metrics.get('epochs', '?'))} training epochs · physical fields unchanged; observer settings recorded separately.")

    def _camera_changed(self, text):
        self.scene.camera_preset(text)
        if self.comparison is not None:
            self.compare_scene.plotter.camera = self.scene.plotter.camera

    def _overlays_changed(self):
        for scene in (self.scene, self.compare_scene):
            changed_cutaway = scene.cutaway != self.cutaway.isChecked()
            scene.cutaway = self.cutaway.isChecked()
            scene.show_phase = self.phase.isChecked()
            scene.show_flow = self.flow.isChecked()
            scene.show_tracers = self.tracers.isChecked()
            scene.show_field = self.field_lines.isChecked()
            scene.slice_enabled = self.slice.isChecked()
            if changed_cutaway:
                scene.rebuild_surface()
            elif scene.result is not None:
                scene.update_index(scene.index)

    def _detail_changed(self, detail):
        for scene in (self.scene, self.compare_scene):
            scene.visual_detail = detail
            scene._field_cache.clear()
            scene._tracer_cache.clear()
        self._render()

    def _opacity_changed(self, value):
        for scene in (self.scene, self.compare_scene):
            scene.opacity = value / 100.0
            if scene._actor is not None:
                scene._actor.prop.opacity = scene.opacity
                scene.plotter.render()
            elif scene.result is None and scene is self.scene:
                scene.preview(scene.config)

    def _section_changed(self, value):
        for scene in (self.scene, self.compare_scene):
            scene.slice_position = value / 1000 * (scene.config.major_radius + scene.config.minor_radius)
            if scene.result is not None:
                scene.update_index(scene.index)

    def _probe_selected(self, index):
        if self.active is None:
            return
        self.probe_index = index
        point = self.active.points[index]
        self.probe_label.setText(f"Probe ({point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f}) m")
        self.refresh_plots()

    def _scrub(self, value):
        if self.active is None:
            return
        end = self._timeline_end()
        target = float(self.active.times[0]) + value / 1000 * (end - self.active.times[0])
        self.seek_time(target)

    def _timeline_end(self):
        if self.active is None:
            return 0.0
        end = float(self.active.times[-1])
        if self.view == "Compare" and self.comparison is not None:
            end = min(end, float(self.comparison.times[-1]))
        return end

    def seek_time(self, value):
        if self.active is None or self.previewing:
            return
        self.current_index = int(np.argmin(np.abs(self.active.times - value)))
        self._play_position = float(self.active.times[self.current_index])
        self._render()

    def _step(self, delta):
        self.stop_play()
        if self.active is not None:
            self.current_index = int(np.clip(self.current_index + delta, 0, len(self.active.times) - 1))
            self._play_position = float(self.active.times[self.current_index])
            self._render()

    def _render(self):
        if self.active is None or self.previewing:
            return
        display_fields = None
        source_label = None
        if self.source.currentIndex() == 3 and self._comparison_difference is not None:
            display_fields = self._comparison_difference
            source_label = "Run difference · interpolated display"
        elif self.source.currentIndex() in {1, 2} and self.pinn is not None:
            status = str(self.pinn.metadata.get("validation_status", "unvalidated"))
            source_label = f"PINN · held-out {status}" if self.source.currentIndex() == 1 else f"PINN − numerical · held-out {status}"
            if self.source.currentIndex() == 1:
                display_fields = self.pinn.fields
            else:
                if self._pinn_difference is None:
                    self._pinn_difference = {key: value - self.active.fields[key] for key, value in self.pinn.fields.items()
                                             if key in self.active.fields and value.shape == self.active.fields[key].shape}
                display_fields = self._pinn_difference
        self.scene.update_index(self.current_index, display_fields, self.source.currentIndex() in {2, 3}, source_label)
        current_time = float(self.active.times[self.current_index])
        self.time_label.setText(f"t = {current_time:.4g} s")
        self.timeline.blockSignals(True)
        span = self._timeline_end() - self.active.times[0]
        self.timeline.setValue(int(1000 * (current_time - self.active.times[0]) / span) if span > 0 else 0)
        self.timeline.blockSignals(False)
        for card in self.charts:
            if card.cursor is not None:
                card.cursor.set_xdata([current_time, current_time])
            state_fields = display_fields if self.source.currentIndex() == 1 and display_fields is not None else self.active.fields
            if card.marker is not None and "enthalpy" in state_fields and "temperature" in state_fields:
                card.marker.set_data([state_fields["enthalpy"][self.current_index, self.probe_index] / 1000],
                                     [state_fields["temperature"][self.current_index, self.probe_index]])
            card.update_artists()
        if self.view == "Thermal" and self.thermal_plot.currentText() == "Temperature profile" and self._profile_samples is not None:
            coordinates, ids = self._profile_samples
            if self.charts[2].ax.collections:
                self.charts[2].ax.collections[0].set_offsets(np.column_stack((coordinates, self.active.fields["temperature"][self.current_index, ids])))
                self.charts[2].canvas.draw_idle()
        if self.view == "Compare" and self.comparison is not None:
            other = int(np.argmin(np.abs(self.comparison.times - current_time)))
            self.compare_scene.update_index(other)

    def toggle_play(self):
        if self.active is None or self.previewing:
            return
        if self.play_timer.isActive():
            self.stop_play()
            return
        self._play_position = float(self.active.times[self.current_index])
        self._last_tick = time.monotonic()
        self.play.setText("Ⅱ Pause")
        self.play_timer.start()

    def stop_play(self):
        self.play_timer.stop()
        if hasattr(self, "play"):
            self.play.setText("▶ Play")

    def _play_tick(self):
        if self.active is None:
            self.stop_play()
            return
        now = time.monotonic()
        speed = float(self.speed.currentText().replace("×", ""))
        self._play_position += min(now - self._last_tick, .2) * speed
        self._last_tick = now
        if self._play_position > self._timeline_end():
            if self.loop.isChecked():
                self._play_position = float(self.active.times[0])
            else:
                self.stop_play()
                return
        index = int(np.argmin(np.abs(self.active.times - self._play_position)))
        if index != self.current_index:
            self.current_index = index
            self._render()

    def toggle_particle_replay(self):
        if self.trajectories is None or self.active is None:
            return
        if self.particle_timer.isActive():
            self.particle_timer.stop()
            self.replay.setText("Particle replay")
        else:
            self.stop_play()
            self.seek_time(float(self.active.times[int(self.trajectories["snapshot_index"])]))
            self._particle_fraction = 0.0
            self.particle_timer.start()
            self.replay.setText("Pause particles")

    def _particle_tick(self):
        self._particle_fraction += .008
        if self._particle_fraction > 1:
            self._particle_fraction = 0.0
        self.scene.draw_particles(self._particle_fraction)

    def _plot_empty(self):
        for card, title, message in zip(self.charts, ("Material T(E)", "Energy & phase", "Probe history"),
                                        ("Run a simulation to inspect the material state", "Computed energy and phase histories", "Right-click a reservoir point after running")):
            card.empty(title, message)

    def refresh_plots(self):
        if self.active is None:
            self._plot_empty()
            return
        if self.view == "Shielding":
            self._shielding_plots()
            return
        if self.view == "PINN":
            self._pinn_plots()
            return
        if self.view == "Dynamo":
            self._dynamo_plots()
            return
        if self.view == "Compare":
            self._comparison_plots()
            return
        result, config = self.active, self.active.config
        current = float(result.times[self.current_index])
        mask = conductor_mask(result)
        temperatures = result.fields.get("temperature")
        if temperatures is None:
            self._plot_empty()
            return
        extent = max(float(np.ptp(temperatures[:, mask])) * .25, 25)
        minimum = min(float(np.min(temperatures[:, mask])), config.melting_temperature - extent)
        maximum = max(float(np.max(temperatures[:, mask])), config.melting_temperature + extent)
        enthalpy = np.array([config.heat_capacity_solid * (minimum - config.melting_temperature), 0,
                             config.latent_heat, config.latent_heat + config.heat_capacity_liquid * (maximum - config.melting_temperature)])
        ax = self.charts[0].reset("Material T(E) · probe state")
        ax.plot(enthalpy / 1000, [minimum, config.melting_temperature, config.melting_temperature, maximum], color="#d27a35", linewidth=2)
        ax.axvline(0, color="#aabac4", linewidth=.8, linestyle=":")
        ax.axvline(config.latent_heat / 1000, color="#aabac4", linewidth=.8, linestyle=":")
        self.charts[0].marker, = ax.plot([], [], "o", color="#167e86", markersize=6)
        ax.set(xlabel="Specific enthalpy [kJ/kg]", ylabel="Temperature [K]")
        self.charts[0].finish()
        ax = self.charts[1].reset("Stored thermal energy", True)
        energy = result.diagnostics.get("recoverable_energy_j", result.diagnostics.get("energy_j"))
        if energy is not None:
            ax.plot(result.times, energy / 3.6e6, color="#187e86", linewidth=1.8, label="Stored")
            ax.plot(result.times, energy * config.efficiency / 3.6e6, color="#d89e39", linewidth=1.4, label=f"Usable at η={config.efficiency:g}")
            ax.legend(fontsize=7, loc="best")
        ax.set(xlabel="Time [s]", ylabel="Energy [kWh]")
        self.charts[1].finish(current)
        choice = self.thermal_plot.currentText()
        ax = self.charts[2].reset(choice, choice != "Temperature profile")
        if choice == "Temperature profile":
            probe = result.points[self.probe_index]
            phi = np.arctan2(probe[1], probe[0])
            angles = np.arctan2(result.points[:, 1], result.points[:, 0])
            delta = np.abs(np.angle(np.exp(1j * (angles - phi))))
            selected = np.flatnonzero(mask & (delta <= max(.04, np.pi / max(config.resolution * 4, 8))))
            coordinate = np.hypot(result.points[selected, 0], result.points[selected, 1]) - config.major_radius
            order = np.argsort(coordinate)
            self._profile_samples = (coordinate[order], selected[order])
            ax.scatter(coordinate[order], temperatures[self.current_index, selected[order]], s=6, color="#d27a35", label="Cross-section samples")
            ax.set(xlabel="Offset from torus centerline [m]", ylabel="Temperature [K]")
            low, high = float(np.min(temperatures[:, mask])), float(np.max(temperatures[:, mask]))
            ax.set_ylim(low - max(1., (high - low) * .05), high + max(1., (high - low) * .05))
        elif choice == "Liquid fraction history":
            values = result.fields.get("liquid_fraction")
            if values is not None:
                ax.plot(result.times, values[:, self.probe_index], color="#187e86", label="Probe")
            if "liquid_fraction" in result.diagnostics:
                ax.plot(result.times, result.diagnostics["liquid_fraction"], color="#d27a35", label="Volume mean")
            ax.set(xlabel="Time [s]", ylabel="Liquid fraction", ylim=(-.02, 1.02))
        elif choice == "Thermal power history":
            for key, label, color in (("net_power_w", "Net thermal input", "#187e86"), ("pump_power_w", "Pumping input", "#d27a35")):
                if key in result.diagnostics:
                    ax.plot(result.times, result.diagnostics[key], color=color, label=label)
            ax.set(xlabel="Time [s]", ylabel="Power [W]")
        else:
            ax.plot(result.times, temperatures[:, self.probe_index], color="#d27a35", linewidth=1.8, label="Numerical")
            ax.axhline(config.melting_temperature, color="#a2b4bf", linestyle=":", linewidth=1, label="Melting point")
            if self.pinn is not None and "temperature" in self.pinn.fields:
                ax.plot(result.times, self.pinn.fields["temperature"][:, self.probe_index], color="#187e86", linestyle="--", label="PINN")
            ax.set(xlabel="Time [s]", ylabel="Temperature [K]")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7, loc="best")
        self.charts[2].finish(current)
        self._render()

    def _dynamo_plots(self):
        result = self.active
        if result is None:
            return
        current = float(result.times[self.current_index])
        for card, key, title, unit, color in ((self.charts[0], "magnetic_energy_j", "Magnetic energy · growth/decay", "J", "#187e86"),
                                              (self.charts[2], "pump_power_w", "Pumping power", "W", "#d27a35")):
            values = result.diagnostics.get(key)
            if values is None:
                card.empty(title, "This field is not computed by the selected backend")
            else:
                ax = card.reset(title, True)
                ax.plot(result.times, values, color=color, linewidth=1.8)
                ax.set(xlabel="Time [s]", ylabel=f"[{unit}]")
                card.finish(current)
        if "magnetic_field" not in result.fields:
            self.charts[1].empty("Magnetic components", "Run a magnetic solver to inspect B")
            return
        ax = self.charts[1].reset("Magnetic field · selected probe", True)
        values = result.fields["magnetic_field"][:, self.probe_index]
        point = result.points[self.probe_index]
        angle = np.arctan2(point[1], point[0])
        toroidal = -np.sin(angle) * values[:, 0] + np.cos(angle) * values[:, 1]
        poloidal = np.sqrt(np.maximum(np.sum(values ** 2, axis=1) - toroidal ** 2, 0))
        ax.plot(result.times, toroidal, label="Toroidal", color="#187e86")
        ax.plot(result.times, poloidal, label="Poloidal magnitude", color="#ce813b")
        ax.set(xlabel="Time [s]", ylabel="Magnetic flux density [T]")
        ax.legend(fontsize=7)
        self.charts[1].finish(current)

    def _shielding_plots(self):
        if self.active is None:
            return
        if self.trajectories is None:
            self.charts[0].empty("Paired transmission", "Trace rays to compare field-on/off trajectories")
            self.charts[2].empty("Incident primary energy", "Supports proton, alpha and heavier-ion inputs")
        else:
            names = ["habitat", "hardware", "ground", "escaped", "window_end"]
            outcomes = self.trajectories["outcomes"]
            baseline = self.trajectories["baseline_outcomes"]
            ax = self.charts[0].reset("First-contact outcomes · paired primaries")
            locations = np.arange(len(names))
            ax.bar(locations - .18, [sum(x == name for x in baseline) for name in names], .35, label="Field off", color="#9caeb9")
            ax.bar(locations + .18, [sum(x == name for x in outcomes) for name in names], .35, label="Field on", color="#228e95")
            ax.set_xticks(locations, names)
            ax.set_ylabel("Primary particles")
            ax.legend(fontsize=7)
            self.charts[0].finish()
            ax = self.charts[2].reset("Incident spectrum · sampled primaries")
            energy = np.asarray(self.trajectories["energies_mev_nucleon"])
            species = np.asarray(self.trajectories["species"])
            for za, color in zip(np.unique(species, axis=0), cycle(("#d29348", "#187e86", "#9c679d", "#4477aa", "#cc6677", "#228833"))):
                selected = np.all(species == za, axis=1)
                ax.hist(energy[selected], bins=min(20, max(3, int(np.sqrt(len(energy))))), color=color, alpha=.6, label=f"Z={za[0]}, A={za[1]}")
            ax.set(xlabel="Energy [MeV/nucleon]", ylabel="Sampled primaries")
            ax.legend(fontsize=6)
            self.charts[2].finish()
            if self.shield_plot.currentText() == "Transmission vs energy":
                ax = self.charts[0].reset("Habitat first-contact fraction · trajectory diagnostic")
                edges = np.geomspace(max(float(energy.min()) * .9, 1e-8), float(energy.max()) * 1.1, 7)
                for za, color in zip(np.unique(species, axis=0), cycle(("#d29348", "#187e86", "#9c679d", "#4477aa", "#cc6677", "#228833"))):
                    selected = np.all(species == za, axis=1)
                    counts, _ = np.histogram(energy[selected], bins=edges)
                    for values, style, label in ((outcomes, "-", "on"), (baseline, "--", "off")):
                        hits = selected & (np.asarray(values) == "habitat")
                        transmitted, _ = np.histogram(energy[hits], bins=edges)
                        centers = np.sqrt(edges[:-1] * edges[1:])
                        available = counts > 0
                        ax.plot(centers[available], transmitted[available] / counts[available], style, marker="o", markersize=3,
                                color=color, label=f"Z={za[0]} A={za[1]} {label}")
                ax.set(xscale="log", xlabel="Energy [MeV/nucleon]", ylabel="Habitat contacts / incident primaries", ylim=(-.03, 1.03))
                ax.legend(fontsize=6)
                self.charts[0].finish()
        if self.dose is None:
            self.charts[1].empty("Absorbed dose & uncertainty", "Run Geant4 for water or anatomical dose\nTrajectories alone do not determine dose")
        else:
            snapshot_time = float(self.active.times[self.dose.snapshot_index])
            ax = self.charts[1].reset(f"{self.dose.target.title()} · t={snapshot_time:.3g} s · Φ={self.active.config.incident_fluence_m2:g}/m²")
            order = np.argsort(self.dose.dose_gy)[-10:]
            positions = np.arange(len(order))
            if "field_off_dose_gy" in self.dose.metadata and "field_off_standard_error_gy" in self.dose.metadata:
                baseline = np.asarray(self.dose.metadata["field_off_dose_gy"])
                baseline_error = np.asarray(self.dose.metadata["field_off_standard_error_gy"])
                ax.barh(positions - .18, baseline[order], height=.35, xerr=baseline_error[order], color="#9caeb9", capsize=2, label="Field off")
            ax.barh(positions + .18, self.dose.dose_gy[order], height=.35, xerr=self.dose.standard_error_gy[order], color="#9c679d", capsize=2, label="Field on")
            ax.set_yticks(positions, [self.dose.organ_names[i] for i in order])
            ax.set_xlabel("Absorbed dose [Gy] ± standard error")
            ax.legend(fontsize=6)
            self.charts[1].finish()
            if self.shield_plot.currentText() == "Secondary dose":
                if "secondary_dose_gy" not in self.dose.metadata or "secondary_standard_error_gy" not in self.dose.metadata:
                    self.charts[0].empty("Secondary contributions", "This saved result has no secondary dose tally")
                    return
                ax = self.charts[0].reset("Secondary contributions · field on")
                values = np.asarray(self.dose.metadata["secondary_dose_gy"])
                error = np.asarray(self.dose.metadata["secondary_standard_error_gy"])
                ax.barh(positions, values[order], xerr=error[order], color="#cc6677", capsize=2)
                ax.set_yticks(positions, [self.dose.organ_names[i] for i in order])
                ax.set_xlabel("Absorbed dose from secondary tracks [Gy]")
                self.charts[0].finish()

    def _pinn_plots(self):
        if self.active is None:
            return
        if self.pinn is None:
            for card, title in zip(self.charts, ("Training convergence", "Held-out validation", "Numerical / PINN")):
                card.empty(title, "Train a PINN for the active saved scenario")
            return
        ax = self.charts[0].reset("Training convergence")
        for key, values in self.pinn.history.items():
            values = np.asarray(values)
            if values.ndim == 1 and len(values) and key not in {"epoch", "epochs", "time", "elapsed_seconds"}:
                ax.plot(np.arange(1, len(values) + 1), np.maximum(values, 1e-16), label=key)
        ax.set(yscale="log", xlabel="Training epoch", ylabel="Nondimensional loss")
        ax.legend(fontsize=6)
        self.charts[0].finish()
        status = self.pinn.metadata.get("validation_status", "unvalidated")
        ax = self.charts[1].reset(f"Held-out fields · {status}")
        metrics = [(key, value) for key, value in self.pinn.metrics.items()
                   if np.isfinite(value) and key.endswith("_normalized_rmse")][:8]
        if metrics:
            ax.barh(np.arange(len(metrics)), [value for key, value in metrics], color="#258d95")
            ax.axvline(.1, color="#c87e37", linestyle=":", linewidth=1, label="Field gate")
            ax.set_yticks(np.arange(len(metrics)), [key.replace("_normalized_rmse", "").replace("_", " ") for key, value in metrics])
            ax.set_xlabel("Held-out normalized RMSE [dimensionless]")
            ax.legend(fontsize=6)
        self.charts[1].finish()
        ax = self.charts[2].reset("Temperature · held scenario", True)
        for fields_, label, color, style in ((self.active.fields, "Numerical", "#d28740", "-"), (self.pinn.fields, "PINN", "#187e86", "--")):
            if "temperature" in fields_:
                ax.plot(self.active.times, fields_["temperature"][:, self.probe_index], color=color, linestyle=style, label=label)
        ax.set(xlabel="Time [s]", ylabel="Temperature [K]")
        ax.legend(fontsize=7)
        self.charts[2].finish(float(self.active.times[self.current_index]))

    def _comparison_plots(self):
        if self.active is None:
            return
        if self.comparison is None:
            for card in self.charts:
                card.empty("Scenario comparison", "Select a second saved run in the sidebar")
            return
        for card, key, title, unit, factor in ((self.charts[0], "recoverable_energy_j", "Recoverable thermal energy", "kWh", 3.6e6),
                                               (self.charts[1], "liquid_fraction", "Mean liquid fraction", "fraction", 1),
                                               (self.charts[2], "magnetic_energy_j", "Magnetic energy", "J", 1)):
            ax = card.reset(title, True)
            for result, color in ((self.active, "#187e86"), (self.comparison, "#d58b40")):
                values = result.diagnostics.get(key)
                if values is not None:
                    ax.plot(result.times, values / factor, label=result.config.name, color=color)
            ax.set(xlabel="Physical time [s]", ylabel=unit)
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(fontsize=6)
            card.finish(float(self.active.times[self.current_index]))
        differences = [key.replace("_", " ") for key, value in self.active.config.to_dict().items()
                       if value != self.comparison.config.to_dict().get(key) and key not in {"name", "preset"}]
        self.message.setText("Compared changes: " + (", ".join(differences[:10]) or "identical settings") + " · colors and camera are shared")

    def export_image(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export scene", "torus-scene.png", "PNG image (*.png)")
        if path:
            self.scene.plotter.screenshot(path, transparent_background=False)
            self._notify(f"Exported scene: {path}")

    def export_charts(self):
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Export chart SVGs")
        if directory:
            for index, card in enumerate(self.charts, 1):
                card.figure.savefig(Path(directory) / f"torus-chart-{index}.svg", bbox_inches="tight")
            self._notify(f"Exported three charts to {directory}")

    def export_data(self):
        if self.active_path is None or self.active is None:
            self._notify("Run a simulation before exporting scientific results.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export result", f"{self.active.run_id}.h5", "HDF5 scientific data (*.h5)")
        if path:
            import shutil
            shutil.copy2(self.active_path, path)
            self._notify(f"Exported SI-valued fields and configuration to {path}")

    def export_config(self):
        try:
            config = self.read_config()
        except ValueError as error:
            self._notify(str(error), error=True)
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export scenario", "torus-scenario.json", "JSON scenario (*.json)")
        if path:
            Path(path).write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")

    def import_config(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Import scenario", "", "JSON scenario (*.json)")
        if path:
            try:
                self.set_config(ScenarioConfig.from_dict(json.loads(Path(path).read_text(encoding="utf-8"))))
            except (OSError, ValueError, TypeError) as error:
                self._notify(f"Cannot import scenario: {error}", error=True)

    def export_movie(self):
        if self.active is None or self.jobs.busy:
            self._notify("Select a saved result and wait for the current job before exporting an animation.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export scene animation", "torus-animation.mp4", "MP4 animation (*.mp4)")
        if path:
            self.begin_movie_export(path)

    def begin_movie_export(self, path):
        if self.active is None or self._movie_writer is not None:
            return
        import imageio_ffmpeg
        self.stop_play()
        self._movie_restore = self.current_index
        self._movie_frames = np.linspace(0, len(self.active.times) - 1, max(60, min(300, len(self.active.times) * 3))).astype(int).tolist()
        self._movie_position = 0
        self.job_widget.show()
        self.progress_bar.show()
        QtWidgets.QApplication.processEvents()
        image = self.scene.plotter.screenshot(return_img=True)
        height, width = image.shape[:2]
        # Even dimensions are required by the H.264 YUV420 format.
        self._movie_size = (width - width % 2, height - height % 2)
        self._movie_writer = imageio_ffmpeg.write_frames(path, self._movie_size, fps=30, codec="libx264", macro_block_size=2,
                                                         pix_fmt_in="rgb24", pix_fmt_out="yuv420p", ffmpeg_log_level="error")
        self._movie_writer.send(None)
        self._movie_path = path
        self.movie_timer.start()
        self._update_actions()

    def _movie_tick(self):
        if self._movie_writer is None:
            return
        if self._movie_position >= len(self._movie_frames):
            self._finish_movie()
            return
        try:
            self.current_index = self._movie_frames[self._movie_position]
            self._render()
            image = self.scene.plotter.screenshot(return_img=True)
            width, height = self._movie_size
            # Dock changes and window resizing can change the render size during export.
            # The encoder requires exactly the dimensions supplied at initialization.
            image = image[:, :, :3]
            if image.shape[:2] != (height, width):
                from PIL import Image
                image = np.asarray(Image.fromarray(image).resize((width, height), Image.Resampling.LANCZOS))
            self._movie_writer.send(np.ascontiguousarray(image))
            self._movie_position += 1
            self._job_progress(self._movie_position / len(self._movie_frames), "Exporting annotated scene animation…")
        except Exception as error:
            self._finish_movie(cancelled=True)
            self._notify(f"Animation export failed: {error}", error=True)

    def _finish_movie(self, cancelled=False):
        self.movie_timer.stop()
        if self._movie_writer is not None:
            self._movie_writer.close()
        self._movie_writer = None
        self.current_index = self._movie_restore
        self._render()
        self._busy_changed(False)
        self._notify("Animation export cancelled; any encoded frames remain in the selected file." if cancelled else f"Exported animation: {self._movie_path}")

    def closeEvent(self, event):
        self.stop_play()
        self.particle_timer.stop()
        if self._movie_writer is not None:
            self._finish_movie(cancelled=True)
        self.jobs.shutdown()
        for card in self.charts:
            if card._dialog is not None:
                card._dialog.close()
        self.scene.close_scene()
        self.compare_scene.close_scene()
        event.accept()


def configure_application(app):
    app.setApplicationName("Magnetic Torus Simulator")
    app.setOrganizationName("HeatEquationSimulator")
    app.setStyle("Fusion")
    palette = QtGui.QPalette()
    for role, color in ((QtGui.QPalette.ColorRole.Window, "#edf2f5"),
                        (QtGui.QPalette.ColorRole.WindowText, "#203547"),
                        (QtGui.QPalette.ColorRole.Base, "#ffffff"),
                        (QtGui.QPalette.ColorRole.AlternateBase, "#f7f9fb"),
                        (QtGui.QPalette.ColorRole.Text, "#203547"),
                        (QtGui.QPalette.ColorRole.Button, "#ffffff"),
                        (QtGui.QPalette.ColorRole.ButtonText, "#203547"),
                        (QtGui.QPalette.ColorRole.Highlight, "#187e86"),
                        (QtGui.QPalette.ColorRole.HighlightedText, "#ffffff")):
        palette.setColor(role, QtGui.QColor(color))
    app.setPalette(palette)


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="Magnetic Torus scientific desktop")
    parser.add_argument("--load", type=Path, help="Open a saved HDF5 result")
    parser.add_argument("--data-dir", type=Path, help="Run history directory")
    options, qt_options = parser.parse_known_args((argv or sys.argv)[1:])
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([sys.argv[0], *qt_options])
    configure_application(app)
    window = TorusWindow(options.data_dir)
    def report_exception(kind, value, tb):
        import traceback
        text = "".join(traceback.format_exception(kind, value, tb))
        window._job_failed(str(value), text)
    sys.excepthook = report_exception
    if options.load is not None:
        try:
            window.activate_result(options.load)
            if window.active is not None:
                window.set_config(window.active.config)
        except (OSError, ValueError, TypeError) as error:
            window._job_failed(f"Cannot open saved run: {error}", str(options.load))
    window.show()
    QtCore.QTimer.singleShot(0, lambda: window.scene.camera_preset(window.camera.currentText()))
    return app.exec()
