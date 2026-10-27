"""Desktop composition and application orchestration; scientific jobs are isolated."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import numpy as np

os.environ.setdefault("QT_API", "pyqt6")
from PyQt6 import QtCore, QtGui, QtWidgets
from .jobs import JobController
from .learning.diagnostics import failed_checks, validation_message
from .learning.support import pinn_training_unavailable_reason
from .models import preset
from .scene import QUANTITIES, SceneWidget, SceneOptions
from .store import RunStore
from .ui.charts import ChartPanel
from .ui.editor import ScenarioEditor
from .ui.exports import ExportController
from .ui.playback import PlaybackController
from .ui.session import WorkspaceSession
from .ui.theme import STYLE


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
        self.session = WorkspaceSession()
        self._pinn_support_result = None
        self._pinn_support_reason = None
        self._build()
        self.exports = ExportController(
            self.session,
            self.scene,
            self.chart_panel,
            self.editor,
            self.playback,
            self._render,
            lambda: self.jobs.busy,
            self,
        )
        self._connect()
        self.editor.set_config(preset("Thermal benchmark"))
        self.scene.preview(self.editor.read_config(), reset_camera=True)
        self.refresh_history()
        self.chart_panel.empty()
        self._update_actions()
        self.statusBar().showMessage(
            "Geometry preview · choose a benchmark or configure a full MHD experiment"
        )

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
        subtitle = QtWidgets.QLabel(
            "Thermal storage  /  magnetic dynamics  /  radiation transport"
        )
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
        self.export_menu = menu
        export.setMenu(menu)
        row.addWidget(export)
        layout.addWidget(header)
        self.body = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.editor = ScenarioEditor()
        self.sidebar = self.editor
        self.body.addWidget(self.sidebar)
        self.body.setCollapsible(0, True)
        center = QtWidgets.QWidget()
        center_layout = QtWidgets.QVBoxLayout(center)
        center_layout.setContentsMargins(10, 8, 12, 8)
        center_layout.setSpacing(7)
        self.message = QtWidgets.QLabel(
            "Preview is geometry only. Run a simulation to display computed fields."
        )
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
        self.thermal_plot.addItems(
            [
                "Probe temperature",
                "Temperature profile",
                "Liquid fraction history",
                "Thermal power history",
            ]
        )
        self.thermal_plot.setToolTip("Choose the third thermal chart")
        quantity_row.addWidget(self.thermal_plot)
        self.shield_plot = QtWidgets.QComboBox()
        self.shield_plot.addItems(
            ["First-contact outcomes", "Transmission vs energy", "Secondary dose"]
        )
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
        self.chart_panel = ChartPanel(self.session)
        self.vertical.addWidget(self.chart_panel)
        self.plots_button.toggled.connect(self.chart_panel.setVisible)
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
        for control in (
            self.cutaway,
            self.phase,
            self.flow,
            self.tracers,
            self.field_lines,
            self.slice,
        ):
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
        self.playback = PlaybackController(self.session, self.scene)
        center_layout.addWidget(self.playback)
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
        diagnostics_button.clicked.connect(
            lambda: self.logs.setVisible(not self.logs.isVisible())
        )
        self.statusBar().addPermanentWidget(diagnostics_button)
        QtGui.QShortcut(
            QtGui.QKeySequence("Ctrl+R"),
            self,
            activated=lambda: self.start_job("simulate"),
        )
        QtGui.QShortcut(
            QtGui.QKeySequence("Space"), self, activated=self.playback.toggle_play
        )

    def _connect(self):
        self.sidebar_button.toggled.connect(self.sidebar.setVisible)
        self.editor.changed.connect(self._draft_changed)
        self.editor.invalid.connect(self._invalid_draft)
        self.editor.action_requested.connect(self.start_job)
        self.cancel_button.clicked.connect(self.cancel_job)
        self.jobs.progress.connect(self._job_progress)
        self.jobs.busy_changed.connect(self._busy_changed)
        self.jobs.completed.connect(self._job_completed)
        self.jobs.failed.connect(self._job_failed)
        self.jobs.cancelled.connect(
            lambda: self._notify(
                "Job cancelled. Previous accepted results are preserved."
            )
        )
        self.jobs.log.connect(self._log)
        self.scene.point_selected.connect(self._probe_selected)
        self.quantity.currentTextChanged.connect(self._quantity_changed)
        self.source.currentIndexChanged.connect(self._source_changed)
        self.thermal_plot.currentTextChanged.connect(self.refresh_plots)
        self.shield_plot.currentTextChanged.connect(self.refresh_plots)
        self.camera.currentTextChanged.connect(self._camera_changed)
        for control in (
            self.cutaway,
            self.phase,
            self.flow,
            self.tracers,
            self.field_lines,
            self.slice,
        ):
            control.toggled.connect(self._scene_options_changed)
        self.detail.currentTextChanged.connect(self._scene_options_changed)
        self.opacity.valueChanged.connect(self._scene_options_changed)
        self.section_slider.valueChanged.connect(self._scene_options_changed)
        self.playback.render_requested.connect(self._render)
        for chart in self.chart_panel.cards:
            chart.time_clicked.connect(self.playback.seek_time)
        self.chart_panel.notice.connect(self.message.setText)
        self.chart_panel.training_settings_requested.connect(
            self._show_training_settings
        )
        self.exports.notified.connect(self._notify)
        self.exports.progress.connect(self._job_progress)
        self.exports.busy_changed.connect(self._busy_changed)
        for label, callback in (
            ("Scene image…", self.exports.export_image),
            ("Charts (SVG)…", self.exports.export_charts),
            ("Scene animation (MP4)…", self.exports.export_movie),
            ("Scientific results (HDF5)…", self.exports.export_data),
            ("Scenario settings (JSON)…", self.exports.export_config),
            ("Import scenario settings…", self.exports.import_config),
        ):
            self.export_menu.addAction(label, callback)
        self.editor.history.activated.connect(self._history_selected)
        self.editor.compare_history.activated.connect(self._compare_selected)

    def _draft_changed(self, config):
        self.editor.run_button.setEnabled(not self.jobs.busy)
        if self.session.active is None or self.session.previewing:
            self.scene.preview(config)
            self.badge.setText("GEOMETRY PREVIEW")
        else:
            self.scene.preview_geometry(config)
        if (
            self.session.active is not None
            and self.session.active.config.config_hash != config.config_hash
        ):
            self.message.show()
            self.message.setText(
                "Changes pending · displayed results belong to the saved scenario. Run/Recalculate to apply physics changes."
            )
            self.editor.run_button.setText("Recalculate scenario")
        elif self.session.active is not None:
            self.message.hide()
            self.message.setText(
                f"{self.session.active.config.name} · {self.session.active.metadata.get('validation', self.session.active.metadata.get('validation_status', 'Unvalidated'))} · right-click to probe"
            )
            self.editor.run_button.setText("Run again")
        else:
            self.message.show()
            self.message.setText(
                "Geometry preview · run the thermal benchmark or select FEniCS for a full coupled MHD experiment."
            )
        self._update_actions()

    def start_job(self, action):
        if self.jobs.busy or self.exports.busy:
            return
        try:
            config = self.editor.read_config()
            if action != "simulate" and self.session.active is None:
                raise ValueError("Run a simulation first.")
            if action == "pinn":
                reason = self._pinn_training_unavailable_reason()
                if reason is not None:
                    raise ValueError(reason)
            self.playback.stop_all()
            pinn_path = None
            if action in {"dose", "trace"} and self.source.currentIndex():
                if self.source.currentIndex() in {2, 3}:
                    raise ValueError(
                        "Select Numerical or PINN fields for radiation analysis; an error map is not a physical field."
                    )
                if self.session.pinn_path is None:
                    raise ValueError(
                        "Train a coupled PINN for this saved scenario first."
                    )
                pinn_path = self.session.pinn_path
            self.session.job_revision = self.session.selection_revision
            self.jobs.start(
                action,
                config,
                self.session.active_path,
                self.session.current_index,
                pinn_path,
            )
            self._notify(
                {
                    "simulate": "Starting numerical solve…",
                    "pinn": "Starting per-scenario PINN training…",
                    "trace": "Integrating paired relativistic trajectories…",
                    "dose": "Starting Geant4 transport…",
                }[action]
            )
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
        busy = self.jobs.busy or self.exports.busy
        self.job_widget.setVisible(busy)
        self.progress_bar.setVisible(busy)
        self.cancel_button.setEnabled(busy)
        self.source.setEnabled(not busy)
        try:
            self.editor.read_config()
            valid_draft = True
        except (ValueError, TypeError):
            valid_draft = False
        self.editor.run_button.setEnabled(not busy and valid_draft)
        has_result = self.session.active is not None
        training_reason = self._pinn_training_unavailable_reason()
        # Unsupported training modes are not presented as available actions.
        # Keep the separate start_job guard for non-UI callers.
        self.editor.train_button.setVisible(not has_result or training_reason is None)
        self.editor.train_button.setEnabled(
            has_result and not busy and training_reason is None
        )
        self.editor.train_button.setToolTip(
            training_reason or "Train a PINN for the accepted simulation."
        )
        self.chart_panel.set_training_unavailable_reason(training_reason)
        for control in (
            self.editor.trace_button,
            self.editor.dose_button,
        ):
            control.setEnabled(has_result and not busy)
        self.playback.set_available(
            has_result, self.session.previewing, self.session.trajectories is not None
        )
        self.source.model().item(1).setEnabled(self.session.pinn is not None)
        self.source.model().item(2).setEnabled(self.session.pinn is not None)
        self.source.model().item(3).setEnabled(
            self.session.comparison_difference is not None
            and self.session.view == "Compare"
        )

    def _pinn_training_unavailable_reason(self):
        result = self.session.active
        if result is not self._pinn_support_result:
            self._pinn_support_result = result
            self._pinn_support_reason = (
                pinn_training_unavailable_reason(result) if result is not None else None
            )
        return self._pinn_support_reason

    def _job_completed(self, event):
        try:
            self.refresh_history()
            if not self.session.job_is_current:
                self._notify(
                    "Saved the completed job to run history. The scenario selected while it ran remains displayed."
                )
                return
            source_path = Path(event["source_path"])
            action = event["action"]
            if action == "simulate" or self.session.active_path != source_path:
                self.activate_result(source_path)
            if action in {"trace", "dose"}:
                selected_source = event.get("selected_source", "Numerical")
                if selected_source == "PINN" and self.session.pinn is None:
                    raise ValueError(
                        "The radiation result has no originating PINN to display."
                    )
                with QtCore.QSignalBlocker(self.source):
                    self.source.setCurrentText(selected_source)
                self.session.source = selected_source
                self._set_quantity_availability()
            if action == "pinn":
                if self.session.active is None:
                    raise ValueError("The trained model has no source scenario.")
                self.session.set_pinn(
                    self.store.load_pinn(event["path"]), Path(event["path"])
                )
                checks = failed_checks(self.session.pinn)
                if checks:
                    self._log(
                        f"PINN {self.session.pinn.run_id} · failed validation checks\n"
                        + "\n".join(
                            f"  {check.label}: {check.value:.6g} (limit {check.limit:g})"
                            for check in checks
                        )
                    )
                self.set_view("PINN")
                self._notify(
                    validation_message(self.session.pinn),
                    sticky=self.session.pinn.metadata.get("validation_status")
                    != "passed",
                )
            elif action == "dose":
                if self.session.active is None:
                    raise ValueError("The dose result has no source scenario.")
                dose = self.store.load_dose(event["path"])
                self.session.set_dose(dose)
                self.playback.seek_time(
                    float(self.session.active.times[self.session.dose.snapshot_index])
                )
                self.scene.set_dose(self.session.dose)
                self.set_view("Shielding")
                precision = self.session.dose.metadata.get(
                    "statistical_precision_met", ()
                )
                self._notify(
                    f"Dose transport complete · {self.session.dose.histories:,} primaries · {sum(precision)}/{len(precision)} regions meet ≤5% statistical uncertainty. Zero-hit regions remain insufficient.",
                    sticky=not all(precision),
                )
            elif action == "trace":
                with np.load(event["path"], allow_pickle=False) as archive:
                    data = {
                        key: archive[key]
                        for key in archive.files
                        if key != "_metadata_json"
                    }
                    data.update(json.loads(str(archive["_metadata_json"])))
                self.session.set_trajectories(data)
                self.playback.seek_time(
                    float(self.session.active.times[int(data["snapshot_index"])])
                )
                self.scene.set_trajectories(data)
                self.set_view("Shielding")
                self._notify(
                    "Paired particle trajectories ready · particle replay uses its own flight-time clock"
                )
            else:
                self._notify(
                    "Simulation complete · edit parameters and recalculate at any time"
                )
            self._update_actions()
            if action == "simulate":
                self.refresh_plots()
                self._render()
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
        if self.exports.busy:
            self.exports.finish_movie(cancelled=True)
        else:
            self.jobs.cancel()

    def refresh_history(self):
        records = self.store.list_runs()
        active_path = (
            str(self.session.active_path) if self.session.active_path else None
        )
        comparison_path = next(
            (
                record["path"]
                for record in records
                if self.session.comparison is not None
                and record.get("run_id") == self.session.comparison.run_id
            ),
            None,
        )
        for combo in (self.editor.history, self.editor.compare_history):
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("Select saved run…", None)
            for record in records:
                label = f"{record['name']} · {record['created'][11:19]}"
                combo.addItem(label, record["path"])
            path = active_path if combo == self.editor.history else comparison_path
            index = combo.findData(path)
            combo.setCurrentIndex(max(0, index))
            combo.blockSignals(False)

    def _history_selected(self, index):
        path = self.editor.history.itemData(index)
        if path:
            try:
                self.activate_result(Path(path))
                if self.session.active is not None:
                    self.editor.set_config(self.session.active.config)
            except Exception as error:
                self._notify(str(error), error=True)

    def set_view(self, view):
        self.session.view = view
        self.playback.clamp_to_timeline()
        self.thermal_plot.setVisible(view == "Thermal")
        self.shield_plot.setVisible(view == "Shielding")
        for button in self.view_group.buttons():
            button.setChecked(button.text() == view)
        self.compare_scene.setVisible(
            view == "Compare" and self.session.comparison is not None
        )
        if view != "Compare" and self.source.currentIndex() == 3:
            self.source.setCurrentIndex(0)
        self._update_actions()
        if view == "Compare" and self.session.comparison is None:
            self._notify(
                "Choose a second saved run below the parameter sidebar to compare scenarios."
            )
        self.refresh_plots()
        self._render()

    def _set_quantity_availability(self):
        if self.session.active is None:
            return
        selection = self.session.display_selection()
        available = (
            selection.fields
            if selection.fields is not None
            else self.session.active.fields
        )
        for index, name in enumerate(QUANTITIES):
            self.quantity.model().item(index).setEnabled(
                QUANTITIES[name][0] in available
            )
        if QUANTITIES[self.quantity.currentText()][0] not in available:
            for name, (key, _, _) in QUANTITIES.items():
                if key in available:
                    self.quantity.setCurrentText(name)
                    break

    def _source_changed(self):
        self.session.source = self.source.currentText()
        self._set_quantity_availability()
        self.refresh_plots()
        self._render()
        if (
            self.source.currentIndex() == 1
            and self.session.pinn is not None
            and self.session.pinn.metadata.get("explicit_prior_model_reuse")
        ):
            self._notify(
                f"Selected prior PINN · {self.session.pinn.metadata.get('training_epochs', self.session.pinn.metrics.get('epochs', '?'))} training epochs · physical fields unchanged; observer settings recorded separately."
            )

    def _camera_changed(self, text):
        self.scene.camera_preset(text)
        if self.session.comparison is not None:
            self.compare_scene.plotter.camera = self.scene.plotter.camera

    def _probe_selected(self, index):
        if self.session.active is None:
            return
        self.session.probe_index = index
        point = self.session.active.points[index]
        self.probe_label.setText(
            f"Probe ({point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f}) m"
        )
        self.refresh_plots()

    def _invalid_draft(self, message):
        self.message.setText(message)
        self.message.show()
        self.editor.run_button.setEnabled(False)

    def _show_training_settings(self):
        self.sidebar_button.setChecked(True)
        self.sidebar.show()
        if self.body.sizes()[0] < self.editor.minimumWidth():
            self.body.setSizes([285, max(1, self.body.width() - 285)])
        self.editor.show_training_settings()

    def activate_result(self, path):
        if self.exports.busy:
            self.exports.finish_movie(cancelled=True)
        self.playback.stop_all()
        old_time = self.session.load_result(self.store, Path(path))
        self.chart_panel.invalidate()
        with QtCore.QSignalBlocker(self.source), QtCore.QSignalBlocker(self.quantity):
            self.source.setCurrentText("Numerical")
            self._set_quantity_availability()
        result = self.session.active
        if self.session.comparison is not None:
            comparison = self.session.comparison
            try:
                self.playback.comparison_indices(comparison)
            except ValueError:
                self._clear_comparison()
            else:
                self.session.set_comparison(comparison)
        self.playback.clamp_to_timeline()
        self.scene.clear_analysis()
        self.scene.set_result(
            result, self.quantity.currentText(), self.session.current_index
        )
        self.scene.set_dose(self.session.dose)
        if self.session.comparison is not None:
            self.compare_scene.set_quantity(self.quantity.currentText())
            self.compare_scene.plotter.camera = self.scene.plotter.camera
            self._share_color_range()
        self.compare_scene.setVisible(
            self.session.view == "Compare" and self.session.comparison is not None
        )
        self.badge.setText(
            "COUPLED MHD" if result.config.solver == "fenics" else "THERMAL BENCHMARK"
        )
        self.badge.setToolTip(str(result.metadata.get("backend", "Numerical")))
        if old_time == 0.0 and result.config.duration > 60:
            candidates = [
                float(self.playback.speed.itemText(i).replace("×", ""))
                for i in range(self.playback.speed.count())
            ]
            self.playback.speed.setCurrentIndex(
                int(
                    np.argmin(
                        np.abs(np.asarray(candidates) - result.config.duration / 20)
                    )
                )
            )
        self.refresh_plots()
        self._render()
        self.refresh_history()
        self.editor.validate()
        self._update_actions()
        self.statusBar().showMessage(
            f"Loaded {result.config.name} · accepted numerical snapshots"
        )

    def _clear_comparison(self):
        self.session.set_comparison(None)
        if self.source.currentText() == "Run difference":
            with QtCore.QSignalBlocker(self.source):
                self.source.setCurrentText("Numerical")
            self.session.source = "Numerical"
        config = (
            self.session.active.config
            if self.session.active is not None
            else self.compare_scene.config
        )
        self.compare_scene.preview(config)
        self.compare_scene.hide()
        self.scene.set_shared_color_range(None)
        self.compare_scene.set_shared_color_range(None)

    def _compare_selected(self, index):
        path = self.editor.compare_history.itemData(index)
        if not path:
            self._clear_comparison()
            self._update_actions()
            self.refresh_plots()
            self._render()
            return
        try:
            comparison = self.store.load_run(path)
            if self.session.active is not None:
                self.playback.comparison_indices(comparison)
            self.session.set_comparison(comparison)
            self.compare_scene.set_result(
                self.session.comparison, self.quantity.currentText()
            )
            if (
                self.session.comparison_difference is None
                and self.source.currentText() == "Run difference"
            ):
                self.source.setCurrentText("Numerical")
            self.compare_scene.plotter.camera = self.scene.plotter.camera
            self._share_color_range()
            self.set_view("Compare")
        except Exception as error:
            self.refresh_history()
            self._notify(str(error), error=True)

    def _quantity_changed(self):
        if self.session.active is None or self.session.previewing:
            return
        self.scene.set_quantity(self.quantity.currentText())
        if self.session.comparison is not None:
            self.compare_scene.set_quantity(self.quantity.currentText())
            self._share_color_range()
        self.refresh_plots()

    def _share_color_range(self):
        if self.session.comparison is None:
            return
        self.scene.set_shared_color_range(None)
        self.compare_scene.set_shared_color_range(None)
        left, right = self.scene.color_range(), self.compare_scene.color_range()
        shared = (min(left[0], right[0]), max(left[1], right[1]))
        self.scene.set_shared_color_range(shared)
        self.compare_scene.set_shared_color_range(shared)

    def _scene_options_changed(self):
        for scene in (self.scene, self.compare_scene):
            options = SceneOptions(
                cutaway=self.cutaway.isChecked(),
                show_phase=self.phase.isChecked(),
                show_flow=self.flow.isChecked(),
                show_tracers=self.tracers.isChecked(),
                show_field=self.field_lines.isChecked(),
                slice_enabled=self.slice.isChecked(),
                slice_position=self.section_slider.value()
                / 1000
                * (scene.config.major_radius + scene.config.minor_radius),
                visual_detail=self.detail.currentText(),
                opacity=self.opacity.value() / 100.0,
            )
            scene.set_options(options)

    def refresh_plots(self):
        self.chart_panel.refresh(
            self.thermal_plot.currentText(), self.shield_plot.currentText()
        )
        if self.session.active is not None:
            self.chart_panel.update_frame(self.session.display_selection())

    def _render(self):
        if self.session.active is None or self.session.previewing:
            return
        if (
            self.scene.set_color_fields(
                self.session.pinn.fields if self.session.pinn is not None else None
            )
            and self.session.comparison is not None
        ):
            self._share_color_range()
        selection = self.session.display_selection()
        self.scene.set_display(selection)
        self.scene.update_index(self.session.current_index)
        self.playback.update_position()
        self.chart_panel.update_frame(selection)
        if self.session.view == "Compare" and self.session.comparison is not None:
            _, indices = self.playback.comparison_indices(self.session.comparison)
            other = int(
                indices[
                    np.argmin(
                        np.abs(
                            self.session.comparison.times[indices]
                            - self.session.current_time
                        )
                    )
                ]
            )
            self.compare_scene.update_index(other)

    def closeEvent(self, event):
        self.playback.stop_all()
        if self.exports.busy:
            self.exports.finish_movie(cancelled=True)
        self.jobs.shutdown()
        self.chart_panel.close_charts()
        self.scene.close_scene()
        self.compare_scene.close_scene()
        event.accept()


def configure_application(app):
    app.setApplicationName("Magnetic Torus Simulator")
    app.setOrganizationName("HeatEquationSimulator")
    app.setStyle("Fusion")
    palette = QtGui.QPalette()
    for role, color in (
        (QtGui.QPalette.ColorRole.Window, "#edf2f5"),
        (QtGui.QPalette.ColorRole.WindowText, "#203547"),
        (QtGui.QPalette.ColorRole.Base, "#ffffff"),
        (QtGui.QPalette.ColorRole.AlternateBase, "#f7f9fb"),
        (QtGui.QPalette.ColorRole.Text, "#203547"),
        (QtGui.QPalette.ColorRole.Button, "#ffffff"),
        (QtGui.QPalette.ColorRole.ButtonText, "#203547"),
        (QtGui.QPalette.ColorRole.Highlight, "#187e86"),
        (QtGui.QPalette.ColorRole.HighlightedText, "#ffffff"),
    ):
        palette.setColor(role, QtGui.QColor(color))
    app.setPalette(palette)


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="Magnetic Torus scientific desktop")
    parser.add_argument("--load", type=Path, help="Open a saved HDF5 result")
    parser.add_argument("--data-dir", type=Path, help="Run history directory")
    options, qt_options = parser.parse_known_args((argv or sys.argv)[1:])
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(
        [sys.argv[0], *qt_options]
    )
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
            if window.session.active is not None:
                window.editor.set_config(window.session.active.config)
        except (OSError, ValueError, TypeError) as error:
            window._job_failed(f"Cannot open saved run: {error}", str(options.load))
    window.show()
    QtCore.QTimer.singleShot(
        0, lambda: window.scene.camera_preset(window.camera.currentText())
    )
    return app.exec()
