"""Editable scenario controls, presets, parsing and debounced validation."""

from __future__ import annotations

from PyQt6 import QtCore, QtWidgets
from ..models import ScenarioConfig, preset


class ScenarioEditor(QtWidgets.QFrame):
    changed = QtCore.pyqtSignal(object)
    invalid = QtCore.pyqtSignal(str)
    action_requested = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._loading = False
        self._base_config = preset("Thermal benchmark")
        self._draft_timer = QtCore.QTimer(self)
        self._draft_timer.setSingleShot(True)
        self._draft_timer.setInterval(160)
        self._draft_timer.timeout.connect(self.validate)
        self._build()
        self.preset_combo.currentTextChanged.connect(self._preset_changed)
        self.name_edit.textEdited.connect(self._queue_draft)
        for control, action in (
            (self.run_button, "simulate"),
            (self.train_button, "pinn"),
            (self.trace_button, "trace"),
            (self.dose_button, "dose"),
        ):
            control.clicked.connect(
                lambda checked=False, action=action: self.action_requested.emit(action)
            )

    def validate(self):
        try:
            config = self.read_config()
        except ValueError as error:
            self.invalid.emit(str(error))
            return None
        self.changed.emit(config)
        return config

    def _build(self):
        frame = self
        frame.setObjectName("sidebar")
        frame.setMinimumWidth(250)
        frame.setMaximumWidth(390)
        layout = QtWidgets.QVBoxLayout(frame)
        layout.setContentsMargins(12, 10, 12, 8)
        self.preset_combo = QtWidgets.QComboBox()
        self.preset_combo.addItems(
            [
                "Thermal benchmark",
                "Iron",
                "Lunar habitat",
                "Regolith composite",
                "Custom",
            ]
        )
        layout.addWidget(QtWidgets.QLabel("SCENARIO"))
        layout.addWidget(self.preset_combo)
        self.name_edit = QtWidgets.QLineEdit()
        self.name_edit.setPlaceholderText("Scenario name")
        layout.addWidget(self.name_edit)
        self.editors: dict[str, QtWidgets.QWidget] = {}
        self._section(
            "Run setup",
            [("duration", "Simulation duration", "s")],
            True,
            target_layout=layout,
            description=(
                "Geometry below defines the reservoir. After running, right-click "
                "the 3D view to inspect a point."
            ),
        )
        self.run_button = QtWidgets.QPushButton("Run simulation")
        self.run_button.setObjectName("run")
        layout.addWidget(self.run_button)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QtWidgets.QWidget()
        self.sections_layout = QtWidgets.QVBoxLayout(scroll_content)
        self.sections_layout.setContentsMargins(0, 0, 0, 0)
        self.sections_layout.setSpacing(0)
        self._section(
            "Geometry",
            [
                ("major_radius", "Major radius", "m"),
                ("minor_radius", "Tube radius", "m"),
                ("habitat_radius", "Habitat radius", "m"),
                ("vessel_thickness", "Vessel wall", "m"),
                ("habitat_thickness", "Habitat wall", "m"),
                ("ground_enabled", "Include ground", ""),
            ],
            True,
        )
        self._section(
            "Thermal battery",
            [
                ("initial_temperature", "Initial temperature", "K"),
                ("melting_temperature", "Melting point", "K"),
                ("heating_power", "Charge power", "W"),
                ("discharge_power", "Discharge power", "W"),
                ("discharge_temperature", "Discharge cutoff", "K"),
                ("efficiency", "Electrical efficiency", "0–1"),
            ],
            True,
        )
        self._section(
            "Material properties",
            [
                ("density", "Density", "kg/m³"),
                ("heat_capacity_solid", "Solid heat capacity", "J/(kg K)"),
                ("heat_capacity_liquid", "Liquid heat capacity", "J/(kg K)"),
                ("latent_heat", "Latent heat", "J/kg"),
                ("thermal_conductivity", "Thermal conductivity", "W/(m K)"),
                ("electrical_conductivity_solid", "Solid conductivity", "S/m"),
                ("electrical_conductivity_liquid", "Liquid conductivity", "S/m"),
                ("dynamic_viscosity", "Dynamic viscosity", "Pa s"),
                ("thermal_expansion", "Thermal expansion", "1/K"),
            ],
        )
        self._section(
            "Flow & magnetic field",
            [
                ("mode", "Experiment", ("driven", "dynamo")),
                ("coil_current", "Coil current", "A"),
                ("coil_turns", "Coil turns", ""),
                ("pump_force", "Pump body force", "N/m³"),
                ("helical_fraction", "Helical forcing", "0–1"),
                ("seed_field", "Seed field", "T"),
                ("seed_duration", "Seed pulse", "s"),
                ("gravity", "Gravity", "m/s²"),
                ("turbulence", "MHD turbulence closure", ""),
            ],
        )
        self._section(
            "Particles & dose",
            [
                ("particle_z", "Atomic number Z", ""),
                ("particle_a", "Mass number A", ""),
                ("particle_energy_mev_nucleon", "Primary energy", "MeV/nucleon"),
                ("source_kind", "Source", ("beam", "isotropic", "spectrum")),
                ("spectrum_path", "Spectrum JSON", "file"),
                ("particle_count", "Preview particles", ""),
                ("radiation_histories", "Transport histories", ""),
                (
                    "transport_physics_list",
                    "Geant4 physics",
                    ("Shielding", "QGSP_BIC_HP"),
                ),
                ("production_cut_m", "Production cut", "m"),
                ("incident_fluence_m2", "Incident fluence", "1/m²"),
                ("target", "Dose target", ("water", "male", "female")),
                ("phantom_path", "ICRP110 data", "directory"),
            ],
        )
        self._section(
            "Advanced solver & training",
            [
                ("solver", "Numerical backend", ("thermal_benchmark", "fenics")),
                ("resolution", "Mesh resolution", ""),
                ("time_steps", "Output/time steps", ""),
                ("transition_width", "Phase regularization", "K"),
                ("exterior_radius_factor", "Exterior radius", "× R"),
                ("thermal_turbulent_prandtl", "Turbulent Prandtl", ""),
                ("ambient_temperature", "Radiation background", "K"),
                ("emissivity", "Emissivity", "0–1"),
                ("charge_fraction", "Charging fraction", "0–1"),
                ("pinn_epochs", "Training epochs", ""),
                ("pinn_batch_size", "Training batch", ""),
                ("seed", "Random seed", ""),
            ],
        )
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

    def _section(
        self,
        title,
        specifications,
        opened=False,
        *,
        target_layout=None,
        description=None,
    ):
        section = QtWidgets.QFrame()
        section.setObjectName("section")
        layout = QtWidgets.QVBoxLayout(section)
        layout.setContentsMargins(0, 0, 0, 3)
        button = QtWidgets.QToolButton()
        button.setObjectName("sectionTitle")
        button.setText(title)
        button.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        button.setArrowType(
            QtCore.Qt.ArrowType.DownArrow if opened else QtCore.Qt.ArrowType.RightArrow
        )
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
                browse.clicked.connect(
                    lambda checked, key=key, directory=units == "directory": (
                        self._browse_input(key, directory)
                    )
                )
                field_layout.addWidget(browse)
            text = label + (
                f"\n[{units}]"
                if isinstance(units, str)
                and units not in {"", "0–1", "file", "directory"}
                else ""
            )
            caption = QtWidgets.QLabel(text)
            caption.setWordWrap(True)
            caption.setMaximumWidth(140)
            form.addRow(caption, field_widget)
        if description:
            help_label = QtWidgets.QLabel(description)
            help_label.setObjectName("subtitle")
            help_label.setWordWrap(True)
            form.addRow(help_label)
        content.setVisible(opened)
        button.toggled.connect(content.setVisible)
        button.toggled.connect(
            lambda checked: button.setArrowType(
                QtCore.Qt.ArrowType.DownArrow
                if checked
                else QtCore.Qt.ArrowType.RightArrow
            )
        )
        layout.addWidget(content)
        (
            target_layout if target_layout is not None else self.sections_layout
        ).addWidget(section)

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
                    base[key] = (
                        int(value)
                        if isinstance(original, int)
                        else float(value)
                        if isinstance(original, float)
                        else value
                    )
                except ValueError:
                    raise ValueError(
                        f"{key.replace('_', ' ').capitalize()}: enter a valid {'integer' if isinstance(original, int) else 'number'}."
                    ) from None
        base["name"] = self.name_edit.text().strip() or "Custom scenario"
        base["preset"] = self.preset_combo.currentText()
        return ScenarioConfig.from_dict(base)

    def set_config(self, config):
        self._loading = True
        self._base_config = config
        self.name_edit.setText(config.name)
        self.preset_combo.setCurrentText(
            config.preset
            if config.preset
            in [self.preset_combo.itemText(i) for i in range(self.preset_combo.count())]
            else "Custom"
        )
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
        self.validate()

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

    def _browse_input(self, key, directory):
        if directory:
            value = QtWidgets.QFileDialog.getExistingDirectory(
                self, "Select phantom data directory"
            )
        else:
            value, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Select spectrum", "", "Spectrum (*.json *.csv);;All files (*)"
            )
        if value:
            self.editors[key].setText(value)
            self._queue_draft()
