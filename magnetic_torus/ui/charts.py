"""Scientific plot construction and lightweight cursor updates."""

from __future__ import annotations

from itertools import cycle
import numpy as np
from PyQt6 import QtCore, QtWidgets
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt import NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from ..geometry import conductor_mask
from .session import WorkspaceSession


_CHART_MARGINS = {"left": 0.25, "right": 0.96, "bottom": 0.33, "top": 0.88}


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
        self._header_controls = QtWidgets.QHBoxLayout()
        self._header_controls.setContentsMargins(0, 0, 0, 0)
        self._header_controls.setSpacing(3)
        toolbar.addLayout(self._header_controls)
        detach = QtWidgets.QToolButton()
        detach.setText("↗")
        detach.setToolTip("Expand this chart")
        detach.clicked.connect(self.detach)
        toolbar.addWidget(detach)
        self.layout_.addLayout(toolbar)
        self.figure = Figure(figsize=(4.0, 2.4), dpi=100, facecolor="white")
        self.figure.subplots_adjust(**_CHART_MARGINS)
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

    def add_header_control(self, control: QtWidgets.QWidget):
        """Embed a focused plot control beside the title and expand button."""
        self._header_controls.addWidget(control)

    def reset(self, title: str, time_axis=False):
        self._background = None
        self.title.setText(title)
        self.figure.clear()
        # Figure.clear resets subplot parameters. Reapply margins for compact
        # cards so absolute Kelvin ticks and axis labels remain inside canvas.
        self.figure.subplots_adjust(**_CHART_MARGINS)
        self.ax = self.figure.add_subplot(111)
        self.ax.tick_params(labelsize=8, colors="#526a79")
        self.ax.grid(True, color="#e7edf1", linewidth=0.6)
        for spine in self.ax.spines.values():
            spine.set_color("#d2dfe6")
        self.time_axis = time_axis
        self.cursor = None
        self.marker = None
        return self.ax

    def empty(self, title, message):
        ax = self.reset(title)
        ax.text(
            0.5,
            0.5,
            message,
            ha="center",
            va="center",
            transform=ax.transAxes,
            color="#8596a1",
            fontsize=9,
            wrap=True,
        )
        ax.set_axis_off()
        self.canvas.draw_idle()

    def finish(self, current_time: float = 0.0):
        if self.time_axis:
            self.cursor = self.ax.axvline(
                current_time, color="#187e86", linewidth=1, alpha=0.8
            )
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
        if (
            not navigating
            and event.button == 1
            and self.time_axis
            and event.inaxes == self.ax
            and event.xdata is not None
        ):
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


class ChartPanel(QtWidgets.QWidget):
    """Rebuild plots only after selection changes; update artists during playback."""

    notice = QtCore.pyqtSignal(str)

    def __init__(self, session: WorkspaceSession, parent=None):
        super().__init__(parent)
        self.session = session
        self.cards = [ChartCard() for _ in range(3)]
        self.energy_mode = QtWidgets.QComboBox()
        self.energy_mode.addItems(["Change (ΔE)", "Stored total"])
        self.energy_mode.setToolTip(
            "Change since the initial accepted snapshot, or total recoverable energy above the discharge cutoff"
        )
        self.energy_mode.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.energy_mode.setMinimumContentsLength(6)
        self.energy_mode.setMaximumWidth(130)
        self.energy_mode.setEnabled(False)
        self.cards[1].add_header_control(self.energy_mode)
        self.energy_mode.currentIndexChanged.connect(self._energy_mode_changed)
        self.profile_samples = None
        self.thermal_mode = "Probe temperature"
        self.shield_mode = "First-contact outcomes"
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(7)
        for card in self.cards:
            layout.addWidget(card)

    def invalidate(self):
        self.profile_samples = None

    def refresh(self, thermal_mode=None, shield_mode=None):
        if thermal_mode is not None:
            self.thermal_mode = thermal_mode
        if shield_mode is not None:
            self.shield_mode = shield_mode
        self._refresh()

    def update_frame(self, selection):
        result = self.session.active
        if result is None:
            return
        index, probe = self.session.current_index, self.session.probe_index
        current_time = float(result.times[index])
        state_fields = (
            selection.fields
            if self.session.source == "PINN" and selection.fields is not None
            else result.fields
        )
        for card in self.cards:
            if card.cursor is not None:
                card.cursor.set_xdata([current_time, current_time])
            if (
                card.marker is not None
                and "enthalpy" in state_fields
                and "temperature" in state_fields
            ):
                card.marker.set_data(
                    [state_fields["enthalpy"][index, probe] / 1000],
                    [state_fields["temperature"][index, probe]],
                )
            card.update_artists()
        if (
            self.session.view == "Thermal"
            and self.thermal_mode == "Temperature profile"
            and self.profile_samples is not None
        ):
            coordinates, ids = self.profile_samples
            if self.cards[2].ax.collections:
                self.cards[2].ax.collections[0].set_offsets(
                    np.column_stack(
                        (coordinates, result.fields["temperature"][index, ids])
                    )
                )
                self.cards[2].canvas.draw_idle()

    def close_charts(self):
        for card in self.cards:
            if card._dialog is not None:
                card._dialog.close()

    def empty(self):
        for card, title, message in zip(
            self.cards,
            ("Material T(E)", "Energy & phase", "Probe history"),
            (
                "Run a simulation to inspect the material state",
                "Computed energy and phase histories",
                "Right-click a reservoir point after running",
            ),
        ):
            card.empty(title, message)

    def _refresh(self):
        self.energy_mode.setVisible(self.session.view == "Thermal")
        self.energy_mode.setEnabled(self.session.active is not None)
        if self.session.active is None:
            self.empty()
            return
        if self.session.view == "Shielding":
            self._shielding_plots()
            return
        if self.session.view == "PINN":
            self._pinn_plots()
            return
        if self.session.view == "Dynamo":
            self._dynamo_plots()
            return
        if self.session.view == "Compare":
            self._comparison_plots()
            return
        result, config = self.session.active, self.session.active.config
        current = float(result.times[self.session.current_index])
        mask = conductor_mask(result)
        temperatures = result.fields.get("temperature")
        if temperatures is None:
            self.empty()
            return
        extent = max(float(np.ptp(temperatures[:, mask])) * 0.25, 25)
        minimum = min(
            float(np.min(temperatures[:, mask])), config.melting_temperature - extent
        )
        maximum = max(
            float(np.max(temperatures[:, mask])), config.melting_temperature + extent
        )
        enthalpy = np.array(
            [
                config.heat_capacity_solid * (minimum - config.melting_temperature),
                0,
                config.latent_heat,
                config.latent_heat
                + config.heat_capacity_liquid * (maximum - config.melting_temperature),
            ]
        )
        regularized = config.solver == "fenics"
        ax = self.cards[0].reset(
            "Material T(E) · sharp / FE"
            if regularized
            else "Material T(E) · probe state"
        )
        ax.plot(
            enthalpy / 1000,
            [minimum, config.melting_temperature, config.melting_temperature, maximum],
            color="#d27a35",
            linewidth=2,
            label="Sharp reference",
        )
        if regularized:
            from ..physics import regularized_enthalpy

            transition = (
                config.melting_temperature
                + np.linspace(-0.5, 0.5, 161) * config.transition_width
            )
            curve_temperature = np.unique(
                np.r_[
                    np.linspace(minimum, maximum, 241),
                    transition,
                    config.melting_temperature,
                ]
            )
            ax.plot(
                regularized_enthalpy(curve_temperature, config) / 1000,
                curve_temperature,
                color="#167e86",
                linestyle="--",
                linewidth=1.4,
                label=f"FE law ({config.transition_width:g} K transition)",
            )
        ax.axvline(0, color="#aabac4", linewidth=0.8, linestyle=":")
        ax.axvline(
            config.latent_heat / 1000, color="#aabac4", linewidth=0.8, linestyle=":"
        )
        (self.cards[0].marker,) = ax.plot(
            [],
            [],
            "o",
            color="#167e86",
            markersize=6,
            label="Computed probe" if regularized else "Probe",
        )
        ax.set(xlabel="Specific enthalpy [kJ/kg]", ylabel="Temperature [K]")
        if regularized:
            ax.legend(fontsize=6, loc="best")
        self.cards[0].finish()
        self._energy_plot()
        choice = self.thermal_mode
        ax = self.cards[2].reset(choice, choice != "Temperature profile")
        if choice == "Temperature profile":
            probe = result.points[self.session.probe_index]
            phi = np.arctan2(probe[1], probe[0])
            angles = np.arctan2(result.points[:, 1], result.points[:, 0])
            delta = np.abs(np.angle(np.exp(1j * (angles - phi))))
            selected = np.flatnonzero(
                mask & (delta <= max(0.04, np.pi / max(config.resolution * 4, 8)))
            )
            coordinate = (
                np.hypot(result.points[selected, 0], result.points[selected, 1])
                - config.major_radius
            )
            order = np.argsort(coordinate)
            self.profile_samples = (coordinate[order], selected[order])
            ax.scatter(
                coordinate[order],
                temperatures[self.session.current_index, selected[order]],
                s=6,
                color="#d27a35",
                label="Cross-section samples",
            )
            ax.set(xlabel="Offset from torus centerline [m]", ylabel="Temperature [K]")
            low, high = (
                float(np.min(temperatures[:, mask])),
                float(np.max(temperatures[:, mask])),
            )
            ax.set_ylim(
                low - max(1.0, (high - low) * 0.05),
                high + max(1.0, (high - low) * 0.05),
            )
        elif choice == "Liquid fraction history":
            values = result.fields.get("liquid_fraction")
            if values is not None:
                ax.plot(
                    result.times,
                    values[:, self.session.probe_index],
                    color="#187e86",
                    label="Probe",
                )
            if "liquid_fraction" in result.diagnostics:
                ax.plot(
                    result.times,
                    result.diagnostics["liquid_fraction"],
                    color="#d27a35",
                    label="Volume mean",
                )
            ax.set(xlabel="Time [s]", ylabel="Liquid fraction", ylim=(-0.02, 1.02))
        elif choice == "Thermal power history":
            for key, label, color in (
                ("net_power_w", "Net thermal input", "#187e86"),
                ("pump_power_w", "Pumping input", "#d27a35"),
            ):
                if key in result.diagnostics:
                    ax.plot(
                        result.times, result.diagnostics[key], color=color, label=label
                    )
            ax.set(xlabel="Time [s]", ylabel="Power [W]")
        else:
            probe_temperature = temperatures[:, self.session.probe_index]
            histories = [probe_temperature]
            ax.plot(
                result.times,
                probe_temperature,
                color="#d27a35",
                linewidth=1.8,
                label="Numerical",
            )
            if (
                self.session.pinn is not None
                and "temperature" in self.session.pinn.fields
            ):
                pinn_temperature = self.session.pinn.fields["temperature"][
                    :, self.session.probe_index
                ]
                histories.append(pinn_temperature)
                ax.plot(
                    result.times,
                    pinn_temperature,
                    color="#187e86",
                    linestyle="--",
                    label="PINN",
                )
            low = min(float(np.min(values)) for values in histories)
            high = max(float(np.max(values)) for values in histories)
            padding = max(0.1 * (high - low), 0.005)
            low, high = low - padding, high + padding
            ax.set_ylim(low, high)
            ax.ticklabel_format(axis="y", style="plain", useOffset=False)
            notes = []
            if low <= config.melting_temperature <= high:
                ax.axhline(
                    config.melting_temperature,
                    color="#a2b4bf",
                    linestyle=":",
                    linewidth=1,
                    label="Melting point",
                )
            else:
                direction = "above" if config.melting_temperature > high else "below"
                notes.append(
                    f"Tm = {config.melting_temperature:g} K ({direction} range)"
                )
            notes.append(
                f"Probe ΔT = {probe_temperature[-1] - probe_temperature[0]:+.6g} K"
            )
            ax.text(
                0.02,
                0.98,
                "\n".join(notes),
                transform=ax.transAxes,
                va="top",
                fontsize=6.2,
                color="#526a79",
                bbox={
                    "facecolor": "white",
                    "alpha": 0.85,
                    "edgecolor": "none",
                    "pad": 1.5,
                },
            )
            ax.set(xlabel="Time [s]", ylabel="Temperature [K]")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7, loc="best")
        self.cards[2].finish(current)

    def _energy_mode_changed(self):
        if self.session.active is not None and self.session.view == "Thermal":
            self._energy_plot()
            self.update_frame(self.session.display_selection())

    def _energy_plot(self):
        result = self.session.active
        if result is None:
            return
        card = self.cards[1]
        recoverable = result.diagnostics.get("recoverable_energy_j")
        energy = (
            recoverable
            if recoverable is not None
            else result.diagnostics.get("energy_j")
        )
        if energy is None:
            card.empty("Thermal energy", "This run has no thermal energy diagnostic")
            return
        config = result.config
        change = self.energy_mode.currentIndex() == 0
        title = (
            ("Energy change" if change else "Stored energy")
            if recoverable is not None
            else ("Enthalpy change" if change else "Enthalpy total")
        )
        ax = card.reset(title, True)
        values = (energy - energy[0]) / 3.6e6 if change else energy / 3.6e6
        ax.plot(
            result.times,
            values,
            color="#187e86",
            linewidth=1.8,
            label=("Stored ΔE" if change else "Stored")
            if recoverable is not None
            else "Enthalpy",
        )
        ax.plot(
            result.times,
            values * config.efficiency,
            color="#d89e39",
            linewidth=1.4,
            label=f"{'Usable' if recoverable is not None else 'Scaled'} at η={config.efficiency:g}",
        )
        initial = float(energy[0]) / 3.6e6
        net_change = float(energy[-1] - energy[0]) / 3.6e6
        initial_text = (
            f"Initial: {initial:.6g} kWh stored; {initial * config.efficiency:.6g} kWh usable"
            if recoverable is not None
            else f"Initial enthalpy integral: {initial:.6g} kWh"
        )
        reference = (
            f"Discharge reference: {config.discharge_temperature:g} K"
            if recoverable is not None
            else f"Enthalpy reference: Tm = {config.melting_temperature:g} K"
        )
        ax.text(
            0.02,
            0.98,
            f"{initial_text}\n{reference}\nNet ΔE: {net_change:+.6g} kWh",
            transform=ax.transAxes,
            va="top",
            fontsize=6.2,
            color="#526a79",
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none", "pad": 1.5},
        )
        ax.margins(y=0.2)
        ax.legend(fontsize=6, loc="lower right")
        ax.set(xlabel="Time [s]", ylabel="ΔEnergy [kWh]" if change else "Energy [kWh]")
        card.finish(self.session.current_time)

    def _dynamo_plots(self):
        result = self.session.active
        if result is None:
            return
        current = float(result.times[self.session.current_index])
        for card, key, title, unit, color in (
            (
                self.cards[0],
                "magnetic_energy_j",
                "Magnetic energy · growth/decay",
                "J",
                "#187e86",
            ),
            (self.cards[2], "pump_power_w", "Pumping power", "W", "#d27a35"),
        ):
            values = result.diagnostics.get(key)
            if values is None:
                card.empty(title, "This field is not computed by the selected backend")
            else:
                ax = card.reset(title, True)
                ax.plot(result.times, values, color=color, linewidth=1.8)
                ax.set(xlabel="Time [s]", ylabel=f"[{unit}]")
                card.finish(current)
        if "magnetic_field" not in result.fields:
            self.cards[1].empty(
                "Magnetic components", "Run a magnetic solver to inspect B"
            )
            return
        ax = self.cards[1].reset("Magnetic field · selected probe", True)
        values = result.fields["magnetic_field"][:, self.session.probe_index]
        point = result.points[self.session.probe_index]
        angle = np.arctan2(point[1], point[0])
        toroidal = -np.sin(angle) * values[:, 0] + np.cos(angle) * values[:, 1]
        poloidal = np.sqrt(np.maximum(np.sum(values**2, axis=1) - toroidal**2, 0))
        ax.plot(result.times, toroidal, label="Toroidal", color="#187e86")
        ax.plot(result.times, poloidal, label="Poloidal magnitude", color="#ce813b")
        ax.set(xlabel="Time [s]", ylabel="Magnetic flux density [T]")
        ax.legend(fontsize=7)
        self.cards[1].finish(current)

    def _shielding_plots(self):
        if self.session.active is None:
            return
        if self.session.trajectories is None:
            self.cards[0].empty(
                "Paired transmission", "Trace rays to compare field-on/off trajectories"
            )
            self.cards[2].empty(
                "Incident primary energy",
                "Supports proton, alpha and heavier-ion inputs",
            )
        else:
            names = ["habitat", "hardware", "ground", "escaped", "window_end"]
            outcomes = self.session.trajectories["outcomes"]
            baseline = self.session.trajectories["baseline_outcomes"]
            ax = self.cards[0].reset("First-contact outcomes · paired primaries")
            locations = np.arange(len(names))
            ax.bar(
                locations - 0.18,
                [sum(x == name for x in baseline) for name in names],
                0.35,
                label="Field off",
                color="#9caeb9",
            )
            ax.bar(
                locations + 0.18,
                [sum(x == name for x in outcomes) for name in names],
                0.35,
                label="Field on",
                color="#228e95",
            )
            ax.set_xticks(locations, names)
            ax.set_ylabel("Primary particles")
            ax.legend(fontsize=7)
            self.cards[0].finish()
            ax = self.cards[2].reset("Incident spectrum · sampled primaries")
            energy = np.asarray(self.session.trajectories["energies_mev_nucleon"])
            species = np.asarray(self.session.trajectories["species"])
            for za, color in zip(
                np.unique(species, axis=0),
                cycle(
                    ("#d29348", "#187e86", "#9c679d", "#4477aa", "#cc6677", "#228833")
                ),
            ):
                selected = np.all(species == za, axis=1)
                ax.hist(
                    energy[selected],
                    bins=min(20, max(3, int(np.sqrt(len(energy))))),
                    color=color,
                    alpha=0.6,
                    label=f"Z={za[0]}, A={za[1]}",
                )
            ax.set(xlabel="Energy [MeV/nucleon]", ylabel="Sampled primaries")
            ax.legend(fontsize=6)
            self.cards[2].finish()
            if self.shield_mode == "Transmission vs energy":
                ax = self.cards[0].reset(
                    "Habitat first-contact fraction · trajectory diagnostic"
                )
                edges = np.geomspace(
                    max(float(energy.min()) * 0.9, 1e-8), float(energy.max()) * 1.1, 7
                )
                for za, color in zip(
                    np.unique(species, axis=0),
                    cycle(
                        (
                            "#d29348",
                            "#187e86",
                            "#9c679d",
                            "#4477aa",
                            "#cc6677",
                            "#228833",
                        )
                    ),
                ):
                    selected = np.all(species == za, axis=1)
                    counts, _ = np.histogram(energy[selected], bins=edges)
                    for values, style, label in (
                        (outcomes, "-", "on"),
                        (baseline, "--", "off"),
                    ):
                        hits = selected & (np.asarray(values) == "habitat")
                        transmitted, _ = np.histogram(energy[hits], bins=edges)
                        centers = np.sqrt(edges[:-1] * edges[1:])
                        available = counts > 0
                        ax.plot(
                            centers[available],
                            transmitted[available] / counts[available],
                            style,
                            marker="o",
                            markersize=3,
                            color=color,
                            label=f"Z={za[0]} A={za[1]} {label}",
                        )
                ax.set(
                    xscale="log",
                    xlabel="Energy [MeV/nucleon]",
                    ylabel="Habitat contacts / incident primaries",
                    ylim=(-0.03, 1.03),
                )
                ax.legend(fontsize=6)
                self.cards[0].finish()
        if self.session.dose is None:
            self.cards[1].empty(
                "Absorbed dose & uncertainty",
                "Run Geant4 for water or anatomical dose\nTrajectories alone do not determine dose",
            )
        else:
            snapshot_time = float(
                self.session.active.times[self.session.dose.snapshot_index]
            )
            ax = self.cards[1].reset(
                f"{self.session.dose.target.title()} · t={snapshot_time:.3g} s · Φ={self.session.active.config.incident_fluence_m2:g}/m²"
            )
            order = np.argsort(self.session.dose.dose_gy)[-10:]
            positions = np.arange(len(order))
            if (
                "field_off_dose_gy" in self.session.dose.metadata
                and "field_off_standard_error_gy" in self.session.dose.metadata
            ):
                baseline = np.asarray(self.session.dose.metadata["field_off_dose_gy"])
                baseline_error = np.asarray(
                    self.session.dose.metadata["field_off_standard_error_gy"]
                )
                ax.barh(
                    positions - 0.18,
                    baseline[order],
                    height=0.35,
                    xerr=baseline_error[order],
                    color="#9caeb9",
                    capsize=2,
                    label="Field off",
                )
            ax.barh(
                positions + 0.18,
                self.session.dose.dose_gy[order],
                height=0.35,
                xerr=self.session.dose.standard_error_gy[order],
                color="#9c679d",
                capsize=2,
                label="Field on",
            )
            ax.set_yticks(positions, [self.session.dose.organ_names[i] for i in order])
            ax.set_xlabel("Absorbed dose [Gy] ± standard error")
            ax.legend(fontsize=6)
            self.cards[1].finish()
            if self.shield_mode == "Secondary dose":
                if (
                    "secondary_dose_gy" not in self.session.dose.metadata
                    or "secondary_standard_error_gy" not in self.session.dose.metadata
                ):
                    self.cards[0].empty(
                        "Secondary contributions",
                        "This saved result has no secondary dose tally",
                    )
                    return
                ax = self.cards[0].reset("Secondary contributions · field on")
                values = np.asarray(self.session.dose.metadata["secondary_dose_gy"])
                error = np.asarray(
                    self.session.dose.metadata["secondary_standard_error_gy"]
                )
                ax.barh(
                    positions,
                    values[order],
                    xerr=error[order],
                    color="#cc6677",
                    capsize=2,
                )
                ax.set_yticks(
                    positions, [self.session.dose.organ_names[i] for i in order]
                )
                ax.set_xlabel("Absorbed dose from secondary tracks [Gy]")
                self.cards[0].finish()

    def _pinn_plots(self):
        if self.session.active is None:
            return
        if self.session.pinn is None:
            for card, title in zip(
                self.cards,
                ("Training convergence", "Held-out validation", "Numerical / PINN"),
            ):
                card.empty(title, "Train a PINN for the active saved scenario")
            return
        ax = self.cards[0].reset("Training convergence")
        for key, values in self.session.pinn.history.items():
            values = np.asarray(values)
            if (
                values.ndim == 1
                and len(values)
                and key not in {"epoch", "epochs", "time", "elapsed_seconds"}
            ):
                ax.plot(
                    np.arange(1, len(values) + 1), np.maximum(values, 1e-16), label=key
                )
        ax.set(yscale="log", xlabel="Training epoch", ylabel="Nondimensional loss")
        ax.legend(fontsize=6)
        self.cards[0].finish()
        status = self.session.pinn.metadata.get("validation_status", "unvalidated")
        ax = self.cards[1].reset(f"Held-out fields · {status}")
        metrics = [
            (key, value)
            for key, value in self.session.pinn.metrics.items()
            if np.isfinite(value) and key.endswith("_normalized_rmse")
        ][:8]
        if metrics:
            ax.barh(
                np.arange(len(metrics)),
                [value for key, value in metrics],
                color="#258d95",
            )
            ax.axvline(
                0.1, color="#c87e37", linestyle=":", linewidth=1, label="Field gate"
            )
            ax.set_yticks(
                np.arange(len(metrics)),
                [
                    key.replace("_normalized_rmse", "").replace("_", " ")
                    for key, value in metrics
                ],
            )
            ax.set_xlabel("Held-out normalized RMSE [dimensionless]")
            ax.legend(fontsize=6)
        self.cards[1].finish()
        ax = self.cards[2].reset("Temperature · held scenario", True)
        for fields_, label, color, style in (
            (self.session.active.fields, "Numerical", "#d28740", "-"),
            (self.session.pinn.fields, "PINN", "#187e86", "--"),
        ):
            if "temperature" in fields_:
                ax.plot(
                    self.session.active.times,
                    fields_["temperature"][:, self.session.probe_index],
                    color=color,
                    linestyle=style,
                    label=label,
                )
        ax.set(xlabel="Time [s]", ylabel="Temperature [K]")
        ax.legend(fontsize=7)
        self.cards[2].finish(
            float(self.session.active.times[self.session.current_index])
        )

    def _comparison_plots(self):
        if self.session.active is None:
            return
        if self.session.comparison is None:
            for card in self.cards:
                card.empty(
                    "Scenario comparison", "Select a second saved run in the sidebar"
                )
            return
        for card, key, title, unit, factor in (
            (
                self.cards[0],
                "recoverable_energy_j",
                "Recoverable thermal energy",
                "kWh",
                3.6e6,
            ),
            (self.cards[1], "liquid_fraction", "Mean liquid fraction", "fraction", 1),
            (self.cards[2], "magnetic_energy_j", "Magnetic energy", "J", 1),
        ):
            ax = card.reset(title, True)
            for result, color in (
                (self.session.active, "#187e86"),
                (self.session.comparison, "#d58b40"),
            ):
                values = result.diagnostics.get(key)
                if values is not None:
                    ax.plot(
                        result.times,
                        values / factor,
                        label=result.config.name,
                        color=color,
                    )
            ax.set(xlabel="Physical time [s]", ylabel=unit)
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(fontsize=6)
            card.finish(float(self.session.active.times[self.session.current_index]))
        differences = [
            key.replace("_", " ")
            for key, value in self.session.active.config.to_dict().items()
            if value != self.session.comparison.config.to_dict().get(key)
            and key not in {"name", "preset"}
        ]
        self.notice.emit(
            "Compared changes: "
            + (", ".join(differences[:10]) or "identical settings")
            + " · colors and camera are shared"
        )
