"""Real Qt/Matplotlib checks for small accepted thermal changes and references.

These synthetic accepted-field fixtures verify plotting, not physical solvers.
"""

from dataclasses import replace
import os
from pathlib import Path
from types import SimpleNamespace
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt6 import QtWidgets

from magnetic_torus.models import PINNResult, RunResult, ScenarioConfig
from magnetic_torus.ui.charts import ChartCard, ChartPanel
from magnetic_torus.ui.session import WorkspaceSession


def iron_style_result():
    config = replace(ScenarioConfig(), duration=5.0)
    times = np.array([0.0, 3.25, 5.0])
    angle = np.arange(4) * np.pi / 2
    points = np.column_stack((np.cos(angle), np.sin(angle), np.zeros(4)))
    temperature = np.broadcast_to(
        np.array([1800.0, 1800.027721, 1800.027721])[:, None], (3, 4)
    ).copy()
    fields = {
        "temperature": temperature,
        "enthalpy": config.heat_capacity_solid
        * (temperature - config.melting_temperature),
        "liquid_fraction": np.zeros_like(temperature),
    }
    energy = 234.0e6 + np.array([0.0, 65000.0, 65000.0])
    return RunResult(
        "thermal-plot-fixture",
        config,
        times,
        points,
        fields,
        {"recoverable_energy_j": energy},
        {"backend": "Synthetic plot fixture"},
    )


class ThermalChartTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = QtWidgets.QApplication.instance() or QtWidgets.QApplication(
            []
        )

    def setUp(self):
        self.result = iron_style_result()
        self.session = WorkspaceSession()
        self.session.activate(self.result, Path("plot-fixture.h5"))
        self.panel = ChartPanel(self.session)
        self.panel.resize(950, 260)
        self.panel.show()
        self.panel.refresh()
        self.application.processEvents()
        self.addCleanup(self.close_panel)

    def close_panel(self):
        self.panel.close_charts()
        self.panel.close()
        self.application.processEvents()

    def test_default_energy_change_exposes_small_input_and_labels_initial_reference(
        self,
    ):
        ax = self.panel.cards[1].ax
        energy = self.result.diagnostics["recoverable_energy_j"]
        expected = (energy - energy[0]) / 3.6e6
        np.testing.assert_allclose(ax.lines[0].get_ydata(), expected)
        np.testing.assert_allclose(
            ax.lines[1].get_ydata(), expected * self.result.config.efficiency
        )
        self.assertEqual(ax.get_ylabel(), "ΔEnergy [kWh]")
        self.assertLess(np.ptp(ax.get_ylim()), 0.1)
        annotation = "\n".join(text.get_text() for text in ax.texts)
        self.assertIn("65 kWh stored", annotation)
        self.assertIn("19.5 kWh usable", annotation)
        self.assertIn("Discharge reference: 1700 K", annotation)
        self.assertIn("Net ΔE: +0.0180556 kWh", annotation)
        np.testing.assert_array_equal(
            energy, 234.0e6 + np.array([0.0, 65000.0, 65000.0])
        )

    def test_stored_total_keeps_absolute_data_and_switching_retains_manual_change_zoom(
        self,
    ):
        change_card = self.panel.cards[1]
        change_card.ax.set_xlim(1.0, 4.0)
        change_card.ax.set_ylim(-0.001, 0.01)
        change_card._remember_view(SimpleNamespace(inaxes=change_card.ax))
        self.panel.energy_mode.setCurrentIndex(1)
        ax = self.panel.cards[1].ax
        expected = self.result.diagnostics["recoverable_energy_j"] / 3.6e6
        np.testing.assert_allclose(ax.lines[0].get_ydata(), expected)
        np.testing.assert_allclose(ax.lines[1].get_ydata(), expected * 0.3)
        self.assertEqual(ax.get_ylabel(), "Energy [kWh]")
        self.assertIn("Net ΔE", ax.texts[0].get_text())
        self.panel.energy_mode.setCurrentIndex(0)
        np.testing.assert_allclose(change_card.ax.get_xlim(), (1.0, 4.0))
        np.testing.assert_allclose(change_card.ax.get_ylim(), (-0.001, 0.01))

    def test_probe_fits_small_temperature_change_without_offscale_melting_line(self):
        ax = self.panel.cards[2].ax
        expected = self.result.fields["temperature"][:, self.session.probe_index]
        np.testing.assert_array_equal(ax.lines[0].get_ydata(), expected)
        low, high = ax.get_ylim()
        self.assertLess(low, 1800.0)
        self.assertGreater(high, 1800.027721)
        self.assertLess(high - low, 0.1)
        self.assertFalse(ax.yaxis.get_major_formatter().get_useOffset())
        annotation = "\n".join(text.get_text() for text in ax.texts)
        self.assertIn("Tm = 1810 K (above range)", annotation)
        self.assertIn("Probe ΔT = +0.027721 K", annotation)
        self.assertNotIn("Melting point", [line.get_label() for line in ax.lines])

    def test_probe_range_includes_pinn_and_draws_inrange_melting_reference(self):
        fields = dict(self.result.fields)
        fields["temperature"] = self.result.fields["temperature"] + 20.0
        self.session.set_pinn(
            PINNResult(
                "plot-pinn",
                self.result.run_id,
                self.result.config.config_hash,
                fields,
                {},
                {},
                {"validation_status": "synthetic fixture"},
            )
        )
        self.panel.refresh()
        ax = self.panel.cards[2].ax
        low, high = ax.get_ylim()
        self.assertLess(low, 1800.0)
        self.assertGreater(high, 1820.027721)
        self.assertIn("PINN", [line.get_label() for line in ax.lines])
        self.assertIn("Melting point", [line.get_label() for line in ax.lines])

    def test_energy_selector_is_hidden_outside_thermal_and_playback_keeps_axes(self):
        for view in ("Dynamo", "Shielding", "PINN", "Compare"):
            self.session.view = view
            self.panel.refresh()
            self.assertTrue(self.panel.energy_mode.isHidden())
        self.session.view = "Thermal"
        self.panel.refresh()
        self.assertFalse(self.panel.energy_mode.isHidden())
        axes = [card.ax for card in self.panel.cards]
        self.session.current_index = 2
        self.panel.update_frame(self.session.display_selection())
        self.assertEqual([card.ax for card in self.panel.cards], axes)
        np.testing.assert_allclose(self.panel.cards[1].cursor.get_xdata(), [5.0, 5.0])

    def test_enthalpy_fallback_does_not_claim_discharge_reference_or_recoverable_energy(
        self,
    ):
        fallback = replace(
            self.result, diagnostics={"energy_j": np.array([-1000.0, 0.0, 1000.0])}
        )
        self.session.activate(fallback, Path("enthalpy-fixture.h5"))
        self.panel.refresh()
        text = self.panel.cards[1].ax.texts[0].get_text()
        self.assertIn("Enthalpy reference: Tm = 1810 K", text)
        self.assertNotIn("Discharge reference", text)
        self.assertNotIn("kWh usable", text)

    def test_small_card_reset_keeps_full_kelvin_ticks_and_axis_labels_inside_figure(
        self,
    ):
        card = ChartCard()
        self.addCleanup(card.close)
        card.resize(330, 150)
        card.show()
        card.reset("Previous plot")
        ax = card.reset("Small temperature change", True)
        ax.plot(self.result.times, self.result.fields["temperature"][:, 0])
        ax.set(
            xlabel="Time [s]", ylabel="Temperature [K]", ylim=(1799.995, 1800.032721)
        )
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        card.finish()
        self.application.processEvents()
        card.canvas.draw()
        renderer = card.canvas.get_renderer()
        figure = card.figure.bbox
        labels = [ax.xaxis.label, ax.yaxis.label]
        for axis, limits in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
            labels.extend(
                tick.label1
                for tick in axis.get_major_ticks()
                if tick.label1.get_visible()
                and limits[0] <= tick.get_loc() <= limits[1]
            )
        for label in labels:
            with self.subTest(label=label.get_text()):
                bounds = label.get_window_extent(renderer)
                self.assertGreaterEqual(bounds.x0, figure.x0)
                self.assertGreaterEqual(bounds.y0, figure.y0)
                self.assertLessEqual(bounds.x1, figure.x1)
                self.assertLessEqual(bounds.y1, figure.y1)
        self.assertTrue(any(label.get_text() == "1800.02" for label in labels))

    def test_small_energy_card_keeps_three_line_reference_annotation_inside_figure(
        self,
    ):
        energy = 65.1327 * 3.6e6 + np.array([0.0, 65000.0, 65000.0])
        self.session.activate(
            replace(self.result, diagnostics={"recoverable_energy_j": energy}),
            Path("energy-reference.h5"),
        )
        self.panel.refresh()
        card = self.panel.cards[1]
        card.setFixedSize(330, 150)
        self.application.processEvents()
        card.canvas.draw()
        annotation = card.ax.texts[0]
        self.assertEqual(len(annotation.get_text().splitlines()), 3)
        bounds = annotation.get_window_extent(card.canvas.get_renderer())
        figure = card.figure.bbox
        self.assertGreaterEqual(bounds.x0, figure.x0)
        self.assertGreaterEqual(bounds.y0, figure.y0)
        self.assertLessEqual(bounds.x1, figure.x1)
        self.assertLessEqual(bounds.y1, figure.y1)


if __name__ == "__main__":
    unittest.main()
