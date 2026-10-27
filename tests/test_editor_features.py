"""Scenario form coverage using real Qt editors without a renderer."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from dataclasses import replace
import tempfile
from pathlib import Path
import unittest

from PyQt6 import QtTest, QtWidgets

from magnetic_torus.models import preset
from magnetic_torus.ui.editor import ScenarioEditor

SUITE_CAPABILITIES = ("desktop",)


class EditorFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.editor = ScenarioEditor()
        self.editor.set_config(preset("Iron"))
        self.editor.show()
        self.addCleanup(self.editor.close)
        self.addCleanup(self.editor._draft_timer.stop)

    def test_all_presets_roundtrip_every_exposed_and_hidden_configuration_field(self):
        for name in (
            "Thermal benchmark",
            "Iron",
            "Regolith composite",
            "Lunar habitat",
        ):
            with self.subTest(name=name):
                original = preset(name)
                self.editor.set_config(original)
                self.assertEqual(
                    self.editor.read_config().to_dict(), original.to_dict()
                )

    def test_custom_float_configuration_roundtrips_without_precision_or_hash_loss(self):
        original = replace(
            preset("Iron"),
            preset="Custom",
            name="High precision Δ",
            duration=1.2345678901234567,
            heat_capacity_solid=449.123456789,
            electrical_conductivity_solid=1000000.123456789,
        )
        self.editor.set_config(original)
        self.assertEqual(self.editor.read_config(), original)
        self.assertEqual(self.editor.read_config().config_hash, original.config_hash)

    def test_real_edits_create_custom_and_preserve_material_composition(self):
        original = preset("Regolith composite")
        self.editor.set_config(original)
        field = self.editor.editors["duration"]
        field.selectAll()
        QtTest.QTest.keyClicks(field, "12.5")
        self.assertEqual(self.editor.preset_combo.currentText(), "Custom")
        edited = self.editor.read_config()
        self.assertEqual(edited.duration, 12.5)
        self.assertEqual(edited.composition, original.composition)
        self.assertEqual(edited.provenance, original.provenance)
        self.assertEqual(original.duration, 5.0)

    def test_dropdown_and_checkbox_inputs_update_the_same_snapshot(self):
        self.editor.editors["mode"].setCurrentText("dynamo")
        self.editor.editors["turbulence"].setChecked(False)
        self.editor.editors["ground_enabled"].setChecked(True)
        result = self.editor.read_config()
        self.assertEqual(result.mode, "dynamo")
        self.assertFalse(result.turbulence)
        self.assertTrue(result.ground_enabled)
        self.assertEqual(result.preset, "Custom")

    def test_invalid_numeric_text_and_geometry_fail_before_a_job_can_start(self):
        for key, value in (
            ("density", "0"),
            ("density", "-1"),
            ("duration", "nan"),
            ("duration", "inf"),
            ("duration", "words"),
            ("time_steps", "2.5"),
            ("efficiency", "1.01"),
            ("charge_fraction", "-.1"),
            ("minor_radius", "2"),
        ):
            with self.subTest(key=key, value=value):
                self.editor.set_config(preset("Iron"))
                invalid = QtTest.QSignalSpy(self.editor.invalid)
                self.editor.editors[key].setText(value)
                self.assertIsNone(self.editor.validate())
                self.assertEqual(len(invalid), 1)
                with self.assertRaises(ValueError):
                    self.editor.read_config()

    def test_allowed_fraction_and_latent_heat_boundaries_are_editable(self):
        for key in ("efficiency", "emissivity", "charge_fraction"):
            for value in (0.0, 1.0):
                with self.subTest(key=key, value=value):
                    self.editor.set_config(preset("Iron"))
                    self.editor.editors[key].setText(str(value))
                    self.assertEqual(getattr(self.editor.read_config(), key), value)
        self.editor.editors["latent_heat"].setText("0")
        self.assertEqual(self.editor.read_config().latent_heat, 0)

    def test_unicode_space_paths_and_browse_cancellation_preserve_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "spectrum Δ with spaces.json"
            path.write_text("{}", encoding="utf-8")
            self.editor.editors["spectrum_path"].setText(str(path))
            self.assertEqual(self.editor.read_config().spectrum_path, str(path))
            from unittest.mock import patch

            with patch(
                "magnetic_torus.ui.editor.QtWidgets.QFileDialog.getOpenFileName",
                return_value=("", ""),
            ):
                self.editor._browse_input("spectrum_path", False)
            self.assertEqual(self.editor.read_config().spectrum_path, str(path))

    def test_multiple_keystrokes_emit_one_debounced_valid_configuration(self):
        changed = QtTest.QSignalSpy(self.editor.changed)
        field = self.editor.editors["duration"]
        field.selectAll()
        QtTest.QTest.keyClicks(field, "12.5")
        self.assertEqual(len(changed), 0)
        QtTest.QTest.qWait(220)
        self.assertEqual(len(changed), 1)
        self.assertEqual(changed[0][0].duration, 12.5)


if __name__ == "__main__":
    unittest.main()
