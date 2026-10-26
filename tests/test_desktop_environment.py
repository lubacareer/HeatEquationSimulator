"""Graphics provenance and cache identity without importing graphics runtimes."""

from __future__ import annotations

import ast
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from simulation_suite.capabilities import _DESKTOP_PROBE, desktop_environment
from simulation_suite.runner import case_fingerprint
from simulation_suite.spec import CaseSpec


class DesktopEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.geometry = [40, 50, 1280, 800]
        self.surface = SimpleNamespace(
            geometry=lambda: SimpleNamespace(getRect=lambda: tuple(self.geometry)),
            name=lambda: "Physical panel",
            devicePixelRatio=lambda: 2.0,
            refreshRate=lambda: 164.999999,
        )
        self.app = SimpleNamespace(
            primaryScreen=lambda: self.surface, platformName=lambda: "windows"
        )
        self.visible, self.minimized, self.active = True, False, True
        self.window = SimpleNamespace(
            screen=lambda: self.surface,
            isVisible=lambda: self.visible,
            isMinimized=lambda: self.minimized,
            isActiveWindow=lambda: self.active,
        )
        self.capabilities = (
            "OpenGL vendor string:  Test vendor\n"
            "OpenGL renderer string:  Integrated GPU\n"
            "OpenGL version string:  4.6 Driver A\n"
            "OpenGL extensions: lots of irrelevant extensions\n"
        )
        self.plotter = SimpleNamespace(
            ren_win=SimpleNamespace(
                GetClassName=lambda: "vtkGenericOpenGLRenderWindow",
                ReportCapabilities=lambda: self.capabilities,
            ),
            window_size=[2012, 668],
        )

    def environment(self, **kwargs):
        return desktop_environment(self.app, self.window, self.plotter, **kwargs)

    def test_records_actual_context_screen_geometry_and_pixel_units(self):
        details = self.environment(include_window_state=True)
        self.assertEqual(details["opengl_vendor"], "Test vendor")
        self.assertEqual(details["opengl_renderer"], "Integrated GPU")
        self.assertEqual(details["opengl_version"], "4.6 Driver A")
        self.assertEqual(
            details["vtk_render_window_class"], "vtkGenericOpenGLRenderWindow"
        )
        self.assertEqual(details["qt_platform"], "windows")
        self.assertEqual(details["screen_name"], "Physical panel")
        self.assertEqual(details["screen_geometry"], [40, 50, 1280, 800])
        self.assertEqual(details["screen"], [1280, 800])
        self.assertEqual(details["device_pixel_ratio"], 2.0)
        self.assertEqual(details["refresh_rate_hz"], 165.0)
        self.assertEqual(details["render_size_px"], [2012, 668])
        self.assertEqual(
            details["window_state"],
            {"visible": True, "minimized": False, "active": True},
        )
        self.assertNotIn("extensions", details)

    def test_uses_actual_window_screen_and_reports_missing_context_data(self):
        primary = self.surface
        other = SimpleNamespace(
            geometry=lambda: SimpleNamespace(getRect=lambda: (-1920, 0, 1920, 1080)),
            name=lambda: "Second monitor",
            devicePixelRatio=lambda: 1.0,
            refreshRate=lambda: 60.0,
        )
        self.window.screen = lambda: other
        details = self.environment()
        self.assertEqual(details["screen_name"], "Second monitor")
        self.assertEqual(details["screen_geometry"], [-1920, 0, 1920, 1080])
        self.window.screen = lambda: None
        self.assertEqual(self.environment()["screen_name"], primary.name())
        self.app.primaryScreen = lambda: None
        self.capabilities = "Not implemented by this rendering backend"
        details = self.environment()
        self.assertIsNone(details["screen_geometry"])
        self.assertIsNone(details["refresh_rate_hz"])
        self.assertEqual(details["opengl_renderer"], "unavailable")

    def test_probe_metadata_excludes_focus_and_per_window_pixel_size(self):
        before = self.environment()
        self.visible, self.minimized, self.active = False, True, False
        self.plotter.window_size = [800, 400]
        self.assertEqual(self.environment(), before)
        self.assertNotIn("render_size_px", before)
        self.assertNotIn("window_state", before)
        diagnostic = self.environment(include_window_state=True)
        self.assertEqual(
            diagnostic["window_state"],
            {"visible": False, "minimized": True, "active": False},
        )
        self.assertEqual(diagnostic["render_size_px"], [800, 400])

    def test_isolated_probe_uses_identical_metadata_without_shared_imports(self):
        tree = ast.parse(_DESKTOP_PROBE)
        helper = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "desktop_environment"
        )
        namespace = {}
        exec(
            compile(
                ast.Module(body=[helper], type_ignores=[]),
                "<desktop environment>",
                "exec",
            ),
            namespace,
        )
        self.assertEqual(
            namespace["desktop_environment"](self.app, self.window, self.plotter),
            self.environment(),
        )
        compile(_DESKTOP_PROBE, "<isolated desktop probe>", "exec")

    def test_gpu_driver_or_display_changes_invalidate_desktop_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            case = CaseSpec(
                "desktop.fixture",
                "desktop",
                "fixture",
                ("{python}",),
                capabilities=("desktop",),
            )
            details = self.environment()

            def fingerprint(value):
                return case_fingerprint(
                    case,
                    Path(temporary),
                    {"desktop": {"available": True, "versions": {}, "details": value}},
                )

            before = fingerprint(details)
            for change in (
                {"opengl_renderer": "Dedicated GPU"},
                {"opengl_version": "4.6 Driver B"},
                {"screen_name": "Remote display"},
                {"screen_geometry": [0, 0, 1920, 1080]},
                {"device_pixel_ratio": 1.0},
                {"refresh_rate_hz": 60.0},
            ):
                with self.subTest(change=change):
                    self.assertNotEqual(fingerprint({**details, **change}), before)
            self.assertEqual(fingerprint(copy.deepcopy(details)), before)


if __name__ == "__main__":
    unittest.main()
