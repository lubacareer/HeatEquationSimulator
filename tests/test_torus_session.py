"""Workspace identity, display-source and physical-time behavior without Qt."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch
import unittest

import numpy as np

from magnetic_torus.models import DoseResult, PINNResult, RunResult, ScenarioConfig
from magnetic_torus.ui.session import WorkspaceSession


def result(run_id="run", times=(0.0, 1.0, 2.0), count=4):
    config = replace(ScenarioConfig(), solver="thermal_benchmark", turbulence=False)
    angle = np.arange(count) * 2 * np.pi / count
    points = np.column_stack(
        (
            config.major_radius * np.cos(angle),
            config.major_radius * np.sin(angle),
            np.zeros(count),
        )
    )
    temperature = 1800.0 + np.asarray(times)[:, None] + np.arange(count)[None, :]
    return RunResult(
        run_id, config, np.asarray(times), points, {"temperature": temperature}, {}, {}
    )


class WorkspaceSessionTests(unittest.TestCase):
    def setUp(self):
        self.session = WorkspaceSession()
        self.run = result()
        self.session.activate(self.run, Path("first.h5"))

    def test_new_mesh_retains_physical_time_and_nearest_probe_and_isolates_stale_job(
        self,
    ):
        self.session.current_index = 1
        self.session.probe_index = 3
        self.session.job_revision = self.session.selection_revision
        replacement = result("smaller", times=(0.0, 0.8, 1.7), count=2)
        self.session.activate(replacement, Path("smaller.h5"))
        self.assertEqual(self.session.current_time, 0.8)
        self.assertLess(self.session.probe_index, 2)
        self.assertFalse(self.session.job_is_current)
        self.assertIs(self.session.active, replacement)

    def test_replacement_model_invalidates_error_fields_and_rejects_foreign_model(self):
        model = PINNResult(
            "model",
            self.run.run_id,
            self.run.config.config_hash,
            {"temperature": self.run.fields["temperature"] + 10.0},
            {},
            {},
            {},
        )
        self.session.set_pinn(model)
        self.session.source = "PINN error"
        first = self.session.display_selection()
        np.testing.assert_allclose(first.fields["temperature"], 10.0)
        replacement = replace(
            model,
            run_id="replacement",
            fields={"temperature": self.run.fields["temperature"] + 20.0},
        )
        self.session.set_pinn(replacement)
        second = self.session.display_selection()
        np.testing.assert_allclose(second.fields["temperature"], 20.0)
        self.assertIsNot(first.fields, second.fields)
        with self.assertRaises(ValueError):
            self.session.set_pinn(replace(model, source_run_id="foreign"))
        self.assertIs(self.session.pinn, replacement)

    def test_comparison_aligns_physical_times_and_rejects_spatial_difference_on_new_mesh(
        self,
    ):
        other = result("other", times=(0.0, 2.0))
        self.session.set_comparison(other)
        self.session.source = "Run difference"
        selection = self.session.display_selection()
        self.assertTrue(selection.difference)
        self.assertIn("interpolated display", selection.source_label)
        np.testing.assert_allclose(selection.fields["temperature"], 0.0)
        self.session.view = "Compare"
        self.session.set_comparison(result("short", times=(0.0, 0.5), count=2))
        self.assertEqual(self.session.timeline_end(), 0.5)
        self.assertIsNone(self.session.comparison_difference)

    def test_invalid_analysis_does_not_replace_accepted_association(self):
        accepted = {
            "source_run_id": self.run.run_id,
            "config_hash": self.run.config.config_hash,
            "snapshot_index": 1,
        }
        self.session.set_trajectories(accepted)
        for changes in (
            {"source_run_id": "foreign"},
            {"config_hash": "foreign"},
            {"snapshot_index": 99},
        ):
            with self.assertRaises(ValueError):
                self.session.set_trajectories({**accepted, **changes})
            self.assertIs(self.session.trajectories, accepted)

    def test_failed_load_preserves_accepted_result(self):
        store = Mock()
        store.load_run.side_effect = ValueError("corrupt saved fields")
        with self.assertRaises(ValueError):
            self.session.load_result(store, Path("corrupt.h5"))
        self.assertIs(self.session.active, self.run)

    def test_unavailable_pinn_and_dose_timestamps_do_not_block_valid_run_and_sidecars(
        self,
    ):
        accepted = result("loadable-run")
        root = Path("saved-run")
        missing_pinn = root / "pinn-disappeared.h5"
        valid_pinn = root / "pinn-valid.h5"
        unreadable_dose = root / "dose-unreadable.json"
        valid_dose = root / "dose-valid.json"
        paths = {
            "pinn-*.h5": [missing_pinn, valid_pinn],
            "dose-*.json": [unreadable_dose, valid_dose],
        }
        model = PINNResult(
            "valid-model",
            accepted.run_id,
            accepted.config.config_hash,
            accepted.fields,
            {},
            {},
            {},
        )
        dose = DoseResult(
            "valid-dose",
            accepted.run_id,
            accepted.config.config_hash,
            0,
            "water",
            ("water",),
            np.array([0.0]),
            np.array([0.0]),
            1,
        )
        store = Mock()
        store.load_run.return_value = accepted
        store.load_pinn.return_value = model
        store.load_dose.return_value = dose

        def stat(path):
            if path == missing_pinn:
                raise FileNotFoundError("Sidecar removed during discovery")
            if path == unreadable_dose:
                raise PermissionError("Sidecar timestamp unavailable")
            return SimpleNamespace(st_mtime=1.0)

        with (
            patch.object(
                Path, "rglob", lambda directory, pattern: iter(paths[pattern])
            ),
            patch.object(Path, "stat", stat),
        ):
            self.session.load_result(store, root / "result.h5")
        self.assertIs(self.session.active, accepted)
        self.assertIs(self.session.pinn, model)
        self.assertIs(self.session.dose, dose)
        store.load_pinn.assert_called_once_with(valid_pinn)
        store.load_dose.assert_called_once_with(valid_dose)

    def test_malformed_or_unreadable_newer_pinn_continues_to_valid_older_model(self):
        root = Path("saved-run")
        newer = root / "pinn-newer-broken.h5"
        older = root / "pinn-older-valid.h5"
        model = PINNResult(
            "valid-model",
            self.run.run_id,
            self.run.config.config_hash,
            self.run.fields,
            {},
            {},
            {},
        )
        for error in (KeyError, TypeError, OSError, ValueError):
            with self.subTest(error=error.__name__):
                store = Mock()
                store.load_run.return_value = self.run
                store.load_pinn.side_effect = [
                    error("Invalid or unreadable model sidecar"),
                    model,
                ]
                with (
                    patch.object(
                        Path,
                        "rglob",
                        lambda directory, pattern: iter(
                            [older, newer] if pattern == "pinn-*.h5" else []
                        ),
                    ),
                    patch.object(
                        Path,
                        "stat",
                        lambda path: SimpleNamespace(
                            st_mtime=2.0 if path == newer else 1.0
                        ),
                    ),
                ):
                    self.session.load_result(store, root / "result.h5")
                self.assertIs(self.session.active, self.run)
                self.assertIs(self.session.pinn, model)
                self.assertEqual(self.session.pinn_path, older)
                self.assertEqual(
                    store.load_pinn.call_args_list, [call(newer), call(older)]
                )


if __name__ == "__main__":
    unittest.main()
