"""Workspace identity, display-source and physical-time behavior without Qt."""

from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock
import unittest

import numpy as np

from magnetic_torus.models import PINNResult, RunResult, ScenarioConfig
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


if __name__ == "__main__":
    unittest.main()
