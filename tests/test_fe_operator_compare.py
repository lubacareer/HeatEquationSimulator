"""Checks for raw FE operator comparison and its tolerance accounting."""
from dataclasses import replace
from contextlib import redirect_stdout
from io import StringIO
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from magnetic_torus.numerics.weak_residual import evaluate_weak_residual_torch
from magnetic_torus.store import RunStore
from scripts import compare_fe_operator
from scripts.compare_fe_operator import comparisons
from test_pinn_research_scaling import tiny_fe_source

SUITE_CAPABILITIES = ("torch_cpu",)


class OperatorComparisonTests(unittest.TestCase):
    def test_comparison_reports_discrepancies_without_reusing_solver_tolerance(self):
        matched = comparisons(np.array([1., 2.]), np.array([1., 2.]))
        self.assertTrue(matched["matches"])
        self.assertEqual(matched["maximum_absolute_difference"], 0.)
        mismatch = comparisons(np.array([1., 2.]), np.array([1., 2.01]))
        self.assertFalse(mismatch["matches"])
        self.assertAlmostEqual(mismatch["difference_l2"], .01)
        for invalid in (np.array([float("nan"), 0.]), np.ones(3)):
            with self.assertRaises(ValueError):
                comparisons(np.array([1., 2.]), invalid)

    def test_raw_components_are_opt_in_and_preserve_boundary_masks_and_gradients(self):
        with tempfile.TemporaryDirectory() as directory:
            original = tiny_fe_source(directory)
            source = replace(original, metadata={**original.metadata, "pinn_physics": {
                "kind": "fenics_a_phi_vms_v1", "bc_nodes": {"0": [0], "1": [1], "2": [2], "3": [3], "4": [0]}}})
            fields = {name: torch.tensor(values[:2], dtype=torch.float64, requires_grad=True)
                      for name, values in source.fields.items()}
            default = evaluate_weak_residual_torch(source.config, source, fields, [0, 1])
            raw = evaluate_weak_residual_torch(source.config, source, fields, [0, 1], include_raw_components=True)
            for key, value in default.items():
                torch.testing.assert_close(raw[key], value, rtol=0., atol=0.)
            for key in ("momentum", "continuity", "induction", "charge", "enthalpy"):
                self.assertEqual(raw[f"raw_{key}"].shape[0], 3)
                self.assertTrue(torch.isfinite(raw[f"raw_{key}"]).all())
            gradient = torch.autograd.grad(raw["raw_momentum"].square().sum(), fields["pressure"])[0]
            self.assertTrue(torch.isfinite(gradient).all())

    def test_raw_pressure_jacobian_action_matches_centered_difference(self):
        with tempfile.TemporaryDirectory() as directory:
            source = tiny_fe_source(directory)
            fields = {name: torch.tensor(values[:2], dtype=torch.float64)
                      for name, values in source.fields.items()}
            pressure = fields["pressure"][1]
            tangent = torch.tensor([0., .1, -.05, .08], dtype=torch.float64)
            def momentum(value):
                selected = dict(fields)
                selected["pressure"] = torch.stack((fields["pressure"][0], value))
                return evaluate_weak_residual_torch(source.config, source, selected, [0, 1],
                                                    include_raw_components=True)["raw_momentum"]
            _, jvp = torch.autograd.functional.jvp(momentum, pressure, tangent)
            epsilon = 1.e-4
            finite = (momentum(pressure + epsilon * tangent) - momentum(pressure - epsilon * tangent)) / (2 * epsilon)
            torch.testing.assert_close(jvp, finite, rtol=1.e-6, atol=1.e-10)


class ProbeIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = RunStore(self.root / "input-store").save_run(tiny_fe_source(self.root))
        self.worker_calls = 0
        self.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, self.previous_threads)

    def run_comparison(self, output, *, reuse=None, seed=None, corrupt_copy=False):
        args = ["compare_fe_operator.py", "--source", str(self.source), "--output", str(output)]
        if reuse is not None:
            args += ["--reuse-worker", str(reuse)]
        if seed is not None:
            args += ["--seed", str(seed)]
        real_copy = shutil.copy2
        def copy(source, target, *extra, **kwargs):
            result = real_copy(source, target, *extra, **kwargs)
            if corrupt_copy and Path(target).name == "operator_probe.npz":
                Path(target).write_bytes(b"corrupted destination")
            return result
        def worker(*_args, **_kwargs):
            self.worker_calls += 1
            destination = Path(output) / "worker"
            destination.mkdir()
            # Deliberately inaccurate values must remain a scientific FAIL.
            names = ("velocity", "pressure", "vector_potential", "electric_potential", "temperature")
            arrays = {}
            for name in names:
                size = (4, 3) if name in {"velocity", "vector_potential"} else (4,)
                arrays[f"residual_{name}"] = np.ones(size)
                arrays[f"jacobian_{name}"] = np.ones(size)
            np.savez_compressed(destination / "operator_probe.npz", **arrays)
            return SimpleNamespace(returncode=0, stdout="worker fixture\n", stderr="")
        with (patch.object(sys, "argv", args),
              patch.object(compare_fe_operator.subprocess, "run", side_effect=worker),
              patch.object(compare_fe_operator.shutil, "copy2", side_effect=copy),
              redirect_stdout(StringIO())):
            return compare_fe_operator.main()

    def test_fresh_fail_and_chained_reuse_record_identical_loaded_probe(self):
        original, first, second = (self.root / name for name in ("original", "first", "second"))
        self.assertEqual(self.run_comparison(original), 1)
        probe = original / "worker" / "operator_probe.npz"
        expected = hashlib.sha256(probe.read_bytes()).hexdigest()
        for output, previous in ((original, None), (first, original), (second, first)):
            if previous:
                self.assertEqual(self.run_comparison(output, reuse=previous), 1)
            report = json.loads((output / "report.json").read_text())
            self.assertEqual(report["status"], "FAIL")
            self.assertEqual(report["operator_probe_sha256"], expected)
            self.assertEqual((output / "operator_probe.sha256").read_text().strip(), expected)
            self.assertEqual(hashlib.sha256((output / "worker" / "operator_probe.npz").read_bytes()).hexdigest(), expected)
            self.assertEqual(report["reused_worker_from"], str(previous.resolve()) if previous else None)
        self.assertEqual(self.worker_calls, 1)

    def test_fresh_passing_comparison_records_probe_digest(self):
        output = self.root / "passing"
        with patch.object(compare_fe_operator, "comparisons", return_value={"matches": True}):
            self.assertEqual(self.run_comparison(output), 0)
        report = json.loads((output / "report.json").read_text())
        expected = hashlib.sha256((output / "worker" / "operator_probe.npz").read_bytes()).hexdigest()
        self.assertEqual((report["status"], report["operator_probe_sha256"]), ("PASS", expected))
        self.assertEqual((output / "operator_probe.sha256").read_text().strip(), expected)

    def test_reuse_rejects_missing_bad_or_changed_digest_and_corrupted_copy(self):
        original = self.root / "original"
        self.assertEqual(self.run_comparison(original), 1)
        for name in ("missing", "invalid", "tampered", "copied"):
            with self.subTest(name=name):
                prior, output = self.root / (name + "-prior"), self.root / (name + "-output")
                shutil.copytree(original, prior)
                if name == "missing":
                    (prior / "operator_probe.sha256").unlink()
                elif name == "invalid":
                    (prior / "operator_probe.sha256").write_text("not a digest")
                elif name == "tampered":
                    (prior / "worker" / "operator_probe.npz").write_bytes(b"tampered")
                with self.assertRaisesRegex(ValueError, "digest|SHA-256"):
                    self.run_comparison(output, reuse=prior, corrupt_copy=name == "copied")
                self.assertFalse((output / "report.json").exists())
        self.assertEqual(self.worker_calls, 1)

    def test_reuse_still_rejects_worker_input_and_scenario_mismatches(self):
        original = self.root / "original"
        self.assertEqual(self.run_comparison(original), 1)
        for name in ("worker", "input", "scenario"):
            with self.subTest(name=name):
                prior, output = self.root / (name + "-prior"), self.root / (name + "-output")
                shutil.copytree(original, prior)
                if name == "worker":
                    (prior / "worker.sha256").write_text("0" * 64)
                elif name == "scenario":
                    (prior / "request.json").write_text("{}")
                with self.assertRaisesRegex(ValueError, "identical"):
                    self.run_comparison(output, reuse=prior, seed=43 if name == "input" else None)
                self.assertFalse((output / "report.json").exists())
        self.assertEqual(self.worker_calls, 1)


if __name__ == "__main__":
    unittest.main()
