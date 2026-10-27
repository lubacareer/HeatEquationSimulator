"""Memory telemetry is opt-in, scalar-only and cannot mask training failures."""
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import torch

from magnetic_torus.learning.memory_telemetry import MemoryTelemetry, optional_memory_telemetry
from magnetic_torus.learning.training import _bounded_lbfgs_step
from magnetic_torus.models import SimulationCancelled
from magnetic_torus.pinn import train_pinn
from test_torus_pinn import uniform_heating_run

SUITE_CAPABILITIES = ("torch_cpu",)


def records(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


class MemoryTelemetryTests(unittest.TestCase):
    def test_disabled_by_default_and_cpu_events_are_json_scalars(self):
        with patch.dict(os.environ, {"MAGNETIC_TORUS_PINN_MEMORY_LOG": ""}):
            self.assertIsNone(optional_memory_telemetry(torch, torch.device("cpu")))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "memory.jsonl"
            telemetry = MemoryTelemetry(str(path), torch, torch.device("cpu"))
            telemetry.stage, telemetry.epoch, telemetry.closures = "Adam", 50, 3
            telemetry.emit("step")
            event = records(path)[0]
            self.assertEqual((event["stage"], event["epoch"], event["lbfgs_closures"]), ("Adam", 50, 3))
            self.assertNotIn("allocated_bytes", event)

    def test_unwritable_destination_and_unavailable_cuda_stats_never_raise(self):
        with tempfile.TemporaryDirectory() as folder:
            MemoryTelemetry(folder, torch, torch.device("cpu")).emit("error")
            path = Path(folder) / "memory.jsonl"
            with patch.object(torch.cuda, "memory_allocated", side_effect=RuntimeError("unavailable")):
                MemoryTelemetry(str(path), torch, torch.device("cuda")).emit("training_error")
            self.assertEqual(records(path)[0]["event"], "training_error")
            self.assertIsNone(records(path)[0]["allocated_bytes"])

    def test_original_optimizer_error_survives_failed_rollback(self):
        parameter = torch.nn.Parameter(torch.tensor([1.], dtype=torch.float64))
        class BrokenOptimizer:
            def state_dict(self):
                return {}
            def step(self, closure):
                raise RuntimeError("original CUDA OOM")
            def load_state_dict(self, state):
                raise RuntimeError("rollback failed")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "memory.jsonl"
            telemetry = MemoryTelemetry(str(path), torch, torch.device("cpu"))
            with self.assertRaisesRegex(RuntimeError, "original CUDA OOM") as caught:
                _bounded_lbfgs_step(BrokenOptimizer(), (parameter,), lambda: None,
                                    threading.Event(), telemetry=telemetry)
            self.assertIn("rollback failed", " ".join(caught.exception.__notes__))
            self.assertEqual([r["event"] for r in records(path)],
                             ["optimizer_error", "rollback_start", "rollback_error"])

    def test_real_training_records_checkpoint_and_completion(self):
        self._training_probe("cpu")

    def test_cancelled_training_records_error_without_a_result(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "memory.jsonl"
            cancel = threading.Event()
            def progress(_, message):
                if message.startswith("Epoch 1 "):
                    cancel.set()
            with patch.dict(os.environ, {"MAGNETIC_TORUS_PINN_DEVICE": "cpu",
                                         "MAGNETIC_TORUS_PINN_MEMORY_LOG": str(path)}):
                with self.assertRaises(SimulationCancelled):
                    train_pinn(uniform_heating_run(), progress, cancel)
            self.assertEqual(records(path)[-1]["event"], "training_error")
            self.assertEqual(records(path)[-1]["error_type"], "SimulationCancelled")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_real_cuda_training_records_allocator_stats(self):
        self._training_probe("cuda")

    def _training_probe(self, device):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "memory.jsonl"
            with patch.dict(os.environ, {"MAGNETIC_TORUS_PINN_DEVICE": device,
                                         "MAGNETIC_TORUS_PINN_MEMORY_LOG": str(path)}):
                result = train_pinn(uniform_heating_run(), lambda *_: None,
                                    threading.Event(), Path(folder) / "models")
            events = records(path)
            self.assertEqual(events[0]["event"], "prepared")
            self.assertEqual(events[-1]["event"], "complete")
            self.assertEqual(events[-1]["validation_status"], result.metadata["validation_status"])
            self.assertIn("checkpoint_start", [row["event"] for row in events])
            self.assertEqual([row["epoch"] for row in events if row["event"] == "step"], [1, 8])
            self.assertTrue(all(isinstance(row["epoch"], int) for row in events))
            if device == "cuda":
                self.assertGreater(events[-1]["peak_allocated_bytes"], 0)
                self.assertGreater(events[-1]["device_total_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
