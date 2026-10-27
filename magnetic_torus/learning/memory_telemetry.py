"""Optional, best-effort PINN allocator telemetry; never affects training."""
from __future__ import annotations

import json
import os
from pathlib import Path
import time
from typing import Any


class MemoryTelemetry:
    """Append scalar events without retaining tensors or synchronizing CUDA."""

    def __init__(self, path: str, torch: Any, device: Any):
        self.path = Path(path)
        self.torch = torch
        self.device = device
        self.stage = "preparation"
        self.epoch = 0
        self.closures = 0

    def emit(self, event: str, **details: Any) -> None:
        # Logging is diagnostic; errors here must never hide an original OOM or
        # change whether model/checkpoint publication succeeds.
        try:
            record = {"event": event, "wall_time_s": time.time(), "pid": os.getpid(),
                      "device": str(self.device), "stage": self.stage, "epoch": self.epoch,
                      "lbfgs_closures": self.closures, **details}
            if self.device.type == "cuda":
                cuda = self.torch.cuda
                for name, getter in (("allocated_bytes", cuda.memory_allocated),
                                     ("reserved_bytes", cuda.memory_reserved),
                                     ("peak_allocated_bytes", cuda.max_memory_allocated),
                                     ("peak_reserved_bytes", cuda.max_memory_reserved)):
                    try:
                        record[name] = int(getter(self.device))
                    except Exception as error:
                        record[name] = None
                        record[f"{name}_error"] = f"{type(error).__name__}: {error}"
                try:
                    free, total = cuda.mem_get_info(self.device)
                    record.update(device_free_bytes=int(free), device_total_bytes=int(total))
                except Exception as error:
                    record["device_memory_error"] = f"{type(error).__name__}: {error}"
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
        except Exception:
            pass


def optional_memory_telemetry(torch: Any, device: Any) -> MemoryTelemetry | None:
    path = os.environ.get("MAGNETIC_TORUS_PINN_MEMORY_LOG", "").strip()
    return MemoryTelemetry(path, torch, device) if path else None
