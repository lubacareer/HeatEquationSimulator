"""Measure complete Qt/VTK playback against a saved reference, without Torch.

Run in a graphical session; both measurements must use the same machine and
settings. The temporary run history is removed after the benchmark.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--resolution", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--frames", type=int, default=60)
    args = parser.parse_args(argv)
    if min(args.resolution, args.repeats, args.frames) < 1:
        parser.error("Resolution, repeats and frames must be positive.")

    from PyQt6 import QtWidgets
    from magnetic_torus.app import TorusWindow, configure_application
    from magnetic_torus.models import preset
    from magnetic_torus.physics import run_simulation
    from simulation_suite.capabilities import desktop_environment

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    configure_application(app)
    with tempfile.TemporaryDirectory(prefix="torus-benchmark-") as temporary:
        window = TorusWindow(temporary)
        try:
            window.resize(1282, 780)
            window.show()
            app.processEvents()
            config = replace(
                preset("Thermal benchmark"), resolution=args.resolution, time_steps=24
            )
            result = run_simulation(config)
            window.activate_result(window.store.save_run(result))
            # Support both the current component ownership and older window
            # attribute layouts without adding compatibility to product code.
            editor = getattr(window, "editor", window)
            session = getattr(window, "session", window)
            options = getattr(window.scene, "options", window.scene)
            editor.set_config(config)
            window.scene.camera_preset("Isometric")
            app.processEvents()

            def frame(index):
                session.current_index = index % len(result.times)
                window._render()
                app.processEvents()

            for index in range(12):
                frame(index)
            repetitions = []
            for _ in range(args.repeats):
                samples = []
                for index in range(args.frames):
                    started = time.perf_counter()
                    frame(index)
                    samples.append(time.perf_counter() - started)
                median = float(np.median(samples))
                repetitions.append({"median_frame_s": median, "fps": 1 / median})
            report = {
                "points": len(result.points),
                "resolution": args.resolution,
                "window": [window.width(), window.height()],
                "frames_per_repeat": args.frames,
                "repetitions": repetitions,
                "median_fps": float(np.median([value["fps"] for value in repetitions])),
                "detail": options.visual_detail,
                "opacity": options.opacity,
                "overlays": {
                    name: getattr(options, name)
                    for name in (
                        "show_phase",
                        "show_flow",
                        "show_tracers",
                        "show_field",
                        "slice_enabled",
                    )
                },
                "includes_qt_events": True,
                "graphics": desktop_environment(
                    app, window, window.scene.plotter, include_window_state=True
                ),
            }
        finally:
            window.close()
            app.processEvents()
    status = 0
    if args.baseline:
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        for key in ("points", "resolution", "window", "frames_per_repeat"):
            if baseline[key] != report[key]:
                raise ValueError(
                    f"Baseline {key} differs: {baseline[key]} != {report[key]}"
                )
        if len(baseline["repetitions"]) != len(report["repetitions"]):
            raise ValueError("Baseline repetition count differs.")
        environment_verified = "graphics" in baseline
        if environment_verified:
            for key, value in report["graphics"].items():
                if key != "window_state" and baseline["graphics"].get(key) != value:
                    raise ValueError(f"Baseline graphics {key} differs.")
        ratio = report["median_fps"] / baseline["median_fps"]
        report.update(
            baseline_fps=baseline["median_fps"],
            speed_ratio=ratio,
            performance_regression=ratio < 0.9,
            meets_30_fps=report["median_fps"] >= 30,
            baseline_environment_verified=environment_verified,
        )
        status = 3 if ratio < 0.9 else 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report), flush=True)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
