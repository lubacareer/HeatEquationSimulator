"""Exercise the real Qt/VTK workspace and isolated worker lifecycle on Windows."""

from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback
import imageio_ffmpeg
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PyQt6 import QtCore, QtGui, QtWidgets
from magnetic_torus.app import TorusWindow, configure_application
from magnetic_torus.models import PINNResult, preset


def _check(condition, message):
    """Keep smoke gates active even when the caller enables Python optimization."""
    if not condition:
        raise AssertionError(message)


def exception_hook(kind, value, tb):
    traceback.print_exception(kind, value, tb)
    sys.stderr.flush()


sys.excepthook = exception_hook

application = QtWidgets.QApplication([])
configure_application(application)
data_root = os.environ.get("MAGNETIC_TORUS_DATA")
if data_root:
    Path(data_root).mkdir(parents=True, exist_ok=True)
directory = Path(tempfile.mkdtemp(prefix="magnetic-torus-desktop-", dir=data_root))
window = TorusWindow(directory)
window.setWindowTitle("Magnetic Torus — automated verification")
window.show()
application.processEvents()
window.scene.camera_preset("Isometric")
errors = []
window.jobs.failed.connect(lambda message, detail: errors.append((message, detail)))


def wait_job(timeout=150):
    deadline = time.monotonic() + timeout
    while window.jobs.busy:
        application.processEvents()
        if time.monotonic() > deadline:
            raise TimeoutError("Desktop worker did not finish")
        time.sleep(0.02)
    application.processEvents()
    if errors:
        raise RuntimeError(str(errors))


try:
    print("START thermal worker", flush=True)
    config = replace(
        preset("Thermal benchmark"), resolution=4, time_steps=16, pinn_epochs=4
    )
    window.editor.set_config(config)
    window.start_job("simulate")
    wait_job()
    print("FIRST result loaded", flush=True)
    _check(
        window.session.active is not None,
        "Check failed: window.session.active is not None",
    )
    first = window.session.active
    first_path = window.session.active_path
    _check(not window.session.previewing, "Check failed: not window.session.previewing")
    window.playback.seek_time(config.duration / 2)
    print("TIMELINE updated", flush=True)
    _check(
        window.session.current_index > 0,
        "Check failed: window.session.current_index > 0",
    )
    camera = np.asarray(list(window.scene.plotter.camera_position))
    window.scene._pick(first.points[5])
    _check(
        window.session.probe_index == 5, "Check failed: window.session.probe_index == 5"
    )
    window.quantity.setCurrentText("Liquid fraction")
    print("QUANTITY updated", flush=True)
    window.cutaway.setChecked(False)
    window.cutaway.setChecked(True)
    window.slice.setChecked(True)
    window.section_slider.setValue(100)
    window.slice.setChecked(False)
    print("OVERLAYS updated", flush=True)
    window.opacity.setValue(75)
    application.processEvents()
    window.scene.plotter.screenshot(str(directory / "thermal-scene.png"))
    screenshot = QtGui.QImage(str(directory / "thermal-scene.png"))
    _check(not screenshot.isNull(), "Scene PNG must decode")
    sample_colors = {
        screenshot.pixel(x * screenshot.width() // 4, y * screenshot.height() // 4)
        for x in (1, 2, 3)
        for y in (1, 2, 3)
    }
    _check(len(sample_colors) > 1, "Scene PNG must contain rendered graphics")
    _check(
        (directory / "thermal-scene.png").stat().st_size > 1000,
        'Check failed: (directory / "thermal-scene.png").stat().st_size > 1000',
    )
    old_time = window.session.current_time
    window.editor.set_config(
        replace(config, latent_heat=config.latent_heat * 1.5, name="Edited latent heat")
    )
    _check(
        window.session.active is first, "Check failed: window.session.active is first"
    )
    window.start_job("simulate")
    _check(
        window.session.active is first, "Check failed: window.session.active is first"
    )
    wait_job()
    print("SECOND result loaded", flush=True)
    np.testing.assert_allclose(list(window.scene.plotter.camera_position), camera)
    _check(
        np.isclose(window.session.current_time, old_time),
        "Recalculation must preserve physical time",
    )
    _check(
        window.session.active.run_id != first.run_id,
        "Check failed: window.session.active.run_id != first.run_id",
    )
    _check(
        window.session.active.config.latent_heat == config.latent_heat * 1.5,
        "Check failed: window.session.active.config.latent_heat == config.latent_heat * 1.5",
    )
    _check(
        len(window.store.list_runs()) == 2,
        "Check failed: len(window.store.list_runs()) == 2",
    )
    window.session.set_comparison(first)
    window.compare_scene.set_result(first, window.quantity.currentText())
    window.compare_scene.plotter.camera = window.scene.plotter.camera
    window.set_view("Compare")
    window._share_color_range()
    window.playback.seek_time(1000)
    _check(
        window.scene.color_range() == window.compare_scene.color_range(),
        "Check failed: window.scene.color_range() == window.compare_scene.color_range()",
    )
    # Invalid editing cannot launch a worker or corrupt an accepted run.
    accepted = window.session.active
    window.editor.editors["density"].setText("nan")
    window.start_job("simulate")
    _check(
        not window.jobs.busy and window.session.active is accepted,
        "Check failed: not window.jobs.busy and window.session.active is accepted",
    )
    window.editor.set_config(accepted.config)
    # Cancellation preserves accepted fields and emits no result publication.
    count = len(window.store.list_runs())
    window.start_job("simulate")
    window.cancel_job()
    wait_job()
    _check(
        window.session.active is accepted,
        "Check failed: window.session.active is accepted",
    )
    _check(
        len(window.store.list_runs()) == count,
        "Check failed: len(window.store.list_runs()) == count",
    )
    # Selecting another accepted run makes a late response stale to the display.
    window.editor.set_config(
        replace(accepted.config, name="Saved but not selected", latent_heat=200000.0)
    )
    window.start_job("simulate")
    window.activate_result(first_path)
    selected_id = window.session.active.run_id
    wait_job()
    _check(
        window.session.active.run_id == selected_id,
        "Check failed: window.session.active.run_id == selected_id",
    )
    _check(
        len(window.store.list_runs()) == count + 1,
        "Check failed: len(window.store.list_runs()) == count + 1",
    )
    # A process failure keeps the previous accepted result intact.
    before_failure = window.session.active
    window.jobs.start(
        "invalid_action", window.session.active.config, window.session.active_path
    )
    deadline = time.monotonic() + 30
    while window.jobs.busy and time.monotonic() < deadline:
        application.processEvents()
        time.sleep(0.02)
    _check(
        errors and window.session.active is before_failure,
        "Check failed: errors and window.session.active is before_failure",
    )
    errors.clear()
    window.logs.hide()
    window.editor.set_config(window.session.active.config)
    window.set_view("Thermal")
    window.quantity.setCurrentText("Temperature")
    timings = []
    for repeat in range(2):
        for index in range(len(window.session.active.times)):
            started = time.perf_counter()
            window.session.current_index = index
            window._render()
            application.processEvents()
            timings.append(time.perf_counter() - started)
    timings.sort()
    median_fps = 1 / timings[len(timings) // 2]
    # A full-size annotated MP4 is encoded using the actual renderer.
    window.exports.begin_movie_export(str(directory / "smoke.mp4"))
    movie_size = tuple(window.exports._movie_size)
    resize_frame_counts = []

    def resize_during_export():
        window.resize(1133, 731)
        resize_frame_counts.append(window.exports._movie_position)

    QtCore.QTimer.singleShot(20, resize_during_export)
    deadline = time.monotonic() + 60
    while window.exports.writer is not None:
        application.processEvents()
        if time.monotonic() > deadline:
            raise TimeoutError("Animation export timed out")
        time.sleep(0.005)
    export_frame_count = window.exports._movie_position
    _check(
        (directory / "smoke.mp4").stat().st_size > 1000,
        'Check failed: (directory / "smoke.mp4").stat().st_size > 1000',
    )
    reader = imageio_ffmpeg.read_frames(str(directory / "smoke.mp4"))
    video_metadata = next(reader)
    encoded_frames = 0
    for frame in reader:
        _check(
            len(frame) == movie_size[0] * movie_size[1] * 3,
            "Every resized MP4 frame must preserve its encoded dimensions",
        )
        encoded_frames += 1
    _check(
        bool(resize_frame_counts), "Window must resize during the real renderer export"
    )
    resize_frame_count = resize_frame_counts[0]
    _check(
        export_frame_count > resize_frame_count,
        "At least one frame must be captured after resizing the export window "
        f"(resize={resize_frame_count}, final={export_frame_count})",
    )
    _check(
        tuple(video_metadata["size"]) == movie_size,
        "MP4 dimensions must match the encoder",
    )
    _check(encoded_frames >= 60, (encoded_frames, video_metadata))
    window.editor.set_config(replace(window.session.active.config, particle_count=8))
    window.start_job("trace")
    wait_job()
    _check(
        window.session.trajectories is not None
        and window.session.trajectories["paths"].shape[0] == 8,
        'Check failed: window.session.trajectories is not None and window.session.trajectories["paths"].shape[0] == 8',
    )
    _check(
        window.session.trajectories["source_run_id"] == window.session.active.run_id,
        'Check failed: window.session.trajectories["source_run_id"] == window.session.active.run_id',
    )
    window.shield_plot.setCurrentText("Transmission vs energy")
    window.playback.toggle_particle_replay()
    window.playback._particle_tick()
    window.playback.toggle_particle_replay()
    snapshot = int(window.session.trajectories["snapshot_index"])
    window.playback.step(-1 if snapshot else 1)
    _check(
        not window.scene.plotter.actors["particle_paths"].GetVisibility(),
        'Check failed: not window.scene.plotter.actors["particle_paths"].GetVisibility()',
    )
    # This is display data with known errors, never a trained or qualified PINN.
    # Training and accuracy are exercised by separate scientific cases.
    source = window.session.active
    fields = {name: values.copy() for name, values in source.fields.items()}
    fields["temperature"] += 0.25
    model = PINNResult(
        "desktop-display-fixture",
        source.run_id,
        source.config.config_hash,
        fields,
        {"loss": np.array([1.0, 0.5])},
        {"epochs": 2.0},
        {"validation_status": "unvalidated display fixture", "fixture": True},
    )
    window.store.save_pinn(model)
    window.activate_result(window.session.active_path)
    _check(
        window.session.pinn is not None,
        "The smoke fixture must be discovered on reload",
    )
    window.set_view("PINN")
    window.source.setCurrentText("PINN")
    window.source.setCurrentText("PINN error")
    _check(
        window.scene.display_selection.difference,
        "PINN error source must use differences",
    )
    np.testing.assert_allclose(
        window.scene.display_selection.fields["temperature"], 0.25
    )
    inspected_pinn = True
    print(
        json.dumps(
            {
                "passed": True,
                "artifact_directory": str(directory),
                "runs": len(window.store.list_runs()),
                "points": len(accepted.points),
                "snapshots": len(accepted.times),
                "encoded_frames": encoded_frames,
                "median_playback_fps": median_fps,
                "particle_replay": True,
                "inspected_saved_pinn": inspected_pinn,
                "pinn_fixture_kind": "display-only; not a trained model",
                "resized_during_mp4": bool(resize_frame_counts),
                "resize_frame_count": resize_frame_count,
                "export_frame_count": export_frame_count,
            }
        ),
        flush=True,
    )
    if "--hold" in sys.argv:
        application.exec()
finally:
    window.close()
    application.processEvents()
