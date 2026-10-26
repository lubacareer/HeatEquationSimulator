"""Exercise the real Qt/VTK workspace and isolated worker lifecycle on Windows."""
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import time
import traceback
import imageio_ffmpeg

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PyQt6 import QtWidgets
from magnetic_torus.app import TorusWindow, configure_application
from magnetic_torus.models import preset

def exception_hook(kind, value, tb):
    traceback.print_exception(kind, value, tb)
    sys.stderr.flush()

sys.excepthook = exception_hook

application = QtWidgets.QApplication([])
configure_application(application)
directory = Path(tempfile.mkdtemp(prefix="magnetic-torus-desktop-"))
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
        time.sleep(.02)
    application.processEvents()
    if errors:
        raise RuntimeError(str(errors))


try:
    print("START thermal worker", flush=True)
    config = replace(preset("Thermal benchmark"), resolution=4, time_steps=16, pinn_epochs=4)
    window.editor.set_config(config)
    window.start_job("simulate")
    wait_job()
    print("FIRST result loaded", flush=True)
    assert window.session.active is not None
    first = window.session.active
    first_path = window.session.active_path
    assert not window.session.previewing
    window.playback.seek_time(config.duration / 2)
    print("TIMELINE updated", flush=True)
    assert window.session.current_index > 0
    camera = window.scene.plotter.camera_position
    window.scene.point_selected.emit(5)
    assert window.session.probe_index == 5
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
    assert (directory / "thermal-scene.png").stat().st_size > 1000
    window.editor.set_config(replace(config, latent_heat=config.latent_heat * 1.5, name="Edited latent heat"))
    assert window.session.active is first  # Draft does not mutate accepted fields.
    window.start_job("simulate")
    assert window.session.active is first  # Remains visible while its replacement runs.
    wait_job()
    print("SECOND result loaded", flush=True)
    assert window.session.active.run_id != first.run_id
    assert window.session.active.config.latent_heat == config.latent_heat * 1.5
    assert len(window.store.list_runs()) == 2
    window.session.set_comparison(first)
    window.compare_scene.set_result(first, window.quantity.currentText())
    window.compare_scene.plotter.camera = window.scene.plotter.camera
    window.set_view("Compare")
    window._share_color_range()
    window.playback.seek_time(1000)
    assert window.scene.color_range() == window.compare_scene.color_range()
    # Invalid editing cannot launch a worker or corrupt an accepted run.
    accepted = window.session.active
    window.editor.editors["density"].setText("nan")
    window.start_job("simulate")
    assert not window.jobs.busy and window.session.active is accepted
    window.editor.set_config(accepted.config)
    # Cancellation preserves accepted fields and emits no result publication.
    count = len(window.store.list_runs())
    window.start_job("simulate")
    window.cancel_job()
    wait_job()
    assert window.session.active is accepted
    assert len(window.store.list_runs()) == count
    # Selecting another accepted run makes a late response stale to the display.
    window.editor.set_config(replace(accepted.config, name="Saved but not selected", latent_heat=200000.))
    window.start_job("simulate")
    window.activate_result(first_path)
    selected_id = window.session.active.run_id
    wait_job()
    assert window.session.active.run_id == selected_id
    assert len(window.store.list_runs()) == count + 1
    # A process failure keeps the previous accepted result intact.
    before_failure = window.session.active
    window.jobs.start("invalid_action", window.session.active.config, window.session.active_path)
    deadline = time.monotonic() + 30
    while window.jobs.busy and time.monotonic() < deadline:
        application.processEvents()
        time.sleep(.02)
    assert errors and window.session.active is before_failure
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
    deadline = time.monotonic() + 60
    while window.exports.writer is not None:
        application.processEvents()
        if time.monotonic() > deadline:
            raise TimeoutError("Animation export timed out")
        time.sleep(.005)
    assert (directory / "smoke.mp4").stat().st_size > 1000
    reader = imageio_ffmpeg.read_frames(str(directory / "smoke.mp4"))
    video_metadata = next(reader)
    encoded_frames = sum(1 for _ in reader)
    assert encoded_frames >= 60, (encoded_frames, video_metadata)
    window.editor.set_config(replace(window.session.active.config, particle_count=8))
    window.start_job("trace")
    wait_job()
    assert window.session.trajectories is not None and window.session.trajectories["paths"].shape[0] == 8
    assert window.session.trajectories["source_run_id"] == window.session.active.run_id
    window.shield_plot.setCurrentText("Transmission vs energy")
    window.playback.toggle_particle_replay()
    window.playback._particle_tick()
    window.playback.toggle_particle_replay()
    snapshot = int(window.session.trajectories["snapshot_index"])
    window.playback.step(-1 if snapshot else 1)
    assert not window.scene.plotter.actors["particle_paths"].GetVisibility()
    native_report = Path(__file__).resolve().parent.parent / ".cache" / "pinn_final_native" / "report.json"
    inspected_pinn = False
    if native_report.is_file():
        trained = json.loads(native_report.read_text())
        window.activate_result(Path(trained["source_path"]))
        assert window.session.pinn is not None
        window.set_view("PINN")
        window.source.setCurrentText("PINN")
        window.source.setCurrentText("PINN error")
        assert window.scene.display_selection.difference
        inspected_pinn = True
    print(json.dumps({"passed": True, "artifact_directory": str(directory),
                      "runs": len(window.store.list_runs()), "points": len(accepted.points),
                      "snapshots": len(accepted.times), "encoded_frames": encoded_frames,
                      "median_playback_fps": median_fps, "particle_replay": True,
                      "inspected_saved_pinn": inspected_pinn}), flush=True)
    if "--hold" in sys.argv:
        application.exec()
finally:
    window.close()
    application.processEvents()
