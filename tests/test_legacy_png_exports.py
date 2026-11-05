"""Exercise bounded per-App snapshots with real PNG writers and concurrency."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import chdir, redirect_stdout
import errno
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import uuid

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from PIL import Image

from interface3 import App


class LegacyPngExportTests(unittest.TestCase):
    def setUp(self):
        # Never let a host operator's override redirect test files outside their
        # owned TemporaryDirectory fixtures.
        environment = patch.dict(os.environ, {"HEAT_PNG_OUTPUT_DIR": ""})
        environment.start()
        self.addCleanup(environment.stop)

    def test_repeated_real_exports_are_bounded_and_images_decode(self):
        with TemporaryDirectory() as directory, chdir(directory):
            previous = Path("PINN_temperature_surface.png")
            previous.write_bytes(b"previous artifact")
            figure, axes = plt.subplots()
            app = App.__new__(App)
            try:
                line, = axes.plot([], [])
                app.line3, app.ax3, app.canvas3 = line, axes, figure.canvas
                paths = set()
                for value in range(6):
                    paths.add(Path(app._export_png(figure.savefig, "PINN_temperature_surface", dpi=30)))
                    App.update_line_plot3(app, [0, 1], [value, value + 1])
                session = Path(app._png_export_directory)
                self.assertEqual(len(paths), 1)
                self.assertEqual(session.parent, Path.cwd())
                self.assertEqual({path.name for path in session.iterdir()},
                                 {"PINN_temperature_surface.png", "Tx.png"})
                for output in session.iterdir():
                    with Image.open(output) as image:
                        image.load()
                        self.assertGreater(min(image.size), 0)
                self.assertEqual(previous.read_bytes(), b"previous artifact")
            finally:
                plt.close(figure)

    def test_concurrent_app_sessions_keep_separate_bounded_snapshots(self):
        with TemporaryDirectory() as directory, chdir(directory):
            def save(index):
                app = App.__new__(App)
                payload = f"session-{index}".encode()

                def write(output, *, format):
                    self.assertEqual(format, "png")
                    output.write(payload)

                first = app._export_png(write, "PINN_temperature_surface")
                self.assertEqual(first, app._export_png(write, "PINN_temperature_surface"))
                second = app._export_png(write, "Tx")
                return Path(app._png_export_directory), (Path(first), Path(second)), payload

            with ThreadPoolExecutor(max_workers=4) as executor:
                results = list(executor.map(save, range(20)))
            self.assertEqual(len({session for session, _, _ in results}), 20)
            for session, paths, payload in results:
                self.assertEqual(len(list(session.iterdir())), 2)
                for path in paths:
                    self.assertEqual(path.read_bytes(), payload)

    def test_failed_render_or_replace_preserves_previous_snapshot(self):
        with TemporaryDirectory() as directory, chdir(directory):
            app = App.__new__(App)

            def good(output, *, format):
                output.write(b"successful snapshot")

            previous = Path(app._export_png(good, "PINN_temperature_surface"))

            def fail(output, *, format):
                output.write(b"partial PNG")
                raise ValueError("fixture exporter failure")

            with self.assertRaisesRegex(ValueError, "fixture exporter failure"):
                app._export_png(fail, "PINN_temperature_surface")
            self.assertEqual(previous.read_bytes(), b"successful snapshot")
            self.assertEqual(list(previous.parent.iterdir()), [previous])
            with patch("interface3.os.replace", side_effect=PermissionError("fixture replace failure")):
                with self.assertRaisesRegex(PermissionError, "fixture replace failure"):
                    app._export_png(good, "PINN_temperature_surface")
            self.assertEqual(previous.read_bytes(), b"successful snapshot")
            self.assertEqual(list(previous.parent.iterdir()), [previous])

    def test_explicit_operator_directory_restores_legacy_filenames(self):
        for configured in (".", "legacy exports"):
            with self.subTest(directory=configured), TemporaryDirectory() as directory, chdir(directory):
                with patch.dict(os.environ, {"HEAT_PNG_OUTPUT_DIR": configured}):
                    app = App.__new__(App)
                    destination = (Path.cwd() / configured).resolve()

                    def write(output, *, format):
                        output.write(b"latest snapshot")

                    for stem in ("PINN_temperature_surface", "Tx"):
                        target = Path(app._export_png(write, stem))
                        self.assertEqual(target, destination / f"{stem}.png")
                        self.assertEqual(app._export_png(write, stem), str(target))
                    self.assertEqual({path.name for path in destination.iterdir()},
                                     {"PINN_temperature_surface.png", "Tx.png"})

                    def fail(output, *, format):
                        output.write(b"partial")
                        raise OSError(errno.ENOSPC, "fixture storage full")

                    with self.assertRaises(OSError):
                        app._export_png(fail, "Tx")
                    self.assertEqual((destination / "Tx.png").read_bytes(), b"latest snapshot")
                    self.assertEqual({path.name for path in destination.iterdir()},
                                     {"PINN_temperature_surface.png", "Tx.png"})

    def test_export_io_failure_still_draws_without_masking_programming_errors(self):
        figure, axes = plt.subplots()
        try:
            line, = axes.plot([], [])
            app = App.__new__(App)
            app.line3, app.ax3, app.canvas3 = line, axes, figure.canvas
            warning = io.StringIO()
            with patch.object(app, "_export_png", side_effect=OSError(errno.ENOSPC, "fixture storage full")), patch.object(
                app.canvas3, "draw", wraps=app.canvas3.draw
            ) as draw, redirect_stdout(warning):
                app.update_line_plot3([0, 1], [2, 3])
                draw.assert_called_once_with()
            self.assertEqual(list(line.get_ydata()), [2, 3])
            self.assertIn("Failed to save Tx figure", warning.getvalue())
            with patch.object(app, "_export_png", side_effect=ValueError("fixture programming error")):
                with self.assertRaisesRegex(ValueError, "fixture programming error"):
                    app.update_line_plot3([0, 1], [4, 5])
        finally:
            plt.close(figure)

    def test_creation_collisions_preserve_preexisting_directories_and_files(self):
        with TemporaryDirectory() as directory, chdir(directory):
            occupied = Path(f"heat-equation-session-{uuid.UUID(int=1).hex}")
            occupied.mkdir()
            marker = occupied / "keep.txt"
            marker.write_bytes(b"preexisting directory")
            app = App.__new__(App)

            def write(output, *, format):
                output.write(b"new snapshot")

            with patch("interface3.uuid.uuid4", side_effect=[uuid.UUID(int=i) for i in (1, 2, 3)]):
                target = Path(app._export_png(write, "Tx"))
            self.assertEqual(marker.read_bytes(), b"preexisting directory")
            collision = target.parent / f".Tx-{uuid.UUID(int=4).hex}.tmp"
            collision.write_bytes(b"preexisting file")
            with patch("interface3.uuid.uuid4", side_effect=[uuid.UUID(int=i) for i in (4, 5)]):
                self.assertEqual(app._export_png(write, "Tx"), str(target))
            self.assertEqual(collision.read_bytes(), b"preexisting file")
            self.assertEqual(target.read_bytes(), b"new snapshot")
            self.assertEqual({path.name for path in target.parent.iterdir()}, {"Tx.png", collision.name})

    @unittest.skipUnless(os.name == "posix", "POSIX permissions are verified in the Linux image")
    def test_export_permissions_honor_normal_and_restrictive_umasks(self):
        # Change umask only in fresh child processes, never in a shared process.
        code = """
import json, os, stat, sys
from pathlib import Path
sys.path.insert(0, sys.argv[2])
from interface3 import App
os.umask(int(sys.argv[1], 8))
app = App.__new__(App)
def write(output, *, format):
    output.write(b'fixture snapshot')
target = Path(app._export_png(write, 'Tx'))
print(json.dumps({'directory': stat.S_IMODE(target.parent.stat().st_mode),
                  'file': stat.S_IMODE(target.stat().st_mode)}))
"""
        for mask, directory_mode, file_mode in (("022", 0o755, 0o644), ("077", 0o700, 0o600)):
            with self.subTest(umask=mask), TemporaryDirectory() as directory, chdir(directory):
                result = subprocess.run([sys.executable, "-c", code, mask,
                                         str(Path(__file__).resolve().parents[1])],
                                        capture_output=True, text=True, check=True, timeout=30)
                actual = json.loads(result.stdout.splitlines()[-1])
                self.assertEqual(actual, {"directory": directory_mode, "file": file_mode})


if __name__ == "__main__":
    unittest.main()
