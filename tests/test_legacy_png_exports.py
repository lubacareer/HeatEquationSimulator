"""Exercise bounded per-App snapshots with real PNG writers and concurrency."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import chdir
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from PIL import Image

from interface3 import App


class LegacyPngExportTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
