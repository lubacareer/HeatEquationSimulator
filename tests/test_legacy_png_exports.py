"""Exercise real plot exporters and concurrent output allocation."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import chdir
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from PIL import Image

from interface3 import App, _save_png_export


class LegacyPngExportTests(unittest.TestCase):
    def test_real_figure_and_canvas_exports_keep_existing_images(self):
        with TemporaryDirectory() as directory, chdir(directory):
            previous = Path("PINN_temperature_surface.png")
            previous.write_bytes(b"previous artifact")
            figure, axes = plt.subplots()
            try:
                axes.plot([0, 1], [0, 1])
                first = Path(_save_png_export(figure.savefig, "PINN_temperature_surface", dpi=30))
                second = Path(_save_png_export(figure.savefig, "PINN_temperature_surface", dpi=30))
                line, = axes.plot([], [])
                app = App.__new__(App)
                app.line3, app.ax3, app.canvas3 = line, axes, figure.canvas
                App.update_line_plot3(app, [0, 1], [2, 3])
                outputs = [first, second, *Path.cwd().glob("Tx-*.png")]
                self.assertEqual(len(outputs), 3)
                self.assertEqual(len(set(outputs)), 3)
                for output in outputs:
                    self.assertEqual(output.parent, Path.cwd())
                    self.assertTrue(output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
                    with Image.open(output) as image:
                        image.load()
                        self.assertGreater(min(image.size), 0)
                self.assertEqual(previous.read_bytes(), b"previous artifact")
            finally:
                plt.close(figure)

    def test_concurrent_exports_preserve_every_payload(self):
        with TemporaryDirectory() as directory, chdir(directory):
            def save(index):
                payload = f"artifact-{index}".encode()

                def write(output, *, format):
                    self.assertEqual(format, "png")
                    output.write(payload)

                return Path(_save_png_export(write, "PINN_temperature_surface")), payload

            with ThreadPoolExecutor(max_workers=4) as executor:
                results = list(executor.map(save, range(20)))
            self.assertEqual(len({path for path, _ in results}), 20)
            for path, payload in results:
                self.assertEqual(path.read_bytes(), payload)

    def test_failed_export_removes_only_its_partial_file(self):
        with TemporaryDirectory() as directory, chdir(directory):
            existing = Path("keep.png")
            existing.write_bytes(b"keep")

            def fail(output, *, format):
                output.write(b"partial PNG")
                raise ValueError("fixture exporter failure")

            with self.assertRaisesRegex(ValueError, "fixture exporter failure"):
                _save_png_export(fail, "PINN_temperature_surface")
            self.assertEqual(list(Path.cwd().iterdir()), [existing.resolve()])
            self.assertEqual(existing.read_bytes(), b"keep")


if __name__ == "__main__":
    unittest.main()
