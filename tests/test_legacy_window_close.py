"""Close the dashboard's own figure without affecting another caller's plot."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from interface3 import App


class LegacyWindowCloseTests(unittest.TestCase):
    def test_closes_owned_figure_before_root_and_preserves_other_figures(self):
        owned = plt.figure()
        other = plt.figure()
        app = App.__new__(App)
        app.fig1 = owned

        def destroy():
            self.assertFalse(plt.fignum_exists(owned.number))
            self.assertTrue(plt.fignum_exists(other.number))

        app.root = SimpleNamespace(destroy=Mock(side_effect=destroy))
        try:
            app.on_closing()
            app.root.destroy.assert_called_once_with()
            self.assertFalse(plt.fignum_exists(owned.number))
            self.assertTrue(plt.fignum_exists(other.number))
        finally:
            plt.close(owned)
            plt.close(other)


if __name__ == "__main__":
    unittest.main()
