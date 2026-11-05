"""Require the smoke check's scheduled event-loop callback, not incidental close."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

SMOKE = Path(__file__).resolve().parents[1] / "docker/legacy_container_smoke.py"
spec = importlib.util.spec_from_file_location("legacy_gui_smoke", SMOKE)
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


class ContainerGuiSmokeTests(unittest.TestCase):
    def application(self):
        """Queue callbacks without executing them as Tk.after does."""
        callbacks = []
        root = SimpleNamespace(after=lambda _delay, callback: callbacks.append(callback))
        return SimpleNamespace(root=root, on_closing=Mock(), run=Mock()), callbacks

    def test_noop_run_cannot_pass_the_smoke_event_loop_check(self):
        """Reproduce the reported silent-return false-positive path."""
        application, callbacks = self.application()
        with self.assertRaisesRegex(AssertionError, "Scheduled Tk close callback"):
            smoke.exercise_event_loop(application)
        self.assertEqual(len(callbacks), 1)
        application.on_closing.assert_not_called()

    def test_direct_close_without_processing_the_scheduled_callback_is_rejected(self):
        """Closing outside the queued event does not prove event-loop progress."""
        application, _ = self.application()
        application.run.side_effect = application.on_closing
        with self.assertRaisesRegex(AssertionError, "Scheduled Tk close callback"):
            smoke.exercise_event_loop(application)
        application.on_closing.assert_called_once_with()

    def test_processed_callback_must_complete_the_real_close_method(self):
        """Accept a completed scheduled close and propagate close-method failure."""
        application, callbacks = self.application()
        application.run.side_effect = lambda: callbacks.pop()()
        smoke.exercise_event_loop(application)
        application.on_closing.assert_called_once_with()
        application, callbacks = self.application()
        application.run.side_effect = lambda: callbacks.pop()()
        application.on_closing.side_effect = RuntimeError("owned close failure")
        with self.assertRaisesRegex(RuntimeError, "owned close failure"):
            smoke.exercise_event_loop(application)


if __name__ == "__main__":
    unittest.main()
