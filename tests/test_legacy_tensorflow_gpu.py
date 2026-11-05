"""Opt-in smoke test for the legacy TensorFlow GPU training path."""
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


@unittest.skipUnless(os.environ.get("RUN_LEGACY_TF_GPU_TEST") == "1", "Requires TensorFlow and an NVIDIA GPU")
class LegacyTensorFlowGpuTests(unittest.TestCase):
    def test_one_training_epoch_on_gpu(self):
        import numpy as np
        import tensorflow as tf
        from tensorflow.tsl.profiler.protobuf import xplane_pb2
        import deepLearning as dl

        self.assertTrue(tf.config.list_physical_devices("GPU"), "TensorFlow did not detect a GPU")
        with tf.device("/GPU:0"):
            product = tf.linalg.matmul(tf.ones((8, 8)), tf.ones((8, 8)))
        self.assertIn("GPU:0", product.device)
        x = np.array([[0., 0.], [.25, .25], [.5, .5], [1., 1.]], dtype=np.float32)
        y_t = np.array([[290.], [295.], [300.], [305.]], dtype=np.float32)
        y_b = np.array([[.1], [.2], [.3], [.4]], dtype=np.float32)
        masks = np.ones((4, 1), dtype=np.float32)
        model = dl.CustomPINNModel(
            input_dim=2, output_dim=1, alpha=.1, T_m=300., T_a=290.,
            boundary_indices=[], x_arr=np.arange(2), t_arr=np.arange(2),
            pcm=SimpleNamespace(rho=1., c=1.), x_input=x,
            bound_mask_array=masks, temp_mask_array=masks, y_T=y_t, y_B=y_b,
        )
        initial_weights = []

        class SnapshotLogger(dl.PINNConsoleLogger):
            def on_train_begin(self, logs=None):
                initial_weights.extend(v.numpy().copy() for v in self.model.trainable_variables)
                super().on_train_begin(logs)

        with tempfile.TemporaryDirectory(prefix="pinn-gpu-profile-") as trace_directory:
            # Capture only training: the standalone GPU operation above cannot
            # satisfy these assertions if fit silently falls back to the CPU.
            tf.profiler.experimental.start(trace_directory)
            try:
                with patch.object(dl, "PINNConsoleLogger", SnapshotLogger), tf.device("/GPU:0"):
                    loss, metrics, temperature, boundary = dl.train_PINN(
                        model, x, x, y_t, y_b, epochs=1,
                        mask_T=masks, mask_B=masks, batch_size=4,
                    )
            finally:
                tf.profiler.experimental.stop()
            gpu_operations = set()
            traces = list(Path(trace_directory).rglob("*.xplane.pb"))
            self.assertTrue(traces, "TensorFlow did not produce a training device trace")
            for trace_path in traces:
                trace = xplane_pb2.XSpace.FromString(trace_path.read_bytes())
                for plane in trace.planes:
                    if not plane.name.startswith("/device:GPU:"):
                        continue
                    for line in plane.lines:
                        for event in line.events:
                            for stat in event.stats:
                                if plane.stat_metadata[stat.metadata_id].name == "tf_op":
                                    operation = (plane.stat_metadata[stat.ref_value].name
                                                 if stat.WhichOneof("value") == "ref_value"
                                                 else stat.str_value)
                                    gpu_operations.add(operation)
            self.assertTrue(any("MatMul" in operation and "gradient_tape" in operation
                                for operation in gpu_operations),
                            "No PINN backpropagation matrix multiplication executed on the GPU")
        self.assertTrue(initial_weights, "Training began without trainable weights")
        final_weights = [v.numpy() for v in model.trainable_variables]
        self.assertEqual(len(initial_weights), len(final_weights))
        self.assertTrue(any(not np.array_equal(before, after)
                            for before, after in zip(initial_weights, final_weights)),
                        "One epoch did not update any trainable weights")
        self.assertGreater(int(model.optimizer.iterations.numpy()), 0)
        for values in (loss, metrics["raw_loss"], metrics["scaled_accuracy_T"],
                       metrics["scaled_accuracy_B"], metrics["physics_loss"]):
            self.assertEqual(len(values), 1)
            self.assertTrue(np.isfinite(values).all())
        for prediction in (temperature, boundary):
            self.assertEqual(prediction.shape, (len(x), 1))
            self.assertTrue(np.isfinite(prediction).all())


if __name__ == "__main__":
    unittest.main()
