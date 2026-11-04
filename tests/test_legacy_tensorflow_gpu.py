"""Opt-in smoke test for the legacy TensorFlow GPU training path."""
import os
from types import SimpleNamespace
import unittest


@unittest.skipUnless(os.environ.get("RUN_LEGACY_TF_GPU_TEST") == "1", "Requires TensorFlow and an NVIDIA GPU")
class LegacyTensorFlowGpuTests(unittest.TestCase):
    def test_one_training_epoch_on_gpu(self):
        import numpy as np
        import tensorflow as tf
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
        with tf.device("/GPU:0"):
            result = dl.train_PINN(
                model, x, x, y_t, y_b, epochs=1,
                mask_T=masks, mask_B=masks, batch_size=4,
            )
        self.assertTrue(result)


if __name__ == "__main__":
    unittest.main()
