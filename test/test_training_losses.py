import importlib.util
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


TENSORFLOW_AVAILABLE = importlib.util.find_spec("tensorflow") is not None


@unittest.skipUnless(TENSORFLOW_AVAILABLE, "TensorFlow is not installed")
class ElementwiseTrainingLossTests(unittest.TestCase):
    def test_only_the_positive_slot_receives_the_large_weight(self):
        import tensorflow as tf

        from causal_note.training_losses import elementwise_binary_crossentropy_v1

        y_true = tf.constant([[[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]])
        y_pred = tf.fill((1, 1, 6), 0.5)
        element_weights = tf.constant([[[64.0, 1.0, 1.0, 1.0, 1.0, 1.0]]])
        inputs = tf.keras.Input(shape=(1, 6))
        model = tf.keras.Model(inputs, inputs)
        model.compile(loss=elementwise_binary_crossentropy_v1)

        loss = model.compiled_loss(
            y_true,
            y_pred,
            sample_weight=element_weights,
            regularization_losses=[],
        )

        expected = ((64.0 + 5.0) / 6.0) * math.log(2.0)
        self.assertAlmostEqual(float(loss), expected, places=5)
        self.assertEqual(
            tuple(elementwise_binary_crossentropy_v1(y_true, y_pred).shape),
            (1, 1, 6),
        )

    def test_saved_loss_round_trips_in_a_fresh_process(self):
        import tensorflow as tf

        from causal_note.training_losses import elementwise_binary_crossentropy_v1

        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "elementwise.keras"
            inputs = tf.keras.Input(shape=(1, 6))
            model = tf.keras.Model(inputs, inputs)
            model.compile(loss=elementwise_binary_crossentropy_v1)
            model.save(path)

            source_root = Path(__file__).resolve().parents[1] / "src"
            code = (
                "import math, sys; "
                f"sys.path.insert(0, {str(source_root)!r}); "
                "import tensorflow as tf; "
                "import causal_note.training_losses; "
                "model=tf.keras.models.load_model(sys.argv[1], compile=True); "
                "y=tf.constant([[[1.,0.,0.,0.,0.,0.]]]); "
                "p=tf.fill((1,1,6),0.5); "
                "w=tf.constant([[[64.,1.,1.,1.,1.,1.]]]); "
                "loss=model.compiled_loss(y,p,sample_weight=w," 
                "regularization_losses=[]); "
                "expected=((64.+5.)/6.)*math.log(2.); "
                "assert abs(float(loss)-expected)<1e-5, (loss, expected)"
            )
            completed = subprocess.run(
                [sys.executable, "-B", "-c", code, str(path)],
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
