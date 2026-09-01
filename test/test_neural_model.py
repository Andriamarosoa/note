import importlib.util
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from causal_note.neural_model import (
    EVENT_SLOTS,
    OUTPUT_NAMES,
    build_causal_boundary_model,
    calculate_receptive_field,
)
from causal_note.keras_predictor import KerasBoundaryPredictor


TENSORFLOW_AVAILABLE = importlib.util.find_spec("tensorflow") is not None


class NeuralModelModuleTests(unittest.TestCase):
    def test_output_names_are_the_two_boundaries(self) -> None:
        self.assertEqual(OUTPUT_NAMES, ("onset", "offset"))

    def test_import_does_not_load_tensorflow(self) -> None:
        source_root = Path(__file__).resolve().parents[1] / "src"
        code = (
            "import sys; "
            f"sys.path.insert(0, {str(source_root)!r}); "
            "import causal_note.neural_model; "
            "raise SystemExit('tensorflow' in sys.modules)"
        )
        completed = subprocess.run(
            [sys.executable, "-c", code],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_receptive_field_matches_dilated_stack(self) -> None:
        self.assertEqual(calculate_receptive_field(5, (1, 2, 4, 8)), 61)
        self.assertEqual(calculate_receptive_field(1, (1,)), 1)
        self.assertEqual(calculate_receptive_field(), 4093)

    def test_invalid_parameters_are_rejected_before_tensorflow_is_needed(self) -> None:
        for invalid in (0, -1, 1.5, True):
            with self.subTest(filters=invalid):
                with self.assertRaises(ValueError):
                    build_causal_boundary_model(filters=invalid)
            with self.subTest(kernel_size=invalid):
                with self.assertRaises(ValueError):
                    calculate_receptive_field(invalid, (1,))

        for invalid_rates in ((), (0,), (1, -2), (True,), "1,2"):
            with self.subTest(dilation_rates=invalid_rates):
                with self.assertRaises(ValueError):
                    calculate_receptive_field(3, invalid_rates)

        with self.assertRaises(ValueError):
            build_causal_boundary_model(name="  ")


@unittest.skipUnless(TENSORFLOW_AVAILABLE, "TensorFlow is not installed")
class NeuralModelTensorFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import tensorflow as tf

        cls.tf = tf

    def tearDown(self) -> None:
        self.tf.keras.backend.clear_session()

    @staticmethod
    def _as_mapping(model, outputs):
        if isinstance(outputs, dict):
            return outputs
        if isinstance(outputs, (tuple, list)):
            return dict(zip(model.output_names, outputs))
        raise AssertionError(f"unexpected model outputs: {type(outputs)!r}")

    def test_output_shapes_and_all_convolutions_are_causal(self) -> None:
        model = build_causal_boundary_model(
            filters=8,
            kernel_size=3,
            dilation_rates=(1, 2, 4),
        )
        outputs = self._as_mapping(
            model,
            model(self.tf.zeros((2, 96, 1)), training=False),
        )

        self.assertEqual(set(outputs), set(OUTPUT_NAMES))
        for output_name in OUTPUT_NAMES:
            self.assertEqual(tuple(outputs[output_name].shape), (2, 96, EVENT_SLOTS))

        convolutions = [
            layer
            for layer in model.layers
            if isinstance(layer, self.tf.keras.layers.Conv1D)
        ]
        self.assertTrue(convolutions)
        self.assertTrue(all(layer.padding == "causal" for layer in convolutions))
        self.assertEqual(model.receptive_field, 15)

    def test_changing_future_audio_does_not_change_past_outputs(self) -> None:
        self.tf.random.set_seed(17)
        model = build_causal_boundary_model(
            filters=8,
            kernel_size=3,
            dilation_rates=(1, 2, 4),
        )
        split = 48
        past = self.tf.random.normal((1, split, 1))
        baseline = self.tf.concat(
            (past, self.tf.zeros((1, 48, 1))),
            axis=1,
        )
        changed = self.tf.concat(
            (past, self.tf.random.normal((1, 48, 1))),
            axis=1,
        )
        baseline_outputs = self._as_mapping(model, model(baseline, training=False))
        changed_outputs = self._as_mapping(model, model(changed, training=False))

        for output_name in OUTPUT_NAMES:
            self.tf.debugging.assert_near(
                baseline_outputs[output_name][:, :split],
                changed_outputs[output_name][:, :split],
                atol=1e-7,
                rtol=1e-7,
            )

    def test_model_can_be_saved_and_reloaded(self) -> None:
        model = build_causal_boundary_model(
            filters=4,
            kernel_size=3,
            dilation_rates=(1, 2),
        )
        probe = self.tf.random.normal((1, 32, 1))
        expected = self._as_mapping(model, model(probe, training=False))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "causal-boundary.keras"
            model.save(path)
            reloaded = self.tf.keras.models.load_model(path)
            actual = self._as_mapping(reloaded, reloaded(probe, training=False))

        self.assertEqual(set(actual), set(OUTPUT_NAMES))
        for output_name in OUTPUT_NAMES:
            self.assertEqual(tuple(actual[output_name].shape), (1, 32, EVENT_SLOTS))
            self.tf.debugging.assert_near(
                expected[output_name],
                actual[output_name],
                atol=1e-7,
                rtol=1e-7,
            )

    def test_one_synthetic_training_step(self) -> None:
        model = build_causal_boundary_model(
            filters=4,
            kernel_size=3,
            dilation_rates=(1, 2),
        )
        losses = {
            output_name: self.tf.keras.losses.BinaryCrossentropy()
            for output_name in OUTPUT_NAMES
        }
        model.compile(
            optimizer=self.tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss=losses,
        )

        batch_size = 2
        time_steps = 64
        starts = self.tf.constant([8, 16], dtype=self.tf.int32)
        ends = self.tf.constant([40, 48], dtype=self.tf.int32)
        slots = self.tf.constant([0, 1], dtype=self.tf.int32)
        slot_mask = self.tf.one_hot(slots, EVENT_SLOTS)[:, None, :]
        onset = self.tf.one_hot(starts, time_steps)[:, :, None] * slot_mask
        offset = self.tf.one_hot(ends, time_steps)[:, :, None] * slot_mask
        audio = self.tf.random.normal((batch_size, time_steps, 1))

        metrics = model.train_on_batch(
            audio,
            {"onset": onset, "offset": offset},
            return_dict=True,
        )
        self.assertIn("loss", metrics)
        self.assertTrue(math.isfinite(float(metrics["loss"])))

    def test_streaming_adapter_accepts_the_real_keras_model(self) -> None:
        model = build_causal_boundary_model(
            filters=2,
            kernel_size=3,
            dilation_rates=(1, 2),
        )
        predictor = KerasBoundaryPredictor(model)
        first = predictor.predict_chunk((0.0,) * 16, start_sample=0)
        second = predictor.predict_chunk((0.0,) * 8, start_sample=16)

        self.assertEqual((first.sample_count, first.slot_count), (16, EVENT_SLOTS))
        self.assertEqual((second.sample_count, second.slot_count), (8, EVENT_SLOTS))

    def test_stateful_streaming_scores_equal_full_causal_inference(self) -> None:
        self.tf.random.set_seed(23)
        model = build_causal_boundary_model(
            filters=3,
            kernel_size=3,
            dilation_rates=(1, 2, 4),
        )
        audio = self.tf.random.normal((1, 79, 1))
        expected = self._as_mapping(model, model(audio, training=False))
        values = tuple(float(value) for value in audio.numpy()[0, :, 0])

        predictor = KerasBoundaryPredictor(model)
        chunks = (
            predictor.predict_chunk(values[:17], start_sample=0),
            predictor.predict_chunk(values[17:48], start_sample=17),
            predictor.predict_chunk(values[48:], start_sample=48),
        )

        for output_name in ("onset", "offset"):
            actual = self.tf.constant(
                [
                    row
                    for chunk in chunks
                    for row in getattr(chunk, output_name)
                ],
                dtype=self.tf.float32,
            )[None, :, :]
            self.tf.debugging.assert_near(
                actual,
                expected[output_name],
                atol=1e-6,
                rtol=1e-6,
            )


if __name__ == "__main__":
    unittest.main()
