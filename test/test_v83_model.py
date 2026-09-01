import unittest


class V83ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import numpy as np
            import tensorflow as tf
        except ImportError as exc:
            raise unittest.SkipTest(str(exc))
        cls.np = np
        cls.tf = tf

    def _models(self):
        from causal_note.v8_model import build_v8_stream_model
        from causal_note.v83_model import build_v83_stream_model, initialize_v83_stream_from_v81

        tf = self.tf
        tf.keras.utils.set_random_seed(1234)
        rates = (1, 2, 4, 8, 16, 32)
        source = build_v8_stream_model(
            filters=4,
            kernel_size=3,
            dilation_rates=rates,
            name="test_v81_source",
        )
        target = build_v83_stream_model(
            filters=4,
            kernel_size=3,
            dilation_rates=rates,
            split_after=3,
            name="test_v83_target",
        )
        initialize_v83_stream_from_v81(source, target, split_after=3)
        return source, target

    def test_v81_copy_is_functionally_exact_before_training(self):
        source, target = self._models()
        tf = self.tf
        probe = tf.reshape(tf.linspace(-0.75, 0.75, 257), (1, 257, 1))
        source_outputs = source(probe, training=False)
        target_outputs = target(probe, training=False)
        for name in source_outputs:
            delta = tf.reduce_max(tf.abs(source_outputs[name] - target_outputs[name]))
            self.assertLessEqual(float(delta.numpy()), 1e-6, name)

    def test_private_towers_are_distinct_weight_objects_with_equal_initial_values(self):
        _source, target = self._models()
        np = self.np
        onset = target.get_layer("v83_onset_feature_4")
        offset = target.get_layer("v83_offset_feature_4")
        self.assertIsNot(onset, offset)
        self.assertIsNot(onset.kernel, offset.kernel)
        np.testing.assert_allclose(onset.get_weights()[0], offset.get_weights()[0], rtol=0, atol=0)
        np.testing.assert_allclose(onset.get_weights()[1], offset.get_weights()[1], rtol=0, atol=0)

    def test_onset_output_has_no_gradient_to_offset_private_tower(self):
        _source, target = self._models()
        tf = self.tf
        probe = tf.ones((1, 96, 1), dtype=tf.float32) * 0.1
        offset_layer = target.get_layer("v83_offset_feature_4")
        with tf.GradientTape() as tape:
            outputs = target(probe, training=True)
            loss = tf.reduce_sum(outputs["onset_presence"])
        grads = tape.gradient(loss, offset_layer.trainable_variables)
        self.assertTrue(all(grad is None for grad in grads))

    def test_stateful_predictor_matches_slow_oracle(self):
        from causal_note.v83_predictor import V83KerasPredictor

        _source, target = self._models()
        np = self.np
        receptive_field = int(target.receptive_field)
        fast = V83KerasPredictor(target, receptive_field=receptive_field, use_stateful=True)
        slow = V83KerasPredictor(target, receptive_field=receptive_field, use_stateful=False)
        self.assertTrue(fast.stateful_enabled)
        values = tuple(float(v) for v in np.linspace(-0.2, 0.3, 173))
        position = 0
        for size in (17, 64, 23, 69):
            chunk = values[position:position + size]
            fast_scores = fast.predict_chunk(chunk, start_sample=position)
            slow_scores = slow.predict_chunk(chunk, start_sample=position)
            np.testing.assert_allclose(fast_scores.onset_presence, slow_scores.onset_presence, rtol=1e-5, atol=1e-6)
            np.testing.assert_allclose(fast_scores.offset_presence, slow_scores.offset_presence, rtol=1e-5, atol=1e-6)
            np.testing.assert_allclose(fast_scores.onset_multiplicity, slow_scores.onset_multiplicity, rtol=1e-5, atol=1e-6)
            np.testing.assert_allclose(fast_scores.offset_multiplicity, slow_scores.offset_multiplicity, rtol=1e-5, atol=1e-6)
            position += size


if __name__ == "__main__":
    unittest.main()
