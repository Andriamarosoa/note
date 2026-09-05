import unittest


class V84ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import tensorflow as tf
        except ImportError as exc:
            raise unittest.SkipTest("TensorFlow unavailable") from exc
        cls.tf = tf

    def test_v81_copy_is_functionally_exact_and_variables_are_disjoint(self):
        from causal_note.v8_model import build_v8_stream_model
        from causal_note.v84_model import build_v84_stream_model, initialize_v84_stream_from_v81

        source = build_v8_stream_model(filters=8)
        target = build_v84_stream_model(filters=8)
        initialize_v84_stream_from_v81(source, target)
        probe = self.tf.reshape(self.tf.linspace(-0.7, 0.7, 257), (1, 257, 1))
        src = source(probe, training=False)
        got = target(probe, training=False)
        for name in ("onset_presence", "offset_presence", "onset_multiplicity", "offset_multiplicity"):
            delta = float(self.tf.reduce_max(self.tf.abs(src[name] - got[name])).numpy())
            self.assertLessEqual(delta, 1e-7, name)

        onset = target.get_layer("v84_onset_stream")
        offset = target.get_layer("v84_offset_stream")
        onset_ids = {id(variable) for variable in onset.trainable_variables}
        offset_ids = {id(variable) for variable in offset.trainable_variables}
        self.assertTrue(onset_ids)
        self.assertTrue(offset_ids)
        self.assertFalse(onset_ids & offset_ids)

    def test_stateful_predictor_matches_slow_oracle(self):
        from causal_note.v8_model import build_v8_stream_model
        from causal_note.v84_model import build_v84_stream_model, initialize_v84_stream_from_v81
        from causal_note.v84_predictor import V84KerasPredictor

        source = build_v8_stream_model(filters=8)
        model = build_v84_stream_model(filters=8)
        initialize_v84_stream_from_v81(source, model)
        fast = V84KerasPredictor(model, use_stateful=True)
        slow = V84KerasPredictor(model, use_stateful=False)
        values = [((i * 17) % 101 - 50) / 50.0 for i in range(777)]
        cursor = 0
        for size in (37, 128, 5, 256, 211, 140):
            chunk = values[cursor:cursor + size]
            a = fast.predict_chunk(chunk, start_sample=cursor)
            b = slow.predict_chunk(chunk, start_sample=cursor)
            self.assertEqual(a.sample_count, b.sample_count)
            for rows_a, rows_b in (
                (a.onset_presence, b.onset_presence),
                (a.offset_presence, b.offset_presence),
            ):
                self.assertLessEqual(max(abs(x-y) for x, y in zip(rows_a, rows_b)), 2e-5)
            cursor += size
        self.assertEqual(cursor, len(values))
        self.assertTrue(fast.stateful_enabled)


if __name__ == "__main__":
    unittest.main()
