import unittest

from causal_note.v8_predictor import V8KerasPredictor


class FakeModel:
    def __call__(self, batch, training=False):
        values = batch.tolist()[0] if hasattr(batch, "tolist") else batch[0]
        flat = [float(row[0]) for row in values]
        return {
            "onset_presence": [[[1.0 if value > 0.5 else 0.0] for value in flat]],
            "offset_presence": [[[1.0 if value < -0.5 else 0.0] for value in flat]],
            "onset_multiplicity": [[
                [1.0, 0.0, 0.0] if value < 0.8 else [0.0, 1.0, 0.0]
                for value in flat
            ]],
            "offset_multiplicity": [[[1.0, 0.0, 0.0] for _ in flat]],
        }


class V8PredictorTests(unittest.TestCase):
    def test_chunk_lengths_and_positions(self):
        predictor = V8KerasPredictor(FakeModel(), receptive_field=4)
        scores = predictor.predict_chunk((0.0, 0.7, 0.9), start_sample=0)
        self.assertEqual(scores.start_sample, 0)
        self.assertEqual(scores.sample_count, 3)
        self.assertEqual(scores.onset_presence, (0.0, 1.0, 1.0))
        second = predictor.predict_chunk((-0.8, 0.1), start_sample=3)
        self.assertEqual(second.offset_presence, (1.0, 0.0))

    def test_noncontiguous_chunk_rejected(self):
        predictor = V8KerasPredictor(FakeModel(), receptive_field=4)
        predictor.predict_chunk((0.0,), start_sample=0)
        with self.assertRaises(ValueError):
            predictor.predict_chunk((0.0,), start_sample=2)

    def test_reset_allows_new_stream(self):
        predictor = V8KerasPredictor(FakeModel(), receptive_field=4)
        predictor.predict_chunk((0.0,), start_sample=0)
        predictor.reset()
        predictor.predict_chunk((0.0,), start_sample=100)
        self.assertEqual(predictor.next_sample, 101)


if __name__ == "__main__":
    unittest.main()
