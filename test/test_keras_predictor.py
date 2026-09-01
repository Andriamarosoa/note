import unittest

from causal_note.detector import BoundaryType, LiveModelDetector
from causal_note.keras_predictor import KerasBoundaryPredictor


class _Array:
    def __init__(self, values):
        self._values = values

    def numpy(self):
        return self

    def tolist(self):
        return self._values


class _CausalFakeModel:
    output_shape = {
        "onset": (None, None, 2),
        "offset": (None, None, 2),
    }
    receptive_field = 3

    def __init__(self):
        self.call_lengths = []

    def __call__(self, batch, training=False):
        self.call_lengths.append(len(batch[0]))
        onset = []
        offset = []
        for (sample,) in batch[0]:
            onset.append([1.0 if sample == 1.0 else 0.0, 0.0])
            offset.append([1.0 if sample == -1.0 else 0.0, 0.0])
        return {
            "onset": _Array([onset]),
            "offset": _Array([offset]),
        }


class KerasBoundaryPredictorTests(unittest.TestCase):
    def test_warm_up_does_not_consume_audio_or_context(self):
        model = _CausalFakeModel()
        predictor = KerasBoundaryPredictor(model)
        predictor.warm_up(4)
        scores = predictor.predict_chunk((1.0, 0.0), start_sample=10)

        self.assertEqual(model.call_lengths, [4, 2])
        self.assertEqual(scores.start_sample, 10)
        self.assertEqual(scores.sample_count, 2)

    def test_context_is_reused_and_only_new_scores_are_decoded(self):
        model = _CausalFakeModel()
        predictor = KerasBoundaryPredictor(model)
        detector = LiveModelDetector(predictor)

        first = detector.process_chunk((0.0, 1.0), start_sample=10)
        second = detector.process_chunk((0.0, -1.0))

        self.assertEqual(model.call_lengths, [2, 4])
        self.assertEqual(
            tuple((event.kind, event.event_id, event.sample) for event in first + second),
            (
                (BoundaryType.ONSET, "event-000001", 11),
                (BoundaryType.OFFSET, "event-000001", 13),
            ),
        )

    def test_noncontiguous_predictor_input_is_rejected(self):
        predictor = KerasBoundaryPredictor(_CausalFakeModel())
        predictor.predict_chunk((0.0,), start_sample=5)
        with self.assertRaises(ValueError):
            predictor.predict_chunk((0.0,), start_sample=7)

    def test_invalid_model_shape_is_rejected(self):
        model = _CausalFakeModel()
        model.output_shape = {"other": (None, None, 2)}
        with self.assertRaises(ValueError):
            KerasBoundaryPredictor(model)

    def test_sequence_output_shapes_are_mapped_by_keras_output_names(self):
        model = _CausalFakeModel()
        model.output_names = ["onset", "offset"]
        model.output_shape = [
            (None, None, 2),
            (None, None, 2),
        ]
        self.assertEqual(KerasBoundaryPredictor(model).slot_count, 2)

    def test_onset_and_offset_slot_counts_must_match(self):
        model = _CausalFakeModel()
        model.output_shape = {
            "onset": (None, None, 2),
            "offset": (None, None, 3),
        }
        with self.assertRaises(ValueError):
            KerasBoundaryPredictor(model)


if __name__ == "__main__":
    unittest.main()
