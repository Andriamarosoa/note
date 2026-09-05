import unittest

from causal_note.v8_model import calculate_receptive_field
from causal_note.v8_sampling import (
    V8PointExample,
    causal_context_bounds,
    hierarchical_targets,
    hierarchical_weights,
)
from causal_note.v8_targets import (
    exact_count_to_hierarchical,
    slot_targets_to_exact_counts,
)
from causal_note.v81_targets import (
    DEFAULT_OFFSET_HORIZON_SAMPLES,
    DEFAULT_ONSET_HORIZON_SAMPLES,
    cluster_fixed_span,
    response_window_is_empty,
    training_context_bounds,
)


class V8TargetTests(unittest.TestCase):
    def test_exact_count_hierarchy(self):
        target = exact_count_to_hierarchical(0)
        self.assertEqual((target.presence, target.multiplicity_class), (0, 0))
        self.assertEqual(exact_count_to_hierarchical(1).multiplicity_class, 0)
        self.assertEqual(exact_count_to_hierarchical(2).multiplicity_class, 1)
        self.assertEqual(exact_count_to_hierarchical(3).multiplicity_class, 2)
        self.assertEqual(exact_count_to_hierarchical(6).multiplicity_class, 2)

    def test_slot_collapse_preserves_exact_multiplicity(self):
        rows = [
            [0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [1, 0, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 1],
        ]
        self.assertEqual(slot_targets_to_exact_counts(rows), [0, 1, 2, 6])

    def test_receptive_field_keeps_v7_temporal_budget(self):
        self.assertEqual(calculate_receptive_field(), 4093)

    def test_context_ends_exactly_at_query_sample(self):
        self.assertEqual(causal_context_bounds(5000, 4093), (908, 5001, 0))
        self.assertEqual(causal_context_bounds(10, 4093), (0, 11, 4082))

    def test_rare_extra_has_no_presence_weight(self):
        example = V8PointExample(
            track_index=0,
            position=100,
            onset_count=3,
            offset_count=0,
            stratum="rare_multiplicity_extra",
            presence_weight=0.0,
            multiplicity_weight=1.0,
        )
        self.assertEqual(
            hierarchical_targets(example),
            {
                "onset_presence": 1,
                "offset_presence": 0,
                "onset_multiplicity": 2,
                "offset_multiplicity": 0,
            },
        )
        self.assertEqual(
            hierarchical_weights(example),
            {
                "onset_presence": 0.0,
                "offset_presence": 0.0,
                "onset_multiplicity": 1.0,
                "offset_multiplicity": 0.0,
            },
        )


class V81BurstTargetTests(unittest.TestCase):
    def test_data_driven_horizons(self):
        self.assertEqual(DEFAULT_ONSET_HORIZON_SAMPLES, 882)
        self.assertEqual(DEFAULT_OFFSET_HORIZON_SAMPLES, 1323)

    def test_fixed_span_bursts_do_not_chain_grow(self):
        bursts = cluster_fixed_span((100, 105, 110, 121), 10)
        self.assertEqual(
            [(item.start_sample, item.end_sample, item.count) for item in bursts],
            [(100, 110, 3), (121, 121, 1)],
        )
        self.assertEqual(bursts[0].count_class, 2)
        self.assertEqual(bursts[1].count_class, 0)

    def test_response_window_rejects_ambiguous_negative(self):
        positions = (1000, 2000)
        self.assertFalse(response_window_is_empty(positions, 900, 200))
        self.assertTrue(response_window_is_empty(positions, 1200, 200))
        self.assertFalse(
            response_window_is_empty(positions, 780, 200, margin_samples=32)
        )

    def test_training_context_contains_past_and_response_horizon(self):
        self.assertEqual(training_context_bounds(5000, 4093, 1323), (908, 6324, 0))
        self.assertEqual(training_context_bounds(10, 4093, 1323), (0, 1334, 4082))


if __name__ == "__main__":
    unittest.main()
