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


if __name__ == "__main__":
    unittest.main()
