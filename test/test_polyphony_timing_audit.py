import unittest

import numpy as np

from scripts.audit_v273_polyphony_timing import assign, features


class PolyphonyTimingTest(unittest.TestCase):
    def describe(self, events, selected):
        ev = np.asarray(events, np.int64)
        return features(*ev.T, np.asarray(selected, bool))

    def test_successive_onsets_can_overlap(self):
        result = self.describe([(0, 2000, 60, 0), (900, 1900, 64, 1)], [True, True])
        self.assertEqual(result["span_samples"], 900)
        self.assertEqual(result["common_overlap_samples"], 1000)
        self.assertEqual(result["overlapping_pairs"], 1)
        self.assertEqual(result["peak_current_notes"], 2)

    def test_exactly_touching_notes_do_not_overlap(self):
        result = self.describe([(0, 900, 60, 0), (900, 1900, 64, 0)], [True, True])
        self.assertEqual(result["common_overlap_samples"], 0)
        self.assertEqual(result["overlapping_pairs"], 0)
        self.assertEqual(result["peak_current_notes"], 1)
        self.assertEqual(result["same_string_extra_attacks"], 1)

    def test_partial_overlap_not_all_together(self):
        result = self.describe([(0, 1000, 60, 0), (900, 2000, 64, 1), (1900, 2900, 67, 2)], [True]*3)
        self.assertEqual(result["common_overlap_samples"], 0)
        self.assertEqual(result["overlapping_pairs"], 2)
        self.assertEqual(result["peak_current_notes"], 2)

    def test_one_new_note_with_old_note_and_offset_boundary(self):
        result = self.describe([(0, 2000, 48, 0), (10, 1000, 55, 1), (1000, 3000, 60, 2)], [False, False, True])
        self.assertEqual(result["carried"], 1)
        self.assertEqual(result["carried_20ms"], 1)
        self.assertEqual(result["carried_50ms"], 0)
        self.assertEqual(result["k"], 1)
        self.assertEqual(result["previous_attack_gap_samples"], 990)

    def test_neighbour_attack_does_not_become_old_note_at_first_onset(self):
        result = self.describe([(0, 2000, 60, 0), (0, 2000, 64, 1), (500, 2000, 67, 2)], [True, False, True])
        self.assertEqual(result["carried"], 0)
        self.assertEqual(result["foreign_during_any_attack"], 1)

    def test_original_radius_and_owner_tie(self):
        candidates = np.array([1000, 1200, 5000])
        owners = np.array([3, 1, 2])
        np.testing.assert_array_equal(assign(np.array([1100, 118, 117, 5882, 5883]), candidates, owners), [1, 3, -1, 2, -1])

    def test_empty_group(self):
        result = self.describe([(0, 2000, 60, 0)], [False])
        self.assertEqual(result["k"], 0)
        self.assertEqual(result["span_samples"], -1)


if __name__ == "__main__":
    unittest.main()
