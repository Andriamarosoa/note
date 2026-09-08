import unittest
import numpy as np
from scripts.audit_v240_selection import score, aggregate, ranked_ids, compare_maps


class SelectionAuditTests(unittest.TestCase):
    def test_aggregate_uses_counts_not_mean_f1(self):
        a = score({'a': (100,)}, {'a': (100,)}, 0)
        b = score({'b': tuple(range(9))}, {'b': ()}, 0)
        total = aggregate([a,b])
        self.assertEqual(total['true_positive'], 1)
        self.assertAlmostEqual(total['f1'], 2/11)

    def test_ranking_masks_padding_and_caps_count(self):
        ids = ranked_ids(np.array([[.1,.9,.8]]), np.array([[1,0,1]]), [3])
        np.testing.assert_array_equal(ids, [[2,0,-1,-1,-1,-1]])

    def test_same_count_can_hide_selection_at_wide_tolerance(self):
        refs = {'a': (1000,)}
        first, second = {'a': (1000,)}, {'a': (2000,)}
        self.assertEqual(score(refs, first, 2205), score(refs, second, 2205))
        self.assertNotEqual(score(refs, first, 220), score(refs, second, 220))
        self.assertEqual(compare_maps(first, second)['tracks_with_different_timestamps'], 1)
        self.assertEqual(compare_maps(first, second)['tracks_with_different_counts'], 0)

    def test_duplicate_times_preserve_event_multiplicity(self):
        r = score({'a': (100,100)}, {'a': (100,100)}, 0)
        self.assertEqual(r['true_positive'], 2)

    def test_zero_count_emits_nothing(self):
        np.testing.assert_array_equal(ranked_ids(np.ones((1,3)), np.ones((1,3)), [0]), [[-1]*6])
