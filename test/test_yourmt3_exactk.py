"""Independent checks for native onset assignment and paired error accounting."""
import unittest
import numpy as np
from scripts.yourmt3_exactk_common import assign_onsets, cluster_ends, metrics, paired


class YourMT3ExactKTests(unittest.TestCase):
    def test_endpoint_bridge_matches_full_original_candidates(self):
        rng = np.random.default_rng(912)
        for _ in range(50):
            groups, start = [], 0
            for _ in range(12):
                width = int(rng.integers(0, 1765))
                group = np.unique(np.r_[start, start+width, rng.integers(start, start+width+1, 100)])
                groups.append(group)
                start += width + int(rng.integers(1, 2000))
            refs = rng.integers(-1000, start+1000, 200)
            expected = []
            for ref in refs:
                choices = [(int(np.abs(g-ref).min()), j) for j, g in enumerate(groups)]
                distance, j = min(choices)
                expected.append(j if distance <= 882 else -1)
            counts, assigned = assign_onsets(refs, [g[0] for g in groups], [g[-1] for g in groups])
            np.testing.assert_array_equal(assigned, expected)
            self.assertEqual(int(counts.sum()), sum(j >= 0 for j in expected))

    def test_retained_geometry_recovers_missing_endpoints_and_singleton(self):
        starts = np.array([100, 2500, 5000])
        ends = np.array([1864, 2900, 5000])
        samples = np.array([[500, 1000], [2650, 2799], [5000, 5000]])
        centers = (starts+ends)/2
        sequence = np.stack([(samples-starts[:, None])/1764,
                             (samples-centers[:, None])/1764], axis=-1).astype(np.float32)
        stats = np.zeros((3, 8), np.float32)
        stats[:, 1] = np.maximum(1, ends-starts)/1764
        result = cluster_ends(sequence, np.ones((3, 2)), starts, stats)
        np.testing.assert_array_equal(result, ends)

    def test_boundary_tie_and_radius_are_exact(self):
        counts, assigned = assign_onsets([-883, -882, 882, 883], [0, 1764], [0, 1764])
        np.testing.assert_array_equal(assigned, [-1, 0, 0, 1])
        np.testing.assert_array_equal(counts, [2, 1])

    def test_sustained_note_is_not_recounted_in_later_groups(self):
        counts, _ = assign_onsets([0], [0, 2205, 4410], [0, 2205, 4410])
        np.testing.assert_array_equal(counts, [1, 0, 0])

    def test_distinct_notes_at_same_onset_keep_multiplicity(self):
        counts, _ = assign_onsets([100, 100, 100], [100], [100])
        np.testing.assert_array_equal(counts, [3])

    def test_overlap_is_rejected(self):
        with self.assertRaises(ValueError):
            assign_onsets([0], [0, 10], [20, 30])

    def test_error_and_net_accounting(self):
        y, base, pred = [0, 1, 2, 3, 6], [0, 0, 2, 2, 5], [1, 1, 3, 3, 6]
        m, d = metrics(y, pred), paired(y, base, pred)
        self.assertEqual((m['correct'], m['under'], m['over']), (3, 0, 2))
        self.assertEqual(d['global'], dict(corrections=3, regressions=2, net=1, changed=5))
        self.assertEqual(d['poly']['net'], 1)
        self.assertIsNone(m['by_k']['4']['exact'])


if __name__ == '__main__':
    unittest.main()
