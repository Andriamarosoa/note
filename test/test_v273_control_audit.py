"""Independent decision accounting and annotation-assignment audit checks."""
import unittest
import numpy as np
from scripts.audit_v273_control_fold3 import scalar_decode, effect, classify_errors, PAIRS
from scripts.audit_v273_control_inputs import nearest_assignment
from scripts.train_v273_native_paired import decode
from scripts.audit_v273_candidate_alignment import possible_origins
from scripts.train_v92_string_factorized_cardinality import _recover_cluster_start


class ControlAuditTests(unittest.TestCase):
    def test_independent_decoder_random_ties_and_threshold_boundaries(self):
        rng = np.random.default_rng(301)
        p = {name: rng.dirichlet(np.ones(size), 1600).astype(np.float32)
             for name, size in [('v260_uniform', 7), ('v260_weighted', 7), ('v272_uniform', 5)]}
        anchor = rng.integers(0, 7, 1600)
        p['v260_uniform'][0] = [0.5, 0.5, 0, 0, 0, 0, 0]
        p['v260_weighted'][0] = [0, 0, 0.75, 0.25, 0, 0, 0]
        anchor[0] = 0
        p['v272_uniform'][0] = [0.25, 0.75, 0, 0, 0]
        for threshold in [0., .5, .75, 1.000001]:
            cal = {'v271': {'threshold': threshold}, 'v273': {
                'thresholds': {f'{a}_to_{b}': threshold for a, b in PAIRS}}}
            got = scalar_decode(anchor, p, cal)
            np.testing.assert_array_equal(got['final'], decode(anchor, p, cal))
            if threshold == .5:
                self.assertEqual(got['final'][0], 3)  # both >= comparisons include equality
            if threshold == .75:
                self.assertEqual(got['base'][0], 2)
                self.assertEqual(got['final'][0], 2)

    def test_invalid_probabilities_are_rejected(self):
        p = {'v260_uniform': np.zeros((1, 7)), 'v260_weighted': np.zeros((1, 7)),
             'v272_uniform': np.zeros((1, 5))}
        with self.assertRaisesRegex(RuntimeError, 'unnormalized'):
            scalar_decode([0], p, {})

    def test_error_partition_and_wrong_to_wrong_accounting(self):
        k = np.array([2, 5, 3, 4, 6, 2])
        stage = {'base': np.array([0, 3, 2, 3, 4, 2]),
                 'final': np.array([0, 3, 2, 3, 4, 2]),
                 'specialist': np.array([2, 5, 3, 3, 4, 2]),
                 'uniform': np.array([0, 1, 1, 4, 2, 2]),
                 'weighted': np.array([0, 1, 1, 3, 3, 2])}
        cats = classify_errors(k, stage)
        self.assertEqual([int(x.sum()) for x in cats.values()], [1, 1, 1, 1, 1])
        got = effect(np.array([3, 3, 3]), np.array([4, 2, 3]), np.array([2, 3, 4]))
        self.assertEqual((got['fixed'], got['broken'], got['net_correct']), (1, 1, 0))

    def test_assignment_boundary_tie_duplicate_notes(self):
        samples = np.array([1000, 2000, 2000, 4000])
        rows = np.array([1, 3, 2, 4])
        self.assertEqual(nearest_assignment(1500, samples, rows), (1, 500))
        self.assertEqual(nearest_assignment(2000, samples, rows), (2, 0))
        self.assertEqual(nearest_assignment(4882, samples, rows), (4, 882))
        self.assertIsNone(nearest_assignment(4883, samples, rows))

    def test_legacy_origin_bug_and_rank_ambiguity(self):
        seq = np.zeros((2, 5), np.float16)
        seq[:, -2] = [0, 200 / 1764]
        seq[:, -3] = [.9, .8]
        mask, top = np.ones(2), np.array([1000] * 6)
        # The real first/highest-score candidate is at 1000; the next is at 1200.
        self.assertEqual(_recover_cluster_start(seq, mask, top), (800, 6))
        self.assertEqual(possible_origins(seq, mask, top), ([800, 1000], [1000]))
        seq[:, -3] = .9
        self.assertEqual(possible_origins(seq, mask, top), ([800, 1000], [800, 1000]))

    def test_all_integer_relative_times_survive_float16(self):
        expected = np.arange(1765)
        stored = (expected.astype(np.float32) / 1764).astype(np.float16)
        np.testing.assert_array_equal(np.rint(stored.astype(np.float64) * 1764), expected)


if __name__ == '__main__':
    unittest.main()
