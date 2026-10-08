"""Decision-theoretic and alignment checks for a forensic, label-blind replay."""
import unittest

import numpy as np

from scripts.audit_v273_selector_design import (
    aligned_positions, expected_net_replay, legacy_decode, metrics,
    producer_dependencies,
)


class SelectorDesignAuditTests(unittest.TestCase):
    def test_calibrated_multiclass_counterexample(self):
        p = np.array([[.30, .40, .20, .06, .04]])
        regress = np.full((1, 5), .40)
        base = np.array([3])
        old, _ = legacy_decode(p, regress, np.array([.40]), base)
        self.assertEqual(old.tolist(), [2])
        self.assertEqual(expected_net_replay(p, regress, base).tolist(), [3])

    def test_expected_gain_agrees_with_optimal_exact_k_for_coherent_probabilities(self):
        rng = np.random.default_rng(971)
        p = rng.dirichlet(np.ones(5), 500)
        base = rng.integers(2, 5, 500)
        regress = np.broadcast_to(p[np.arange(500), base-2, None], p.shape)
        np.testing.assert_array_equal(expected_net_replay(p, regress, base), p.argmax(1)+2)

    def test_keep_ties_and_noop_mask(self):
        p = np.full((3, 5), .2)
        base = np.array([2, 3, 4])
        np.testing.assert_array_equal(expected_net_replay(p, p, base), base)
        # Own-K auxiliary score must not create a correction candidate.
        correct = np.zeros((3, 5)); correct[np.arange(3), base-2] = 1.
        np.testing.assert_array_equal(expected_net_replay(correct, p, base), base)

    def test_id_order_is_explicit_and_invalid_ids_rejected(self):
        np.testing.assert_array_equal(aligned_positions(np.array([20, 10, 30]),
                                                        np.array([30, 20])), [2, 0])
        for ids, selected in [([20, 20], [20]), ([10, 20], [20, 20]), ([10, 20], [30])]:
            with self.assertRaises(ValueError):
                aligned_positions(np.array(ids), np.array(selected))

    def test_corrections_regressions_and_neutral_changes(self):
        y = np.array([3, 3, 4, 1]); b = np.array([2, 3, 2, 1]); p = np.array([3, 2, 3, 1])
        m = metrics(y, b, p)["global_metrics"]
        self.assertEqual((m["corrections"], m["regressions"], m["changes"], m["net"]),
                         (1, 1, 3, 0))

    def test_provenance_detects_inner_reentry_without_outer_leakage(self):
        folds = np.repeat([0, 1, 2, 4], 3)
        truth = np.tile([0, 2, 3], 4)
        rows = producer_dependencies(truth, folds, np.arange(12))
        self.assertEqual(len(rows), 24)
        self.assertTrue(all(r["receiver_ids_in_producer_train"] == 2 for r in rows))
        self.assertTrue(all(r["outer_ids_in_producer_train"] == 0 for r in rows))


if __name__ == "__main__":
    unittest.main()
