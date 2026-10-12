"""Contracts for experimental S60-S64 K-specific risk heads."""
import unittest
import numpy as np

from scripts.experimental_v273_regression_heads import (
    TRANSITIONS, head_masks, audited_training_targets, countervote_features,
    new_nonlinear_head, apply_veto
)


class TestExperimentalRiskHeads(unittest.TestCase):
    def test_six_nonoverlapping_heads(self):
        base = np.array([3,2,4,3,0,1,4,2])
        prop = np.array([2,3,3,4,1,0,2,2])
        groups = head_masks(base,prop)
        self.assertEqual(set(groups), set(TRANSITIONS)|{"H_other"})
        for i,head in enumerate(("H3_under","H3_over","H4_under","H4_over","H01","H01","H_other")):
            self.assertTrue(groups[head][i])
        self.assertFalse(any(mask[7] for mask in groups.values()))

    def test_training_target_encoding(self):
        base = np.array([3,2,4,1,2])
        prop = np.array([2,3,3,0,2])
        truth = np.array([3,3,1,2,2])
        # Regression +1; correction -1; neutral 0; unchanged 0.
        np.testing.assert_array_equal(
            audited_training_targets(truth,base,prop),[1,-1,0,0,0])

    def test_veto_preserves_other_decisions(self):
        base = np.array([3,2,4,0])
        prop = np.array([2,3,3,1])
        out = apply_veto(base,prop,np.array([True,False,True,False]))
        np.testing.assert_array_equal(out,[3,3,4,1])
        with self.assertRaises(ValueError):
            apply_veto(np.array([0]),np.array([0]),np.array([True]))

    def test_vote_counts_do_not_use_truth(self):
        policies = np.tile(np.array([3]*21+[2]*15), (2,1))
        base=np.array([3,2]);proposal=np.array([2,3])
        info=countervote_features(policies,base,proposal)
        np.testing.assert_array_equal(info["baseline_count"],[21,15])
        np.testing.assert_array_equal(info["candidate_count"],[15,21])
        np.testing.assert_array_equal(info["baseline_advantage"],[6,-6])

    def test_new_head_is_unfitted(self):
        estimator=new_nonlinear_head()
        self.assertFalse(hasattr(estimator,"n_iter_"))
        self.assertEqual(estimator.max_iter,55)


if __name__ == "__main__":
    unittest.main()
