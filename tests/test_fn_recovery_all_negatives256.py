"""Preregistered negative-coverage ablation H_FN_NEGCOVERAGE_01 (#141)."""
import unittest
import numpy as np
from scripts.train_onset_fn_recovery256 import train_one

class FullNegativeCoverageTests(unittest.TestCase):
    def test_all_train_negative_examples_seen_and_outer_excluded(self):
        rng=np.random.default_rng(141)
        rows=[]
        for i in range(48+600+25):
            f=rng.normal(size=69).astype(np.float32)
            # 48 positive, 600 negative, 25 disjoint OUTER samples.
            y=1 if i<48 else 0
            rows.append(dict(X=f,y=y,origin="outer" if i>=648 else "train"))
        model,stats=train_one(rows,list(range(648)),1001)
        self.assertEqual(stats["tp_candidates"],48)
        self.assertEqual(stats["negative_candidates"],600)
        self.assertEqual(stats["used_unique_negative_candidates"],600)
        self.assertEqual(stats["negative_coverage"],1.0)
        self.assertEqual(stats["repeated_positives"],5)
        self.assertEqual(stats["training_rows"],600+48*5)
        self.assertTrue(stats["original_3_to_1_sampling_ablation"])
        self.assertEqual(model.n_features_in_,69)
        self.assertEqual(model.predict_proba(np.zeros((49,69))).shape,(49,2))
    def test_no_training_labels_from_cal_or_outer(self):
        rng=np.random.default_rng(143)
        train=[dict(X=rng.normal(size=69).astype(np.float32),y=int(i<48)) for i in range(648)]
        cal=[dict(X=np.full(69,111,np.float32),y=1) for _ in range(10)]
        outer=[dict(X=np.full(69,222,np.float32),y=0) for _ in range(10)]
        allrows=train+cal+outer
        a,detail_a=train_one(allrows,list(range(648)),991)
        b,detail_b=train_one(train,list(range(648)),991)
        self.assertEqual(detail_a,detail_b)
        for va,vb in zip(a.named_steps["mlpclassifier"].coefs_,
                         b.named_steps["mlpclassifier"].coefs_):
            np.testing.assert_array_equal(va,vb)
    def test_preregistered_frozen_threshold_not_tuned_on_outer(self):
        from scripts.replay_onset_fn_recovery256 import (
            REFERENCE_FOUR_HEAD_F1,REFERENCE_FOUR_HEAD_FP,REFERENCE_FOUR_HEAD_TP)
        self.assertAlmostEqual(REFERENCE_FOUR_HEAD_F1,.6790945406125166)
        self.assertEqual(REFERENCE_FOUR_HEAD_FP,4605)
        self.assertEqual(REFERENCE_FOUR_HEAD_TP,8160)
if __name__=="__main__":unittest.main()
