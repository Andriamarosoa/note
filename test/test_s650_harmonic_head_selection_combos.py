"""S650 tests for head-level combinations, abstention, and composition CV."""
import unittest
import numpy as np

from scripts.s650_harmonic_head_selection_combos import (
    HEADS,HEAD_ALIAS,COMBO_KINDS,FLOORS,head_policies,
    binary_utility,model_train_cv,
)
from scripts.s649_exclusive_harmonic_attack import matrices,WIDTH
from scripts.s640_learned_overlap_rescue import FEATURE_NAMES
from scripts.s641_fundamental_consistency import ADDITIONAL_NAMES


class HeadSelectionCompositionTests(unittest.TestCase):
    def test_all_selectors_use_actual_independent_head_predictions(self):
        predictions=dict(
            B=np.array([.99,.99,.4,.9,.01]),
            T=np.array([.99,.02,.99,.02,.99]),
            E=np.array([.99,.02,.02,.99,.99]),
            J=np.array([.99,.1,.1,.99,.05]))
        floors={k:.8 for k in HEAD_ALIAS}
        blends={name:.8 for name in (
            "weighted_B60_T20_E20","weighted_B40_T30_E30",
            "weighted_B40_T20_E20_J20","geometric_BTE")}
        out=head_policies(predictions,floors,blends)
        self.assertEqual(set(out),set(COMBO_KINDS))
        np.testing.assert_array_equal(out["single_B"],[1,1,0,1,0])
        np.testing.assert_array_equal(out["single_E"],[1,0,0,1,1])
        np.testing.assert_array_equal(out["B_AND_T"],[1,0,0,0,0])
        np.testing.assert_array_equal(out["B_AND_T_OR_E"],[1,0,0,1,0])
        np.testing.assert_array_equal(out["B_OR_T_AND_E"],[1,1,0,1,1])
        np.testing.assert_array_equal(out["vote_two_of_BTE"],[1,0,0,1,1])
        np.testing.assert_array_equal(out["vote_three_of_BTE"],[1,0,0,0,0])
        for name in COMBO_KINDS:
            self.assertEqual(len(out[name]),5)
            self.assertEqual(out[name].dtype,np.bool_)

    def test_agreement_can_reduce_both_corrections_and_regressions(self):
        truth=np.asarray([1,0,1,0,0,0],bool)
        base=np.asarray([1,1,1,1,0,0],bool)
        veto=np.asarray([1,0,0,0,0,0],bool)
        b=binary_utility(base,truth)
        v=binary_utility(veto,truth)
        self.assertEqual((b["TP"],b["FP"]),(2,2))
        self.assertEqual((v["TP"],v["FP"]),(1,0))
        self.assertEqual(b["utility"],v["utility"])
        self.assertEqual(binary_utility(np.zeros_like(truth),truth)["accepted"],0)

    def test_original_features_and_four_head_train_without_data_leakage(self):
        f=np.zeros(len(FEATURE_NAMES),np.float32)
        p=np.zeros(len(ADDITIONAL_NAMES),np.float32)
        n=np.zeros(WIDTH,np.float32)
        e=np.zeros(24,np.float32)
        prototype=matrices(f,p,n,e)
        self.assertTrue(set(HEADS).issubset(set(prototype)))
        rng=np.random.default_rng(650)
        records=[]
        for i in range(9):
            y=np.zeros(80,np.int8)
            y[:6]=1
            arr={}
            for name in HEADS:
                x=rng.normal(size=(80,len(prototype[name]))).astype(np.float32)
                x[:6,0]+=3
                arr[name]=x
            records.append(dict(composition=f"source{i}",y=y,matrices=arr))
        models,thresholds,blends,winner,cv=model_train_cv(records)
        self.assertEqual(cv["sample_count"],720)
        self.assertEqual(cv["positive"],54)
        self.assertEqual(len(cv["cv_folds"]),5)
        for fold in cv["cv_folds"]:
            self.assertFalse(set(fold["train"])&set(fold["held"]))
        self.assertEqual(set(models),set(HEAD_ALIAS))
        for value in [*thresholds.values(),*blends.values()]:
            self.assertIn(value,FLOORS)
        self.assertIn(winner,COMBO_KINDS)
        self.assertEqual(set(cv["selection_utility"]),set(COMBO_KINDS))


if __name__=="__main__":
    unittest.main()
