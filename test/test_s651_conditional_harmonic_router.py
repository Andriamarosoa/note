"""S651 verify conditional router, leakage-free nested composition splits and causal inputs."""
import unittest
import numpy as np

from scripts.s651_conditional_harmonic_router import (
    ROUTERS,FLOORS,HEADS,HEAD_ALIAS,
    CONTEXT_FEATURE_INDICES,PHYSICAL_FEATURE_INDICES,
    meta_features,nested_router_cv,HEAD_FLOORS
)
from scripts.s649_exclusive_harmonic_attack import matrices,WIDTH
from scripts.s640_learned_overlap_rescue import FEATURE_NAMES
from scripts.s641_fundamental_consistency import ADDITIONAL_NAMES


class ConditionalSelectionRouterTests(unittest.TestCase):
    def test_router_features_per_candidate_never_use_labels(self):
        scores=np.asarray([[.99,.25,.8,.7],[.11,.92,.5,.43]],np.float32)
        features=dict(control41=np.zeros((2,len(FEATURE_NAMES)),np.float32),
                      exclusive65=np.zeros((2,len(FEATURE_NAMES)+len(ADDITIONAL_NAMES)+24),np.float32))
        features["control41"][:,0]=[.1,.8]
        features["control41"][:,9]=[.6,.2]
        features["exclusive65"][:,len(FEATURE_NAMES)+2]=[2.,4.]
        a=meta_features(scores,features,"opinions_only")
        b=meta_features(scores,features,"conditional_context")
        self.assertEqual(a.shape,(2,14))
        self.assertEqual(b.shape,(2,14+len(CONTEXT_FEATURE_INDICES)+len(PHYSICAL_FEATURE_INDICES)))
        np.testing.assert_array_equal(a,b[:,:14])
        self.assertFalse(np.array_equal(b[0],b[1]))
        self.assertTrue(np.isfinite(b).all())
        with self.assertRaises(ValueError):
            meta_features(scores,features,"wrong_router")

    def test_nested_composition_cv_excludes_held_group_from_both_levels(self):
        rng=np.random.default_rng(651)
        patterns=matrices(np.zeros(len(FEATURE_NAMES),np.float32),
                          np.zeros(len(ADDITIONAL_NAMES),np.float32),
                          np.zeros(WIDTH,np.float32),np.zeros(24,np.float32))
        records=[]
        for group in range(9):
            count=85
            y=np.zeros(count,np.int8);y[:7]=1
            mats={}
            for name in set(HEADS)|{"control41","exclusive65"}:
                x=rng.normal(size=(count,len(patterns[name]))).astype(np.float32)
                x[:7,:3]+=1.7
                mats[name]=x
            records.append(dict(composition=f"G{group}",y=y,matrices=mats))
        base,meta,winner,choices,cv=nested_router_cv(records)
        self.assertEqual(cv["train_examples"],765)
        self.assertEqual(cv["train_positive"],63)
        self.assertEqual(len(cv["nested_folds"]),5)
        self.assertIn(winner,("S641_only",*ROUTERS))
        for info in cv["nested_folds"]:
            self.assertFalse(set(info["train_compositions"])&
                             set(info["held_compositions"]))
        for name in ROUTERS:
            self.assertIn(name,meta)
            self.assertIn(choices[name]["chosen"]["floor"],FLOORS)
        self.assertEqual(set(base),set(HEAD_ALIAS))

    def test_original_head_and_exact_k_retained_unchanged(self):
        self.assertEqual(HEAD_FLOORS["B"],.90)
        self.assertEqual(HEADS[0],"control41")
        self.assertEqual(HEADS[1],"temporal65")
        self.assertEqual(HEADS[2],"exclusive65")
        self.assertEqual(HEADS[3],"exclusive_temporal65")


if __name__=="__main__":
    unittest.main()
