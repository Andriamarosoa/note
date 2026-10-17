"""Unit tests for S661 joint causal selective rescue.
No tests depend on GuitarSet labels or DEV results.
"""
from types import SimpleNamespace
import unittest
import numpy as np
from scripts.s661_joint_causal_selector import joined, held_winner, POLICIES

class JointCausalTests(unittest.TestCase):
    def make_case(self):
        cfg=SimpleNamespace(onset=.7,active_on=.5)
        state=SimpleNamespace(config=cfg,
            active=np.zeros(49,dtype=bool),
            armed=np.ones(49,dtype=bool))
        native=np.zeros((49,3),dtype=np.float32)
        bass=native.copy()
        tone=native.copy()
        bass[5,1]=.95
        tone[7,0]=.9
        tone[7,1]=.9
        scores={5:.94}
        contrasts={45:{"long":3.,"short":3.,"recent":3.},
                   47:{"long":3.,"short":3.,"recent":3.}}
        return native,bass,tone,state,scores,contrasts

    def test_union_restores_two_independent_candidates(self):
        native,bass,tone,state,scores,contrasts=self.make_case()
        p=joined(native,bass,tone,state,{},scores,
                 .9,"S661_union_recent",contrasts)
        self.assertAlmostEqual(float(p[5,0]),.7,places=6)
        self.assertAlmostEqual(float(p[7,0]),.9,places=6)
        self.assertEqual(float(native.sum()),0.)

    def test_physical_veto_preserves_original_predictions(self):
        native,bass,tone,state,scores,contrasts=self.make_case()
        native[15,0]=.99
        contr={47:contrasts[47]} # candidate pitch 45 has no support
        p=joined(native,bass,tone,state,{},scores,
                 .9,"S661_s641_physical_veto",contr)
        self.assertEqual(p[15,0],native[15,0])
        self.assertEqual(p[5,0],0.)
        self.assertEqual(float(native[5,0]),0.)

    def test_no_backdated_signal_needed(self):
        native,bass,tone,state,scores,contrasts=self.make_case()
        p=joined(native,bass,tone,state,{},scores,
                 .9,"S660_recent",contrasts)
        self.assertEqual(p[5,0],0.)
        self.assertAlmostEqual(float(p[7,0]),.9,places=6)

    def test_training_holdout_cannot_promote_without_positive_groups(self):
        def make_score(tp,fp):
            return dict(TP=tp,FP=fp,FN=100-tp,
                F1=2*tp/(tp+fp+100),latency_ms=[])
        records=[]
        for c in ("A","B","C","D"):
            stats={name:{"score":make_score(60,20),
                         "onset_poly":0,"K3plus_TP":0,"E2_TP":0,
                         "new_true":0,"lost_true":0,"attempts":0}
                   for name in POLICIES}
            records.append(dict(composition=c,stats=stats))
        winner,proof=held_winner(records)
        self.assertEqual(winner,"S641_41D")
        self.assertEqual(proof["selected_without_DEV"],"S641_41D")

if __name__=="__main__":
    unittest.main()
