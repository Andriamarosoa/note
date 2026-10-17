"""S655 per-hop K polyphonic Tone selector preserves true original K and pitch clock."""
import unittest
import numpy as np
from causal_note.strict_live import Selection
from scripts.s655_k_conditioned_tone_selector import (
    local_probs,POLICIES,train_choice
)

class NativeKConditionedToneTests(unittest.TestCase):
    def setUp(self):
        self.cfg=Selection()
        self.orig=np.zeros((49,3),np.float32)
        self.mild=np.ones_like(self.orig)*.9
        self.dark=np.ones_like(self.orig)*.8
        self.k=np.zeros(7,np.float32)
        self.k[1]=.95;self.k[0]=.05

    def test_native_K1_controls_original_when_polyprob_small(self):
        before=self.orig.copy();kbefore=self.k.copy()
        for name in ("tone_poly_ge025","tone_poly_ge050","tone_poly_ge075",
                     "tone_K3_ge025","tone_K3_ge050",
                     "tone_poly_high_register","tone_poly_low_register",
                     "tone_poly_mild_half","tone_poly_original_or_filtered"):
            out=local_probs(name,self.orig,self.mild,self.dark,self.k,self.cfg)
            np.testing.assert_array_equal(out,before)
        np.testing.assert_array_equal(self.orig,before)
        np.testing.assert_array_equal(self.k,kbefore)

    def test_true_poly_context_changes_only_pre_registered_register(self):
        self.k[:]=0;self.k[3]=.90;self.k[1]=.10
        out=local_probs("tone_poly_low_register",
                        self.orig,self.mild,self.dark,self.k,self.cfg)
        np.testing.assert_array_equal(out[17:],self.orig[17:])
        self.assertGreater(float(out[0,0]),0.)
        out=local_probs("tone_poly_high_register",
                        self.orig,self.mild,self.dark,self.k,self.cfg)
        np.testing.assert_array_equal(out[:17],self.orig[:17])
        self.assertGreater(float(out[24,0]),0.)
        self.assertEqual(len(POLICIES),12)

    def test_training_selection_rejects_group_regressions(self):
        def item(group,baseline=(5,5),other=(8,4)):
            d={}
            for p in POLICIES:
                a,b=(baseline if p=="S614_original" else other)
                d[p]=dict(score=dict(TP=a,FP=b,FN=10-a,
                                     latency_ms=[]),
                          K2_TP=a,K3plus_TP=a,overlap_TP=a,
                          E2_TP=0,E3_TP=0,new=0,lost=0)
            return dict(composition=group,results=d)
        calibration=[item(k) for k in "ABCDE"]
        held=[item(k,other=(4,8)) for k in "FGHI"]
        choice,decision=train_choice(calibration,held)
        self.assertEqual(choice,"S614_original")
        self.assertEqual(len(decision["held_delta_by_composition"]),4)

if __name__=="__main__":
    unittest.main()
