"""S658 natural GuitarSet tone gate evaluation and approval tests."""
import unittest
from scripts.s658_natural_tone_gate_validation import (
    POLICIES,select_train_only
)

def make(comp,baseline=(10,3),candidate=(12,3)):
    outcomes={}
    for name in POLICIES:
        tp,fp=baseline if name=="S614_original" else candidate
        outcomes[name]=dict(score=dict(TP=tp,FP=fp,
                                      FN=20-tp,latency_ms=[]),
                            K2_TP=tp,K3plus_TP=tp,overlap_TP=tp,
                            E2_TP=0,E3_TP=0,new_true=0,lost_true=0)
    return dict(composition=comp,results=outcomes)

class NaturalToneGateTests(unittest.TestCase):
    def test_requires_three_held_group_improvements(self):
        calib=[make(k) for k in "ABCDE"]
        held=[make(k,baseline=(10,3),candidate=(9,8)) for k in "FGHI"]
        selected,cv=select_train_only(calib,held)
        self.assertEqual(selected,"S614_original")
        self.assertEqual(len(cv["held_delta_by_composition"]),4)

    def test_conditional_head_controls_native_original_selection(self):
        self.assertEqual(POLICIES[0],"S614_original")
        self.assertIn("gate_last4_050",POLICIES)
        self.assertEqual(len(POLICIES),6)

if __name__=="__main__":
    unittest.main()
