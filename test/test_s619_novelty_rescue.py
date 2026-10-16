"""Causal pitch onset rescue gate safety tests, no learned ground truth input."""
import unittest

import numpy as np

from causal_note.strict_live import HOP, Selection
from scripts.s619_novelty_onset_gate import (
    RescueGate, weak_note_boost, causal_replay,
)


class CausalRescueTests(unittest.TestCase):
    def test_does_not_promote_low_novelty_even_with_weak_attack(self):
        x=np.zeros((49,126),np.float32)
        p=np.zeros((49,3),np.float32)
        p[12,:2]=[.52,.8]
        result,count=weak_note_boost(p,x,RescueGate())
        np.testing.assert_array_equal(result,p)
        self.assertEqual(count,0)

    def test_independent_novelty_promotes_only_correct_pitch(self):
        x=np.zeros((49,126),np.float32)
        p=np.zeros((49,3),np.float32)
        p[12,:2]=[.52,.80]
        p[20,:2]=[.53,.80]
        p[30,:2]=[.85,.90]
        x[12,114]=.8
        x[20,114]=.02
        result,count=weak_note_boost(p,x,RescueGate(
            weak_onset_min=.45,relative_novelty_min=.35,
            absolute_novelty_min=.05))
        self.assertEqual(count,1)
        self.assertGreaterEqual(result[12,0],.6)
        self.assertEqual(result[20,0],p[20,0])
        self.assertEqual(result[30,0],p[30,0])

    def test_causal_replay_attacks_without_future_audio_or_extra_hop(self):
        ends=np.arange(1,9,dtype=np.int64)*HOP
        x=np.zeros((len(ends),49,126),np.float32)
        p=np.zeros((len(ends),49,3),np.float32)
        k=np.zeros((len(ends),7),np.float32); k[:,1]=1
        p[2,12,:2]=[.52,.8]
        x[2,12,114]=.8
        event,n=causal_replay(ends,x,p,k,int(ends[-1]),
                             RescueGate(weak_onset_min=.45,
                                        relative_novelty_min=.3,
                                        absolute_novelty_min=.05),
                             selection=Selection(onset=.6,active_on=.5))
        ons=[e for e in event if e["type"]=="note_on"]
        self.assertEqual(n,1)
        self.assertEqual(len(ons),1)
        self.assertEqual(ons[0]["pitch"],52)
        self.assertEqual(ons[0]["decision_sample"],3*HOP)
        self.assertEqual(ons[0]["available_sample"],3*HOP)


if __name__=="__main__":
    unittest.main()
