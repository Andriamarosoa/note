"""S672 pre-registered causal hazard timing variants and horizon invariance."""
import unittest
import numpy as np
from causal_note.strict_live import HOP
from scripts.s672_causal_offset_timing_grid import (
    PERSISTENCES,GATES,choose_offset
)
class S672HazardPersistenceTests(unittest.TestCase):
    def test_precommitted_search(self):
        self.assertEqual(PERSISTENCES,(1,2,3,4))
        self.assertTrue(.5 in GATES)
    def test_waiting_for_consecutive_evidence(self):
        ends=np.arange(1,121)*HOP
        item={"ends":ends,"notes":[dict(start=10,pitch=40,
                    native_end=90,next_start=115)]}
        frames=np.arange(20,100)
        probs=np.zeros(len(frames),np.float32)
        probs[8:]=1.
        t1=choose_offset(item,0,frames,probs,.5,"hazard",
                         batch_hops=1,persistence=1)
        t4=choose_offset(item,0,frames,probs,.5,"hazard",
                         batch_hops=1,persistence=4)
        self.assertGreater(t4,t1)
        b4=choose_offset(item,0,frames,probs,.5,"hazard",
                         batch_hops=4,persistence=4)
        self.assertGreaterEqual(b4,t4)
    def test_native_ignores_other_hazard_parameters(self):
        ends=np.arange(1,121)*HOP
        item={"ends":ends,"notes":[dict(start=10,pitch=40,
                    native_end=90,next_start=115)]}
        frames=np.arange(20,100)
        probs=np.ones(len(frames),np.float32)
        for persistence in PERSISTENCES:
            v=choose_offset(item,0,frames,probs,.6,"native",1,persistence)
            self.assertEqual(v,91*HOP/44100)
if __name__=="__main__":unittest.main()
