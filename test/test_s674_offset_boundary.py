"""S674 boundary persistence and model temporal-causality checks."""
import unittest
import numpy as np
from scripts.s674_true_noteoff_boundary_classifier import (
    choose_offset,native_rhythm_clock,GATES,PERSISTENCES)
from causal_note.strict_live import HOP

class BoundaryTests(unittest.TestCase):
    def test_boundaries_are_narrow_and_thresholds_preregistered(self):
        stop=80
        indices=np.arange(50,115)
        y=(np.abs(indices-stop)<=5)
        self.assertEqual(y.sum(),11)
        self.assertEqual(len(PERSISTENCES),4)
        self.assertIn(.50,GATES)
    def test_no_backdating_batch_delivery(self):
        ends=np.arange(1,180)*HOP
        note={"start":20,"native_end":140,"next_start":170,"pitch":40}
        item={"ends":ends,"notes":[note]}
        frames=np.arange(35,150)
        p=np.zeros(len(frames),np.float32);p[31:]=1.
        q=choose_offset(item,0,frames,p,.50,"hazard",batch_hops=4,persistence=1)
        self.assertGreaterEqual(q,(35+31+1)*HOP/44100)
    def test_new_rhythm_clock_without_events(self):
        a=native_rhythm_clock([],np.arange(1,100)*HOP)
        self.assertTrue(np.isfinite(a).all())
if __name__=="__main__": unittest.main()
