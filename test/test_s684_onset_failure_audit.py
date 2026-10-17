"""S684 errors are grouped into disjoint audit categories."""
import unittest
import numpy as np
from causal_note.strict_live import HOP,SR,Selection
from scripts.s684_onset_failure_cause_audit import classify,register
class S684Tests(unittest.TestCase):
    def test_missing_onset_low_posterior(self):
        n=60
        ends=np.arange(1,n+1)*HOP
        p=np.zeros((n,49,3),np.float32)
        p[:,:,0]=.05
        gt=[dict(pitch=40,time=10*HOP/SR,duration=.2)]
        row=classify(dict(truth=gt,events=[],p=p,ends=ends),Selection())
        self.assertEqual(row["baseline"]["FN"],1)
        self.assertEqual(row["FN_types"]["onset_posterior_below_threshold"],1)
        self.assertEqual(register(40),"bass_E2_B2")
    def test_harmonic_neighbor_annotated_as_false_positive(self):
        n=80;ends=np.arange(1,n+1)*HOP
        p=np.zeros((n,49,3),np.float32)
        truth=[dict(pitch=40,time=10*HOP/SR,duration=.3)]
        event=dict(type="note_on",pitch=52,time=11*HOP/SR,
                   decision_sample=11*HOP)
        row=classify(dict(truth=truth,events=[event],p=p,ends=ends),Selection())
        self.assertEqual(row["baseline"]["FP"],1)
        self.assertEqual(row["FP_types"]["harmonic_competitor_near_gt"],1)
        self.assertEqual(row["FN_types"]["near_octave_or_harmonic_noteon"],1)
if __name__=="__main__":unittest.main()
