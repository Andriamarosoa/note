"""S670 past-only rhythm clock cannot see future NoteOns or PCM."""
import unittest
import numpy as np
from causal_note.strict_live import HOP
from scripts.s670_causal_rhythm_offset_hazard import native_rhythm_clock,frame_matrix

class S670RhythmTests(unittest.TestCase):
    def test_future_noteon_does_not_change_past_rhythm(self):
        n=200;ends=np.arange(1,n+1)*HOP
        event=lambda j:dict(type="note_on",pitch=40,decision_sample=j*HOP)
        first=[event(3),event(20),event(42),event(63)]
        altered=first+[event(160)]
        a=native_rhythm_clock(first,ends)
        b=native_rhythm_clock(altered,ends)
        np.testing.assert_array_equal(a[:150],b[:150])

    def test_hazard_features_causal_given_precomputed_clock(self):
        n=100;ends=np.arange(1,n+1)*HOP
        rnd=np.random.default_rng(670)
        f=rnd.random((n,49,126),dtype=np.float32)
        p=rnd.random((n,49,3),dtype=np.float32)
        k=np.zeros((n,7),np.float32);k[:,1]=1.
        clock=native_rhythm_clock([
            dict(type="note_on",pitch=40,decision_sample=4*HOP),
            dict(type="note_on",pitch=42,decision_sample=21*HOP),
        ],ends)
        old=frame_matrix(f,p,k,ends,40,4,np.array([20,25]),clock)
        f[35:]=99;p[35:]=99
        new=frame_matrix(f,p,k,ends,40,4,np.array([20,25]),clock)
        self.assertEqual(old.shape[1],75)
        np.testing.assert_array_equal(old,new)

    def test_clock_no_label_input(self):
        n=80
        a=native_rhythm_clock([],np.arange(1,n+1)*HOP)
        self.assertEqual(a.shape,(n,4))
        self.assertFalse(a.any())

if __name__=="__main__":
    unittest.main()
