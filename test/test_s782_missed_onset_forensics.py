"""S782 tests strictly diagnostic onset evidence, no model decision mutation."""
import unittest
import numpy as np
from causal_note.strict_live import SR
from scripts.s782_missed_onset_acoustic_forensics import (
    acoustic_evidence,overlap_harmonic,true_polyphony,classify_pitch,summarize)

class S782MissedOnsetTests(unittest.TestCase):
    def test_causal_evidence_window_and_no_future_leak(self):
        clock=np.array([1024,2048,3072,4096,5120,6144],np.int64)
        post=np.zeros((6,49,3),np.float32)
        spec=np.zeros((6,49),np.float32)
        note=dict(pitch=52,time=1025/SR,duration=.15)
        post[2,12,0]=.4;spec[2,12]=.7
        a=acoustic_evidence(note,clock,post,spec)
        self.assertAlmostEqual(a["onset"],.4)
        p2=post.copy();s2=spec.copy();p2[5,12,0]=.99;s2[5,12]=99
        b=acoustic_evidence(note,clock,p2,s2)
        self.assertEqual(a,b)

    def test_missing_harmonic_neighbor(self):
        n=dict(pitch=52,time=0.,duration=.2)
        a=dict(pitch=40,time=0.,duration=.2)
        b=dict(pitch=54,time=0.,duration=.2)
        self.assertTrue(overlap_harmonic(n,[n,a]))
        self.assertFalse(overlap_harmonic(n,[n,b]))
        self.assertEqual(true_polyphony(n,[n,a]),2)
        self.assertEqual(classify_pitch(40),"E2_B2")

    def test_signal_without_attack_is_weak(self):
        r=[dict(matched=False,evidence=dict(onset=.01,rise=.02)),
           dict(matched=False,evidence=dict(onset=.2,rise=.5)),
           dict(matched=True,evidence=dict(onset=.3,rise=.5))]
        out=summarize(r)
        self.assertEqual(out["missed"],2)
        self.assertEqual(out["missed_weak_both"],1)
        self.assertEqual(out["missed_strong_both"],1)
        self.assertEqual(out["matched_strong_both"],1)

if __name__=="__main__":
    unittest.main()
