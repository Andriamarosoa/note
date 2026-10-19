"""S770 regression tests; source MIDI is immutable and all hypotheses causal."""
import unittest
from unittest.mock import patch
import numpy as np
from scripts import s770_joint_onset_fit80 as s770

class TestJointOnsetShadow(unittest.TestCase):
    def setUp(self):
        self.ends=np.array([1024,2048,3072],dtype=np.int64)
        self.feat=np.zeros((3,49,87),dtype=np.float32)
        self.source=[dict(type="note_on",pitch=40,available_sample=1024),
                     dict(type="note_off",pitch=40,available_sample=1800)]
        self.native=[self.source[0]]
        self.added=[dict(type="note_on",pitch=41,available_sample=2048)]
        self.pred=np.zeros((3,49),dtype=np.float32)

    def test_rise_gate_uses_only_available_audio(self):
        self.feat[1,1,63:71]=0.05
        self.assertEqual(len(s770.rise_only(self.added,self.ends,self.feat,0.025)),1)
        self.assertEqual(len(s770.rise_only(self.added,self.ends,self.feat,0.10)),0)
        self.feat[2,1,63:71]=999.0
        self.assertEqual(len(s770.rise_only(self.added,self.ends,self.feat,0.10)),0)

    def test_invalid_candidate_clock_fails_closed(self):
        bad=[dict(type="note_on",pitch=41,available_sample=2049)]
        with self.assertRaises(AssertionError):
            s770.rise_only(bad,self.ends,self.feat,0.025)

    def test_original_midi_untouched(self):
        original=[x.copy() for x in self.source]
        with patch.object(s770,"proposals",return_value=self.added):
            out=s770.shadow(self.source,self.native,np.array([0.1]),
                            self.pred,self.ends,self.feat,(0.2,0.9,0.0))
        self.assertEqual(out,self.added)
        self.assertEqual(self.source,original)

    def test_two_independent_heads(self):
        with patch.object(s770,"proposals",return_value=self.added):
            a=s770.shadow(self.source,self.native,np.array([0.8]),
                          self.pred,self.ends,self.feat,(0.2,0.9,0.0))
            b=s770.shadow(self.source,self.native,np.array([0.1]),
                          self.pred,self.ends,self.feat,(0.2,0.9,0.0))
        self.assertEqual(len(a),2)
        self.assertEqual(len(b),1)

    def test_native_only_control_and_registered_grid(self):
        self.assertEqual(len(s770.GRID),65)
        base=s770.shadow(self.source,self.native,np.array([1.]),
                         self.pred,self.ends,self.feat,(0.,None,0.))
        self.assertEqual(base,self.native)
        with self.assertRaises(ValueError):
            s770.shadow(self.source,self.native,np.array([1.]),
                        self.pred,self.ends,self.feat,(0.,0.95,0.))

if __name__=="__main__":
    unittest.main()
