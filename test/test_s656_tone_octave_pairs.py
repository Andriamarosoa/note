"""S656 Tone octave pair tests independent F0 control and audit aggregate."""
import unittest
import numpy as np
from causal_note.strict_live import SR,Selection
from scripts.s656_tone_true_octave_pairs import (
    eval_condition,POLICIES,CASES,LOW,HIGH
)
from scripts.s642_real_octave_pair_comparison import conditions,BASE_T,DURATION

class ToneTrueOctavePairsTests(unittest.TestCase):
    def test_same_physical_E3_source_controls_mix(self):
        duration=round(DURATION*SR)
        t=np.arange(duration)/SR
        z=np.where(t>=BASE_T,np.exp(-4*(t-BASE_T)),0)
        e3=np.asarray(.03*z*np.sin(2*np.pi*164.81*t),np.float32)
        e4=np.asarray(.02*z*np.sin(2*np.pi*329.63*t),np.float32)
        originals=dict(wave=e3)
        upper=dict(wave=e4)
        clips,labels=conditions(originals,upper)
        np.testing.assert_array_equal(clips["E3_alone"],e3)
        self.assertEqual(set(clips),set(CASES))
        self.assertEqual(labels["E3_plus_E4_simultaneous"][0]["pitch"],LOW)
        self.assertEqual(labels["E3_plus_E4_simultaneous"][1]["pitch"],HIGH)
        self.assertNotEqual(np.max(np.abs(clips["E3_alone"])),
                            np.max(np.abs(clips["E3_plus_E4_simultaneous"])))

    def test_policy_set_explicit_original_and_tone_views(self):
        self.assertIn("S614_original",POLICIES)
        self.assertIn("avg_three",POLICIES)
        self.assertEqual(len(POLICIES),8)

if __name__=="__main__":
    unittest.main()
