"""S632 pinned native-feature and original-selection invariants."""
from __future__ import annotations
import unittest

import numpy as np

from causal_note.strict_live import CausalSelection, Selection, SR, feature_track
from causal_note.register_rescue import (
    RescuePolicy, BASS_MODERATE, HIGH_MODERATE, propose, variants,
)


class RegisterRescueTests(unittest.TestCase):
    def setUp(self):
        self.state = CausalSelection(Selection(onset=.6,active_on=.5))
        self.x = np.zeros((49,126),dtype=np.float32)
        self.p = np.zeros((49,3),dtype=np.float32)
        self.p[:,1]=.7
        self.policy=RescuePolicy(BASS_MODERATE,HIGH_MODERATE)

    def strong_fundamental(self,pitch):
        ix=pitch-40
        self.p[ix,0]=.45
        self.x[ix,[0,8,16]]=.8
        self.x[ix,[1,9,17]]=.3
        self.x[ix,54:60]=.3

    def test_bass_and_high_have_distinct_gates(self):
        self.strong_fundamental(52)
        self.strong_fundamental(80)
        self.strong_fundamental(70)  # The middle register is always untouched.
        output,choices=propose(self.x,self.p,self.state,self.policy)
        self.assertTrue(choices["bass"][52-40])
        self.assertTrue(choices["high"][80-40])
        self.assertFalse(choices["bass"][70-40])
        self.assertFalse(choices["high"][70-40])
        self.assertAlmostEqual(float(output[52-40,0]),self.state.config.onset,places=6)
        self.assertAlmostEqual(float(output[80-40,0]),self.state.config.onset,places=6)
        self.assertAlmostEqual(float(output[70-40,0]),.45,places=5)
        np.testing.assert_allclose(self.p[[52-40,80-40],0],[.45,.45],atol=1e-7)

    def test_harmonics_without_new_fundamental_cannot_trigger(self):
        self.strong_fundamental(80)
        self.x[80-40,[0,8,16]]=.02
        self.x[80-40,[1,9,17]]=3.0
        out,choices=propose(self.x,self.p,self.state,self.policy)
        self.assertFalse(choices["high"].any())
        self.assertAlmostEqual(float(out[80-40,0]),.45,places=5)

    def test_sustain_alone_or_unarmed_notes_not_rescued(self):
        self.strong_fundamental(52)
        self.p[52-40,0]=.0
        _,choices=propose(self.x,self.p,self.state,self.policy)
        self.assertFalse(choices["bass"].any())
        self.p[52-40,0]=.45
        self.state.active[52-40]=True
        _,choices=propose(self.x,self.p,self.state,self.policy)
        self.assertFalse(choices["bass"].any())
        self.state.active[52-40]=False
        self.state.armed[52-40]=False
        _,choices=propose(self.x,self.p,self.state,self.policy)
        self.assertFalse(choices["bass"].any())

    def test_control_bitwise_unchanged_and_strict_input_validation(self):
        self.strong_fundamental(52)
        result,masks=propose(self.x,self.p,self.state,RescuePolicy())
        np.testing.assert_array_equal(result,self.p)
        self.assertFalse(masks["bass"].any())
        self.assertFalse(masks["high"].any())
        with self.assertRaises(ValueError):
            propose(np.ones((49,110),dtype=np.float32),self.p,self.state,self.policy)
        with self.assertRaises(ValueError):
            propose(self.x,np.zeros((49,2)),self.state,self.policy)

    def test_native_K_posterior_vetoes_false_bass_proposals(self):
        from causal_note.register_rescue import BASS_K50
        self.strong_fundamental(52)
        self.p[52-40,0] = .52  # Meets this conservative gate's .48 onset floor.
        policy = RescuePolicy(bass=BASS_K50)
        blocked = np.array([.9,.1,0,0,0,0,0],dtype=np.float32)
        admitted = np.array([.1,.9,0,0,0,0,0],dtype=np.float32)
        unchanged, gates = propose(self.x,self.p,self.state,policy,blocked)
        self.assertFalse(gates["bass"][52-40])
        self.assertAlmostEqual(float(unchanged[52-40,0]),.52,places=6)
        modified, gates = propose(self.x,self.p,self.state,policy,admitted)
        self.assertTrue(gates["bass"][52-40])
        self.assertAlmostEqual(float(modified[52-40,0]),.6,places=6)
        with self.assertRaises(ValueError):
            propose(self.x,self.p,self.state,policy, np.ones(3))

    def test_real_causal_frontend_width_and_independent_candidate_list(self):
        audio=np.zeros(round(.12*SR),np.float32)
        audio[1024:]+=np.sin(2*np.pi*82.407*np.arange(len(audio)-1024)/SR).astype(np.float32)*.07
        ends,features=feature_track(audio)
        self.assertEqual(features.shape[1:],(49,126))
        self.assertEqual(ends[-1] % 256,0)
        self.assertEqual(len(variants()),22)


if __name__ == "__main__":
    unittest.main()
