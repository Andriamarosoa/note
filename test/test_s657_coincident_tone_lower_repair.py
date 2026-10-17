"""S657 ensure only candidate lower pitch can be modified and timing is causal."""
import unittest
import numpy as np
from causal_note.strict_live import Selection
from scripts.s657_coincident_tone_lower_repair import condition,POLICIES

class ControlledToneOctaveGate(unittest.TestCase):
    def setUp(self):
        self.cfg=Selection()
        self.p=np.zeros((49,3),np.float32)
        self.m=np.zeros_like(self.p)
        self.d=np.zeros_like(self.p)
        self.age=np.full(49,10**7,np.int32)
        self.armed=np.ones(49,bool)
        self.active=np.zeros(49,bool)

    def test_strong_high_attack_enables_only_missing_low_pitch(self):
        low=52-40;high=64-40
        self.p[high,:2]=(.9,.9)
        self.d[low,:2]=(.82,.85)
        self.age[high]=0
        out=condition(self.p,self.m,self.d,self.cfg,self.age,
                      "gate_last4_050",self.armed,self.active)
        self.assertGreaterEqual(float(out[low,0]),self.cfg.onset)
        self.assertEqual(float(out[high,0]),float(self.p[high,0]))
        self.assertEqual(np.count_nonzero(out[:,0]-self.p[:,0]),1)
        np.testing.assert_array_equal(self.p[low,:2],[0.,0.])

    def test_no_future_attacks_delayed_upper_and_no_double_bass(self):
        low=52-40;high=64-40
        self.d[low,:2]=(.9,.9)
        self.age[high]=10**7
        out=condition(self.p,self.m,self.d,self.cfg,self.age,
                      "gate_last8_050",self.armed,self.active)
        np.testing.assert_array_equal(out,self.p)
        self.age[high]=0
        self.active[low]=True
        out=condition(self.p,self.m,self.d,self.cfg,self.age,
                      "gate_last8_050",self.armed,self.active)
        np.testing.assert_array_equal(out,self.p)

    def test_all_gates_keep_original_as_control(self):
        self.d[0,:2]=(.96,.95)
        self.age[12]=0
        self.assertEqual(len(POLICIES),6)
        for policy in POLICIES:
            out=condition(self.p,self.m,self.d,self.cfg,self.age,
                          policy,self.armed,self.active)
            self.assertTrue(np.isfinite(out).all())
        out=condition(self.p,self.m,self.d,self.cfg,self.age,
                      "S614_original",self.armed,self.active)
        np.testing.assert_array_equal(out,self.p)

if __name__=="__main__":
    unittest.main()
