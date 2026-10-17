"""S695 time-domain onset selector must never consult future frames."""
import unittest
import numpy as np
from scripts.s695_causal_persistent_onset import causal_streak,rescue,POLICIES
class S695Tests(unittest.TestCase):
    def test_no_backdating_streak(self):
        x=np.array([[0,0],[1,0],[1,1],[1,1],[0,1],[1,1]],dtype=bool)
        y=causal_streak(x,3)
        self.assertFalse(y[:3].any())
        self.assertTrue(y[3,0])
        self.assertFalse(y[3,1])
        self.assertFalse(y[4,0])
        self.assertTrue(y[4,1])
    def test_native_source_conservation(self):
        rng=np.random.default_rng(695)
        p=rng.random((70,49,3),dtype=np.float32)
        f=rng.random((70,49,126),dtype=np.float32)
        k=rng.random((70,7),dtype=np.float32)
        for policy in POLICIES:
            q=rescue(p,f,policy,.60,k)
            np.testing.assert_array_equal(q[:,:,1:],p[:,:,1:])
            self.assertTrue(np.all(q[:,:,0]>=p[:,:,0]))
        self.assertIs(rescue(p,f,"native",.60,k),p)
    def test_causal_prefix_stability(self):
        rng=np.random.default_rng(695)
        p=rng.random((90,49,3),dtype=np.float32)
        f=rng.random((90,49,126),dtype=np.float32)
        k=rng.random((90,7),dtype=np.float32)
        a=rescue(p,f,"p3_floor_0.35_ratio_1.2",.6,k)
        p[70:]=0;f[70:]=20;k[70:]=0
        b=rescue(p,f,"p3_floor_0.35_ratio_1.2",.6,k)
        np.testing.assert_array_equal(a[:69],b[:69])
if __name__=="__main__":unittest.main()
