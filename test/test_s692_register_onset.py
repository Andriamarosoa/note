"""S692 register-specific variants alter only selected pitch Onset."""
import unittest
import numpy as np
from scripts.s692_register_onset_selection import rescue,POLICIES
class S692Tests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(692)
        self.p=rng.random((80,49,3),dtype=np.float32)
        self.f=rng.random((80,49,126),dtype=np.float32)
        self.k=rng.random((80,7),dtype=np.float32)
    def test_no_register_bleed(self):
        low=rescue(self.p,self.f,"bass_0.35_1.2",.6,self.k)
        np.testing.assert_array_equal(low[:,8:,0],self.p[:,8:,0])
        high=rescue(self.p,self.f,"high_0.35_1.2",.6,self.k)
        np.testing.assert_array_equal(high[:,:20,0],self.p[:,:20,0])
        middle=rescue(self.p,self.f,"mid_0.35_1.2",.6,self.k)
        np.testing.assert_array_equal(middle[:,:8,0],self.p[:,:8,0])
        np.testing.assert_array_equal(middle[:,20:,0],self.p[:,20:,0])
    def test_frozen_k_sustain_release(self):
        for policy in POLICIES:
            altered=rescue(self.p,self.f,policy,.6,self.k)
            np.testing.assert_array_equal(altered[:,:,1:],self.p[:,:,1:])
            self.assertTrue(np.all(altered[:,:,0]>=self.p[:,:,0]))
        self.assertIs(rescue(self.p,self.f,"native",.6,self.k),self.p)
    def test_prefix_stability(self):
        a=rescue(self.p,self.f,"basshigh_0.35_1.2",.6,self.k)
        self.f[60:]=1;self.p[60:]=0;self.k[60:]=0
        b=rescue(self.p,self.f,"basshigh_0.35_1.2",.6,self.k)
        np.testing.assert_array_equal(a[:59],b[:59])
if __name__=="__main__":unittest.main()
