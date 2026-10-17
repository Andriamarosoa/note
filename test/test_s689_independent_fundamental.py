"""S689 audit causal independent harmonic rise and immutable native heads."""
import unittest
import numpy as np
from scripts.s689_independent_fundamental_rescue import rescue,POLICIES
class S689Tests(unittest.TestCase):
    def test_native_immutable_even_for_every_variant(self):
        rng=np.random.default_rng(689)
        p=rng.random((100,49,3),dtype=np.float32)
        f=rng.random((100,49,126),dtype=np.float32)
        k=rng.random((100,7),dtype=np.float32)
        for option in POLICIES:
            modified=rescue(p,f,option,.60,k)
            np.testing.assert_array_equal(modified[:,:,1:],p[:,:,1:])
            self.assertTrue(np.all(modified[:,:,0]>=p[:,:,0]))
        self.assertIs(rescue(p,f,"native",.60,k),p)
    def test_future_audio_cannot_alter_prior_decisions(self):
        rng=np.random.default_rng(689)
        p=rng.random((90,49,3),dtype=np.float32)
        f=rng.random((90,49,126),dtype=np.float32)
        k=rng.random((90,7),dtype=np.float32)
        a=rescue(p,f,"independent_floor_0.27_ratio_1.2",.6,k)
        p[70:]=.01;f[70:]=99;k[70:]=0
        b=rescue(p,f,"independent_floor_0.27_ratio_1.2",.6,k)
        np.testing.assert_array_equal(a[:69],b[:69])
    def test_unvoiced_empty_signal_never_generates_notes(self):
        p=np.zeros((16,49,3),np.float32)
        f=np.zeros((16,49,126),np.float32)
        k=np.zeros((16,7),np.float32)
        for policy in POLICIES:
            self.assertFalse(rescue(p,f,policy,.6,k).any())
if __name__=="__main__":unittest.main()
