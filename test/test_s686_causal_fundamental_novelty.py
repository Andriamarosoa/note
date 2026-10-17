"""S686 signal is causal, pitch-specific and never modifies NOTE K/S675."""
import unittest
import numpy as np
from scripts.s686_causal_fundamental_novelty import rescue,POLICIES
class S686Tests(unittest.TestCase):
    def test_no_future_pcm_used(self):
        rng=np.random.default_rng(686)
        p=rng.random((120,49,3),dtype=np.float32)*.5
        f=rng.random((120,49,126),dtype=np.float32)
        name="rescue_0.20_rise_0.12"
        a=rescue(p,f,name,.60)
        f[80:]=99;p[80:]=99
        b=rescue(p,f,name,.60)
        np.testing.assert_array_equal(a[:80],b[:80])
    def test_exact_same_sustain_release_and_pitch(self):
        rng=np.random.default_rng(686)
        p=rng.random((40,49,3),dtype=np.float32)
        f=rng.random((40,49,126),dtype=np.float32)
        for policy in POLICIES:
            out=rescue(p,f,policy,.60)
            np.testing.assert_array_equal(out[:,:,1:],p[:,:,1:])
        self.assertIs(rescue(p,f,"native",.60),p)
    def test_new_onset_requires_original_support(self):
        p=np.zeros((8,49,3),np.float32); f=np.ones((8,49,126),np.float32)
        x=rescue(p,f,"rescue_0.20_rise_0.12",.6)
        self.assertFalse(x[:,:,0].any())
if __name__=="__main__":
    unittest.main()
