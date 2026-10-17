"""S688 octave-note acoustic contrast tests are causal and K/pitch safe."""
import unittest
import numpy as np
from scripts.s688_octave_aware_novelty import rescue,POLICIES
from scripts.s688_octave_robust_four_lock import choose_train_only

class OctaveTests(unittest.TestCase):
    def test_suppress_low_evidence_upper_harmonic_only(self):
        p=np.zeros((8,49,3),np.float32)
        f=np.zeros((8,49,126),np.float32)
        p[:,0,1]=.9
        p[:,12,0]=.35;p[:,12,1]=.9
        f[:,0,54:57]=1
        f[:,12,54:57]=.5
        base=rescue(p,f,"rescue_base",.60)
        self.assertTrue(np.all(base[:,12,0]>.60))
        gated=rescue(p,f,"guard_0.55_ratio_1.00",.60)
        np.testing.assert_array_equal(gated[:,12,0],p[:,12,0])
        np.testing.assert_array_equal(gated[:,0,0],p[:,0,0])

    def test_cannot_suppress_original_onset_or_change_release(self):
        rng=np.random.default_rng(688)
        p=rng.random((26,49,3),dtype=np.float32)
        f=rng.random((26,49,126),dtype=np.float32)
        original=p.copy()
        for mode in POLICIES:
            q=rescue(p,f,mode,.6)
            np.testing.assert_array_equal(q[:,:,1:],p[:,:,1:])
            self.assertTrue(np.all(q[:,:,0][p[:,:,0]>=.6]>=.6))
        np.testing.assert_array_equal(p,original)

    def test_no_future_audio(self):
        rng=np.random.default_rng(688)
        f=rng.random((105,49,126),dtype=np.float32)
        p=rng.random((105,49,3),dtype=np.float32)
        a=rescue(p,f,"guard_0.70_ratio_1.25",.6)
        p[81:]=0;f[81:]=99
        b=rescue(p,f,"guard_0.70_ratio_1.25",.6)
        np.testing.assert_array_equal(a[:80],b[:80])
if __name__=="__main__":unittest.main()
