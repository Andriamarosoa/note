"""S701 multi-overtone learned K features are strictly past-only and 104D."""
import unittest
import numpy as np
from scripts.s701_harmonic_independence_K import (
    shadow_features,HARMONIC_K_EXTRA_DIMS
)
class S701FeatureTests(unittest.TestCase):
    def sample(self,n=80):
        rng=np.random.default_rng(701)
        feat=rng.random((n,49,126),dtype=np.float32)
        p=rng.random((n,49,3),dtype=np.float32)
        k=rng.random((n,7),dtype=np.float32)
        count=np.arange(n,dtype=np.int32)%5
        return feat,p,k,count
    def test_future_frames_never_used(self):
        feat,p,k,count=self.sample()
        a=shadow_features(feat,p,k,count)
        self.assertEqual(HARMONIC_K_EXTRA_DIMS,41)
        self.assertEqual(a.shape,(80,104))
        feat[60:]=45;p[60:]=0;k[60:]=0;count[60:]=6
        b=shadow_features(feat,p,k,count)
        np.testing.assert_array_equal(a[:60],b[:60])
    def test_frozen_source_inputs(self):
        feat,p,k,count=self.sample()
        save=(feat.copy(),p.copy(),k.copy(),count.copy())
        _=shadow_features(feat,p,k,count)
        for src,orig in zip((feat,p,k,count),save):
            np.testing.assert_array_equal(src,orig)
    def test_empty(self):
        feat=np.zeros((0,49,126),np.float32)
        p=np.zeros((0,49,3),np.float32)
        k=np.zeros((0,7),np.float32)
        c=np.zeros(0,np.int32)
        self.assertEqual(shadow_features(feat,p,k,c).shape,(0,104))
if __name__=="__main__":unittest.main()
