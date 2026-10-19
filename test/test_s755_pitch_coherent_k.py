"""S755: verify pitch-aligned rare-polyphony K specialist remains causal."""
import unittest
import numpy as np
from scripts.s755_pitch_coherent_k_fit_cal32 import (
    coherent_k_features,causal_k_features,calibrated_k,WEIGHTS,GRID)

class S755PitchCoherenceTests(unittest.TestCase):
    def setUp(self):
        self.T=22
        self.x=np.zeros((self.T,49,288),np.float32)
        self.p=np.zeros((self.T,49,3),np.float32)
        self.native=np.zeros_like(self.p)
        self.k=np.zeros((self.T,7),np.float32)
        self.k[:,1]=1.
        self.x[:,:,192:216]=.25
        self.p[:,0,:]=[.8,.9,.1]
        self.p[:,12,:]=[.2,.5,.7]

    def features(self):
        return coherent_k_features(self.x,self.p,self.k,self.native)

    def test_feature_contract_and_grid(self):
        self.assertEqual(self.features().shape,(self.T,240))
        self.assertEqual(len(GRID),12)
        self.assertGreater(WEIGHTS["rare"][3],WEIGHTS["rare"][2])
        self.assertGreater(WEIGHTS["rare_strong"][3],WEIGHTS["rare"][3])

    def test_cooccurrence_preserved_unlike_per_channel_marginals(self):
        # Same sorted marginal values per Onset and Sustain, but
        # different pitch association. S754 features cannot distinguish.
        a=self.features()
        old=causal_k_features(self.x,self.p,self.k,self.native)
        swapped=self.p.copy()
        swapped[:,0,1],swapped[:,12,1]=(
            self.p[:,12,1].copy(),self.p[:,0,1].copy())
        self.p=swapped
        b=self.features()
        legacy=causal_k_features(self.x,self.p,self.k,self.native)
        np.testing.assert_array_equal(old,legacy)
        self.assertGreater(float(np.abs(a-b).sum()),0.)

    def test_mutating_future_does_not_change_past(self):
        old=self.features()
        self.x[16:,:,:]=2.
        self.p[16:,:,:]=1.
        self.k[16:,:]=np.eye(7,dtype=np.float32)[6]
        self.native[16:,:,:]=.75
        np.testing.assert_array_equal(self.features()[:16],old[:16])

    def test_zero_blend_preserves_baseline_k(self):
        other=np.eye(7,dtype=np.float32)[6][None].repeat(self.T,axis=0)
        np.testing.assert_array_equal(calibrated_k(self.k,other,0),self.k)

    def test_bad_acoustic_shapes_rejected(self):
        with self.assertRaises(ValueError):
            self.features_from_broken_x()

    def features_from_broken_x(self):
        return coherent_k_features(self.x[:4],self.p,self.k,self.native)

if __name__=="__main__":
    unittest.main()
