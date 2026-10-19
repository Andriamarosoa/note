"""S754: causal K specialist input/fusion invariants, no fitted model needed."""
import unittest
import numpy as np
from scripts.s754_fit_k_specialist_cal32_loops import (
    ARCHITECTURES,WEIGHTS,GRID,causal_k_features,calibrated_k)

class KSpecialistTests(unittest.TestCase):
    def setUp(self):
        self.n=20
        self.x=np.zeros((self.n,49,288),np.float32)
        self.post=np.zeros((self.n,49,3),np.float32)
        self.native=np.zeros_like(self.post)
        self.k=np.zeros((self.n,7),np.float32)
        self.k[:,1]=1.
        self.x[:,:,192:216]=.35
        self.post[:,0,0]=.5
        self.post[:,1,1]=.4

    def feat(self):
        return causal_k_features(self.x,self.post,self.k,self.native)

    def test_all_twelve_independent_k_head_blends_are_declared(self):
        self.assertEqual(len(GRID),12)
        self.assertEqual(len(ARCHITECTURES),2)
        self.assertEqual(len(WEIGHTS),2)

    def test_causal_input_abi(self):
        x=self.feat()
        self.assertEqual(x.shape,(self.n,104))
        self.assertTrue(np.isfinite(x).all())

    def test_future_spectra_does_not_change_past_features(self):
        old=self.feat()
        self.x[15:]*=0
        self.x[15:,:,192:216]=100
        self.k[15:]=np.eye(7,dtype=np.float32)[6]
        self.post[15:]=1
        self.native[15:]=1
        altered=self.feat()
        np.testing.assert_array_equal(old[:15],altered[:15])

    def test_zero_blend_is_identical_to_original_s708(self):
        new=np.eye(7,dtype=np.float32)[5][None].repeat(self.n,axis=0)
        out=calibrated_k(self.k,new,0)
        np.testing.assert_array_equal(out,self.k)

    def test_independent_k6_specialist_is_not_forced_at_low_blend(self):
        specialist=np.eye(7,dtype=np.float32)[6][None].repeat(self.n,axis=0)
        lo=calibrated_k(self.k,specialist,.15)
        hi=calibrated_k(self.k,specialist,.55)
        self.assertTrue(np.all(lo.argmax(1)==1))
        self.assertTrue(np.all(hi.argmax(1)==6))
        np.testing.assert_allclose(lo.sum(1),1,atol=1e-6)

    def test_invalid_or_future_shape_fails_closed(self):
        with self.assertRaises(ValueError):
            causal_k_features(self.x[1:],self.post,self.k,self.native)
        with self.assertRaises(ValueError):
            calibrated_k(self.k,np.zeros((self.n,6),np.float32),.25)
        with self.assertRaises(ValueError):
            calibrated_k(self.k,self.k,1.25)

if __name__=="__main__":
    unittest.main()
