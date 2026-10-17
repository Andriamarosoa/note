"""S683 strictly causal NOTE onset features and immutable K/sustain/release."""
import unittest
import numpy as np
from scripts.s683_causal_onset_residual_head import (
    onset_feature_rows,fuse,VARIANTS,FEATURE_CHOICES
)
class S683CausalOnsetTests(unittest.TestCase):
    def fixtures(self):
        n=120;npitches=49
        rng=np.random.default_rng(683)
        f=rng.random((n,npitches,126),dtype=np.float32)
        p=rng.random((n,npitches,3),dtype=np.float32)
        k=np.zeros((n,7),np.float32);k[:,1]=1
        return f,p,k

    def test_prefix_invariance_all_native_inputs(self):
        f,p,k=self.fixtures()
        t=np.array([25,33,60],np.int64)
        q=np.array([0,20,48],np.int64)
        x=onset_feature_rows(f,p,k,t,q)
        f[61:]=99;p[61:]=99;k[61:]=99
        y=onset_feature_rows(f,p,k,t,q)
        self.assertEqual(x.shape,(3,75))
        np.testing.assert_array_equal(x,y)

    def test_note_heads_and_K_posterior_unchanged_for_every_variant(self):
        _,p,k=self.fixtures()
        novelty=np.full((len(p),49),.96,np.float32)
        original=p.copy()
        frozen_k=k.copy()
        for policy in VARIANTS:
            modified=fuse(p,novelty,policy)
            np.testing.assert_array_equal(modified[:,:,1:],original[:,:,1:])
            np.testing.assert_array_equal(k,frozen_k)
        np.testing.assert_array_equal(p,original)

    def test_native_is_exact_copy(self):
        _,p,k=self.fixtures()
        self.assertIs(fuse(p,np.zeros((len(p),49),np.float32),"native"),p)
        self.assertEqual(FEATURE_CHOICES.max(),121)

if __name__=="__main__":
    unittest.main()
