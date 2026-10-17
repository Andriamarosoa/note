"""S691 dual-head causal evidence must agree before new Onset."""
import unittest
import numpy as np
from scripts.s691_two_head_onset_consensus import rescue,POLICIES
class TwoHeadTests(unittest.TestCase):
    def fixtures(self):
        rng=np.random.default_rng(691)
        return (rng.random((120,49,3),dtype=np.float32),
                rng.random((120,49,126),dtype=np.float32),
                rng.random((120,7),dtype=np.float32))
    def test_no_new_notes_without_both_heads(self):
        p=np.zeros((30,49,3),np.float32)
        f=np.ones((30,49,126),np.float32)
        k=np.zeros((30,7),np.float32);k[:,1]=1
        for policy in POLICIES:
            self.assertFalse(rescue(p,f,policy,.60,k).any())
    def test_never_edit_native_other_heads(self):
        p,f,k=self.fixtures()
        for policy in POLICIES:
            out=rescue(p,f,policy,.6,k)
            np.testing.assert_array_equal(out[:,:,1:],p[:,:,1:])
            self.assertTrue(np.all(out[:,:,0]>=p[:,:,0]))
    def test_future_window_has_no_effect(self):
        p,f,k=self.fixtures()
        a=rescue(p,f,"vote_k2_0.27_0.8",.6,k)
        p[71:]=0;f[71:]=9;k[71:]=0
        b=rescue(p,f,"vote_k2_0.27_0.8",.6,k)
        np.testing.assert_array_equal(a[:70],b[:70])
if __name__=="__main__":unittest.main()
