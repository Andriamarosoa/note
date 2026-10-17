"""S693 genuine K posterior gating, not GT K or future PCM."""
import unittest
import numpy as np
from scripts.s693_poly_predicted_k_onset import rescue,POLICIES
class S693Test(unittest.TestCase):
    def test_no_added_notes_when_original_K1_certain(self):
        p=np.zeros((30,49,3),np.float32)
        p[:,:,0]=.37;p[:,:,1]=.9
        f=np.ones((30,49,126),np.float32)
        f[:,:,54:57]=4;f[:,:,69:72]=3
        k=np.zeros((30,7),np.float32);k[:,1]=1.
        for policy in POLICIES:
            np.testing.assert_array_equal(rescue(p,f,policy,.6,k),p)
    def test_immutable_native_release_and_K(self):
        rng=np.random.default_rng(693)
        p=rng.random((80,49,3),dtype=np.float32)
        f=rng.random((80,49,126),dtype=np.float32)
        k=rng.random((80,7),dtype=np.float32)
        ref_k=k.copy();ref_p=p.copy()
        for policy in POLICIES:
            trial=rescue(p,f,policy,.6,k)
            np.testing.assert_array_equal(trial[:,:,1:],p[:,:,1:])
            self.assertTrue(np.all(trial[:,:,0]>=p[:,:,0]))
        np.testing.assert_array_equal(k,ref_k)
        np.testing.assert_array_equal(p,ref_p)
    def test_no_future_frame_dependency(self):
        rng=np.random.default_rng(693)
        p=rng.random((90,49,3),dtype=np.float32)
        f=rng.random((90,49,126),dtype=np.float32)
        k=rng.random((90,7),dtype=np.float32)
        a=rescue(p,f,"poly_0.30",.6,k)
        p[60:]=0;f[60:]=12;k[60:]=0
        b=rescue(p,f,"poly_0.30",.6,k)
        np.testing.assert_array_equal(a[:59],b[:59])
if __name__=="__main__":unittest.main()
