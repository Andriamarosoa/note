"""S697 63D future-free learned K features and original NOTE MIDI parity."""
import unittest
import numpy as np
from scripts.s697_causal_learned_K_shadow import (
    shadow_features,rank_cols,predictions
)
class ConstantK:
    classes_=np.array([0,1,2,3],dtype=np.int32)
    def predict_proba(self,X):
        y=np.tile([.05,.05,.85,.05],(len(X),1))
        return y
class S697Tests(unittest.TestCase):
    def data(self):
        n=90;rng=np.random.default_rng(697)
        f=rng.random((n,49,126),dtype=np.float32)
        p=rng.random((n,49,3),dtype=np.float32)
        k=rng.random((n,7),dtype=np.float32)
        active=np.asarray(np.arange(n)//10%4,dtype=np.int32)
        return f,p,k,active
    def test_63D_and_source_future_invariance(self):
        f,p,k,active=self.data()
        a=shadow_features(f,p,k,active)
        self.assertEqual(a.shape,(90,63))
        f[65:]=9;p[65:]=0;k[65:]=0;active[65:]=6
        b=shadow_features(f,p,k,active)
        np.testing.assert_array_equal(a[:65],b[:65])
    def test_original_K_and_decoded_note_frames_unchanged(self):
        f,p,k,active=self.data()
        orig_k=k.copy();orig_p=p.copy();orig_active=active.copy()
        X=shadow_features(f,p,k,active)
        item=dict(features=X,k=k,count=active)
        result=predictions(item,ConstantK())
        np.testing.assert_array_equal(result["native_decoded"],orig_active)
        np.testing.assert_array_equal(result["frozen_direct_K"],orig_k.argmax(axis=1))
        np.testing.assert_array_equal(result["learned"],np.full(90,2))
        np.testing.assert_array_equal(k,orig_k)
        np.testing.assert_array_equal(p,orig_p)
        np.testing.assert_array_equal(active,orig_active)
    def test_empty_stream(self):
        f=np.empty((0,49,126),np.float32);p=np.empty((0,49,3),np.float32)
        k=np.empty((0,7),np.float32)
        x=shadow_features(f,p,k,np.empty(0,np.int32))
        self.assertEqual(x.shape,(0,63))
if __name__=="__main__": unittest.main()
