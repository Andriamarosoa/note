"""Only true trained-256 weights and native256 causal Onset output are allowed."""
import unittest
import numpy as np
from onset256.acoustics import HOP,SR
from onset256.s701_features import LiveS701Features,onset_features
from onset256.runtime import Stream
from onset256.model import S613_SHA256,S701_SHA256

class DummyNative:
    def __init__(self):self.calls=0;self.metadata={"selection":{}}
    def predict(self,feat):
        self.calls+=1
        p=np.zeros((49,3),np.float32);p[0]=(.25,.85,.01)
        return p,np.ones(7,np.float32)/7
class DummyS701:
    def __init__(self):self.calls=0
    def predict_proba(self,x):
        assert x.shape==(49,39)
        self.calls+=1
        return np.tile([.2,.8],(49,1))

class OnsetTrained256(unittest.TestCase):
    def test_four_real_decisions_and_axis_isolation(self):
        n,m=DummyNative(),DummyS701();s=Stream(n,m)
        rows=tuple(d for i in range(4) for d in s.push(np.zeros(256,np.float32),start_sample=i*256))
        self.assertEqual([d.sample for d in rows],[256,512,768,1024])
        self.assertEqual((s.calls,n.calls,m.calls),(4,4,4))
        self.assertEqual(rows[0].intents,())
        self.assertEqual([(v.kind,v.pitch,v.sample) for v in rows[0].candidate_intents],[("note_on",40,256)])
        self.assertTrue(all(not d.candidate_intents for d in rows[1:]))
        with self.assertRaises(ValueError):s.push(np.zeros(256,np.float32),start_sample=0)
    def test_partial_blocks_have_no_lookahead(self):
        n,m=DummyNative(),DummyS701();s=Stream(n,m)
        self.assertEqual(s.push(np.zeros(255,np.float32),start_sample=0),())
        self.assertEqual([d.sample for d in s.push(np.zeros(1,np.float32),start_sample=255)],[256])
    def test_transport_partition_invariance(self):
        a=Stream(DummyNative(),DummyS701());b=Stream(DummyNative(),DummyS701())
        x=np.zeros(1024,np.float32)
        self.assertEqual(a.push(x,start_sample=0),tuple(z for i in range(4) for z in b.push(x[i*256:(i+1)*256],start_sample=i*256)))
    def test_s701_causal_exact_full_track_parity(self):
        rng=np.random.default_rng(701)
        f=rng.standard_normal((19,49,126)).astype(np.float32)
        p=rng.random((19,49,3)).astype(np.float32)
        k=rng.random((19,7)).astype(np.float32)
        cache=LiveS701Features()
        observed=np.stack([cache.step(f[i],p[i],k[i]) for i in range(19)])
        np.testing.assert_array_equal(observed,onset_features(f,p,k))
    def test_checkpoint_sha_256_and_pace(self):
        self.assertEqual(HOP,256)
        self.assertAlmostEqual(HOP/SR*1000,5.804988662,places=7)
        self.assertEqual(len(S613_SHA256),64)
        self.assertEqual(len(S701_SHA256),64)
if __name__=="__main__":unittest.main()
