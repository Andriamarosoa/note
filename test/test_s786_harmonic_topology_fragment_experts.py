"""S786 causal lattice integrity, harmonic topology and fragment-level gate."""
import unittest
import numpy as np
from scripts.s786_harmonic_topology_fragment_experts import (
    harmonic_features,gated_candidate,robust_score,DIM,GATES,WEIGHTS,INTERVALS)

class S786TopologyTests(unittest.TestCase):
    def fixture(self,n=7):
        item=dict(x=np.zeros((n,49,288),np.float32),
                  decision_ends=(np.arange(n)+1)*1024)
        post=np.zeros((n,49,3),np.float32)
        k=np.ones((n,7),np.float32)/7
        return item,post,k

    def test_200d_and_no_future_leakage(self):
        item,post,k=self.fixture()
        x,h=harmonic_features(item,post,k,[])
        self.assertEqual(x.shape,(7,DIM))
        self.assertEqual(h.shape,(7,))
        item2=dict(item,x=item["x"].copy())
        item2["x"][4:,:,:]=200
        later=post.copy();later[4:,:,:]=1
        k2=k.copy();k2[4:,:]=.5
        y,h2=harmonic_features(item2,later,k2,[])
        np.testing.assert_array_equal(x[:4],y[:4])
        np.testing.assert_array_equal(h[:4],h2[:4])

    def test_pitch_identity_preserved(self):
        item,p,k=self.fixture()
        p[2,0,1]=.8;p[2,12,1]=.7
        x,h=harmonic_features(item,p,k,[])
        self.assertAlmostEqual(x[2,31],.8,places=6)
        self.assertAlmostEqual(x[2,31+12],.7,places=6)
        self.assertGreater(h[2],0)
        p2=p.copy();p2[2,12,1]=0
        x2,h2=harmonic_features(item,p2,k,[])
        self.assertEqual(h2[2],0)
        self.assertTrue(np.isfinite(x).all())

    def test_gate_switches_per_fragment_without_labels(self):
        class Base:
            def predict(self,x):return np.array([4,4,4],np.int8)
        class Topology:
            def predict(self,x):return np.array([5,5,6],np.int8)
        n=np.array([4,4,4],np.int8)
        a=np.zeros((3,200),np.float32)
        b=np.zeros((3,31),np.float32)
        harm=np.array([0,.17,.35],np.float32)
        result=gated_candidate(Topology(),Base(),a,b,harm,.15,n)
        self.assertEqual(result.tolist(),[4,5,6])

    def test_low_native_tail_guard(self):
        class Base:
            def predict(self,x):return np.array([4,4],np.int8)
        class Topology:
            def predict(self,x):return np.array([6,5],np.int8)
        native=np.array([0,4],np.int8)
        out=gated_candidate(Topology(),Base(),np.zeros((2,DIM)),
                  np.zeros((2,31)),np.ones(2),.15,native)
        self.assertEqual(out.tolist(),[0,5])

    def test_deterministic_policy_space(self):
        self.assertEqual(len(GATES)*len(WEIGHTS),6)
        self.assertEqual(len(INTERVALS),6)
        groups={"BN1":dict(per_k=[dict(n=1,accuracy=.5) for _ in range(7)],
                           exact_accuracy=.8)}
        self.assertEqual(len(robust_score(groups,groups["BN1"])),3)

if __name__=="__main__":
    unittest.main()
