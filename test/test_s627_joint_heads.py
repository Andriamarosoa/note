"""S627 regression protection for strict causal jointly trained NOTE heads."""
from __future__ import annotations
import unittest
import numpy as np
from causal_note.s627_joint_heads import frames, targets, Fusion, fuse,variants,INPUT_DIM
from causal_note.strict_live import HOP,SR


class S627Tests(unittest.TestCase):
    def test_targets_cardinality_new_notes_not_sustain(self):
        ends=np.arange(1,80,dtype=np.int64)*HOP
        notes=[{"pitch":40,"time":.05,"duration":.3},
               {"pitch":43,"time":.05,"duration":.4},
               {"pitch":52,"time":.19,"duration":.3}]
        attack,k=targets(ends,notes)
        assert attack.shape==(len(ends),49)
        self.assertEqual(int(k.max()),2)
        self.assertEqual(int(k[np.searchsorted(ends,round(.34*SR))]),0)
        self.assertEqual(float(attack[np.searchsorted(ends,round(.34*SR)),0]),0.)
        self.assertTrue(np.any(k==1))
    def test_no_future_can_change_previous_frame_feature(self):
        rng=np.random.default_rng(627)
        x=rng.random((20,49,126)).astype(np.float32)
        p=rng.random((20,49,3)).astype(np.float32)
        current=frames(x,p)
        a=x.copy();b=p.copy()
        a[12:]*=3.;b[12:]=.01
        altered=frames(a,b)
        np.testing.assert_array_equal(current[:12],altered[:12])
        self.assertEqual(current.shape,(20,49,INPUT_DIM))
    def test_zero_fusion_exact_passthrough(self):
        rng=np.random.default_rng(627)
        p=rng.random((49,3)).astype(np.float32)
        q=rng.random(49).astype(np.float32)
        k=np.eye(7,dtype=np.float32)[2]
        np.testing.assert_array_equal(fuse(p,q,k,Fusion()),p)
        self.assertEqual(len(variants()),12)
        self.assertFalse(np.array_equal(fuse(p,q,k,Fusion(.6,.25,.25)),p))
    def test_never_use_active_count_as_event_count(self):
        p=np.zeros((49,3),np.float32)
        p[10,0]=.9;p[10,1]=.6
        q=np.zeros(49,np.float32);q[10]=.9
        activeK=np.eye(7,dtype=np.float32)[5]
        new_eventK=np.eye(7,dtype=np.float32)[0]
        a=fuse(p,q,activeK,Fusion(.5,0,1.))
        b=fuse(p,q,new_eventK,Fusion(.5,0,1.))
        self.assertGreater(a[10,0],b[10,0])
    def test_bad_values_fail(self):
        with self.assertRaises(ValueError):Fusion(1.2,0,0)
        with self.assertRaises(ValueError):fuse(np.zeros((49,2)),np.zeros(49),np.ones(7),Fusion())


if __name__=="__main__":
    unittest.main()
