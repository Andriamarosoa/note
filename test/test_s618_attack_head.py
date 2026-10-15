"""S618 no-future conditioning, native ABI, and unchanged reference tests."""
from __future__ import annotations
import unittest
import numpy as np
from causal_note.strict_live import Selection,CausalSelection,HOP
from causal_note.s618_attack_head import (
    DIM, input_features, candidate_mask, NativeLearnedSelection,TrainedPolicy,variants)


class AttackHeadTests(unittest.TestCase):
    def test_schema_and_past_only_features(self):
        rng=np.random.default_rng(618)
        x=rng.random((30,49,126)).astype(np.float32)
        p=rng.random((30,49,3)).astype(np.float32)
        features=input_features(x,p)
        self.assertEqual(features.shape,(30,49,DIM))
        x2=x.copy();p2=p.copy()
        x2[20:]=rng.random(x2[20:].shape)
        p2[20:]=rng.random(p2[20:].shape)
        self.assertTrue(np.array_equal(features[:20],input_features(x2,p2)[:20]))
        self.assertEqual(candidate_mask(x,p).shape,(30,49))

    def test_noop_equal_original_s614_gate(self):
        rng=np.random.default_rng(619)
        original=CausalSelection(Selection(onset=.6,active_on=.5))
        new=NativeLearnedSelection(Selection(onset=.6,active_on=.5),TrainedPolicy())
        for i in range(100):
            p=rng.random((49,3)).astype(np.float32)
            k=np.eye(7,dtype=np.float32)[i%7]
            x=rng.random((49,126)).astype(np.float32)
            q=rng.random(49).astype(np.float32)
            a=original.step(p,k,HOP*(i+1),HOP*(i+1))
            b=new.step(p,k,q,x,HOP*(i+1))
            self.assertEqual(a,b)
        self.assertEqual(original.panic(101*HOP),new.finish(101*HOP))
        self.assertEqual(len(variants()),11)

    def test_inference_uses_only_current_trained_confidence(self):
        model=NativeLearnedSelection(Selection(onset=.6,active_on=.5),
                                     TrainedPolicy(veto_below=.4,rescue_above=.8))
        p=np.zeros((49,3),np.float32)
        x=np.zeros((49,126),np.float32)
        k=np.eye(7,dtype=np.float32)[1]
        q=np.zeros(49,np.float32)
        p[2]=(.9,.9,.0)
        self.assertEqual(model.step(p,k,q,x,HOP),[])
        q[2]=.98
        emitted=model.step(p,k,q,x,2*HOP)
        self.assertEqual([(e["type"],e["pitch"]) for e in emitted],
                         [("note_on",42)])
        self.assertGreater(model.changed,0)

    def test_fail_closed_on_shapes(self):
        with self.assertRaises(ValueError):
            input_features(np.zeros((4,49,125)),np.zeros((4,49,3)))
        with self.assertRaises(ValueError):
            TrainedPolicy(veto_below=1.2)

if __name__=="__main__":unittest.main()
