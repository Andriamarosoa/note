"""S825 structural/prefix, strict OOF and private head tests; synthetic only."""
import unittest
import numpy as np
from causal_note.s825_crossfit_activek import (
    SCHEMA,MODES,cardinality_features,pitch_proba,fit_teacher
)

def recording(name,comp,seed):
    rng=np.random.default_rng(seed)
    X=rng.uniform(0,2,(56,49,222)).astype(np.float16)
    y=(X[:,:,0]>.95).astype(np.uint8)
    k=(np.arange(56)%7).astype(np.uint8)
    return dict(name=name,composition=comp,X=X,y=y,k=k)

class TestS825IndependentOOF(unittest.TestCase):
    def test_three_modes_have_same_pcm256_causal_prefix(self):
        r=recording("test","EXT",7)
        rng=np.random.default_rng(17)
        p=rng.uniform(0,1,(56,49)).astype(np.float32)
        changed=r["X"].copy();changed[20:]+=10
        cp=p.copy();cp[20:]=1.
        dims=[]
        for mode in MODES:
            before=cardinality_features(p,r["X"],mode)
            after=cardinality_features(cp,changed,mode)
            np.testing.assert_array_equal(before[:20],after[:20])
            self.assertEqual(before.shape[0],56)
            self.assertTrue(np.isfinite(before).all())
            dims.append(before.shape[1])
        self.assertEqual(len(set(dims)),3)
        self.assertEqual(SCHEMA,"S825_OOF_PRIVATE_PITCH_ACTIVEK_NATIVE256_V1")

    def test_strict_three_way_oof_and_no_peer_teacher(self):
        train=[recording("a","A",1),recording("b","B",2),recording("c","C",3)]
        cal=[recording("d","D",4)]
        model=fit_teacher(train,cal,seed=825,pitch_max=56,count_max=56)
        self.assertEqual(set(model.models),set(MODES))
        self.assertEqual(len({id(v) for v in model.models.values()}),3)
        self.assertEqual(model.metadata["trained_private_heads"],4)
        self.assertEqual(len(model.metadata["crossfit_oof_composition_proofs"]),3)
        for comp,proof in model.metadata["crossfit_oof_composition_proofs"].items():
            self.assertNotIn(comp,proof["trained_compositions"])
            self.assertEqual(proof["withheld"],comp)
        self.assertFalse(model.metadata["S708_posteriors_or_MIDI_used"])
        self.assertFalse(model.metadata["other_axis_checkpoint_used"])
        self.assertEqual(model.metadata["lookahead_samples"],0)
        self.assertEqual(model.metadata["hold_samples"],0)
        x=cal[0]["X"]
        predictions=model.predict_all(x)
        for z in predictions.values():
            self.assertFalse(z.flags.writeable)
            self.assertEqual(z.shape,(len(x),))
            self.assertTrue(np.isin(z,range(7)).all())
        corrupted=cal[0].copy()
        corrupted["y"]=np.zeros_like(corrupted["y"])
        corrupted["k"]=np.zeros_like(corrupted["k"])
        np.testing.assert_array_equal(model.predict(x),model.predict(corrupted["X"]))
        with self.assertRaisesRegex(ValueError,"strict 3-composition"):
            fit_teacher(train,[recording("same","A",8)],pitch_max=56,count_max=56)

if __name__=="__main__":
    unittest.main()
