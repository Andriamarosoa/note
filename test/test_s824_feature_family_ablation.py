"""S824 signed 23 S795 feature-family ablation contract tests."""
import unittest
import numpy as np
from causal_note.s824_feature_family_ablation import (
    PRED,MIDI,ACOUSTIC,MASKS,MODES,UNIQUE_MODES,ALIASES,
    masked_matrix,train_all,choose_on_cal
)
from causal_note.s823_legacy23_asof_midi256 import asof_indices

def piece(name,comp,seed):
    rng=np.random.default_rng(seed)
    x=rng.normal(size=(14,49,23)).astype(np.float16)
    d=1024*np.arange(1,len(x)+1)
    ends=256*np.arange(1,57)
    return dict(name=name,composition=comp,X=x,asof=asof_indices(d,ends),
       k=(np.arange(len(ends),dtype=np.uint8)%7))

class S824FeatureFamiliesTests(unittest.TestCase):
    def test_exact_origin_families_and_masks(self):
        self.assertEqual(set(PRED)|set(MIDI)|set(ACOUSTIC),set(range(23)))
        self.assertEqual(len(PRED),14)
        self.assertEqual(MIDI,(12,15))
        self.assertEqual(ACOUSTIC,tuple(range(16,23)))
        self.assertEqual(len(MODES),11)
        self.assertEqual(len(UNIQUE_MODES),9)
        self.assertEqual(set(MASKS["without_s708"]),set(MIDI+ACOUSTIC))
        self.assertEqual(set(MASKS["without_midi"]),set(PRED+ACOUSTIC))
        self.assertEqual(set(MASKS["acoustic_only"]),set(ACOUSTIC))
        self.assertTrue(all(set(MASKS[a])==set(MASKS[b]) for a,b in ALIASES.items()))

    def test_forbidden_feature_perturbation_does_not_affect_projection(self):
        rng=np.random.default_rng(824)
        src=rng.normal(size=(20,23)).astype(np.float32)
        for mode in MODES:
            changed=src.copy()
            removed=sorted(set(range(23))-set(MASKS[mode]))
            changed[:,removed]+=12345.
            np.testing.assert_array_equal(masked_matrix(src,mode),
                                          masked_matrix(changed,mode))
            result=masked_matrix(src,mode)
            if removed:self.assertTrue(np.all(result[:,removed]==0))
        with self.assertRaisesRegex(ValueError,"genuine S795"):
            masked_matrix(src[:,:22],"full23")

    def test_causal_asof_and_nonshared_models(self):
        train=[piece(f"train{i}","A" if i<16 else "B" if i<32 else "C",i)
               for i in range(48)]
        cal=[piece(f"cal{i}","D",100+i) for i in range(16)]
        models=train_all(train,cap=84)
        self.assertEqual(set(models),set(MODES))
        self.assertEqual(len({id(models[x].model) for x in UNIQUE_MODES}),9)
        for mode,teacher in models.items():
            self.assertEqual(teacher.metadata["decision_hop"],256)
            self.assertEqual(teacher.metadata["source_observation_hop"],1024)
            self.assertFalse(teacher.metadata["inference_uses_truth"])
            self.assertFalse(teacher.metadata["promotion"])
            self.assertEqual(teacher.metadata["added_lookahead_samples"],0)
            self.assertFalse(teacher.metadata["production_LIVE_unchanged"] is False)
            self.assertEqual(teacher.metadata["train_compositions"],["A","B","C"])
        decision,results=choose_on_cal(cal,models)
        self.assertIn(decision,UNIQUE_MODES)
        self.assertTrue(all(r["frames"]>0 for r in results.values()))
        with self.assertRaisesRegex(ValueError,"CAL composition touched TRAIN"):
            choose_on_cal(train[:16],models)

if __name__=="__main__":
    unittest.main()
