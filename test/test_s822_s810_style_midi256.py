"""S822 controlled supervised S810-style re-training; no audio network in tests."""
import unittest
import numpy as np
from causal_note.s822_s810_style_teacher import (
    SOURCE_COLS,original_23,count_features_61,feats,
    sample_active,fit_teacher,choose,MODES,SCHEMA,
)
from causal_note.s810_trained_six_axis_shadow import count_features as S810_COUNT

def track(name,composition,seed):
    rng=np.random.default_rng(seed)
    X=rng.normal(1.,.4,(84,49,222)).astype(np.float16)
    # Active classes K1..6 and K0 with no future label feature.
    k=(np.arange(84)%7).astype(np.uint8)
    return dict(name=name,composition=composition,X=X,k=k)

class TestNativeS810FeatureReplication(unittest.TestCase):
    def test_exact_identical_s810_aggregation_mathematics(self):
        x=track("a","T",1)["X"]
        a=original_23(x)
        self.assertEqual(len(SOURCE_COLS),23)
        self.assertEqual(len(set(SOURCE_COLS)),23)
        self.assertTrue(all(0<=j<222 for j in SOURCE_COLS))
        np.testing.assert_array_equal(count_features_61(a),S810_COUNT(a))
        self.assertEqual(feats(x,"agg61").shape,(84,61))
        self.assertEqual(feats(x,"raw23").shape,(84,23))

    def test_no_future_and_data_isolation(self):
        x=track("a","T",1)["X"]
        original=feats(x,"agg61")
        future=x.copy()
        future[41:]=200.
        np.testing.assert_array_equal(feats(future[:41],"agg61"),original[:41])
        with self.assertRaisesRegex(ValueError,"PCM MIDI256"):
            feats(np.zeros((20,49,23),np.float16),"agg61")
        with self.assertRaisesRegex(ValueError,"ablation"):
            feats(x,"invalid")

    def test_true_k_active_only_weighted_sampling(self):
        rows=[track("a","T",1),track("b","T",2)]
        for mode in MODES:
            X,y,w,audit=sample_active(rows,mode,cap=24)
            self.assertEqual(X.shape[1],23 if mode=="raw23" else 61)
            self.assertTrue(np.isin(y,np.arange(1,7)).all())
            self.assertTrue(all(item["class_prior_verified"] for item in audit))
            self.assertAlmostEqual(np.mean(w),1.)
            self.assertTrue(len(w)>0)

    def test_active_k_inference_has_no_ground_truth_access(self):
        source=track("train-one","TRAIN",6)
        model=fit_teacher([source,track("train-two","TRAIN",7)],"agg61",cap=120)
        x=track("held-out","EXTERNAL",8)["X"]
        first=model.predict(x)
        independent_truth=np.full(len(x),6,dtype=np.uint8)
        independent_truth[:]=1
        second=model.predict(x)
        np.testing.assert_array_equal(first,second)
        self.assertNotIn("EXTERNAL",model.metadata["train_compositions"])
        self.assertFalse(model.metadata["labels_used_during_inference"])
        self.assertEqual(model.metadata["additional_hold_samples"],0)

    def test_two_owned_models_same_train_new_composition_calibration(self):
        train=[track("trA","A",1),track("trB","B",2)]
        cal=[track("cal","C",3)]
        teachers={mode:fit_teacher(train,mode,cap=120) for mode in MODES}
        self.assertIsNot(teachers["raw23"].model,teachers["agg61"].model)
        select,counts=choose(cal,teachers)
        self.assertIn(select,MODES)
        self.assertEqual(set(counts),set(MODES))
        for model in teachers.values():
            self.assertEqual(model.metadata["hop_samples"],256)
            self.assertEqual(model.metadata["additional_future_samples"],0)
            self.assertFalse(model.metadata["s810_original_channels_equal"])
            output=model.predict(cal[0]["X"])
            self.assertTrue(np.isin(output,np.arange(1,7)).all())
            self.assertFalse(output.flags.writeable)
            self.assertNotIn("C",model.metadata["train_compositions"])
        with self.assertRaisesRegex(ValueError,"CAL and TRAIN"):
            choose(train,teachers)
        self.assertTrue(SCHEMA.startswith("S822_"))
if __name__=="__main__":
    unittest.main()
