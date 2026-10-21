"""S823 signed legacy S795 as-of test; never use future 1024 sample data."""
import unittest
import numpy as np
from causal_note.s823_legacy23_asof_midi256 import (
    SCHEMA,asof_indices,feature_at,balanced_signed_sample,
    fit_teacher,calibrate,MODES
)
from causal_note.s810_trained_six_axis_shadow import count_features

def make_record(name,composition,seed):
    rng=np.random.default_rng(seed)
    X=rng.normal(0,1,(12,49,23)).astype(np.float16)
    d=np.arange(1,13,dtype=np.int64)*1024
    ends=np.arange(1,49,dtype=np.int64)*256
    k=(np.arange(len(ends))%7).astype(np.uint8)
    return dict(X=X,k=k,name=name,composition=composition,asof=asof_indices(d,ends))

class TestLegacy23AsOf256(unittest.TestCase):
    def test_strict_no_future(self):
        d=np.arange(1,5,dtype=np.int64)*1024
        ends=np.arange(1,17,dtype=np.int64)*256
        idx=asof_indices(d,ends)
        np.testing.assert_array_equal(idx,[-1,-1,-1,0,0,0,0,1,1,1,1,2,2,2,2,3])
        assert np.all(d[idx[idx>=0]]<=ends[idx>=0])
        features=np.zeros((4,49,23),dtype=np.float16)
        features[:]=np.arange(4,dtype=np.float16)[:,None,None]
        original=feature_at(features,idx,"agg61")
        revised=features.copy();revised[2:]=999
        np.testing.assert_array_equal(original[:11],feature_at(revised,idx[:11],"agg61"))
        self.assertTrue(np.all(original[:3]==0))
        self.assertEqual(SCHEMA,"S823_S795_SIGNED_LEGACY_ASOF1024_TO_MIDI256_SHADOW_V1")
        with self.assertRaisesRegex(ValueError,"S795 original"):
            asof_indices(np.arange(1,5)*256,ends)

    def test_exact_genuine_s810_61_feature_math(self):
        r=make_record("a","TRAIN",1)
        idx=r["asof"]
        a=np.zeros((len(idx),49,23),np.float32)
        active=idx>=0
        a[active]=r["X"][idx[active]]
        np.testing.assert_array_equal(feature_at(r["X"],idx,"agg61"),count_features(a))
        self.assertEqual(feature_at(r["X"],idx,"raw23").shape,(48,23))
        self.assertEqual(feature_at(r["X"],idx,"agg61").shape,(48,61))

    def test_prior_train_cal_separation_two_distinct_models(self):
        tr=[make_record("trainA","TRAIN_A",3),make_record("trainB","TRAIN_B",4)]
        cal=[make_record("cal","CAL",5)]
        for m in MODES:
            x,y,w,proof=balanced_signed_sample(tr,m,cap=140)
            self.assertTrue(np.isin(y,np.arange(7)).all())
            self.assertEqual(x.shape[1],23 if m=="raw23" else 61)
            self.assertTrue(all(z["asof_legacy_only"] for z in proof))
            self.assertAlmostEqual(w.mean(),1.)
        models={m:fit_teacher(tr,m,cap=140) for m in MODES}
        self.assertIsNot(models["raw23"].model,models["agg61"].model)
        selected,results=calibrate(cal,models)
        self.assertIn(selected,MODES)
        self.assertEqual(results[selected]["frames"],int(np.count_nonzero(cal[0]["k"])))
        for t in models.values():
            self.assertTrue(t.metadata["depends_on_original_S708_posteriors_and_MIDI"])
            self.assertFalse(t.metadata["reextracted_PCM_native256"])
            self.assertFalse(t.metadata["peer_axis_independence"])
            self.assertFalse(t.metadata["promotion"])
            prediction=t.predict(cal[0])
            self.assertFalse(prediction.flags.writeable)
            self.assertTrue(np.isin(prediction,np.arange(7)).all())
        with self.assertRaisesRegex(ValueError,"CAL overlaps TRAIN"):
            calibrate(tr,models)

if __name__=="__main__":
    unittest.main()
