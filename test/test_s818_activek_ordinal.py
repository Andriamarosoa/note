"""S818 structural synthetic tests; never counted as GuitarSet accuracy."""
from copy import deepcopy
import unittest
import numpy as np
from causal_note.s818_activek_ordinal import (
    LEVELS, FEATURES, OrdinalTeacher, audio_features,
    training_samples, ordinal_probabilities, decide, policy_grid, fit_teacher
)

def fixture(name,comp,seed,T=35):
    rng=np.random.default_rng(seed)
    X=rng.uniform(0,2,(T,49,222)).astype(np.float16)
    k=np.arange(T,dtype=np.uint8)%7
    rng.shuffle(k)
    return dict(name=name,composition=comp,X=X,k=k)

class S818OrdinalActiveKTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train=[fixture("a","TR1",1),fixture("b","TR2",2)]
        cls.cal=[fixture("c","CAL",3)]
        cls.outer=fixture("d","OUT",4)
        cls.teacher=fit_teacher(cls.train,cls.cal,cap=9,model_iter=7)

    def test_six_independent_heads(self):
        model=self.teacher
        self.assertEqual(len(model.heads),6)
        self.assertEqual(tuple(model.heads.keys()),tuple(f"k_ge_{i}" for i in range(1,7)))
        self.assertEqual(len({id(m) for m in model.models}),6)
        self.assertTrue(model.metadata["independent_model_object_ids_verified"])
        with self.assertRaises(TypeError):
            model.heads["frame_pitch"]=object()

    def test_dsp_shape_and_current_past_causality(self):
        data=self.outer["X"]
        y=audio_features(data)
        self.assertEqual(y.shape,(len(data),FEATURES))
        altered=data.copy()
        altered[17:]=42
        np.testing.assert_array_equal(y[:17],audio_features(altered)[:17])
        with self.assertRaisesRegex(ValueError,"MIDI256"):
            audio_features(np.zeros((3,49,123)))

    def test_predict_future_suffix_and_labels_cannot_leak(self):
        x=self.outer["X"]
        old=self.teacher.predict(x)
        modified=x.copy();modified[22:]=0
        np.testing.assert_array_equal(old[:22],self.teacher.predict(modified)[:22])
        other=deepcopy(self.outer)
        other["k"][:]=6
        np.testing.assert_array_equal(old,self.teacher.predict(other["X"]))
        self.assertFalse(old.flags.writeable)
        self.assertTrue(np.all(old<=6))
        with self.assertRaises(TypeError):
            self.teacher.predict(x,other["k"])

    def test_ordinal_probabilities_monotone_and_partition_causal(self):
        p=self.teacher.predict_proba(self.outer["X"])
        self.assertEqual(p.shape,(35,6))
        self.assertTrue(np.all(np.diff(p,axis=1)<=1e-8))
        self.assertTrue(np.all((p>=0)&(p<=1)))
        self.assertFalse(p.flags.writeable)

    def test_actual_class_prevalence_restored(self):
        row=fixture("rare","TR",34,T=350)
        row["k"][:]=0
        row["k"][-7:]=np.arange(7,dtype=np.uint8)
        X,y,weights,proof=training_samples([row],6,cap_per_class=8)
        self.assertEqual(X.shape[1],FEATURES)
        self.assertEqual(len(proof),1)
        expected=np.mean(row["k"]>=6)
        weighted=np.dot(y,weights)/weights.sum()
        self.assertAlmostEqual(float(weighted),float(expected),places=9)

    def test_calibration_disjointness(self):
        with self.assertRaisesRegex(ValueError,"composition-disjoint"):
            fit_teacher(self.train,[fixture("overlap","TR1",51)],cap=8,model_iter=7)
        with self.assertRaisesRegex(ValueError,"distinct"):
            fit_teacher(self.train,[fixture("a","DIF",52)],cap=8,model_iter=7)

    def test_predeclared_dynamic_threshold_selection_only(self):
        proba=np.array([[.91,.81,.71,.61,.51,.41],[.90,.60,.30,.20,.10,.02]],dtype=np.float32)
        np.testing.assert_array_equal(decide(proba,"crossing",.60),[4,2])
        np.testing.assert_array_equal(decide(proba,"expected",0.0),[4,2])
        self.assertEqual(len(policy_grid()),7)
        with self.assertRaisesRegex(ValueError,"unknown"):
            decide(proba,"oracle",.5)
        self.assertEqual(len(self.teacher.metadata["calibration_results"]),7)
        self.assertEqual(self.teacher.metadata["train_compositions"],["TR1","TR2"])
        self.assertEqual(self.teacher.metadata["calibration_compositions"],["CAL"])

if __name__=="__main__":
    unittest.main()
