"""S821 synthetic contract: correct only K4 using independently trained acoustics."""
import unittest
import numpy as np
from causal_note.s821_k4_guard import (
    SCHEMA,_column_signature,sample_binary,policy_grid,apply_policy,fit_teacher
)

def record(name,comp,seed):
    rng=np.random.default_rng(seed)
    X=rng.uniform(0,2,(49,49,222)).astype(np.float16)
    k=np.arange(49,dtype=np.uint8)%7
    return dict(name=name,composition=comp,X=X,k=k)

class Base:
    def predict(self,X):
        return np.where(np.arange(len(X))%3==0,4,3).astype(np.uint8)

class FakeHead:
    def __init__(self,classes):
        self.classes_=np.asarray(classes)
    def predict_proba(self,X):
        v=np.zeros((len(X),len(self.classes_)),np.float32)
        v[:,-1]=.85
        v[:,0]=.15
        return v

class High:
    gate=FakeHead([0,1])
    high=FakeHead([4,5,6])

class K4GuardTest(unittest.TestCase):
    def test_no_future_and_preserve_pitchwise_information(self):
        x=record("a","TR",1)["X"]
        before=_column_signature(x)
        mutated=x.copy();mutated[20:]=50
        self.assertEqual(before.shape[0],len(x))
        np.testing.assert_array_equal(before[:20],_column_signature(mutated)[:20])
        self.assertGreater(before.shape[1],80)
        with self.assertRaisesRegex(ValueError,"MIDI256"):
            _column_signature(np.zeros((10,48,222),np.float16))

    def test_exact_non_k4_invariance_and_restricted_fifth_sixth(self):
        k=np.array([0,1,2,3,4,4,4,5,6],np.uint8)
        hyp=np.full(len(k),6,np.uint8)
        conf=np.full(len(k),.9,np.float32)
        p5=np.array([1,1,1,1,.2,.9,.9,1,1],np.float32)
        p6=np.array([1,1,1,1,.1,.2,.95,1,1],np.float32)
        a=apply_policy(k,hyp,conf,p5,p6,"verify56",.7)
        np.testing.assert_array_equal(a,[0,1,2,3,4,5,6,5,6])
        b=apply_policy(k,hyp,conf,p5,p6,"base",0.)
        np.testing.assert_array_equal(b,k)
        with self.assertRaisesRegex(ValueError,"undeclared"):
            apply_policy(k,hyp,conf,p5,p6,"oracle",.01)
        self.assertEqual(len(policy_grid()),13)

    def test_two_new_training_heads_do_not_use_outer_composition(self):
        train=[record("tr1","TR1",1),record("tr2","TR2",2)]
        cal=[record("cal1","CAL",3)]
        teacher=fit_teacher(train,cal,Base(),High(),seed=821)
        self.assertEqual(len(teacher.heads),2)
        self.assertIsNot(teacher.five,teacher.six)
        self.assertEqual(teacher.metadata["train_compositions"],["TR1","TR2"])
        self.assertEqual(teacher.metadata["calibration_compositions"],["CAL"])
        self.assertEqual(len(teacher.metadata["calibration_results"]),13)
        self.assertIn(teacher.mode,("base","confirm5","verify5","verify56"))
        self.assertEqual(teacher.metadata["added_lookahead_samples"],0)
        self.assertEqual(teacher.metadata["added_hold_samples"],0)
        base=Base().predict(cal[0]["X"])
        high=np.full(len(base),6,np.uint8)
        got=teacher.predict(cal[0]["X"],base,high,np.ones(len(base),np.float32))
        np.testing.assert_array_equal(got[base!=4],base[base!=4])
        with self.assertRaisesRegex(ValueError,"leakage"):
            fit_teacher(train,[record("c","TR1",3)],Base(),High())
        with self.assertRaises(TypeError):
            teacher.heads["other_axis"]=object()

    def test_binary_weights_preserve_true_training_prior(self):
        a=record("a","TRAIN",3)
        for level in (5,6):
            X,y,w,proof=sample_binary([a],level,cap=4,seed=821)
            self.assertEqual(len(proof),1)
            actual=np.mean(a["k"][a["k"] >= (3 if level==5 else 4)]>=level)
            self.assertAlmostEqual(float(np.dot(y,w)/w.sum()),float(actual),places=10)

if __name__=="__main__":
    unittest.main()
