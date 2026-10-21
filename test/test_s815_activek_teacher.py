"""Synthetic S815 contract tests, never counted as GuitarSet validation."""
from copy import deepcopy
import unittest
import numpy as np
from causal_note.s815_activek_teacher import (
    fit_teacher, SELECTORS, SHIFTS, SCHEMA
)
from causal_note.s814_activek_teacher import active_score

def row(name,comp,seed,T=49):
    rng=np.random.default_rng(seed)
    X=rng.uniform(0,2,(T,49,222)).astype(np.float16)
    y=(X[:,:,0]>.95)
    k=np.arange(T,dtype=np.uint8)%7
    return dict(X=X,y=y,k=k,composition=comp,name=name)

class S815PrivatePitchActiveKTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.training=[row("a","A",1),row("b","B",2)]
        cls.cal=[row("c","C",3)]
        cls.model=fit_teacher(cls.training,cls.cal,cap=35,pitch_cap=120)
        cls.outer=row("d","D",4)

    def test_all_three_models_owned(self):
        m=self.model
        self.assertEqual(len({id(z) for z in m.heads.values()}),3)
        self.assertEqual(len(m.metadata["trained_heads"]),3)
        self.assertEqual(m.metadata["schema"],SCHEMA)
        self.assertIn(m.selector,SELECTORS)
        self.assertIn(m.shift,SHIFTS)
        self.assertEqual(len(m.metadata["calibration_results"]),15)
        with self.assertRaises(TypeError):
            m.heads["onset"]=1

    def test_no_future_and_frame_partition_invariance(self):
        x=self.outer["X"]
        a=self.model.predict(x)
        changed=x.copy()
        changed[20:]=0
        np.testing.assert_array_equal(a[:20],self.model.predict(changed)[:20])
        np.testing.assert_array_equal(a,np.concatenate([
            self.model.predict(x[:10]),self.model.predict(x[10:])]))
        self.assertFalse(a.flags.writeable)
        self.assertTrue(np.all(a<=6))

    def test_no_label_uses_at_inference(self):
        a=self.model.predict(self.outer["X"])
        r=deepcopy(self.outer)
        r["k"][:]=0
        r["y"][:]=False
        np.testing.assert_array_equal(a,self.model.predict(r["X"]))
        with self.assertRaises(TypeError):
            self.model.predict(r["X"],r["y"])

    def test_forbid_train_cal_overlap(self):
        with self.assertRaisesRegex(ValueError,"disjoint"):
            fit_teacher(self.training,[row("x","A",5)],cap=35,pitch_cap=120)

    def test_true_active_denominator(self):
        d=active_score(np.array([0,2,1,3]),np.array([0,1,2,3]))
        self.assertEqual(d["frames"],3)
        self.assertEqual(d["correct"],1)
        self.assertEqual(d["overcount"],1)
        self.assertEqual(d["undercount"],1)

if __name__=="__main__":
    unittest.main()
