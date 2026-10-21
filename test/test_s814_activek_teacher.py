"""S814 tests: synthetic fixtures prove contracts, never a musical benchmark."""
import copy
import inspect
import unittest
import numpy as np

from causal_note.s814_activek_teacher import (
    summarize, sample_frames, active_score, fit_teacher, SELECTORS, SHIFTS
)
from causal_note.s813_teacher_contract import unification_gate


def fake(name, comp, seed, frames=133):
    r=np.random.default_rng(seed)
    x=r.uniform(0,2,(frames,49,222)).astype(np.float16)
    # All K0..K6 included; these labels are not passed to predict().
    k=np.arange(frames,dtype=np.uint8)%7
    r.shuffle(k)
    return dict(name=name,composition=comp,X=x,k=k)


class S814IndependentActiveKTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.training=[fake("t1","training_a",1),fake("t2","training_b",2)]
        cls.cal=[fake("cal","cal_comp",3)]
        cls.teacher=fit_teacher(cls.training,cls.cal,cap=70)
        cls.outer=fake("out","never_seen",4)

    def test_native256_acoustic_shape_and_finite(self):
        y=summarize(self.outer["X"])
        self.assertEqual(y.shape,(133,220))
        for bad in (np.zeros((9,49,23)),np.zeros((9,49,126))):
            with self.assertRaisesRegex(ValueError,"256"):
                summarize(bad)
        x=self.outer["X"].copy()
        x[0,0,0]=np.nan
        with self.assertRaisesRegex(ValueError,"nonfinite"):
            summarize(x)

    def test_no_future_and_no_delayed_state(self):
        X=self.outer["X"]
        ref=self.teacher.predict(X)
        mutated=X.copy()
        mutated[50:]=42
        np.testing.assert_array_equal(ref[:50],self.teacher.predict(mutated[:50]))
        # The unchanged prefix must also match when a future suffix differs.
        np.testing.assert_array_equal(ref[:50],self.teacher.predict(mutated)[:50])
        piecewise=np.concatenate([self.teacher.predict(X[:25]),
                                  self.teacher.predict(X[25:72]),
                                  self.teacher.predict(X[72:])])
        np.testing.assert_array_equal(ref,piecewise)
        self.assertEqual(ref.dtype,np.dtype("uint8"))
        self.assertTrue(np.all(ref<=6))
        self.assertFalse(ref.flags.writeable)

    def test_two_private_fitted_heads_and_private_selection(self):
        self.assertIsNot(self.teacher.heads["direct"],
                         self.teacher.heads["regression"])
        self.assertIn(self.teacher.selector,SELECTORS)
        self.assertIn(self.teacher.shift,SHIFTS)
        self.assertEqual(len(self.teacher.metadata["calibration_results"]),9)
        self.assertEqual(self.teacher.metadata["train_compositions"],
                         ["training_a","training_b"])
        self.assertEqual(self.teacher.metadata["calibration_compositions"],
                         ["cal_comp"])
        self.assertFalse(self.teacher.metadata["supervised_inference_inputs"])
        with self.assertRaises(TypeError):
            self.teacher.heads["frame_pitch"]=object()

    def test_no_label_or_peer_teacher_in_predict_api(self):
        params=list(inspect.signature(self.teacher.predict).parameters)
        self.assertEqual(params,["X"])
        with self.assertRaises(TypeError):
            self.teacher.predict(self.outer["X"],self.outer["k"])
        before=self.teacher.predict(self.outer["X"])
        changed=copy.deepcopy(self.outer)
        changed["k"][:]=6
        np.testing.assert_array_equal(before,self.teacher.predict(changed["X"]))
        self.assertFalse(unification_gate([])["allow_unification"])

    def test_disjoint_composition_required(self):
        with self.assertRaisesRegex(ValueError,"disjoint"):
            fit_teacher(self.training,[fake("another","training_a",10)],cap=70)
        with self.assertRaisesRegex(ValueError,"disjoint"):
            fit_teacher(self.training,[fake("t1","other",11)],cap=70)

    def test_stratified_sample_preserves_unbalanced_prior(self):
        row=fake("rare","fit",12,frames=420)
        row["k"][:]=0
        row["k"][-7:]=np.arange(7,dtype=np.uint8)
        X,y,w,_=sample_frames([row],cap=70)
        self.assertEqual(X.shape[1],220)
        natural=np.bincount(row["k"],minlength=7)/420
        weighted=np.bincount(y,weights=w,minlength=7)/sum(w)
        np.testing.assert_allclose(natural,weighted,rtol=0,atol=1e-9)
        self.assertLess(float(natural[-1]),.01)

    def test_active_metric_excludes_zero_and_counts_direction(self):
        c=active_score(np.array([4,1,2,0,6],np.uint8),
                       np.array([0,2,2,1,5],np.uint8))
        self.assertEqual(c,dict(correct=1,frames=4,undercount=2,overcount=1))
        with self.assertRaises(ValueError):
            active_score([1,2],[2])


if __name__=="__main__":
    unittest.main()
