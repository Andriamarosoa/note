"""S817 contract tests use synthetic fixtures only, not musical scores."""
from copy import deepcopy
import unittest
import numpy as np
from causal_note.s817_highpoly_temporal import (
    CONTEXT_INDICES,DIM,LAGS,THRESHOLDS,SELECTORS,
    PrivateTemporalContext,temporal,fit_teacher,select
)

def record(name,comp,seed,T=93):
    rng=np.random.default_rng(seed)
    X=rng.uniform(0,2,(T,49,222)).astype(np.float16)
    k=np.arange(T,dtype=np.uint8)%7
    rng.shuffle(k)
    return dict(name=name,composition=comp,X=X,k=k)

class FrozenBase:
    def predict(self,X):
        return np.full(len(X),2,dtype=np.uint8)

class S817CausalHighPolyphonyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train=[record("train1","TR1",1),record("train2","TR2",2)]
        cls.cal=[record("cal","CAL",3)]
        cls.outer=record("outer","HOLD",4)
        cls.teacher=fit_teacher(cls.train,cls.cal,FrozenBase(),seed=817)

    def test_fixed_feature_shape_and_causal_suffix(self):
        X=self.outer["X"]
        full=temporal(X)
        self.assertEqual(full.shape,(len(X),DIM))
        other=X.copy()
        other[40:]=0
        np.testing.assert_array_equal(full[:40],temporal(other)[:40])
        self.assertEqual(DIM,len(CONTEXT_INDICES)*(1+len(LAGS)))

    def test_partition_invariant_streaming_private_state(self):
        X=self.outer["X"]
        state=PrivateTemporalContext()
        got=np.concatenate((state.push(X[:17]),state.push(X[17:50]),
                            state.push(X[50:])))
        np.testing.assert_array_equal(temporal(X),got)
        another=PrivateTemporalContext()
        np.testing.assert_array_equal(another.push(X),got)
        self.assertEqual(state.past.shape,(16,len(CONTEXT_INDICES)))

    def test_two_owned_heads_and_no_peer_characteristic(self):
        self.assertIsNot(self.teacher.heads["high_gate"],
                         self.teacher.heads["high_cardinality"])
        self.assertEqual(len(self.teacher.heads),2)
        self.assertTrue(self.teacher.metadata["no_peer_characteristic_model_or_state"])
        self.assertEqual(self.teacher.metadata["train_compositions"],["TR1","TR2"])
        self.assertEqual(self.teacher.metadata["calibration_compositions"],["CAL"])
        self.assertEqual(len(self.teacher.metadata["calibration_results"]),9)
        with self.assertRaises(TypeError):
            self.teacher.heads["frame_pitch"]=1

    def test_calibration_disjointness(self):
        with self.assertRaisesRegex(ValueError,"disjoint"):
            fit_teacher(self.train,[record("different","TR1",8)],FrozenBase())

    def test_predictions_use_only_private_base_plus_pcm(self):
        x=self.outer["X"]
        base=np.full(len(x),2,np.uint8)
        original=self.teacher.predict(x,base)
        mutated=deepcopy(self.outer)
        mutated["k"][:]=6
        np.testing.assert_array_equal(original,self.teacher.predict(mutated["X"],base))
        self.assertFalse(original.flags.writeable)
        self.assertTrue(np.all(original<=6))
        with self.assertRaisesRegex(ValueError,"base counts"):
            self.teacher.predict(x,np.zeros(len(x)-1,np.uint8))

    def test_past_not_extra_latency_future_persistence(self):
        self.assertEqual(self.teacher.metadata["added_lookahead_samples"],0)
        self.assertEqual(self.teacher.metadata["post_onset_hold_samples"],0)
        self.assertEqual(self.teacher.metadata["context_lags_samples"],[256,1024,4096])
        x=self.outer["X"]
        base=np.full(len(x),2,np.uint8)
        changed=x.copy()
        changed[60:]=42
        np.testing.assert_array_equal(
            self.teacher.predict(x,base)[:60],self.teacher.predict(changed,base)[:60])

    def test_selector_never_uses_labels(self):
        base=np.array([1,2,3,4],np.uint8)
        pro=np.array([4,5,6,4],np.uint8)
        confidence=np.array([.95,.2,.8,.5])
        np.testing.assert_array_equal(
            select(base,confidence,pro,"max_high",.7),[4,2,6,4])
        np.testing.assert_array_equal(
            select(base,confidence,pro,"base",.7),base)
        with self.assertRaisesRegex(ValueError,"undeclared"):
            select(base,confidence,pro,"future_oracle",.5)
        self.assertEqual(THRESHOLDS,(.5,.7,.9))
        self.assertEqual(SELECTORS,("base","override","max_high"))

if __name__=="__main__":
    unittest.main()
