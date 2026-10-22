"""A REAL neural predictor must compute risk before the NOTE-ON decision.

Safety/correctness: PCM256 at 44.1kHz, no S708, no GT, separate private
Onset-only states, correct 63D S613/S701 ABI, original baseline bit-identical.
"""
import unittest
from types import SimpleNamespace
import numpy as np
from onset256.acoustics import HOP,SR
from onset256.runtime import Stream
from onset256.harmonic_evidence import onset_event_evidence,HARMONIC_FIELDS
from onset256.predecision_risk import (
    DIM,SCHEMA,FEATURE_NAMES,predecision_matrix,event_vector,FrozenRiskModel)
from scripts.train_onset_predecision256 import train_one

class DummyNative:
    def __init__(self):self.calls=0;self.metadata={"selection":{}}
    def predict(self,feat):
        self.calls+=1
        p=np.zeros((49,3),np.float32)
        p[0]=(.75,.9,.0)
        return p,np.ones(7,np.float32)/7

class DummyS701:
    def __init__(self):self.calls=0
    def predict_proba(self,x):
        self.calls+=1
        if x.shape!=(49,39):raise AssertionError("wrong S701 ABI")
        return np.tile([.4,.6],(49,1))

class DummyRisk:
    cut=.5
    def __init__(self):self.calls=0;self.stream=None
    def predict_correct_probability(self,f,p,k,s,x):
        assert self.stream is not None
        assert self.stream.base.last_end==self.calls*HOP
        assert self.stream.shadow.last_end==self.calls*HOP
        assert self.stream.risk_selector.last_end==self.calls*HOP
        self.calls+=1
        result=np.ones(49,np.float32)
        result[0]=.1
        return result

class OnsetPredecisionMLPTests(unittest.TestCase):
    def test_training_inference_feature_exact_equivalence(self):
        rng=np.random.default_rng(256)
        f=np.maximum(rng.normal(size=(49,126)),0).astype(np.float32)
        p=rng.uniform(.01,.99,(49,3)).astype(np.float32)
        k=rng.uniform(.01,.99,7).astype(np.float32)
        k=k/k.sum()
        q=rng.uniform(.05,.95,49).astype(np.float32)
        history=rng.normal(size=(49,39)).astype(np.float32)
        mat=predecision_matrix(f,p,k,q,history)
        self.assertEqual(mat.shape,(49,63))
        self.assertEqual(len(FEATURE_NAMES),63)
        for pitch in (40,42,52,59,71,88):
            evidence=onset_event_evidence(f,p,k,q,pitch,256)
            j=pitch-40
            expected=np.asarray([
                *(evidence["context"][name] for name in HARMONIC_FIELDS),
                *p[j].tolist(),float(q[j]),*k.tolist(),pitch/127.,
                *history[j].tolist()],np.float32)
            np.testing.assert_allclose(mat[j],expected,rtol=1e-6,atol=1e-6)
            evidence["predecision_vector"]=mat[j].tolist()
            np.testing.assert_array_equal(event_vector(evidence),mat[j])
    def test_before_selector_risk_and_unchanged_native(self):
        n,m=DummyNative(),DummyS701()
        baseline=Stream(DummyNative(),DummyS701())
        risk=DummyRisk()
        s=Stream(n,m,risk_model=risk)
        risk.stream=s
        b=baseline.push(np.zeros(3*HOP,np.float32),start_sample=0)
        tested=s.push(np.zeros(3*HOP,np.float32),start_sample=0)
        self.assertEqual([d.sample for d in tested],[256,512,768])
        self.assertEqual([(d.sample,d.intents,d.candidate_intents) for d in b],
                         [(d.sample,d.intents,d.candidate_intents) for d in tested])
        self.assertEqual((n.calls,m.calls,risk.calls,s.calls,s.risk_calls),(3,3,3,3,3))
        self.assertEqual([(e.kind,e.pitch,e.sample) for e in tested[0].intents],
                         [("note_on",40,256)])
        self.assertFalse(any(d.predecision_intents for d in tested))
        self.assertEqual(HOP,256)
        self.assertAlmostEqual(1000*HOP/SR,5.804988662,places=7)
    def test_missing_future_audio_emits_nothing(self):
        risk=DummyRisk()
        s=Stream(DummyNative(),DummyS701(),risk_model=risk)
        risk.stream=s
        self.assertEqual(s.push(np.zeros(HOP-1,np.float32),start_sample=0),())
        self.assertEqual(risk.calls,0)
        a=s.push(np.zeros(1,np.float32),start_sample=HOP-1)
        self.assertEqual([d.sample for d in a],[HOP])
        self.assertEqual(risk.calls,1)
    def test_real_neural_network_fits_binary_risk_labels(self):
        # Real sklearn MLP weights (not mocked), three hidden layers including
        # an internal scaler. These 128 synthetic training examples certify API.
        rng=np.random.default_rng(926)
        rows=[]
        for i in range(128):
            x=rng.standard_normal(DIM).astype(np.float32)
            label=int(x[0]+.5*x[3]>0)
            rows.append(dict(X=x,y=label))
        m=train_one(rows,list(range(len(rows))),926)
        self.assertEqual(m.n_features_in_,63)
        p=m.predict_proba(rng.standard_normal((49,DIM)).astype(np.float32))
        self.assertEqual(p.shape,(49,2))
        self.assertTrue(np.isfinite(p).all())
        head=FrozenRiskModel(m,cut=.6)
        self.assertEqual(head.cut,.6)
        self.assertEqual(head.model.n_features_in_,DIM)
    def test_bad_abi_rejected(self):
        f=np.zeros((49,126),np.float32)
        with self.assertRaises(ValueError):
            predecision_matrix(f,np.zeros((49,3),np.float32),
                               np.zeros(7,np.float32),np.zeros(49,np.float32),
                               np.zeros((48,39),np.float32))
if __name__=="__main__":
    unittest.main()
