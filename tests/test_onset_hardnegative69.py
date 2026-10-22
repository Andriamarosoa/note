"""Independent Onset-only 69D ErrorPredictionHead and native256 research tests."""
import unittest
from types import SimpleNamespace
import numpy as np
from onset256.acoustics import HOP
from onset256.error_evidence69 import (
    DIM,STATE_NAMES,capture_selector_state,hard_matrix,hard_event_vector,
    HardNegativeRiskModel)
from onset256.selector import CausalSelection
from onset256.runtime import Stream
from scripts.train_onset_hardnegative69 import weighted_training_indices

class DummyMain:
    metadata={"selection":{}}
    def predict(self,features):
        p=np.zeros((49,3),np.float32);p[0]=(.85,.9,.02)
        return p,np.ones(7,np.float32)/7

class DummyRescue:
    def predict_proba(self,x):
        assert x.shape==(49,39)
        return np.tile(np.array([.4,.6],np.float32),(49,1))

class TrulyTrained69Risk:
    """Testing adapter mimics an independent fitted 69D neural checkpoint."""
    requires_selector_state=True
    trained_hop_samples=256
    cut=.5
    def __init__(self):self.calls=0;self.input_history=[]
    def predict_correct_probability(self,f,p,k,r,x,selector_state):
        matrix=hard_matrix(f,p,k,r,x,selector_state)
        self.input_history.append(matrix.copy())
        self.calls+=1
        result=np.full(49,.9,np.float32)
        result[0]=.1
        return result

class FakeEstimator:
    n_features_in_=69
    def predict_proba(self,features):
        assert features.shape==(49,69)
        return np.tile(np.array([.2,.8]),(49,1))

class Native69Tests(unittest.TestCase):
    def test_selector_snapshot_is_predecision_and_no_future(self):
        selector=CausalSelection()
        s1=capture_selector_state(selector,HOP)
        self.assertEqual(s1.shape,(49,6))
        np.testing.assert_array_equal(s1[:,0],0.)
        np.testing.assert_array_equal(s1[:,1],1.)
        self.assertEqual(len(STATE_NAMES),6)
        probs=np.zeros((49,3),np.float32);probs[0]=(.9,.8,0.)
        k=np.ones(7,np.float32)/7
        e=selector.step(probs,k,256,256)
        self.assertEqual([x["type"] for x in e],["note_on"])
        s2=capture_selector_state(selector,2*HOP)
        self.assertEqual(s2[0,0],1.)
        self.assertEqual(s2[0,1],0.)
        self.assertAlmostEqual(float(s2[0,4]),1./256.)
        self.assertAlmostEqual(float(s2[0,5]),.1)
        # Taking a risk snapshot must NOT modify active/rearm states.
        np.testing.assert_array_equal(selector.active,[True]+[False]*48)
        with self.assertRaises(AssertionError):
            capture_selector_state(selector,4*HOP)
    def test_train_69D_equals_causal_live_69D(self):
        x=np.zeros((49,126),np.float32)
        p=np.zeros((49,3),np.float32)
        q=np.zeros(49,np.float32)
        k=np.ones(7,np.float32)/7
        history=np.zeros((49,39),np.float32)
        selector=CausalSelection()
        states=capture_selector_state(selector,HOP)
        a=hard_matrix(x,p,k,q,history,states)
        self.assertEqual(a.shape,(49,69))
        np.testing.assert_array_equal(a[:,-6:],states)
        n=hard_event_vector({"hard_predecision_vector":a[0].tolist()})
        np.testing.assert_array_equal(n,a[0])
        with self.assertRaises(ValueError):
            hard_matrix(x,p,k,q,history,states[:1])
    def test_real_three_heads_predict_doubt_prior_to_note_on(self):
        risk=TrulyTrained69Risk()
        native=Stream(DummyMain(),DummyRescue())
        improved=Stream(DummyMain(),DummyRescue(),risk_model=risk,audit_events=True)
        samples=np.zeros(3*HOP,np.float32)
        base=native.push(samples,start_sample=0)
        tested=improved.push(samples,start_sample=0)
        self.assertEqual([(x.sample,x.intents,x.candidate_intents) for x in base],
                         [(x.sample,x.intents,x.candidate_intents) for x in tested])
        self.assertEqual(len(tested),3)
        self.assertEqual(risk.calls,3)
        self.assertEqual(improved.risk_calls,3)
        self.assertEqual(improved.onset_heads.names,
                         ("main_attack","rescue_attack","error_prediction"))
        self.assertTrue(tested[0].head_doubts[0].contested)
        self.assertEqual(tested[0].head_doubts[0].pitch,40)
        self.assertEqual(len(tested[0].predecision_intents),0)
        self.assertEqual(len(tested[0].evidence[0]["hard_predecision_vector"]),69)
        self.assertEqual(float(risk.input_history[0][0,63]),0.)
        self.assertEqual(float(risk.input_history[1][0,63]),1.)
        with self.assertRaises(ValueError):
            improved.push(np.zeros(1,np.float32),start_sample=0)
    def test_hard_negative_sampling_uses_only_train_examples(self):
        def row(is_tp,hard,active,name):
            x=np.zeros(DIM,np.float32);x[63]=int(active)
            x[68]=.5 if active else 1.
            return dict(X=x,y=int(is_tp),hard_harmonic=bool(hard),
                        texture="comp",pitch=71,name=name)
        rows=[row(True,True,True,"trainA"),
              row(False,True,False,"trainB"),
              row(False,False,True,"trainC"),
              row(False,False,False,"outerNever")]
        h,_=weighted_training_indices(rows,[0,1,2],"harmonic",8)
        t,_=weighted_training_indices(rows,[0,1,2],"restrike",8)
        self.assertEqual(h.count(1),5)
        self.assertEqual(h.count(2),2)
        self.assertEqual(t.count(1),2)
        self.assertEqual(t.count(2),5)
        self.assertNotIn(3,h)
        self.assertNotIn(3,t)
    def test_neural_weights_independent_shape(self):
        n=HardNegativeRiskModel(FakeEstimator(),cut=.45)
        f=np.zeros((49,126),np.float32);p=np.zeros((49,3),np.float32)
        k=np.ones(7,np.float32)/7
        q=np.zeros(49,np.float32)
        history=np.zeros((49,39),np.float32)
        st=capture_selector_state(CausalSelection(),HOP)
        result=n.predict_correct_probability(f,p,k,q,history,st)
        self.assertEqual(result.shape,(49,))
        self.assertEqual(n.calls,1)
        self.assertTrue(np.allclose(result,.8))
        self.assertEqual(DIM,69)
if __name__=="__main__":unittest.main()
