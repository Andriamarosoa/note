"""A trained ERROR-PREDICTION HEAD, not a post-hoc filter, lives inside Onset."""
import unittest
import numpy as np
from onset256.acoustics import HOP,SR
from onset256.runtime import Stream
from onset256.heads import (
    HEAD_NAMES,OnsetHeadStack,OnsetHeadSelector,ErrorPredictionHead,
    HeadOpinions,Doubt)
from onset256.predecision_risk import DIM

class DummyMain:
    metadata={"selection":{}}
    def __init__(self):self.calls=0
    def predict(self,features):
        self.calls+=1
        p=np.zeros((49,3),np.float32)
        p[0]=[.8,.9,.0]
        k=np.ones(7,np.float32)/7
        return p,k

class DummyRescue:
    n_features_in_=39
    def __init__(self):self.calls=0
    def predict_proba(self,x):
        self.calls+=1
        assert x.shape==(49,39)
        return np.tile([.4,.6],(49,1))

class DummyErrorHead:
    cut=.5
    def __init__(self):self.calls=0;self.stream=None
    def predict_correct_probability(self,features,post,k,rescue,history):
        # This MUST be evaluated BEFORE primary, rescue and doubt selectors.
        if self.stream is not None:
            for s in (self.stream.base,self.stream.shadow,
                      self.stream.risk_selector):
                assert s.last_end==self.calls*256
        assert features.shape==(49,126)
        assert post.shape==(49,3)
        assert k.shape==(7,)
        assert rescue.shape==(49,)
        assert history.shape==(49,39)
        self.calls+=1
        p=np.ones(49,np.float32)*.9
        p[0]=.05  # learned head contests E2
        return p

class EmbeddedErrorHeadTests(unittest.TestCase):
    def test_learned_error_head_is_third_independent_onset_head(self):
        n,r,e=DummyMain(),DummyRescue(),DummyErrorHead()
        stream=Stream(n,r,risk_model=e)
        e.stream=stream
        self.assertEqual(stream.onset_heads.names,HEAD_NAMES)
        self.assertEqual(stream.onset_heads.main.name,"main_attack")
        self.assertEqual(stream.onset_heads.rescue.name,"rescue_attack")
        self.assertEqual(stream.onset_heads.error.name,"error_prediction")
        self.assertIsNot(stream.onset_heads.main.model,stream.onset_heads.error.model)
        self.assertIsNot(stream.onset_heads.rescue.model,stream.onset_heads.error.model)
        out=stream.push(np.zeros(256,np.float32),start_sample=0)
        self.assertEqual(len(out),1)
        d=out[0]
        self.assertEqual((n.calls,r.calls,e.calls),(1,1,1))
        self.assertEqual(d.sample,256)
        self.assertEqual([(x.kind,x.pitch,x.sample) for x in d.intents],
                         [("note_on",40,256)])
        self.assertEqual(d.predecision_intents,())
        self.assertEqual(len(d.head_doubts),1)
        doubt=d.head_doubts[0]
        self.assertIsInstance(doubt,Doubt)
        self.assertEqual(doubt.origin,"learned_error_prediction_head")
        self.assertEqual(doubt.pitch,40)
        self.assertTrue(doubt.contested)
        self.assertAlmostEqual(doubt.predicted_error_probability,.95,places=5)
        self.assertAlmostEqual(doubt.predicted_correct_probability,.05,places=5)
        self.assertEqual(stream.risk_calls,1)
    def test_doubt_never_mutates_main_or_rescue_onset(self):
        x=np.zeros(3*HOP,np.float32)
        native=Stream(DummyMain(),DummyRescue())
        risk=DummyErrorHead()
        scrutinized=Stream(DummyMain(),DummyRescue(),risk_model=risk)
        risk.stream=scrutinized
        before=native.push(x,start_sample=0)
        after=scrutinized.push(x,start_sample=0)
        self.assertEqual([(d.sample,d.intents,d.candidate_intents) for d in before],
                         [(d.sample,d.intents,d.candidate_intents) for d in after])
        self.assertEqual([d.sample for d in after],[256,512,768])
        self.assertEqual((scrutinized.calls,scrutinized.risk_calls,risk.calls),
                         (3,3,3))
        self.assertTrue(all(d.sample%256==0 for d in after))
        self.assertTrue(all(d.predecision_intents==() for d in after))
        self.assertEqual(sum(len(d.head_doubts) for d in after),1)
    def test_doubt_requires_real_256_audio_not_lookahead(self):
        risk=DummyErrorHead()
        stream=Stream(DummyMain(),DummyRescue(),risk_model=risk)
        risk.stream=stream
        self.assertEqual(stream.push(np.zeros(255,np.float32),start_sample=0),())
        self.assertEqual(risk.calls,0)
        decision=stream.push(np.zeros(1,np.float32),start_sample=255)
        self.assertEqual(decision[0].sample,256)
        self.assertEqual(risk.calls,1)
        self.assertAlmostEqual(1000*256/44100,5.80498866,places=7)
    def test_selector_contests_risk_without_editing_main_posterior(self):
        selector=OnsetHeadSelector(.5)
        main=np.zeros((49,3),np.float32)
        main[0]=(.85,.9,0)
        correct=np.ones(49,np.float32)
        correct[0]=.05
        selected=selector.challenged_candidate(main,correct)
        self.assertAlmostEqual(float(main[0,0]),.85,places=6)
        self.assertEqual(selected[0,0],0)
        np.testing.assert_array_equal(main[:,1:],selected[:,1:])
        np.testing.assert_array_equal(selected[1:],main[1:])
        base=selector.challenged_candidate(main,None)
        np.testing.assert_array_equal(base,main)
    def test_every_head_rechecks_256_cadence(self):
        stack=OnsetHeadStack(DummyMain(),DummyRescue(),DummyErrorHead())
        f=np.zeros((49,126),np.float32)
        out=stack.predict(f,end_sample=256)
        self.assertIsInstance(out,HeadOpinions)
        self.assertEqual(out.sample,256)
        self.assertEqual(out.error_probability.shape,(49,))
        self.assertEqual(out.correctness_probability.shape,(49,))
        with self.assertRaises(AssertionError):
            stack.predict(f,end_sample=1024)
        stack.predict(f,end_sample=512)
        self.assertEqual(stack.main.calls,2)
        self.assertEqual(stack.rescue.calls,2)
        self.assertEqual(stack.error.calls,2)
    def test_trained_feature_vector_remains_exactly_63(self):
        from onset256.predecision_risk import FEATURE_NAMES,SCHEMA
        self.assertEqual(DIM,63)
        self.assertEqual(len(FEATURE_NAMES),DIM)
        self.assertIn("TRAINED_256",SCHEMA)
    def test_without_risk_weight_fallback_is_untouched(self):
        stream=Stream(DummyMain(),DummyRescue())
        self.assertEqual(stream.onset_heads.names,("main_attack","rescue_attack"))
        out=stream.push(np.zeros(256,np.float32),start_sample=0)
        self.assertEqual(len(out[0].intents),1)
        self.assertEqual(out[0].head_doubts,())
        self.assertEqual(out[0].predecision_intents,())
        self.assertEqual(stream.risk_calls,0)
if __name__=="__main__":unittest.main()
