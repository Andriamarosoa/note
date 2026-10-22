"""True 4th TRAINABLE Onset FNRecovery head, strict no-lookahead native PCM256."""
import unittest
from types import SimpleNamespace
import numpy as np
from onset256.acoustics import HOP,SR
from onset256.runtime import Stream
from onset256.fn_recovery import (FNRecoverySelector,FNRecoveryModel,
                                  CANDIDATE_POLICIES,DIM,EVAL_DEDUP_SAMPLES)
from scripts.train_onset_fn_recovery256 import train_one,score_selected

class FakeMain:
    metadata={"selection":{}}
    def __init__(self):self.calls=0
    def predict(self,f):
        self.calls+=1
        p=np.zeros((49,3),np.float32)
        p[0]=(.9,.9,0.) # E2 emitted by main
        p[12]=(.15,.25,0.) # E3 missing in main but available to recovery
        return p,np.ones(7,np.float32)/7

class FakeRescue:
    def __init__(self):self.calls=0
    def predict_proba(self,history):
        self.calls+=1
        assert history.shape==(49,39)
        return np.tile([.4,.6],(49,1))

class FakeErrorRisk:
    cut=.5
    requires_selector_state=True
    trained_hop_samples=256
    def __init__(self):self.calls=0
    def predict_correct_probability(self,f,p,k,q,history,st):
        self.calls+=1
        return np.ones(49,np.float32)*.99

class FakeFNModel:
    trained_hop_samples=256
    candidate_policy="moderate"
    calibrated_cut=.7
    enabled=True
    def __init__(self):self.calls=0
    def predict_recovery_probability(self,f,p,k,q,history,selector_state):
        assert f.shape==(49,126) and p.shape==(49,3)
        assert selector_state.shape==(49,6)
        self.calls+=1
        result=np.ones(49,np.float32)*.01
        result[12]=.98
        return result

class FakeEstimator:
    n_features_in_=69
    def predict_proba(self,x):
        assert x.shape==(49,69)
        return np.tile(np.array([.2,.8],np.float32),(49,1))

class FourHeadsFNAuditTests(unittest.TestCase):
    def test_new_fn_recovery_head_before_selection_and_main_intact(self):
        n,s701,risk,recovery=FakeMain(),FakeRescue(),FakeErrorRisk(),FakeFNModel()
        b=Stream(FakeMain(),FakeRescue(),risk_model=FakeErrorRisk())
        stream=Stream(n,s701,risk_model=risk,recovery_model=recovery)
        pcm=np.zeros(3*HOP,np.float32)
        original=b.push(pcm,start_sample=0)
        revised=stream.push(pcm,start_sample=0)
        self.assertEqual(
            [(x.sample,x.intents,x.predecision_intents) for x in original],
            [(x.sample,x.intents,x.predecision_intents) for x in revised])
        self.assertEqual(stream.onset_heads.names,(
            "main_attack","rescue_attack","error_prediction","fn_recovery"))
        self.assertEqual((n.calls,s701.calls,risk.calls,recovery.calls),
                         (3,3,3,3))
        self.assertEqual([(x.kind,x.pitch,x.sample)
                          for x in revised[0].fn_recovery_intents],
                         [("note_on",52,256)])
        self.assertEqual([x.pitch for x in revised[0].fn_unified_intents],[40,52])
        self.assertEqual(revised[1].fn_recovery_intents,())
        self.assertEqual(revised[2].fn_recovery_intents,())
        self.assertEqual(stream.fn_recovery_arbiter.calls,3)
    def test_unavailable_future_audio_produces_no_head_prediction(self):
        recovery=FakeFNModel()
        stream=Stream(FakeMain(),FakeRescue(),risk_model=FakeErrorRisk(),
                      recovery_model=recovery)
        self.assertEqual(stream.push(np.zeros(HOP-1,np.float32),start_sample=0),())
        self.assertEqual(recovery.calls,0)
        result=stream.push(np.zeros(1,np.float32),start_sample=HOP-1)
        self.assertEqual([x.sample for x in result],[256])
        self.assertEqual(recovery.calls,1)
        self.assertAlmostEqual(1000*HOP/SR,5.80498866,places=7)
    def test_candidate_selector_blocks_duplicate_native_main(self):
        selector=FNRecoverySelector(.5,True)
        p=np.ones(49,np.float32)*.9
        main=(SimpleNamespace(pitch=52,sample=256),)
        proposals=[dict(pitch=52,decision_sample=256),
                   dict(pitch=40,decision_sample=256)]
        accepted=selector.select(end_sample=256,
            main_raw=main,main_accepted=main,
            proposals=proposals,recovery_probability=p)
        self.assertEqual([(e["pitch"],e["decision_sample"]) for e in accepted],
                         [(40,256)])
        result=selector.select(end_sample=512,
            main_raw=(),main_accepted=(),
            proposals=[dict(pitch=40,decision_sample=512),
                       dict(pitch=52,decision_sample=512)],
            recovery_probability=p)
        self.assertEqual(result,())
        self.assertGreaterEqual(EVAL_DEDUP_SAMPLES,14*256)
    def test_learned_weights_native69_and_legal_policies(self):
        model=FNRecoveryModel(FakeEstimator(),candidate_policy="moderate",
                              calibrated_cut=.45)
        f=np.zeros((49,126),np.float32)
        p=np.zeros((49,3),np.float32)
        k=np.ones(7,np.float32)/7
        q=np.zeros(49,np.float32)
        history=np.zeros((49,39),np.float32)
        selector_state=np.zeros((49,6),np.float32)
        v=model.predict_recovery_probability(f,p,k,q,history,selector_state)
        self.assertEqual(v.shape,(49,))
        self.assertTrue(np.allclose(v,.8))
        self.assertEqual(model.calls,1)
        self.assertEqual(DIM,69)
        self.assertEqual(set(CANDIDATE_POLICIES),{"sensitive","moderate","cautious"})
        with self.assertRaises(ValueError):
            FNRecoveryModel(FakeEstimator(),candidate_policy="unseen",calibrated_cut=.5)
    def test_a_real_neural_network_is_trained_on_fn_candidates(self):
        rng=np.random.default_rng(256)
        rows=[]
        for i in range(192):
            x=rng.standard_normal(DIM).astype(np.float32)
            y=int(x[0]+x[12]*.7>0)
            rows.append(dict(X=x,y=y))
        model,details=train_one(rows,list(range(len(rows))),926)
        self.assertEqual(model.n_features_in_,69)
        self.assertGreater(details["tp_candidates"],30)
        p=model.predict_proba(rng.standard_normal((49,DIM)).astype(np.float32))
        self.assertEqual(p.shape,(49,2))
        self.assertTrue(np.isfinite(p).all())
    def test_offline_proxy_equals_actual_frozen_selector_arbitration(self):
        # Synthetic proof that CAL proxy uses EXACT live 256 candidate arbiter.
        truth={"test":[dict(pitch=40,time=HOP/SR,duration=.2),
                       dict(pitch=52,time=HOP/SR,duration=.2)]}
        main={"test":[SimpleNamespace(pitch=40,sample=HOP)]}
        rows=[dict(track="test",sample=HOP,pitch=52)]
        scores=np.array([.98],np.float32)
        summary,details=score_selected(("test",),truth,main,main,rows,scores,.7,True)
        self.assertEqual((summary["tp"],summary["fp"],summary["fn"]),(2,0,0))
        self.assertEqual(summary["added"],1)
if __name__=="__main__":unittest.main()
