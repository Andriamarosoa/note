"""Regressions for supervised sixth Onset FP expert gate, real causal hop256."""
import unittest
import numpy as np
from onset256.acoustics import HOP,SR
from onset256.fp_discriminator import FPDiscriminatorModel,VARIANTS
from onset256.fp_expert_gate import (
    FPExpertGateModel,GATE_FEATURES,gate_matrix,training_events,
    select_gate_onsets)
from onset256.runtime import Stream
from scripts.train_onset_fp_expert_gate256 import (
    train_small_network,selection,MIN_DISAGREEMENTS)

class Main:
    metadata={"selection":{}}
    def __init__(self):self.calls=0
    def predict(self,features):
        self.calls+=1
        p=np.zeros((49,3),np.float32)
        p[0]=(.9,.9,0.)
        p[12]=(.15,.25,0.)
        return p,np.ones(7,np.float32)/7

class Rescue:
    def __init__(self):self.calls=0
    def predict_proba(self,h):
        self.calls+=1
        return np.tile([.4,.6],(49,1))

class Error:
    trained_hop_samples=256
    cut=.5
    requires_selector_state=True
    def __init__(self):self.calls=0
    def predict_correct_probability(self,f,p,k,r,h,s):
        self.calls+=1
        return np.ones(49,np.float32)*.99

class FN:
    trained_hop_samples=256
    candidate_policy="moderate"
    calibrated_cut=.7
    enabled=True
    def __init__(self):self.calls=0
    def predict_recovery_probability(self,f,p,k,q,h,s):
        self.calls+=1
        pred=np.zeros(49,np.float32);pred[12]=.99
        return pred

class FPNet:
    def __init__(self,dim,variant):
        self.n_features_in_=dim
        self.variant=variant
        self.calls=0
    def predict_proba(self,X):
        self.calls+=1
        q=np.ones(len(X),np.float32)*.99
        if self.variant=="temporal74":q[0]=.01
        return np.column_stack((1-q,q))

class GateNet:
    n_features_in_=11
    def predict_proba(self,X):
        # Keep Main E2 as genuine despite hostile temporal74 veto.
        q=np.ones(len(X),np.float32)*.02
        q[0]=.98
        return np.column_stack((1-q,q))

def all_fp():
    return {key:FPDiscriminatorModel(FPNet(dim,key),key,.5)
            for key,dim in VARIANTS.items()}

class ExpertGateNative256Tests(unittest.TestCase):
    def test_gate_features_11d_from_current_frame_and_old_parent_only(self):
        p=np.ones(49,np.float32)*.9
        old=np.ones((49,4),np.float32)
        x=gate_matrix(p,p,p,p,old)
        self.assertEqual(x.shape,(49,11))
        self.assertEqual(float(x[0,10]),0.)
        self.assertEqual(float(x[48,10]),1.)
        self.assertEqual(float(x[12,6]),1.)
        self.assertEqual(GATE_FEATURES,11)
        self.assertEqual(HOP,256)
        self.assertEqual(SR,44100)
    def test_gate_learned_applies_only_to_disagreement(self):
        gate=FPExpertGateModel(GateNet(),enabled=True,
            cut=.5,cut69=.5,cut74=.5)
        q69=np.ones(49,np.float32)*.9
        q74=np.ones(49,np.float32)*.9
        q74[0]=.1
        prob=np.zeros(49,np.float32);prob[0]=.98
        keep=gate.accepted_mask(q69,q74,prob)
        self.assertTrue(keep[0])
        self.assertTrue(keep[12])
        # When both experts accept, network cannot suppress this NOTE-ON.
        prob[12]=0.
        self.assertTrue(gate.accepted_mask(q69,q74,prob)[12])
        # When both reject, the network cannot create a new NOTE-ON.
        q69[1]=0.;q74[1]=0.;prob[1]=1.
        self.assertFalse(gate.accepted_mask(q69,q74,prob)[1])
    def test_disabled_gate_matches_original_69d_even_with_74d_disagreement(self):
        gate=FPExpertGateModel(None,enabled=False,
            cut=1.,cut69=.5,cut74=.5)
        q69=np.linspace(0,1,49,dtype=np.float32)
        q74=1-q69
        out=gate.predict(q69,q69,q74,q69,np.ones((49,4),np.float32))
        np.testing.assert_array_equal(out,np.ones(49,np.float32))
        np.testing.assert_array_equal(
            gate.accepted_mask(q69,q74,out),q69>=.5)
    def test_original_four_heads_and_three_controls_remain_bitwise_unchanged(self):
        models=all_fp()
        baseline=Stream(Main(),Rescue(),risk_model=Error(),
                        recovery_model=FN(),fp_models=all_fp())
        gate_net=FPExpertGateModel(GateNet(),enabled=True,
                                  cut=.5,cut69=.5,cut74=.5)
        gated=Stream(Main(),Rescue(),risk_model=Error(),
                     recovery_model=FN(),fp_models=models,gate_model=gate_net)
        audio=np.zeros(4*HOP,np.float32)
        old=baseline.push(audio,start_sample=0)
        new=gated.push(audio,start_sample=0)
        self.assertEqual(len(old),4)
        for a,b in zip(old,new):
            self.assertEqual(a.sample,b.sample)
            self.assertEqual(a.intents,b.intents)
            self.assertEqual(a.predecision_intents,b.predecision_intents)
            self.assertEqual(a.fn_recovery_intents,b.fn_recovery_intents)
            self.assertEqual(a.fn_unified_intents,b.fn_unified_intents)
            self.assertEqual(a.fp_discriminator_intents,b.fp_discriminator_intents)
            self.assertEqual(a.fp_discriminator_unified,b.fp_discriminator_unified)
        self.assertEqual(new[0].fp_discriminator_intents["temporal74"],())
        self.assertEqual([x.pitch for x in new[0].fp_gated_unified],[40,52])
        self.assertEqual(gated.fp_gate_calls,4)
        self.assertEqual(gate_net.calls,4)
        np.testing.assert_array_equal(gated.base.active,baseline.base.active)
        np.testing.assert_array_equal(gated.base.armed,baseline.base.armed)
        np.testing.assert_array_equal(gated.base.last_on,baseline.base.last_on)
    def test_no_future_or_partial_frame(self):
        gate=FPExpertGateModel(GateNet(),enabled=True,
            cut=.5,cut69=.5,cut74=.5)
        s=Stream(Main(),Rescue(),risk_model=Error(),
                 recovery_model=FN(),fp_models=all_fp(),gate_model=gate)
        self.assertEqual(s.push(np.zeros(HOP-1,np.float32),start_sample=0),())
        self.assertEqual(gate.calls,0)
        end=s.push(np.zeros(1,np.float32),start_sample=HOP-1)
        self.assertEqual([d.sample for d in end],[256])
        self.assertEqual(gate.calls,1)
    def test_true_genuine_CAL_only_small_neural_train(self):
        rng=np.random.default_rng(143)
        X=rng.standard_normal((95,11)).astype(np.float32)
        y=(X[:,0]+X[:,1]*.2>0).astype(np.int8)
        m,stats=train_small_network(X,y,143)
        self.assertEqual(stats["status"],"genuine_neural_trained_CAL_only")
        self.assertEqual(m.n_features_in_,11)
        self.assertEqual(m.predict_proba(X[:49]).shape,(49,2))
        self.assertTrue(np.isfinite(m.predict_proba(X[:49])).all())
        self.assertGreaterEqual(MIN_DISAGREEMENTS,20)
    def test_one_record_compiles_event_11d_without_truth_label(self):
        r=np.zeros((2,74),np.float32)
        r[:,73]=.83
        parent=np.array([[.125,.25,.5,1.],[.125,.25,.5,1.]],np.float32)
        r[:,69:73]=parent
        features=training_events(
            r,np.array([.8,.2]),np.array([.8,.9]),
            np.array([.1,.9]),np.array([40,88]))
        self.assertEqual(features.shape,(2,11))
        self.assertEqual(float(features[0,-1]),0.)
        self.assertEqual(float(features[1,-1]),1.)
        self.assertAlmostEqual(float(features[0,6]),.125)
if __name__=="__main__":unittest.main()
