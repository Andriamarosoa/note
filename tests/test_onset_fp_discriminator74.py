"""Executable invariant tests: independent genuine learned 69/70/74D Onset heads."""
import unittest
import numpy as np
from onset256.acoustics import HOP,SR
from onset256.fp_discriminator import (
    causal_parent_matrix,feature_variant,discriminator_matrix,
    FPDiscriminatorModel,select_known_onsets,VARIANTS)
from onset256.runtime import Stream
from scripts.train_onset_fp_discriminator256 import (
    train_network,approximate_score,COMPS,CUTS,BASE_4,BASE_3)

class Main:
    metadata={"selection":{}}
    def __init__(self):self.calls=0
    def predict(self,f):
        self.calls+=1
        a=np.zeros((49,3),np.float32)
        a[0]=[.91,.91,.0]
        a[12]=[.15,.25,.0]
        return a,np.ones(7,np.float32)/7
class Rescue:
    def __init__(self):self.calls=0
    def predict_proba(self,features):
        self.calls+=1
        return np.tile([.4,.6],(49,1))
class Risk:
    cut=.5
    requires_selector_state=True
    trained_hop_samples=256
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
    def predict_recovery_probability(self,f,p,k,r,h,s):
        self.calls+=1
        a=np.full(49,.01,np.float32);a[12]=.99
        return a
class Estimator:
    def __init__(self,dim):self.n_features_in_=dim;self.incoming=[]
    def predict_proba(self,x):
        self.incoming.append(x.copy())
        q=np.full(len(x),.99,np.float32)
        q[0]=.01 # just the native E2 event is challenged
        return np.column_stack((1.-q,q))
class FPCausalRegressionTests(unittest.TestCase):
    def test_69_70_74_contract_and_parent_timelines(self):
        m=np.full(49,-10**12,np.int64)
        np.testing.assert_array_equal(causal_parent_matrix(m,256),np.ones((49,4)))
        m[0]=256
        a=causal_parent_matrix(m,512)
        self.assertAlmostEqual(float(a[12,0]),1/32)
        self.assertEqual(float(a[12,1]),1.)
        self.assertEqual(float(a[0,0]),1.)
        self.assertEqual(float(causal_parent_matrix(m,256)[12,0]),1.)
        m[0]=768
        self.assertEqual(float(causal_parent_matrix(m,512)[12,0]),1.)
        with self.assertRaises(ValueError):
            causal_parent_matrix(m,257)
        x=np.arange(74*49,dtype=np.float32).reshape(49,74)
        self.assertEqual(feature_variant(x,"control69").shape,(49,69))
        self.assertEqual(feature_variant(x,"control70").shape,(49,70))
        self.assertEqual(feature_variant(x,"temporal74").shape,(49,74))
        np.testing.assert_array_equal(feature_variant(x,"control70")[:,-1],x[:,73])
        self.assertEqual(COMPS,("BN1","Funk2","Jazz1","Rock3","SS1"))
        self.assertEqual(BASE_4["fp"],4605)
        self.assertEqual(BASE_3["fp"],4411)
    def test_only_frozen_four_head_onset_emissions_can_be_contested(self):
        models={name:FPDiscriminatorModel(Estimator(dim),name,.5)
                for name,dim in VARIANTS.items()}
        main,rsc,risk,fn=Main(),Rescue(),Risk(),FN()
        current=Stream(main,rsc,risk_model=risk,recovery_model=fn,
                       fp_models=models)
        baseline=Stream(Main(),Rescue(),risk_model=Risk(),recovery_model=FN())
        x=np.zeros(3*HOP,np.float32)
        old=baseline.push(x,start_sample=0)
        new=current.push(x,start_sample=0)
        self.assertEqual([d.intents for d in old],[d.intents for d in new])
        self.assertEqual([d.predecision_intents for d in old],
                         [d.predecision_intents for d in new])
        self.assertEqual([d.fn_recovery_intents for d in old],
                         [d.fn_recovery_intents for d in new])
        self.assertEqual([d.fn_unified_intents for d in old],
                         [d.fn_unified_intents for d in new])
        self.assertEqual(len(new),3)
        self.assertEqual([e.pitch for e in new[0].fn_unified_intents],[40,52])
        for name in VARIANTS:
            self.assertEqual(new[0].fp_discriminator_intents[name],())
            self.assertEqual([e.pitch for e in
                             new[0].fp_discriminator_unified[name]],[52])
            self.assertEqual(models[name].calls,3)
        self.assertEqual(current.fp_calls,3)
        np.testing.assert_array_equal(current.base.active,baseline.base.active)
        np.testing.assert_array_equal(current.base.last_on,baseline.base.last_on)
        self.assertEqual(current.fp_last_native_on[0],256)
        self.assertAlmostEqual(models["temporal74"].model.incoming[1][12,69],1/32)
    def test_no_prediction_when_only_partial_256_received(self):
        models={n:FPDiscriminatorModel(Estimator(d),n,.5)
                for n,d in VARIANTS.items()}
        s=Stream(Main(),Rescue(),risk_model=Risk(),recovery_model=FN(),
                 fp_models=models)
        self.assertEqual(s.push(np.zeros(255,np.float32),start_sample=0),())
        self.assertEqual(s.fp_calls,0)
        end=s.push(np.zeros(1,np.float32),start_sample=255)
        self.assertEqual([x.sample for x in end],[256])
        self.assertEqual(s.fp_calls,1)
        self.assertAlmostEqual(HOP/SR*1000,5.80498866,places=7)
    def test_training_on_train_only_exact_same_seed(self):
        rng=np.random.default_rng(142)
        rows=[]
        for i in range(180):
            f=rng.normal(size=74).astype(np.float32)
            y=int(f[0]*.75+f[1]*.5>0)
            rows.append(dict(X=f,y=y))
        train=list(range(140))
        a=train_network(rows,train,"control69",77)
        b=train_network(rows[:140],train,"control69",77)
        self.assertEqual(a.n_features_in_,69)
        for x,y in zip(a.named_steps["mlpclassifier"].coefs_,
                       b.named_steps["mlpclassifier"].coefs_):
            np.testing.assert_array_equal(x,y)
        c=train_network(rows,train,"control70",77)
        d=train_network(rows,train,"temporal74",77)
        self.assertEqual(c.n_features_in_,70)
        self.assertEqual(d.n_features_in_,74)
    def test_event_proxy_is_only_for_calibration_not_live_report(self):
        rows=[dict(comp="BN1",track="a",y=1),
              dict(comp="BN1",track="a",y=0)]
        refs={"a":dict(comp="BN1",
                       base=dict(tp=1,fp=1,fn=0))}
        x=np.array([.9,.1],np.float32)
        result=approximate_score(rows,refs,"BN1",x,.5)
        self.assertEqual(result["removed_fp"],1)
        self.assertEqual(result["removed_tp"],0)
        self.assertEqual((result["tp"],result["fp"],result["fn"]),(1,0,0))
        self.assertIn(0.,CUTS)
if __name__=="__main__":unittest.main()
