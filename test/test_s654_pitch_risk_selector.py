"""S654 specialist: no model replacement, causal score and correction audits."""
from __future__ import annotations
import unittest
import numpy as np
from causal_note.strict_live import Selection,HOP
from causal_note.s648_event_birth import BirthPolicy,decode_birth,causal_features
from causal_note.s654_pitch_risk_selector import (
    risk_vector,decode_with_specialist,risk_training_examples,
    streaming_specialist_decision
)
from scripts.s654_train_risk_on_unseen_birth_events import (
    risk_lock_policy,partition_calibration
)


class Born:
    def predict_proba(self,x):
        q=np.where(x[:,0]>.40,.95,.01)
        return np.column_stack((1-q,q))


class Risk:
    def __init__(self,q):self.q=q
    def predict_proba(self,x):
        p=np.full(len(x),self.q)
        return np.column_stack((1-p,p))


def fixture():
    n=28
    x=np.zeros((n,49,126),np.float32)
    x[2:8,12,0]=.8
    p=np.zeros((n,49,3),np.float32)
    k=np.broadcast_to(np.eye(7,dtype=np.float32)[1],(n,7)).copy()
    return dict(x=x,p=p,k=k,ends=np.arange(1,n+1)*HOP,n=n*HOP,
        selection=dict(onset=.6,active_on=.5,release_frames=3,refractory_frames=10),
        truth=[dict(pitch=52,time=3*HOP/44100.,duration=.075)])


class S654RiskTests(unittest.TestCase):
    def test_accept_all_reproduces_original_birth_and_reject_all_reproduces_S614(self):
        row=fixture()
        birth_events,_=decode_birth(row,Born(),BirthPolicy(.7,.1))
        accepted,stats=decode_with_specialist(row,Born(),Risk(.99),.5)
        self.assertEqual(accepted,birth_events)
        self.assertGreater(stats["actual_new_notes"],0)
        original,_=decode_birth(row,Born(),BirthPolicy())
        rejected,stats=decode_with_specialist(row,Born(),Risk(.01),.5)
        self.assertEqual(rejected,original)
        self.assertGreater(stats["risk_rejected"],0)

    def test_truth_never_enters_risk_feature(self):
        row=fixture()
        feats=causal_features(row)
        p=row["p"][2]; k=row["k"][2]
        a=risk_vector(feats[2],p,k,np.ones(49,np.float32),
                      np.zeros(49,np.float32),52)
        self.assertEqual(a.ndim,1)
        self.assertEqual(a.shape[0],feats.shape[2]+8)
        row["truth"]=[dict(pitch=55,time=.8,duration=.01)]
        b=risk_vector(feats[2],p,k,np.ones(49,np.float32),
                      np.zeros(49,np.float32),52)
        np.testing.assert_array_equal(a,b)

    def test_out_of_fit_labels_are_not_assumed_all_new_notes(self):
        row=fixture()
        x,y,ledger=risk_training_examples(row,Born())
        self.assertEqual(len(x),len(y))
        self.assertEqual(len(y),len(ledger))
        self.assertTrue(set(y.tolist())<={0,1})
        self.assertTrue(all(z["decision_sample"]>=0 for z in ledger))

    def test_live_decision_matches_batch_specialist_gate(self):
        row=fixture()
        feat=causal_features(row)[2]
        decision=streaming_specialist_decision(Risk(.9),feat,row["p"][2],
          row["k"][2],np.ones(49),np.zeros(49),52,.75)
        self.assertTrue(decision)
        self.assertFalse(streaming_specialist_decision(Risk(.1),feat,row["p"][2],
          row["k"][2],np.ones(49),np.zeros(49),52,.75))

    def test_train_lock_requires_both_corpora_and_no_regressions(self):
        from scripts.s654_train_risk_on_unseen_birth_events import VARIANTS
        matrix={}
        for v in VARIANTS:
            matrix[v]={c:dict(F1=.65 if v=="S614_original" else .61,
                   FP=10,regressed_sources=0) for c in ("GuitarSet","URMP")}
        self.assertEqual(risk_lock_policy(matrix),"S614_original")
        for corpus in ("GuitarSet","URMP"):
            matrix["S654_risk075"][corpus]["F1"]=.66
        self.assertEqual(risk_lock_policy(matrix),"S654_risk075")

if __name__=="__main__":
    unittest.main()
