"""S645 tests for past-only rising-edge calibration and TRAIN-only lock."""
import unittest
import numpy as np
from causal_note.strict_live import HOP,Selection
from scripts.s645_causal_score_rise_loop import evaluate,locked_policy,PREDECLARED,CORPORA


class RisingGateTests(unittest.TestCase):
    def row(self):
        n=28
        p=np.zeros((n,49,3),np.float32)
        p[:12,7,0]=.07;p[:12,7,1]=.15
        k=np.broadcast_to(np.eye(7,dtype=np.float32)[1],(n,7)).copy()
        return dict(ends=np.arange(1,n+1)*HOP,p=p,k=k,
                    x=np.zeros((n,49,126),np.float32),n=n*HOP,
                    truth=[dict(pitch=47,time=HOP/44100.,duration=.06)])
    def test_edge_adds_only_when_acoustic_and_q_rise_agree(self):
        row=self.row()
        q=np.zeros((28,49),np.float32)
        a=evaluate(row,q,Selection(onset=.6),PREDECLARED[1][1])
        self.assertEqual(a["actual_rescued_onsets"],0)
        q[:12,7]=.99
        b=evaluate(row,q,Selection(onset=.6),PREDECLARED[1][1])
        self.assertGreater(b["actual_rescued_onsets"],0)
        self.assertEqual(b["score"]["TP"],1)
        self.assertGreater(b["signal_rises"],0)
    def test_no_lookahead_after_first_true_candidate(self):
        row=self.row()
        q=np.zeros((28,49),np.float32);q[:12,7]=.99
        a=evaluate(row,q,Selection(onset=.6),PREDECLARED[1][1])
        q[14:]=np.ones_like(q[14:])
        b=evaluate(row,q,Selection(onset=.6),PREDECLARED[1][1])
        self.assertEqual(a["score"]["TP"],b["score"]["TP"])
    def test_train_lock_requires_both_improve(self):
        t={}
        for name,_ in PREDECLARED:
            t[name]={c:dict(F1=.6 if name=="S614_original" else .58,
                            regressed_sources=0) for c in CORPORA}
        self.assertEqual(locked_policy(t),"S614_original")
        for c in CORPORA:t["rise099_010"][c]["F1"]=.61
        self.assertEqual(locked_policy(t),"rise099_010")

if __name__=="__main__":unittest.main()
