"""S644 fixed gate anatomy and TRAIN-only decision proofs."""
from __future__ import annotations
import unittest
import numpy as np
from causal_note.strict_live import HOP,Selection
from scripts.s644_gate_rescue_loop import measure,lock_on_train,POLICIES,CORPORA


class GateTests(unittest.TestCase):
    def row(self):
        n=28
        p=np.zeros((n,49,3),np.float32)
        p[:12,7,0]=.25;p[:12,7,1]=.47
        k=np.broadcast_to(np.eye(7,dtype=np.float32)[1],(n,7)).copy()
        return dict(ends=np.arange(1,n+1)*HOP,p=p,k=k,
                    x=np.zeros((n,49,126),np.float32),n=n*HOP,
                    truth=[dict(pitch=47,time=HOP/44100.,duration=.06)])
    def test_each_gate_explicit_no_output_for_extreme_q(self):
        row=self.row()
        q=np.zeros((28,49),np.float32)
        base,counts,profile,weak=measure(row,q,Selection(onset=.6),None)
        self.assertEqual(base["TP"],0)
        self.assertGreater(counts["above_sustain_026"],0)
        self.assertEqual(counts["q965_pass"],0)
        self.assertEqual(counts["attempted_frames"],0)
        q[:12,7]=.8
        new,a,profile,weak=measure(row,q,Selection(onset=.6),.7)
        self.assertGreater(a["attempted_frames"],0)
        self.assertGreater(a["actual_candidate_noteons"],0)
        self.assertEqual(new["TP"],1)
        self.assertEqual(a["q965_pass"],0)
    def test_train_lock_never_looks_at_dev(self):
        case={}
        for name,threshold in POLICIES:
            case[name]={}
            for corpus in CORPORA:
                f=.60 if name=="S614" else .50
                case[name][corpus]=dict(F1=f,source_regressions=0)
        self.assertEqual(lock_on_train(case),"S614")
        for corpus in CORPORA:case["q85"][corpus]["F1"]=.63
        self.assertEqual(lock_on_train(case),"q85")
if __name__=="__main__":
    unittest.main()
