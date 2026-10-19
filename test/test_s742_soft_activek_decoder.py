"""S742 soft-K rescue test; only tested acoustic events, never K-only notes."""
import unittest
import numpy as np
from causal_note.strict_live import HOP,SR
from causal_note.s708_cardinality_selection import GuidedPolicy,decode_guided
from causal_note.s742_soft_activek_decoder import (
    SoftTailPolicy,SoftTailGuidedSelector,decode_soft_tail)

class TestS742SoftK(unittest.TestCase):
    def setUp(self):
        self.policy=GuidedPolicy(count_confidence=1.1,guide_confidence=.7)
        self.ends=np.arange(1,5,dtype=np.int64)*(4*HOP)
        self.p=np.zeros((4,49,3),np.float32)
        self.k=np.zeros((4,7),np.float32)
        self.k[:,0]=.44
        self.k[:,1]=.40
        self.k[:,2]=.16
        self.p[0,0,0]=.30
        self.p[0,0,1]=.75
        self.p[1:,0,0]=.25
        self.p[1:,0,1]=.75
        self.e=SoftTailPolicy(
            tail_threshold=.55,onset=.22,sustain=.65,
            rise=.04,max_new_per_fragment=1)

    def test_extra_birth_when_baseline_argmax_K0_but_soft_tail_plus_acoustics(self):
        initial=decode_guided(
            self.ends,self.p,self.k,self.policy,int(self.ends[-1]))
        self.assertFalse(any(e["type"]=="note_on" for e in initial))
        trial,diag=decode_soft_tail(
            self.ends,self.p,self.k,self.policy,int(self.ends[-1]),self.e)
        ons=[e for e in trial if e["type"]=="note_on"]
        self.assertEqual([e["pitch"] for e in ons],[40])
        self.assertEqual(diag["s742_soft_births"],1)
        self.assertTrue(all(e["available_sample"]>=e["decision_sample"]
                            for e in trial))
        self.assertTrue(all(abs(e["time"]-e["available_sample"]/SR)<1e-10
                            for e in trial))

    def test_k_alone_never_emits(self):
        self.p[:]=0
        emitted,diag=decode_soft_tail(
            self.ends,self.p,self.k,self.policy,int(self.ends[-1]),self.e)
        self.assertFalse(any(e["type"]=="note_on" for e in emitted))
        self.assertEqual(diag["s742_soft_births"],0)

    def test_do_not_backdate_prior_decisions_when_future_changes(self):
        orig,_=decode_soft_tail(
            self.ends,self.p,self.k,self.policy,int(self.ends[-1]),self.e)
        altered=self.p.copy()
        altered[-1,1,0]=.3
        altered[-1,1,1]=.9
        current,_=decode_soft_tail(
            self.ends,altered,self.k,self.policy,int(self.ends[-1]),self.e)
        before=lambda ev:[(x["type"],x["pitch"],x["decision_sample"])
                          for x in ev if x["decision_sample"]<self.ends[-1]]
        self.assertEqual(before(orig),before(current))

    def test_unchanged_reference_when_no_soft_tail_threshold(self):
        extra=SoftTailPolicy(tail_threshold=1.,onset=.22,sustain=.65)
        before=decode_guided(
            self.ends,self.p,self.k,self.policy,int(self.ends[-1]))
        current,_=decode_soft_tail(
            self.ends,self.p,self.k,self.policy,int(self.ends[-1]),extra)
        self.assertEqual(before,current)

    def test_reject_wrong_K_and_excessive_new_notes(self):
        with self.assertRaises(ValueError):
            SoftTailPolicy(max_new_per_fragment=7).validate()
        with self.assertRaises(ValueError):
            SoftTailGuidedSelector(self.policy,self.e).step(
                self.p[0],np.zeros(7),self.ends[0])


if __name__=="__main__":
    unittest.main()
