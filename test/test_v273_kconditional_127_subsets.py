"""Focused forensic contracts for per-K subset enumeration and direct-vote limitations."""
import unittest
import numpy as np

from scripts.audit_v273_kconditional_127_subsets import (
    structural_heads, all_nonempty, votes, probe, count, SCENARIOS
)


class KConditionalAuditTests(unittest.TestCase):
    def test_exact_127_vs_63_vs_one(self):
        self.assertEqual(len(all_nonempty(structural_heads(2,3))),127)
        self.assertEqual(len(all_nonempty(structural_heads(3,2))),127)
        self.assertEqual(len(all_nonempty(structural_heads(3,4))),127)
        self.assertEqual(len(all_nonempty(structural_heads(4,3))),127)
        self.assertEqual(len(all_nonempty(structural_heads(2,4))),63)
        self.assertEqual(len(all_nonempty(structural_heads(2,1))),1)
        self.assertEqual(len(all_nonempty(structural_heads(2,0))),1)
        self.assertEqual(sum(len(all_nonempty(structural_heads(b,k)))
                             for b,k in SCENARIOS),1018)

    def test_nonpoly_cannot_be_proposed_by_head(self):
        P=np.full((11,6,5),.2,dtype="float32")
        for source in (2,3,4):
            for target in (0,1):
                heads=structural_heads(source,target)
                self.assertEqual(heads,(0,))
                keep,new=votes(P,source,target,heads)
                self.assertTrue(np.all(new==0))
                out,_=probe(P,source,target,heads,all_nonempty(heads))
                self.assertFalse(out.any())

    def test_transition_specific_fix(self):
        self.assertIn(6,structural_heads(2,3))
        self.assertNotIn(6,structural_heads(3,2))
        self.assertIn(7,structural_heads(3,2))
        self.assertIn(8,structural_heads(3,4))
        self.assertIn(9,structural_heads(4,3))
        P=np.full((2,6,5),.02,dtype="float32")
        P[:,:,1]=.92  # K3 strong from all experts
        P/=P.sum(axis=2,keepdims=True)
        out,_=probe(P,2,3,(6,),((6,),))
        self.assertTrue(out.all())
        y=np.array([2,3])
        d=count(out[:,0],y,2,3)
        self.assertEqual(d["corrections"],1)
        self.assertEqual(d["regressions"],1)
        self.assertEqual(d["net"],0)
        self.assertEqual(d["true_k2_regressions"],1)
        self.assertEqual(d["true_k3_corrections"],1)

    def test_masked_combinations_may_be_empty_actions(self):
        P=np.full((5,6,5),.2,dtype="float32")
        heads=structural_heads(3,2)
        ids=all_nonempty(heads)
        pred,margin=probe(P,3,2,heads,ids)
        self.assertEqual(pred.shape,(5,127))
        self.assertTrue(np.isfinite(margin).all())
        self.assertFalse(pred.any())

if __name__=="__main__":
    unittest.main()
