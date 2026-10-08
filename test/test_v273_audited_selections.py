"""Regression tests for audit-informed selection and no true-K inference leak."""
import unittest
import numpy as np
from scripts.evaluate_v273_audited_selections import (
    audit_one, audit_all, audited_priors, weight_subsets,
    combo_probabilities, SUBSETS, MASKS, N_HEADS
)


class AuditConditionedTests(unittest.TestCase):
    def test_corrections_regressions_are_counted_separately(self):
        # Same proposal direction K3->K2, but different TRUE K:
        # K2 is corrected; K3 is regressed; K4 stays neutral.
        y=np.array([2,3,4,2,3,2,3,5])
        base=np.array([3,3,3,2,3,3,3,4])
        proposal=np.array([2,2,2,2,3,2,2,3])
        a=audit_one(y,base,proposal)
        self.assertEqual(a["total"]["corrections"],2)
        self.assertEqual(a["total"]["regressions"],2)
        self.assertEqual(a["total"]["neutral"],2)
        self.assertEqual(a["by_true_k"]["2"]["corrections"],2)
        self.assertEqual(a["by_true_k"]["3"]["regressions"],2)
        self.assertEqual(a["by_predicted_transition"]["3->2"]["net"],0)

    def test_rare_cells_are_shrunk_and_audit_changes_weight(self):
        # 50 corrections / 5 regressions for proposal 3->2 gives
        # more utility than 5 corrections / 50 regressions.
        n=110
        base=np.full(n,3,int)
        y=np.r_[np.full(50,2),np.full(5,3),
                 np.full(5,2),np.full(50,3)]
        p=np.full((n,N_HEADS,5),.025,float)
        p[:,0,:]=.025
        p[:,0,1]=.9
        # Head1 votes for 2 on first 55 cases, head2 on next 55.
        p[:,1,0]=.9
        p[:,2,1]=.9
        p[55:,1,0]=.025
        p[55:,1,1]=.9
        p[55:,2,0]=.9
        p[55:,2,1]=.025
        for k in range(3,N_HEADS):p[:,k,:]=.2
        p/=p.sum(axis=2,keepdims=True)
        audits=audit_all(y,base,p)
        i1=SUBSETS.index((1,))
        i2=SUBSETS.index((2,))
        proposed=np.full((1,len(SUBSETS)),3,int)
        proposed[0,i1]=2
        proposed[0,i2]=2
        score=audited_priors(audits,np.array([3]),proposed)
        self.assertGreater(score[0,i1],score[0,i2])
        w=weight_subsets(np.ones((1,N_HEADS))/N_HEADS,score)
        self.assertGreater(w[0,i1],w[0,i2])
        self.assertAlmostEqual(w.sum(),1)

    def test_all_sixty_three_nonempty_selections(self):
        self.assertEqual(len(SUBSETS),2**N_HEADS-1)
        self.assertTrue(np.all(MASKS.sum(axis=1)>0))
        p=np.ones((2,N_HEADS,5),float)/5
        q=combo_probabilities(p)
        self.assertEqual(q.shape,(2,63,5))
        self.assertTrue(np.allclose(q.sum(axis=2),1))


if __name__=="__main__":
    unittest.main()
