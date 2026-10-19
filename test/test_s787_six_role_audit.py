"""S787 negative and positive tests: no single role can approve promotion."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from scripts.s787_audit_consensus import consensus,ROLES
from scripts.s787_six_role_audit import confusion_audit,robust_score

class S787AuditGateTests(unittest.TestCase):
    def role(self,k,passed=True,errors=None):
        return dict(schema="NOTE_SIX_ROLE_AUDIT_V1",role=k,passed=passed,
                    experimental_promotion_allowed=False,errors=errors or [],
                    findings={"evidence":"checked"})

    def test_missing_agent_is_blocker(self):
        with TemporaryDirectory() as d:
            path=Path(d)
            for role in ROLES[:-1]:
                (path/f"role_{role}.json").write_text(json.dumps(self.role(role)))
            result=consensus(path)
            self.assertFalse(result["passed"])
            self.assertEqual(result["audits_received"],5)
            self.assertFalse(result["official_promotion_eligible"])

    def test_one_failing_agent_blocks_all(self):
        with TemporaryDirectory() as d:
            path=Path(d)
            for role in ROLES:
                value=self.role(role,passed=role!="metrics",
                                errors=["loss > guard"] if role=="metrics" else [])
                (path/f"role_{role}.json").write_text(json.dumps(value))
            self.assertFalse(consensus(path)["passed"])

    def test_six_proper_agents_produce_consensus_not_promotion(self):
        with TemporaryDirectory() as d:
            path=Path(d)
            for role in ROLES:
                (path/f"role_{role}.json").write_text(json.dumps(self.role(role)))
            result=consensus(path)
            self.assertTrue(result["passed"])
            self.assertEqual(result["audits_received"],6)
            self.assertFalse(result["official_promotion_eligible"])

    def test_confusion_is_independently_recomputed(self):
        n=[[0]*7 for _ in range(7)]
        for k in range(7):n[k][k]=10
        n[6][5]=5
        rows=[]
        for k in range(7):
            rows.append(dict(k=k,n=sum(n[k]),correct=n[k][k],
                             accuracy=n[k][k]/sum(n[k]),under=sum(n[k][:k]),
                             over=sum(n[k][k+1:]),major_under=sum(n[k][:max(0,k-1)])))
        m=dict(confusion=n,per_k=rows,frames=75,exact_accuracy=70/75)
        result=confusion_audit(m,"fixture")
        self.assertEqual(result["correct"],70)
        self.assertEqual(result["frames"],75)
        self.assertAlmostEqual(result["k6"],10/15)
        m["exact_accuracy"]=1.
        with self.assertRaises(AssertionError):confusion_audit(m,"tampered")

    def test_fit_selection_worst_composition_scoring(self):
        def dummy(k5,k6,acc):
            p=[dict(k=i,n=10,accuracy=.1) for i in range(7)]
            p[5]["accuracy"]=k5;p[6]["accuracy"]=k6
            return dict(per_k=p,exact_accuracy=acc)
        a={"foo":dummy(.9,.3,.95),"bar":dummy(.7,.9,.94)}
        b={"foo":dummy(.7,.8,.93),"bar":dummy(.8,.7,.92)}
        self.assertGreater(robust_score(b,dummy(.7,.7,.9)),
                           robust_score(a,dummy(.8,.6,.95)))

if __name__=="__main__":
    unittest.main()
