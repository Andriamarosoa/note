"""S632 tests for learned cost-to-go targets and truly causal emission."""
from __future__ import annotations
import unittest
import numpy as np

from causal_note.strict_live import HOP,SR,Selection
from causal_note.s632_risk_policy import (
    FEATURE_DIM,training_costs,causal_episodes,apply_policy,
    RiskPolicy,policies)


def small_record():
    n=34
    x=np.zeros((n,49,126),np.float32)
    p=np.zeros((n,49,3),np.float32)
    k=np.broadcast_to(np.eye(7,dtype=np.float32)[1],(n,7)).copy()
    p[0:28,10,0]=.31
    p[0:28,10,1]=.55
    p[0:28,22,0]=.31
    p[0:28,22,1]=.55
    return dict(ends=np.arange(1,n+1)*HOP,x=x,p=p,k=k,
        truth=[dict(pitch=50,time=HOP/SR,duration=.12)],
        n=n*HOP)


class S632RiskTests(unittest.TestCase):
    def test_all_costs_train_only_and_candidate_negative(self):
        record=small_record()
        x,cost,proof=training_costs(record)
        self.assertEqual(x.shape[1],FEATURE_DIM)
        self.assertEqual(cost.shape,(len(x),3))
        self.assertTrue(np.isfinite(cost).all())
        self.assertTrue((cost>=0).all())
        self.assertEqual(proof["positive_episodes"],1)
        self.assertEqual(proof["negative_episodes"],1)
        self.assertGreater(proof["Q_rows"],10)
        # Positive and negative episode are represented in cost supervision.
        self.assertTrue(np.any(cost[:,1]>1))
        self.assertTrue(np.any(cost[:,1]==0))

    def test_proposals_invariant_to_changes_in_future(self):
        original=small_record()
        modified={k:(v.copy() if isinstance(v,np.ndarray) else v)
                  for k,v in original.items()}
        modified["p"][16:]=0
        modified["x"][16:]=5.
        a,_=causal_episodes(original)
        b,_=causal_episodes(modified)
        rows_a={(pitch,birth):[v for v in seq if v[0]<16]
                 for (pitch,birth),seq in a.items() if birth<16}
        rows_b={(pitch,birth):[v for v in seq if v[0]<16]
                 for (pitch,birth),seq in b.items() if birth<16}
        self.assertEqual(rows_a.keys(),rows_b.keys())
        for key in rows_a:
            for left,right in zip(rows_a[key],rows_b[key]):
                self.assertEqual(left[:4],right[:4])
                np.testing.assert_array_equal(left[4],right[4])

    def test_can_emit_additional_note_without_backdating(self):
        record=small_record()
        costs=[]
        for index in range(len(record["ends"])):
            if index==0:
                costs.append([(10,0,0,np.array([.06,1.,2.],np.float32))])
            else:costs.append([])
        events,ledger=apply_policy(record,costs,Selection(),
                  RiskPolicy(max_emit_cost=.45))
        emitted=[e for e in events if e["type"]=="note_on"]
        self.assertEqual([(e["pitch"],e["decision_sample"]) for e in emitted],[(50,HOP)])
        self.assertEqual(ledger["emits"],1)
        self.assertEqual([e["type"] for e in events],["note_on","note_off"])

    def test_noop_original_reference_and_invalid_config(self):
        record=small_record()
        events,ledger=apply_policy(record,[[] for _ in record["ends"]],Selection(),None)
        from scripts.s613_train_strict_live import decode
        self.assertEqual(events,decode(record["ends"],record["p"],record["k"],Selection(),record["n"]))
        self.assertEqual(len(policies()),10)
        with self.assertRaises(ValueError):RiskPolicy(extra_wait_cost=.6)
        with self.assertRaises(ValueError):RiskPolicy(mode="rewind")


if __name__=="__main__":
    unittest.main()
