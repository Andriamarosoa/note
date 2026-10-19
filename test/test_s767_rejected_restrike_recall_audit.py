"""S767 offline labels never reach the causal candidate classifier."""
import unittest
import numpy as np

from test.test_s765_fit_independent_arbiter import ArbiterTests
from scripts.s767_rejected_restrike_recall_audit import (
    CAUSAL_GROUPS,causal_candidates,audit_labels,aggregate_audits)
from causal_note.strict_live import SR

class FilteredCandidateAuditTests(unittest.TestCase):
    def fixture(self):
        item,post,k,source,shadow,ctx=ArbiterTests().fixture()
        item=dict(item)
        item["_k"]=k
        acoustic=np.zeros((len(k),49,5),np.float32)
        return item,post,source,acoustic,ctx

    def test_exact_partition_and_original_source_priority(self):
        item,post,source,acoustic,ctx=self.fixture()
        original=[e.copy() for e in source]
        x,diag=causal_candidates(item,post,source,acoustic,ctx)
        self.assertEqual(source,original)
        self.assertGreaterEqual(len(x),2)
        self.assertIn("source_active",[z["group"] for z in x])
        self.assertIn("eligible",[z["group"] for z in x])
        self.assertEqual(len(x),sum(diag.get(g,0) for g in CAUSAL_GROUPS))

    def test_future_signed_baseline_does_not_change_earlier_group(self):
        item,post,source,acoustic,ctx=self.fixture()
        a,_=causal_candidates(item,post,source,acoustic,ctx)
        future=dict(type="note_on",pitch=75,
                    available_sample=int(item["samples"]),
                    decision_sample=int(item["samples"]),
                    time=item["samples"]/SR,velocity=96,
                    reason="future")
        future_end=dict(future,type="note_off",velocity=0)
        source_after=source+[future,future_end]
        b,_=causal_candidates(item,post,source_after,acoustic,ctx)
        self.assertEqual(
            [(z["on"]["pitch"],z["on"]["available_sample"],z["group"]) for z in a],
            [(z["on"]["pitch"],z["on"]["available_sample"],z["group"]) for z in b])

    def test_only_baseline_missed_truth_counts_as_recovered(self):
        item,post,source,acoustic,ctx=self.fixture()
        proposals,diag=causal_candidates(item,post,source,acoustic,ctx)
        truth=[
            dict(pitch=40,time=int(item["decision_ends"][0])/SR,
                 duration=.04),
            dict(pitch=52,time=int(item["decision_ends"][3])/SR,
                 duration=.04)]
        label=audit_labels(proposals,truth,baseline_matches=[0])
        self.assertEqual(label["missing_truth"],1)
        self.assertEqual(label["matched_missing_truth_unique"],1)
        self.assertEqual(label["group"]["eligible"]["missed_truth_onset_matched"],1)
        sums=aggregate_audits([label,label])
        self.assertEqual(sums["missing_truth"],2)
        self.assertEqual(sums["recoverable_by_all_existing_proposals"],2)

    def test_release_is_diagnostic_not_a_birth_gate(self):
        item,post,source,acoustic,ctx=self.fixture()
        baseline,_=causal_candidates(item,post,source,acoustic,ctx)
        future_post=post.copy()
        future_post[-1,:,2]=1.
        changed,_=causal_candidates(item,future_post,source,acoustic,ctx)
        before=[(z["on"]["pitch"],z["on"]["available_sample"],z["group"])
                for z in baseline]
        after=[(z["on"]["pitch"],z["on"]["available_sample"],z["group"])
               for z in changed]
        self.assertEqual(before,after)

if __name__=="__main__":
    unittest.main()
