"""S651: correction and regression must be audited within each variant loop."""
from __future__ import annotations
import copy
import unittest
from causal_note.s651_loop_guard import GateLimits,audit_policy_records,frozen_selection
from scripts.s651_replay_audit_all_loops import audit_historical


def row(tp,fp,fn):
    return dict(TP=tp,FP=fp,FN=fn,F1=2*tp/max(1,2*tp+fp+fn))


def evidence():
    return [
      dict(name="04_Test-G_comp",corpus="GuitarSet",policies={
        "S614_original":row(10,4,10),
        "candidate_good":row(12,4,8),
        "candidate_bad":row(12,30,8)}),
      dict(name="04_Test-G_solo",corpus="GuitarSet",policies={
        "S614_original":row(10,4,10),
        "candidate_good":row(11,4,9),
        "candidate_bad":row(9,4,11)}),
      dict(name="01_Test-URMP",corpus="URMP",policies={
        "S614_original":row(10,4,10),
        "candidate_good":row(11,4,9),
        "candidate_bad":row(11,15,9)}),
    ]


class LoopGuardTests(unittest.TestCase):
    def test_store_every_source_and_loss_and_role(self):
        result=audit_policy_records(evidence(),"S614_original","candidate_bad")
        self.assertFalse(result["eligible_for_next_stage"])
        self.assertIn("GuitarSet:FP_budget_exceeded",result["rejected_reasons"])
        self.assertEqual(len(result["per_source"]),3)
        self.assertEqual(result["by_corpus_and_role"]["GuitarSet"]["solo"]["delta"]["TP"],-1)
        self.assertIn("04_Test-G_solo",result["by_corpus_and_role"]["GuitarSet"]["ALL"]["losses"])
        self.assertEqual(result["by_corpus_and_role"]["GuitarSet"]["ALL"]["net_fp_added"],26)

    def test_correction_but_never_erase_adverse_evidence(self):
        result=audit_policy_records(evidence(),"S614_original","candidate_good")
        self.assertTrue(result["eligible_for_next_stage"])
        self.assertEqual(result["by_corpus_and_role"]["GuitarSet"]["ALL"]["net_tp_gained"],3)
        self.assertEqual(result["by_corpus_and_role"]["URMP"]["ALL"]["net_fp_added"],0)
        self.assertEqual(frozen_selection([result]),"candidate_good")
        self.assertEqual(result["per_source"][0]["state"],"correction")

    def test_no_promotion_if_truth_rewritten(self):
        changed=evidence()
        changed[0]["policies"]["candidate_good"]=row(12,4,9)
        with self.assertRaises(AssertionError):
            audit_policy_records(changed,"S614_original","candidate_good")

    def test_no_promotion_if_f1_is_fabricated(self):
        changed=evidence()
        changed[0]["policies"]["candidate_good"]["F1"]=.99
        with self.assertRaises(AssertionError):
            audit_policy_records(changed,"S614_original","candidate_good")

    def test_frozen_stage_cannot_be_changed_by_dev(self):
        calibration=audit_policy_records(evidence(),"S614_original","candidate_good")
        lock=frozen_selection([calibration])
        future=copy.deepcopy(evidence())
        for r in future:
            r["policies"]["candidate_good"]=row(0,500,20)
        dev=audit_policy_records(future,"S614_original","candidate_good")
        self.assertFalse(dev["eligible_for_next_stage"])
        self.assertEqual(lock,"candidate_good")

    def test_no_silent_duplicate_source(self):
        with self.assertRaises(AssertionError):
            audit_policy_records(evidence()*2,"S614_original","candidate_good")

    def test_no_regression_of_zero_noop(self):
        a=evidence()
        for r in a:
            r["policies"]["copy"]=dict(r["policies"]["S614_original"])
        test=audit_policy_records(a,"S614_original","copy")
        self.assertFalse(test["eligible_for_next_stage"])
        self.assertEqual(len(test["rejected_reasons"]),2)
        self.assertEqual(len(test["per_source"]),3)


if __name__=="__main__":unittest.main()
