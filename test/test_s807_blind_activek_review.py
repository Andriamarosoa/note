"""S807 malicious/invalid model-answer negative tests. No provider calls."""
import json,unittest
from scripts.s807_blind_activek_evidence import digest
from scripts.s807_blind_nvidia_review import MODELS,PROMPT
from scripts.s807_blind_critic import check
class BlindReviews(unittest.TestCase):
    def evidence(self):
        e=dict(schema="S807_BLIND_ACTIVEK_FIT80_EVIDENCE_V1",
               complete=True,numeric={"split":"FIT80_ONLY",
                 "under_active_frames":80654,"over_active_frames":26544},
               source_code_excerpts=[
                 dict(path="scripts/s794_activek_failure_taxonomy.py",sha256="a",
                      total_lines=100,excerpt="65: x = 1")],
               CAL_contents_sent=False,HELD_contents_sent=False,
               TEST_contents_sent=False,limitations="bounded windows")
        e["evidence_sha256"]=digest(json.dumps(e,sort_keys=True,ensure_ascii=False))
        return e
    def test_three_distinct_model_names(self):
        self.assertEqual(len(set(MODELS.values())),3)
    def test_model_cannot_claim_three_consensus_by_identity_only(self):
        e=self.evidence()
        result=check(e,{})
        self.assertEqual(result["independent_reviews_accepted"],0)
        self.assertFalse(result["all_three_substantive_review_received"])
        self.assertFalse(result["S708_promotion_allowed"])
    def test_evidence_sha_tampering_veto(self):
        e=self.evidence()
        e["numeric"]["under_active_frames"]=1
        with self.assertRaises(AssertionError):check(e,{})
    def test_invalid_model_response_is_not_accepted(self):
        e=self.evidence()
        r={k:dict(reviewer=k,model=v,status="received",quality_pass=True,
                  evidence_sha256="fake",shared_context_sha256="wrong")
           for k,v in MODELS.items()}
        out=check(e,r)
        self.assertEqual(out["independent_reviews_accepted"],0)
        self.assertFalse(out["code_and_metric_fact_checked_afterward"])
    def test_one_shared_prompt_and_no_suggestions(self):
        self.assertNotIn("sustain_decoder",PROMPT.lower())
        self.assertNotIn("independent_head",PROMPT.lower())
        self.assertIn("evidence",PROMPT.lower())
if __name__=="__main__":unittest.main()
