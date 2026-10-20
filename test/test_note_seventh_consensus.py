"""Fail-closed integration test for six local + one NVIDIA independent reviewers."""
import unittest
from scripts.note_seventh_consensus import verified
def original_six():
    roles=("data","acoustics","architecture","training","metrics","live")
    return dict(schema="NOTE_SIX_ROLE_CONSENSUS_V1",passed=True,
        audits_received=6,issues=[],official_promotion_eligible=False,
        auditors={x:dict(passed=True,errors=[]) for x in roles})
def external():
    return dict(schema="NOTE_SEVENTH_SCIENTIFIC_ADVISER_V1",
        series="S802",model="z-ai/glm-5.3-flash",provider="NVIDIA NIM",
        status="review_received",seventh_adviser_executed=True,
        independent_opinion_only=True,official_promotion_allowed=False,
        real_guitarset_scores_measured=False,live_code_modified=False,
        context_sha256="a"*64,
        review_text="Conseils tests causaux, aucun gain validé en CAL32.")
class SevenGate(unittest.TestCase):
    def test_seven_pass_still_no_promotion(self):
        r=verified(original_six(),external(),"S802")
        self.assertTrue(r["passed"])
        self.assertFalse(r["promotion_eligible"])
    def test_missing_nvidia_review_fails(self):
        for changed in (dict(review_text=""),dict(series="S801"),
                        dict(seventh_adviser_executed=False),
                        dict(official_promotion_allowed=True)):
            o=external();o.update(changed)
            with self.assertRaises(AssertionError):
                verified(original_six(),o,"S802")
    def test_six_failure_rejects_even_when_nvidia_succeeds(self):
        z=original_six();z["auditors"]["live"]["passed"]=False
        with self.assertRaises(AssertionError):
            verified(z,external(),"S802")
        z=original_six();z["issues"]=["bad FIT source parity"]
        with self.assertRaises(AssertionError):
            verified(z,external(),"S802")
if __name__=="__main__":
    unittest.main()
