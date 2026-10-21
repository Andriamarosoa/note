"""S819 independent external adviser contract tests -- MOCKED, no NVIDIA use."""
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch
from scripts.s819_blind_nvidia import (SCHEMA,MODELS,SYSTEM,hash_text,
                                      payload,evidence_text,consult)
class TestS819BlindNvidia(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.brief=evidence_text(Path("series/S819/model_brief.md"))
    def test_three_distinct_models_same_inputs(self):
        self.assertEqual(len(MODELS),len(set(MODELS.values()))==3)
        for role in MODELS:
            x=payload(role,self.brief)
            self.assertEqual(x["messages"][0]["content"],SYSTEM)
            self.assertEqual(x["messages"][1]["content"],self.brief)
            self.assertEqual(x["model"],MODELS[role])
            self.assertEqual(x["temperature"],.35)
        self.assertNotIn("assistant_hypothesis",self.brief)
        self.assertNotIn("prior_nvidia_response",self.brief)
        self.assertEqual(hash_text(self.brief),hash_text(self.brief))
    def test_missing_secret_never_counts_as_consultation(self):
        z=consult("model_a",self.brief,"")
        self.assertEqual(z["schema"],SCHEMA)
        self.assertEqual(z["status"],"unavailable")
        self.assertFalse(z["advice_substantive"])
        self.assertFalse(z["response_received"])
        self.assertFalse(z["promotion_allowed"])
        self.assertEqual(z["provider_errors"],["NVIDIA_API_KEY_missing_or_invalid"])
    def test_third_model_is_real_glm_flash_not_kimi_retired(self):
        self.assertEqual(MODELS["model_c"],"z-ai/glm-5.3-flash")
    def test_reject_injected_peer_opinion_or_secret(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"evidence.md"
            p.write_text(self.brief+"\nprior_nvidia_response")
            with self.assertRaisesRegex(ValueError,"secret/peer"):
                evidence_text(p)
            p.write_text(self.brief+"\nnvapi-xxxx")
            with self.assertRaisesRegex(ValueError,"secret/peer"):
                evidence_text(p)
    def test_no_external_call_in_all_tests(self):
        with patch("urllib.request.build_opener") as opener:
            consult("model_a",self.brief,"")
            opener.assert_not_called()
    def test_every_adviser_evaluated_only_on_s819(self):
        self.assertIn("S819",self.brief)
        self.assertIn("S818",self.brief)
        self.assertIn("S815",self.brief)
        self.assertIn("test",self.brief)
if __name__=="__main__":
    unittest.main()
