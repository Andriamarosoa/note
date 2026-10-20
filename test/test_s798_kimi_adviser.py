"""S798: provider interface safely mocked; no request to NVIDIA in tests."""
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import urllib.error

from scripts import s798_kimi_adviser as s

class KimiAdviserTests(unittest.TestCase):
    def test_research_context_is_allowlisted_summaries_only(self):
        self.assertEqual(len(s.ALLOWLIST), 5)
        self.assertTrue(all(p.endswith(".md") and p.startswith("series/S79")
                            for p in s.ALLOWLIST))
        self.assertNotIn("test.py", s.ALLOWLIST)
        with tempfile.TemporaryDirectory() as td:
            for name in s.ALLOWLIST:
                p = Path(td)/name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text("synthetic scientific observation")
            self.assertEqual(s.safe_context(Path(td)).count("synthetic scientific"), 5)
            secret = Path(td)/s.ALLOWLIST[0]
            secret.write_text("nvapi-012345678901234567890123")
            with self.assertRaises(ValueError):
                s.safe_context(Path(td))

    def test_payload_non_stream_kimi_expert_review(self):
        p=s.make_payload("synthetic")
        self.assertEqual(p["model"], "moonshotai/kimi-k3")
        self.assertFalse(p["stream"])
        self.assertEqual(p["reasoning_effort"], "max")
        self.assertIn("synthetic", p["messages"][1]["content"])
        self.assertIn("CAL32", p["messages"][0]["content"])

    def test_provider_call_redacts_accidental_credentials(self):
        api_key = "nvapi-EXAMPLE_INVALID_FOR_TESTS_0000"
        answer = {"choices":[{"message":{"content":
            "Ablation causal; ignore nvapi-EXPOSED_0123456789012345"}}],
            "usage":{"total_tokens":120},
            "reasoning_content":"never archive"}
        mock_response=io.BytesIO(json.dumps(answer).encode())
        class ContextResponse:
            def __enter__(self):return mock_response
            def __exit__(self,*args):return None
        class Opener:
            def open(self, req, timeout=0):
                self_req=req
                self_calls.append((req,timeout))
                return ContextResponse()
        self_calls=[]
        with patch.object(s.urllib.request, "build_opener", return_value=Opener()):
            x=s.request_review(api_key, "fixture")
        self.assertEqual(len(self_calls),1)
        self.assertEqual(self_calls[0][0].full_url,s.ENDPOINT)
        self.assertNotIn(api_key,json.dumps(x))
        self.assertNotIn("EXPOSED",x["review_text"])
        self.assertNotIn("reasoning_content",x)
        self.assertFalse(x["promotion_authorized"])
        self.assertTrue(x["recommendations_unvalidated"])

    def test_network_or_key_errors_redact_provider_body(self):
        with self.assertRaises(ValueError):
            s.request_review("", "fixture")
        class HttpErrorOpener:
            def open(self, req, timeout=0):
                raise urllib.error.HTTPError(s.ENDPOINT,401,
                    "INVALID SECRET nvapi-EXAMPLE_DONT_PRINT",None,
                    io.BytesIO(b"SECRET"))
        with patch.object(s.urllib.request, "build_opener",
                          return_value=HttpErrorOpener()):
            with self.assertRaises(RuntimeError) as caught:
                s.request_review("nvapi-EXAMPLE_INVALID_FOR_TESTS_0000","fixture")
        self.assertNotIn("SECRET",str(caught.exception))
        self.assertNotIn("nvapi-",str(caught.exception))

if __name__ == "__main__":
    unittest.main()
