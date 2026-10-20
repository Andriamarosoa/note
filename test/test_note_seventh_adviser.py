"""No-network test for mandatory seventh independent NVIDIA adviser."""
import io
import json
from pathlib import Path
import tempfile
import unittest
import urllib.error
from unittest.mock import patch
from scripts import note_seventh_adviser as a

KEY="nvapi-UNIT_TEST_DUMMY_NOT_A_REAL_SECRET_0000"
class Reply:
    def __init__(self,txt):
        self.payload=io.BytesIO(json.dumps({"choices":[{"message":
            {"content":txt}}],"usage":{"total_tokens":11}}).encode())
    def __enter__(self):return self
    def __exit__(self,*a):return False
    def read(self,n):return self.payload.read(n)

class Adviser(unittest.TestCase):
    def test_only_research_result_summaries(self):
        self.assertEqual(a.summary_paths("S802"),(
            "series/S800/resultat.md","series/S801/resultat.md",
            "series/S802/resultat.md"))
        with self.assertRaises(ValueError):a.summary_paths("S801")
        with self.assertRaises(ValueError):a.summary_paths("../../src")
        with tempfile.TemporaryDirectory() as path:
            for name in a.summary_paths("S802"):
                f=Path(path)/name;f.parent.mkdir(parents=True,exist_ok=True)
                f.write_text("signed scientific results")
            self.assertEqual(a.safe_context(path,"S802").count(
                "signed scientific results"),3)
            (Path(path)/a.summary_paths("S802")[1]).write_text(
                "nvapi-SENSITIVE_DUMMY_STRING_DO_NOT_SEND")
            with self.assertRaises(ValueError):a.safe_context(path,"S802")
    def test_success_is_one_request_and_never_promotes(self):
        calls=[]
        class Opener:
            def open(self,req,timeout):
                data=json.loads(req.data)
                calls.append(data)
                self.assert_unused=True
                return Reply("Proposer un audit harmoniques par fragment FIT")
        with patch.object(a.urllib.request,"build_opener",return_value=Opener()):
            o=a.consult("S802",KEY,"ONLY FAKE SUMMARY")
        self.assertEqual(len(calls),1)
        self.assertEqual(calls[0]["max_tokens"],1600)
        self.assertEqual(calls[0]["model"],"z-ai/glm-5.3-flash")
        self.assertIn("S802",calls[0]["messages"][1]["content"])
        self.assertFalse(o["official_promotion_allowed"])
        self.assertFalse(o["real_guitarset_scores_measured"])
        self.assertNotIn(KEY,json.dumps(o))
    def test_fallback_only_404_or_410(self):
        attempts=[]
        class Opener:
            def open(self,req,timeout):
                model=json.loads(req.data)["model"]
                attempts.append(model)
                if len(attempts)==1:
                    raise urllib.error.HTTPError(a.ENDPOINT,410,"private",None,
                                                  io.BytesIO(b"private"))
                return Reply("Etudier Offset sans le retarder.")
        with patch.object(a.urllib.request,"build_opener",return_value=Opener()):
            result=a.consult("S802",KEY,"safe")
        self.assertEqual(len(attempts),2)
        self.assertEqual(result["model"],"z-ai/glm-5.3")
        self.assertEqual(result["attempts"][0]["code"],410)
    def test_401_and_429_never_retry_on_another_model(self):
        for code in (401,429):
            attempts=[]
            class Opener:
                def open(self,req,timeout):
                    attempts.append(1)
                    raise urllib.error.HTTPError(a.ENDPOINT,code,"no",None,
                                                  io.BytesIO(b"sensitive"))
            with patch.object(a.urllib.request,"build_opener",return_value=Opener()):
                with self.assertRaises(a.HTTPStatus) as e:
                    a.consult("S802",KEY,"test")
            self.assertEqual(e.exception.code,code)
            self.assertEqual(len(attempts),1)
    def test_wrong_secret_headers_rejected(self):
        for key in ("", "Bearer "+KEY, "nvapi-short"):
            with self.assertRaises(ValueError):a.consult("S802",key,"test")
if __name__=="__main__":
    unittest.main()
