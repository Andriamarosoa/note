"""Offline tests of NVIDIA seventh adviser. NEVER call real API here."""
import io
import json
import unittest
import urllib.error
from unittest.mock import patch
from scripts import s799_nvidia_adviser as s
KEY="nvapi-FAKE_TEST_KEY_0123456789"
class Response:
    def __init__(self,text):self.text=io.BytesIO(json.dumps({"choices":[{"message":{"content":text}}],"usage":{"total_tokens":13}}).encode())
    def __enter__(self):return self
    def __exit__(self,*args):return False
    def read(self,n):return self.text.read(n)
class ProviderTest(unittest.TestCase):
    def test_model_parameters_no_k3_settings(self):
        a=s.make_payload(s.CANDIDATES[0],"synthetic")
        b=s.make_payload(s.CANDIDATES[1],"synthetic")
        self.assertEqual(a["thinking"],{"type":"disabled"})
        self.assertNotIn("reasoning_effort",a)
        self.assertNotIn("thinking",b)
        self.assertEqual(a["model"],"moonshotai/kimi-k2.5")
    def test_success_primary(self):
        calls=[]
        class Opener:
            def open(self,req,timeout):
                calls.append(json.loads(req.data)["model"])
                return Response("harmoniques: auditer differents fragments")
        with patch.object(s.urllib.request,"build_opener",return_value=Opener()):
            data=s.consult(KEY,"fixture")
        self.assertEqual(calls,[s.CANDIDATES[0]])
        self.assertFalse(data["official_promotion_allowed"])
        self.assertFalse(data["real_guitarset_scores_measured"])
        self.assertNotIn(KEY,json.dumps(data))
    def test_410_fallback(self):
        calls=[]
        class Opener:
            def open(self,req,timeout):
                model=json.loads(req.data)["model"]
                calls.append(model)
                if len(calls)==1:
                    raise urllib.error.HTTPError(s.ENDPOINT,410,"secret",{},io.BytesIO(b"secret"))
                return Response("Ablation: onset vs offset")
        with patch.object(s.urllib.request,"build_opener",return_value=Opener()):
            data=s.consult(KEY,"fixture")
        self.assertEqual(calls,list(s.CANDIDATES))
        self.assertEqual(data["model"],s.CANDIDATES[1])
        self.assertEqual(data["attempts"][0]["code"],410)
    def test_no_401_fallback(self):
        calls=[]
        class Opener:
            def open(self,req,timeout):
                calls.append(1)
                raise urllib.error.HTTPError(s.ENDPOINT,401,"secret",{},io.BytesIO(b"secret"))
        with patch.object(s.urllib.request,"build_opener",return_value=Opener()):
            with self.assertRaises(s.ProviderUnavailable):
                s.consult(KEY,"fixture")
        self.assertEqual(len(calls),1)
    def test_invalid_bearer_prefix_and_model(self):
        for key in ("Bearer "+KEY,"nvapi-short",""):
            with self.assertRaises(ValueError):s.consult(key,"test")
        with self.assertRaises(ValueError):s.make_payload("moonshotai/kimi-k3","fixture")
if __name__=="__main__":unittest.main()
