"""S799 research-only NVIDIA external adviser. Max two model calls, no LIVE edits."""
import argparse
import json
import os
from pathlib import Path
import re
import sys
import urllib.error
import urllib.request
from scripts.s798_kimi_adviser import ENDPOINT, NoRedirect, SYSTEM, SECRET_RE, safe_context
CANDIDATES=("moonshotai/kimi-k2.5","meta/llama-3.3-70b-instruct")
FALLBACK_CODES=(404,410)
def make_payload(model, context):
    if model not in CANDIDATES: raise ValueError("unapproved model")
    req={"model":model,
         "messages":[{"role":"system","content":SYSTEM},
                     {"role":"user","content":"Analyse S795-S797 comme septieme avis independant. Cinq hypotheses ablatables sur les harmoniques, confusions octaves, attaques onset et extinctions offset, anti regression six scores, zero lookahead/latence. Toute hypothese non verifiee doit etre identifiee.\n"+context}],
         "stream":False,"max_tokens":4500,
         "temperature":0.6 if model==CANDIDATES[0] else 0.4}
    if model==CANDIDATES[0]:
        req["thinking"]={"type":"disabled"}
    return req
class ProviderUnavailable(RuntimeError):
    def __init__(self,model,code):
        self.model,self.code=model,code
        super().__init__("NVIDIA HTTP "+str(code)+" for "+model)
def fetch_one(model,key,context,timeout=115):
    req=urllib.request.Request(ENDPOINT,method="POST",
         data=json.dumps(make_payload(model,context),ensure_ascii=False).encode(),
         headers={"Authorization":"Bearer "+key,"Content-Type":"application/json","Accept":"application/json"})
    try:
        with urllib.request.build_opener(NoRedirect).open(req,timeout=timeout) as response:
            data=json.loads(response.read(4_000_000).decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise ProviderUnavailable(model,exc.code) from None
    except urllib.error.URLError:
        raise RuntimeError("NVIDIA network failure") from None
    options=data.get("choices") or []
    if not options or not isinstance(options[0].get("message"),dict):
        raise ValueError("NVIDIA returned no assistant message")
    content=options[0]["message"].get("content")
    if isinstance(content,list):
        content="\n".join(str(c.get("text","")) for c in content if isinstance(c,dict) and c.get("type")=="text")
    if not isinstance(content,str) or not content.strip():
        raise ValueError("NVIDIA returned empty answer")
    usage=data.get("usage") or {}
    return {"schema":"NOTE_S799_EXTERNAL_OPINION_V1",
            "model":model,"provider":"NVIDIA NIM","status":"review_received",
            "independent_opinion_only":True,"recommendations_unvalidated":True,
            "six_deterministic_audits_unchanged":True,"official_promotion_allowed":False,
            "real_guitarset_scores_measured":False,"physical_mic_latency_measured":False,
            "live_code_modified":False,
            "review_text":SECRET_RE.sub("[REDACTED]",content)[:32000],
            "token_usage":{k:v for k,v in usage.items() if k in ("prompt_tokens","completion_tokens","total_tokens") and isinstance(v,int)}}
def consult(key,context):
    if not key.startswith("nvapi-") or len(key)<20 or key.strip()!=key:
        raise ValueError("invalid NVIDIA_API_KEY")
    attempts=[]
    for idx,model in enumerate(CANDIDATES):
        try:
            out=fetch_one(model,key,context)
            attempts.append({"model":model,"result":"success"})
            out["attempts"]=attempts
            return out
        except ProviderUnavailable as exc:
            attempts.append({"model":model,"result":"http_error","code":exc.code})
            if idx!=0 or exc.code not in FALLBACK_CODES: raise
    raise RuntimeError("No supported NVIDIA model")
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,default=Path("."))
    parser.add_argument("--out",type=Path,required=True)
    a=parser.parse_args()
    try:
        if a.out.exists(): raise FileExistsError("Refuse overwrite")
        data=consult(os.environ.get("NVIDIA_API_KEY",""),safe_context(a.root))
        a.out.parent.mkdir(parents=True,exist_ok=True)
        a.out.write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        print("S799_EXTERNAL_SUCCESS model="+data["model"]+" opinion_only=true promoted=false")
    except ProviderUnavailable as e:
        print("S799_EXTERNAL_FAIL model="+e.model+" HTTP "+str(e.code),file=sys.stderr)
        raise SystemExit(1) from None
    except (ValueError,OSError,RuntimeError):
        print("S799_EXTERNAL_FAIL invalid input, output or network",file=sys.stderr)
        raise SystemExit(1) from None
if __name__=="__main__":main()
