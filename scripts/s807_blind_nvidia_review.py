"""S807 blind NVIDIA consultation. Same prompt and evidence bytes, 3 separate models.

No role-specific prompts, prior guesses, recommendations, peer outputs, or
assistant preferred architecture. A response is a hypothesis, NOT a finding.
One request per provider; failures saved for audit, no automatic rerun.
"""
from __future__ import annotations
import argparse,hashlib,json,os,re,sys,urllib.request,urllib.error
from pathlib import Path
from scripts.s798_kimi_adviser import ENDPOINT,NoRedirect
from scripts.s807_blind_activek_evidence import SCHEMA,digest,SECRET
MODELS={
 "model_a":"google/gemma-4-31b-it",
 "model_b":"nvidia/nemotron-3.5-lightning-30b-a3b",
 "model_c":"poolside/laguna-xs-2.1",
}
PROMPT=(
 "You are an independent scientific reviewer. Investigate the factual cause "
 "of the low active-note cardinality accuracy in the attached measured FIT80 "
 "data and the numbered code excerpts. You are one of several isolated reviewers "
 "but you receive NO information about other reviewers or any preferred solution. "
 "Do not infer unseen code. Begin by independently analysing the numerical "
 "distribution and the metric definition, then trace plausible causal mechanisms "
 "to visible file paths and line numbers. For each of at least THREE competing "
 "hypotheses give: evidence FOR and AGAINST, an explicit reproducible "
 "counterexample or falsification test, expected direction of all SIX metrics, "
 "and potential increase in physical runtime or decision delay. Rank candidates "
 "based on strongest direct evidence, NOT intuition. Distinguish observation "
 "from conjecture, mathematical upper bound from achievable performance, "
 "and a pretrained head from an experimental untrained proposal. Consider "
 "alternative explanations including metric artefacts, threshold definitions, "
 "training distribution shift, polyphonic source signals, and temporal "
 "note-state errors, without assuming any of them are correct. "
 "Do not propose opening reserved splits or using future frames. "
 "No truth annotations at runtime. No backdating, hold or lookahead. "
 "No changes to official S708 before independent controlled validation. "
 "Cite at least TWO exact visible paths and line numbers. "
 "Write substantial independent technical critique with headings "
 "EVIDENCE, THREE COMPETING HYPOTHESES, DISCONFIRMATION TESTS, "
 "EXPERIMENT DESIGN and LIMITATIONS. No invented F1 improvement."
)
def safe_digest(obj):
 other=dict(obj);given=other.pop("evidence_sha256",None)
 actual=digest(json.dumps(other,sort_keys=True,ensure_ascii=False))
 if given!=actual:raise AssertionError("evidence tampered")
 return given
def frame(obj):
 if obj.get("schema")!=SCHEMA or not obj.get("complete"):raise AssertionError("not verified S807 FIT")
 if obj.get("CAL_contents_sent") or obj.get("HELD_contents_sent") or obj.get("TEST_contents_sent"):
  raise AssertionError("reserved evaluation data blocked")
 if obj.get("numeric",{}).get("split")!="FIT80_ONLY":raise AssertionError("not FIT")
 # Explicitly identical for all three models: no role/model name in content.
 text="BLIND_SCIENTIFIC_EVIDENCE_JSON\n"+json.dumps(obj["numeric"],ensure_ascii=False,sort_keys=True)
 for x in obj["source_code_excerpts"]:
  text+="\nFILE "+x["path"]+" SHA256="+x["sha256"]+" TOTAL_LINES="+str(x["total_lines"])+"\n"+x["excerpt"]
 text+="\nLIMITS: "+obj["limitations"]
 if SECRET.search(text):raise AssertionError("secret-like outbound text")
 return text
def one(role,evidence,key):
 result=dict(schema="S807_THREE_INDEPENDENT_NVIDIA_REVIEWS_V1",
  series="S807",reviewer=role,model=MODELS[role],
  provider="NVIDIA NIM",status="unavailable",quality_pass=False,
  shared_context_sha256=digest(evidence),prompt_sha256=digest(PROMPT),
  response_sha256=None,review_text="",errors=[],no_peer_context=True,
  no_preferred_hypothesis=True,scientific_promotion=False)
 if not key or not key.startswith("nvapi-"):
  result["errors"]=["missing_provider_secret"]
  return result
 request=urllib.request.Request(ENDPOINT,
  data=json.dumps({"model":MODELS[role],
     "messages":[{"role":"system","content":PROMPT},
                 {"role":"user","content":evidence}],
     "stream":False,"temperature":0.4,"max_tokens":3300}).encode("utf-8"),
  headers={"Authorization":"Bearer "+key,"Content-Type":"application/json",
           "Accept":"application/json"},method="POST")
 try:
  with urllib.request.build_opener(NoRedirect).open(request,timeout=185) as response:
   payload=json.loads(response.read(4_000_000).decode("utf-8"))
  choices=payload.get("choices") or []
  if not choices:raise ValueError("provider_no_choices")
  response=choices[0].get("message") or {}
  t=response.get("content") or ""
  if isinstance(t,list):
   t="\n".join(p.get("text","") for p in t if isinstance(p,dict) and p.get("type")=="text")
  t=SECRET.sub("[REDACTED]",str(t))[:60000]
  result["review_text"]=t
  result["response_sha256"]=digest(t)
  result["status"]="received"
  paths=[x["path"] for x in json.loads(evidence.split("BLIND_SCIENTIFIC_EVIDENCE_JSON\n",1)[1].split("\nFILE ",1)[0])["source_code_excerpts"]] if False else []
  # Independent source paths verified against literal LINE references in report.
  cited=set(re.findall(r"(?:src|scripts)/[A-Za-z0-9_/.]+\.py",t))
  available=set(re.findall(r"FILE ((?:src|scripts)/[^\s]+\.py) SHA256=",evidence))
  grounded=sorted(cited&available)
  result["grounded_paths"]=grounded
  result["quality_pass"]=(len(t.strip())>=1200 and len(grounded)>=2 and
    bool(re.search(r"(?i)(hypothes|hypothè|experiment|test)",t)))
  if not result["quality_pass"]:result["errors"].append("shallow_or_uncited_review")
  usage=payload.get("usage") or {}
  result["usage"]={k:v for k,v in usage.items()
                   if k in ("prompt_tokens","completion_tokens","total_tokens") and isinstance(v,int)}
 except urllib.error.HTTPError as e:
  result["errors"]=["provider_http_"+str(e.code)]
 except (urllib.error.URLError,TimeoutError,ValueError,UnicodeError,
         ConnectionError,KeyError,OSError) as e:
  result["errors"]=["provider_"+type(e).__name__]
 return result
def main():
 p=argparse.ArgumentParser()
 p.add_argument("--role",choices=tuple(MODELS),required=True)
 p.add_argument("--bundle",type=Path,required=True)
 p.add_argument("--out",type=Path,required=True)
 a=p.parse_args()
 if a.out.exists():raise FileExistsError("immutable review")
 e=json.loads(a.bundle.read_text())
 safe_digest(e)
 c=frame(e)
 r=one(a.role,c,os.getenv("NVIDIA_API_KEY",""))
 r["evidence_sha256"]=e["evidence_sha256"]
 a.out.parent.mkdir(parents=True,exist_ok=True)
 a.out.write_text(json.dumps(r,ensure_ascii=False,indent=2)+"\n")
 print("S807_INDEPENDENT_REVIEW",a.role,r["model"],r["status"],
       "substantive="+str(r["quality_pass"]),
       "shared_context="+r["shared_context_sha256"],
       "errors="+",".join(r["errors"]),flush=True)
 # Fail closed in *consensus*, not in this capture step: retain errors rather
 # than needlessly requery unreliable vendors multiple times.
if __name__=="__main__":main()
