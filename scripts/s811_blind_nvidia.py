"""S811 isolated, unbiased S810-NIM consultation; no prior adviser opinion in context.

Each provider sees exactly the same verified read-only evidence and system/user
messages. Three distinct MODEL IDS; provider-specific inference settings allowed
only to deliver an answer (e.g. Nemotron disable internal thought). Failures
are reported rather than passing a missing/error result as a model opinion.
No CAL32/HELD/TEST content, personal information, secrets or audio shared.
"""
from __future__ import annotations
import argparse,hashlib,json,os,re,urllib.error,urllib.request
from pathlib import Path
from scripts.s798_kimi_adviser import ENDPOINT,NoRedirect

SCHEMA="S811_S810_BLIND_NVIDIA_REVIEW_V1"
MODELS={
 "model_a":"google/gemma-4-31b-it",
 "model_b":"nvidia/nemotron-3.5-lightning-30b-a3b",
 "model_c":"z-ai/glm-5.3",
}
PROMPT=(
 "You are an independent machine-learning code reviewer. You will receive raw "
 "measurements and bounded code excerpts from a guitar audio machine-learning "
 "system. Independently diagnose why the three S810 shadow axes onset, offset "
 "and full_note perform much worse on their respective reported diagnostics "
 "than the prior official model metrics might superficially suggest. "
 "Inspect the target definitions, examples and sampling, training objectives, "
 "calibration, validation scheme, evaluation definitions and possible code bugs "
 "using only visible evidence; do not assume any one mechanism is the cause. "
 "First separate what the data PROVE from conjecture, and distinguish "
 "shadow 1024-sample frame labels from real event-based six-score MIDI256. "
 "Present at least three mutually distinguishable explanations, EVIDENCE FOR "
 "and EVIDENCE AGAINST each, trace code by named function and line numbers "
 "included in the evidence, and identify concrete checks that could FALSIFY "
 "each hypothesis. Explicitly mention any interpretation that cannot be "
 "supported by the supplied data. Offer minimally invasive controlled "
 "ablation experiments without changing the official S708 reference, using "
 "any reserved split, waiting for future frames, holding attack/offset, "
 "using ground truth during runtime or claiming unmeasured gains. "
 "Critique even intuitive explanations; estimate whether the low F1 is "
 "dominated by FP vs FN from the supplied integer counts. "
 "Never invent results or ask another model's opinion. "
 "Use headings FACTS, COMPETING HYPOTHESES, COUNTEREVIDENCE, "
 "FALSIFICATION EXPERIMENTS, and UNVERIFIED CLAIMS. "
 "Be detailed and technically skeptical. Respond in French."
)
SECRET=re.compile(r"nvapi-[a-zA-Z0-9_-]{8,}|github_pat_[a-zA-Z0-9_]{12,}|ghp_[a-zA-Z0-9]{12,}")

def h(x):return hashlib.sha256(x.encode("utf-8")).hexdigest()

def context(file):
 t=Path(file).read_text("utf-8")
 if "S811 — Blind input" not in t or "154245" not in t or "142827" not in t:
  raise AssertionError("expected original evidence missing")
 if SECRET.search(t):raise AssertionError("credential-looking token blocked")
 if any(z in t for z in ("assistant_hypothesis","prior_nvidia_response","model_a_response")):
  raise AssertionError("biased or peer context found")
 return t

def call_model(role,evidence):
 model=MODELS[role]
 record=dict(schema=SCHEMA,role=role,model=model,provider="NVIDIA_NIM",
             evidence_sha256=h(evidence),prompt_sha256=h(PROMPT),
             no_peer_context=True,no_assistant_recommendations=True,
             status="unavailable",success=False,response="",response_sha256=None,
             reasoning_content_saved=False,review_not_scientific_proof=True,
             provider_errors=[],grounded_functions=[],usage={})
 key=os.getenv("NVIDIA_API_KEY","")
 if not key.startswith("nvapi-"):
  record["provider_errors"]=["missing_nvidia_api_key"]
  return record
 payload=dict(model=model,messages=[
    {"role":"system","content":PROMPT},
    {"role":"user","content":evidence}],
    max_tokens=6200 if role=="model_c" else 5500,
    stream=False,temperature=.5 if role=="model_c" else .4)
 if role=="model_b":
  # Only disables provider's non-visible reasoning consumption.
  payload["chat_template_kwargs"]={"enable_thinking":False}
 request=urllib.request.Request(ENDPOINT,
      data=json.dumps(payload,ensure_ascii=False).encode("utf-8"),
      headers={"Authorization":"Bearer "+key,"Content-Type":"application/json",
               "Accept":"application/json"},
      method="POST")
 try:
  with urllib.request.build_opener(NoRedirect).open(request,timeout=200) as reply:
   result=json.loads(reply.read(5_000_000).decode("utf-8"))
  if not result.get("choices"):raise ValueError("provider_no_choices")
  message=result["choices"][0].get("message") or {}
  answer=message.get("content") or ""
  if isinstance(answer,list):
   answer="\n".join(x.get("text","") for x in answer if isinstance(x,dict))
  answer=SECRET.sub("[REDACTED]",str(answer))[:80000]
  record["response"]=answer
  record["response_sha256"]=h(answer)
  record["status"]="received" if answer.strip() else "empty"
  named=[s for s in ("truth_arrays","select_train_rows","fit_axis",
                     "sample_predictions","counts_for_axis","make_training_track")
         if s in answer]
  record["grounded_functions"]=named
  record["success"]=len(answer)>1700 and len(named)>=2
  record["usage"]={k:v for k,v in (result.get("usage") or {}).items()
     if k in ("prompt_tokens","completion_tokens","total_tokens") and isinstance(v,int)}
  if not record["success"]:record["provider_errors"].append("too_short_or_unsupported_code")
 except urllib.error.HTTPError as e:
  record["provider_errors"]=["HTTP_"+str(e.code)]
 except (TimeoutError,ValueError,UnicodeError,urllib.error.URLError,OSError) as e:
  record["provider_errors"]=[type(e).__name__]
 return record

def main():
 parser=argparse.ArgumentParser()
 parser.add_argument("--role",required=True,choices=tuple(MODELS))
 parser.add_argument("--evidence",required=True,type=Path)
 parser.add_argument("--out",required=True,type=Path)
 a=parser.parse_args()
 if a.out.exists():raise FileExistsError("immutable model answer already exists")
 evidence=context(a.evidence)
 record=call_model(a.role,evidence)
 a.out.parent.mkdir(parents=True,exist_ok=True)
 a.out.write_text(json.dumps(record,indent=2,ensure_ascii=False)+"\n")
 print("S811_NVIDIA_INDEPENDENT",a.role,record["model"],record["status"],
       "substantive="+str(record["success"]),
       "evidence_sha="+record["evidence_sha256"],
       "error="+",".join(record["provider_errors"]),flush=True)
if __name__=="__main__":main()
