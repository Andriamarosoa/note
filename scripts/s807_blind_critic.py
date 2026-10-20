"""S807 blind consultation independent evidence integrity + skeptical triage.

External model text is UNVERIFIED. Never label 3 responses 'correct' just because
they were returned. This validator only checks independence, citations, scope,
and numeric consistency against the unexposed signed FIT80 evidence.
"""
from __future__ import annotations
import argparse,json,re
from pathlib import Path
from scripts.s807_blind_activek_evidence import digest
from scripts.s807_blind_nvidia_review import MODELS,PROMPT,frame,safe_digest
def check(evidence,results):
 accepted={};issues=[]
 base=safe_digest(evidence)
 same_input=digest(frame(evidence))
 for role,model in MODELS.items():
  r=results.get(role)
  if not isinstance(r,dict):
   issues.append(role+": missing model response")
   continue
  if (r.get("reviewer")!=role or r.get("model")!=model or
      r.get("evidence_sha256")!=base or
      r.get("shared_context_sha256")!=same_input or
      r.get("prompt_sha256")!=digest(PROMPT)):
   issues.append(role+": mismatched prompt, evidence or identity")
   continue
  if (r.get("no_peer_context") is not True or
      r.get("no_preferred_hypothesis") is not True or
      r.get("scientific_promotion") is not False):
   issues.append(role+": biased/cross-influenced or unauthorized promotion")
   continue
  if r.get("status")!="received" or not r.get("quality_pass"):
   issues.append(role+": provider unavailable or insufficient real grounded text")
   continue
  if digest(r.get("review_text",""))!=r.get("response_sha256"):
   issues.append(role+": response data changed")
   continue
  accepted[role]={"model":model,"response_sha256":r["response_sha256"],
                  "grounded_paths":r.get("grounded_paths",[])}
 # Certain unsupported claims can be detected, but other claims require human/
 # code-traced critique. We do NOT call these semantic reviewers 'correct'.
 critical={"statements":"Manual line-by-line checks of diagnoses and tests REQUIRED",
   "prohibited_shortcuts":[
     "no training on target composition labels",
     "no post-onset future confirmation",
     "no mixing CAL32 with FIT80",
     "no unchanged S708 claim without byte-level parity",
     "no promotion from FIT or unmeasured physical CPU timing"],
   "known_FIT_truth":evidence["numeric"]["under_active_frames"],
   "signed_fit_k_gap":evidence["numeric"]["under_active_frames"]-
                       evidence["numeric"]["over_active_frames"]}
 return {"schema":"S807_BLIND_REVIEW_TRIAGE_V1",
         "evidence_sha256":base,"shared_context_sha256":same_input,
         "same_system_prompt_sha256":digest(PROMPT),
         "three_distinct_model_ids":len(set(MODELS.values()))==3,
         "independent_reviews_accepted":len(accepted),
         "all_three_substantive_review_received":len(accepted)==3,
         "review_validity_not_scientific_validation":True,
         "code_and_metric_fact_checked_afterward":False,
         "S708_promotion_allowed":False,
         "accepted_reports":accepted,"issues":issues,"critical_protocol":critical}
def main():
 a=argparse.ArgumentParser()
 a.add_argument("--bundle",required=True,type=Path)
 a.add_argument("--reviews-dir",required=True,type=Path)
 a.add_argument("--out",required=True,type=Path)
 opts=a.parse_args()
 src=json.loads(opts.bundle.read_text())
 docs={}
 for role in MODELS:
  p=opts.reviews_dir/("review_"+role+".json")
  docs[role]=json.loads(p.read_text()) if p.exists() else None
 verdict=check(src,docs)
 opts.out.parent.mkdir(parents=True,exist_ok=True)
 opts.out.write_text(json.dumps(verdict,indent=2,ensure_ascii=False)+"\n")
 print("S807_BLIND_MODEL_TRIAGE",verdict["independent_reviews_accepted"],
       "/3; does_not_certify_scientific_correctness",flush=True)
if __name__=="__main__":main()
