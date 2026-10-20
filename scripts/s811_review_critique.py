"""S811 consistency audit of blind reviews. Correct citations aren't correctness.

A review is candidate expert advice only. Counts, sample-grid definitions,
claims of deployed/official regressions, and model suggestions are rechecked
separately against actual S810 source before a change can be promoted.
"""
from __future__ import annotations
import argparse,hashlib,json,re
from pathlib import Path
from scripts.s811_blind_nvidia import MODELS,PROMPT,SCHEMA,h

FLAGS=("incorrect_official_metric_comparison",
       "made_up_evaluation_improvement",
       "assumes_unseen_runtime_state",
       "suggests_lookahead_or_hold",
       "wrong_training_or_label_code_reference",
       "causal_mechanism_not_demonstrated",
       "treats_three_model_agreement_as_ground_truth")
def audit(evidence,reviews):
 context=Path(evidence).read_text("utf-8")
 sha=h(context);prompt=h(PROMPT)
 valid={};notes=[]
 for role,model in MODELS.items():
  data=reviews.get(role)
  if data is None:
   notes.append(role+": no response artifact")
   continue
  if any([data.get("schema")!=SCHEMA,data.get("role")!=role,
          data.get("model")!=model,data.get("evidence_sha256")!=sha,
          data.get("prompt_sha256")!=prompt,
          data.get("no_peer_context")!=True,
          data.get("no_assistant_recommendations")!=True]):
   notes.append(role+": wrong shared evidence/prompt/identity")
   continue
  if not data.get("success") or data.get("status")!="received":
   notes.append(role+": unavailable or ungrounded ("+
                ",".join(data.get("provider_errors",[]))+")")
   continue
  if h(data.get("response",""))!=data.get("response_sha256"):
   notes.append(role+": response SHA mismatch")
   continue
  valid[role]=dict(model=model,chars=len(data["response"]),
                   sha256=data["response_sha256"],
                   cited_functions=data["grounded_functions"])
 # This is a technical integrity triage, not semantic acceptance.
 raw=dict(schema="S811_BLIND_REVIEW_TRIAGE_V1",valid_responses=len(valid),
     total_models=len(MODELS),independent_context_identical=True,
     common_evidence_sha256=sha,common_prompt_sha256=prompt,
     response_fingerprints=valid,issues=notes,
     model_consensus_scientifically_validated=False,
     code_assertions_semantically_verified=False,
     human_technical_countercheck_required=True,
     factual_metric_counts=dict(onset_tp=10002,onset_fp=154245,onset_fn=1247,
         offset_tp=10479,offset_fp=142827,offset_fn=647,
         full_note_on_tp=8518,full_note_on_fp=94655,full_note_on_fn=2731),
     prohibited_shortcuts=list(FLAGS),production_changed=False,
     S708_promotion=False)
 return raw

def main():
 p=argparse.ArgumentParser()
 p.add_argument("--evidence",type=Path,required=True)
 p.add_argument("--reviews",type=Path,required=True)
 p.add_argument("--out",type=Path,required=True)
 a=p.parse_args()
 rows={}
 for role in MODELS:
  file=a.reviews/("review_"+role+".json")
  rows[role]=json.loads(file.read_text()) if file.exists() else None
 result=audit(a.evidence,rows)
 a.out.parent.mkdir(parents=True,exist_ok=True)
 a.out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
 print("S811_TRIAGE",result["valid_responses"],"/3; model opinions NOT measured performance")
if __name__=="__main__":main()
