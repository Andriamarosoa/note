"""S819 fact-check receipt of six deterministic audits and three real NIM reviews.

This is evidence provenance verification, NOT validation of model hypotheses.
Fail closed on provider errors, duplicates, absent model output or mismatched
input hashes. Never promote music systems based on opinion agreement.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from scripts.s819_blind_nvidia import SCHEMA as NVIDIA_SCHEMA,MODELS,SYSTEM,hash_text,evidence_text
from scripts.s819_six_checks import SCHEMA as CHECK_SCHEMA, ROLES

SCHEMA="S819_THREE_REAL_NVIDIA_AND_SIX_CODE_CHECKS_RECEIPT_V1"
def validate(evidence,reviews,audits):
    if set(reviews)!=set(MODELS) or set(audits)!=set(ROLES):
        raise AssertionError("3 independent model roles + 6 local roles required")
    content_hash=hash_text(evidence)
    prompt_hash=hash_text(SYSTEM)
    for role,model in MODELS.items():
        r=reviews[role]
        if (r.get("schema")!=NVIDIA_SCHEMA or r.get("role")!=role
            or r.get("model")!=model or r.get("provider")!="NVIDIA NIM"
            or r.get("status")!="received" or r.get("response_received") is not True
            or r.get("advice_substantive") is not True
            or r.get("evidence_sha256")!=content_hash
            or r.get("system_prompt_sha256")!=prompt_hash
            or r.get("no_peer_context") is not True
            or r.get("production_modified") is not False
            or r.get("promotion_allowed") is not False
            or r.get("provider_errors")):
            raise AssertionError("NVIDIA provider reviewer is missing or unverified: "+role)
        if hashlib.sha256(r["response"].encode()).hexdigest()!=r["response_sha256"]:
            raise AssertionError("NVIDIA adviser output response SHA invalid: "+role)
    for role in ROLES:
        a=audits[role]
        if (a.get("schema")!=CHECK_SCHEMA or a.get("role")!=role
            or a.get("passed") is not True or a.get("model_or_work_agent") is not False
            or a.get("verifier_type")!="deterministic_code_check"
            or a.get("evidence_sha256")!=content_hash
            or a.get("errors") or a.get("promotion_allowed") is not False):
            raise AssertionError("missing independent static audit: "+role)
    return dict(schema=SCHEMA,provider_review_received=3,
        provider_model_ids=[MODELS[x] for x in sorted(MODELS)],
        six_deterministic_audits_verified=6,
        actual_personal_Work_agents_invoked=0,
        evidence_sha256=content_hash,
        identical_external_prompt_sha256=prompt_hash,
        no_peer_opinion_leakage_verified_by_message_constructor=True,
        provider_responses_are_unverified_hypotheses=True,
        composition_OOF_new_S819_score_measured=False,
        production_modified=False,promotion_allowed=False,
        independent_validation_done=False,
        model_response_text_files=[f"review_{x}.json" for x in sorted(MODELS)])
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--evidence",type=Path,required=True)
    p.add_argument("--reviews-dir",type=Path,required=True)
    p.add_argument("--checks-dir",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    data=evidence_text(args.evidence)
    reviews={}
    audits={}
    for role in MODELS:
        reviews[role]=json.loads((args.reviews_dir/f"review_{role}.json").read_text())
    for role in ROLES:
        audits[role]=json.loads((args.checks_dir/f"check_{role}.json").read_text())
    report=validate(data,reviews,audits)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2)+"\n")
    print("S819_REAL_NVIDIA_RECEIPT",report["provider_review_received"],
          "SIX_LOCAL_AUDITS",report["six_deterministic_audits_verified"],
          "WORK_AGENTS",report["actual_personal_Work_agents_invoked"],
          "NO_PROMOTION",not report["promotion_allowed"])
if __name__=="__main__":main()
