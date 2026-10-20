"""S8xx seven-reviewer final gate: six local audits plus real NVIDIA advice.

A scientific series is NEVER allowed to claim seven audits if NVIDIA
was skipped, unavailable, empty, fabricated, or from another series.
External advice cannot authorize model promotion or metric improvements.
"""
from __future__ import annotations
import argparse,json,re
from pathlib import Path

SCHEMA="NOTE_SEVENTH_SCIENTIFIC_ADVISER_V1"
SIX_SCHEMA="NOTE_SIX_ROLE_CONSENSUS_V1"
ALLOWED_MODELS={"z-ai/glm-5.3-flash","z-ai/glm-5.3"}
def verified(six,advice,series):
    if not re.fullmatch(r"S8\d\d",series) or int(series[1:])<802:
        raise ValueError("Unsupported S8xx series")
    if six.get("schema")!=SIX_SCHEMA or six.get("passed") is not True or (
        six.get("audits_received")!=6 or six.get("issues") or
        six.get("official_promotion_eligible") is not False):
        raise AssertionError("Six audit consensus must PASS before final 7/7 verdict")
    auditors=six.get("auditors",{})
    if set(auditors)!={"data","acoustics","architecture","training","metrics","live"}:
        raise AssertionError("Six independent audit identities missing")
    if any(v.get("passed") is not True or v.get("errors") for v in auditors.values()):
        raise AssertionError("One of the six local reviewers failed")
    if advice.get("schema")!=SCHEMA or advice.get("series")!=series or (
            advice.get("status")!="review_received"):
        raise AssertionError("Wrong, missing, or stale NVIDIA review")
    if advice.get("model") not in ALLOWED_MODELS or (
        advice.get("provider")!="NVIDIA NIM"):
        raise AssertionError("Unexpected external reviewer model/provider")
    if advice.get("seventh_adviser_executed") is not True or (
        advice.get("independent_opinion_only") is not True):
        raise AssertionError("NVIDIA did not really return advisory-only content")
    if advice.get("official_promotion_allowed") is not False or (
        advice.get("real_guitarset_scores_measured") is not False or (
        advice.get("live_code_modified") is not False)):
        raise AssertionError("External reviewer unexpectedly granted promotion")
    if not isinstance(advice.get("review_text"),str) or (
        len(advice["review_text"].strip())<40):
        raise AssertionError("NVIDIA review is empty/invalid")
    sha=advice.get("context_sha256","")
    if not re.fullmatch(r"[0-9a-f]{64}",sha):
        raise AssertionError("External research context checksum missing")
    return {"schema":"NOTE_SEVEN_ROLE_FINAL_GATE_V1","series":series,
            "passed":True,"six_audits":6,"seventh_review_received":True,
            "reviewer_model":advice["model"],
            "promotion_eligible":False,"all_six_metrics_preserved_not_measured":True}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--series",required=True)
    p.add_argument("--six-consensus",type=Path,required=True)
    p.add_argument("--seventh-advice",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    six=json.loads(args.six_consensus.read_text())
    review=json.loads(args.seventh_advice.read_text())
    output=verified(six,review,args.series)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(output,indent=2,sort_keys=True)+"\n")
    print("NOTE_SEVEN_OF_SEVEN_PASS",output["series"],
          output["reviewer_model"],"promotion=false")
if __name__=="__main__":main()
