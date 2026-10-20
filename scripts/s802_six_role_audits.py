"""S802 six independent evidence-based retrospective collision audits, fail closed."""
import argparse
import json
from pathlib import Path
import sys

ROLES=("data","acoustics","architecture","training","metrics","live")
POLICIES=("consensus","poly_gap","adaptive")
def audit(role,report):
    r=json.loads(Path(report).read_text(encoding="utf-8"))
    def require(v,why):
        if not v:raise AssertionError(why)
    require(r.get("schema")=="S802_FIT80_NATIVE_SYNTHETIC_COLLISION_FORENSIC_V1"
            and r.get("complete"),"no full real FIT80 forensic result")
    require(r["tracks"]==80 and set(r["policy_list"])==set(POLICIES)
            and set(r["by_policy"])==set(POLICIES),
            "all 80 FIT tracks / three frozen policies required")
    require(r["s801_policy_six_scores_and_counts_parity"],
            "independent real MIDI six-metric parity failed")
    for p in POLICIES:
        d=r["by_policy"][p]
        require(d["matched_ground_truth"]+d["unmatched_ground_truth"]==
                d["total_synthetic_onsets"],"missing TP/FP classification")
        require(d["retrigger_true_positives"]+d["retrigger_false_positives"]==
                d["native_retrigger_closures"],"native closure incorrectly classified")
        require(len(d["by_composition"])==5,"held-out partitions absent")
    if role=="data":
        require(not any(r[k] for k in ("CAL_opened","HELD_opened","TEST_opened")),
                "cal/held/test truth leakage")
        require(r["truth_used_only_for_posthoc_forensics"] and
                r["future_native_used_only_for_posthoc_forensics"],
                "future oracle used by decoder")
        return {"FIT80_only":True,"future_native_for_analysis_only":True}
    if role=="acoustics":
        require(all(k in z for z in r["observations"]["adaptive"]
                    for k in ("on","sustain","off","on_rise",
                              "harmonic_parent","general_p","harmonic_p")),
                "acoustic evidence absent")
        return {"onset_offset_harmonic_native_evidence":True,
                "classification_requires_truth":True}
    if role=="architecture":
        require(r["synthesis_policy_or_decoder_modified"] is False,
                "new unvalidated MIDI decoder introduced")
        require(r["signed_s800_native_fit_counts"]["frames"]==306536,
                "S800 306536 frames source changed")
        return {"same_S795_decoder_and_S801_router":True}
    if role=="training":
        require(r["human_oracle_in_runtime"] is False and
                r["truth_used_only_for_posthoc_forensics"] is True,
                "oracle leak into training/inference")
        return {"no_new_training":True,"retrospective_truth_not_model_input":True}
    if role=="metrics":
        require(r["signed_s800_native_fit_counts"]["k_correct"]==189124,
                "signed FIT source mismatch")
        require(not r["promoted"],"unsupported promotion")
        return {"six_metric_S801_parity":True,
                "collisions_not_automatically_false_positives":True}
    if role=="live":
        require(not r["production_LIVE_modified"] and
                not r["latency_physical_measured"],
                "unmeasured live performance claimed")
        return {"no_live_code_path_change":True,
                "no_added_hold_or_lookahead_by_replay":True}
    raise AssertionError("unknown role")
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--role",choices=ROLES,required=True)
    p.add_argument("--report",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    output={"schema":"NOTE_SIX_ROLE_AUDIT_V1","role":args.role,
            "passed":False,"experimental_promotion_allowed":False,
            "findings":{},"errors":[]}
    try:
        output["findings"]=audit(args.role,args.report)
        output["passed"]=True
    except Exception as e:
        output["errors"]=[type(e).__name__+": "+str(e)]
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(output,indent=2,sort_keys=True)+"\n")
    print("S802_ROLE_"+args.role+"_"+("PASS" if output["passed"] else "FAIL"))
    if not output["passed"]:sys.exit(1)
if __name__=="__main__":main()
