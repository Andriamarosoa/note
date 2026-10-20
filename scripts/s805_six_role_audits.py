"""S805 six evidence audits for true MIDI256 and composition-disjoint learning."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
SCORES=("onset_f1","exact_k","active_exact_k",
        "offset_accuracy","full_note_f1","frame_pitch_f1")
COUNT_KEYS=("tp","fp","fn","frames","k_correct","active_frames",
            "active_k_correct","frame_pitch_tp","frame_pitch_fp",
            "frame_pitch_fn","offset_eligible","offset_correct")
ROLES=("data","acoustics","architecture","training","metrics","live")
POLICIES=("spectral_balance","risk_acoustic","risk_joint")

def audit(role,r):
    def need(check,msg):
        if not check:raise AssertionError(msg)
    need(r.get("schema")=="S805_FIT80_CROSSFIT_CAUSAL_ONSET_RISK_V1" and
         r.get("complete"),"not a complete real S805 experiment")
    need(r["tracks"]==80 and len(r["compositions"])==5,"GuitarSet FIT80 absent")
    need(set(r["policies"])==set(POLICIES),"expected three predeclared policies")
    need(r["source_counts"]["frames"]==306536 and
         r["source_counts"]["k_correct"]==189124,
         "signed S800/S803 baseline count drift")
    need(set(r["source_counts"])==set(COUNT_KEYS),"12 canonical counters missing")
    need(set(r["baseline_fit"])==set(SCORES),"six baseline scores missing")
    need(not any(r[k] for k in ("CAL_opened","HELD_opened","TEST_opened")),
         "reserved split opened")
    need(not r["official_promotion"] and r["official_reference"]=="S708",
         "S708 reference unauthorized promotion")
    if role=="data":
        need(r["source_runs"]=={"S800":39352880035,"S802":39359182007,
            "S803":39363062723},"evidence provenance drift")
        return {"signed80_FIT":True,"source_raw_count_parity":True}
    if role=="acoustics":
        need(r["features_current_frame_only"] and
             r["S802_truth_used_for_crossfold_training_only"],
             "noncausal runtime risk classifier")
        return {"current_frame_features_only":True}
    if role=="architecture":
        need(r["risk_heads_independent_per_fold"] and
             not r["production_LIVE_modified"],"heads or production drift")
        return {"independent_risk_heads":2,"frozen_five_experts":5}
    if role=="training":
        need(not r["training_on_evaluation_composition"] and
             not r["threshold_search_performed"],"FIT fold used for risk training")
        for comp,heads in r["train_provenance"].items():
            need(comp in r["compositions"] and
                 set(heads)=={"risk_acoustic","risk_joint"},"missing OOF head")
            for head,proof in heads.items():
                need(proof["excluded_evaluation_composition"]==comp and
                     comp not in proof["training_compositions"] and
                     len(proof["training_compositions"])==4 and
                     proof["training_rows"]==proof["pos"]+proof["neg"] and
                     proof["pos"]>0 and proof["neg"]>0,
                     "cross-composition training leakage")
        return {"leave_one_composition_out_training_verified":True}
    if role=="metrics":
        for policy,row in r["policy_results"].items():
            need(set(row["counts"])==set(COUNT_KEYS) and
                 set(row["six_scores"])==set(SCORES) and
                 set(row["delta"])==set(SCORES) and
                 set(row["by_composition_delta"])==set(r["compositions"]),
                 "six ratios or composition comparisons missing")
            safe=all(float(x)>=-1e-12 for x in row["delta"].values()) and all(
                all(float(x)>=-1e-12 for x in per.values())
                for per in row["by_composition_delta"].values())
            need(row["no_regression_on_all_six_and_all_five_compositions"]==safe,
                 "regression veto not evaluated correctly")
            need(row["fit_exploratory_eligible"]==
                 (safe and any(x>1e-12 for x in row["delta"].values())),
                 "illegitimate FIT promotion flag")
        return {"all_six_MIDI256_metrics_checked":True,
                "fit_eligible":r["fit_provisional_eligible"]}
    if role=="live":
        need(not r["production_LIVE_modified"] and
             not r["physical_live_latency_measured"] and
             r["added_hold_samples"]==0 and
             r["added_lookahead_samples"]==0,"unproven latency or LIVE change")
        return {"no_extra_algorithmic_hold":True,
                "physical_mic_test_still_missing":True}
    raise ValueError("unexpected role")
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--report",type=Path,required=True)
    p.add_argument("--role",choices=ROLES,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    out=dict(schema="NOTE_SIX_ROLE_AUDIT_V1",role=a.role,passed=False,
             findings={},errors=[],experimental_promotion_allowed=False)
    try:
        out["findings"]=audit(a.role,json.loads(a.report.read_text()))
        out["passed"]=True
    except Exception as exc:
        out["errors"]=[type(exc).__name__+": "+str(exc)]
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(out,indent=2)+"\n")
    print("S805_SIX_ROLE",a.role,"PASS" if out["passed"] else "FAIL")
    if not out["passed"]:sys.exit(1)
if __name__=="__main__":main()
