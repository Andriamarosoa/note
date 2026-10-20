"""Six independent fail-closed S803 evidence audits (not six Work sessions)."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
# Audit stays dependency-light: canonical raw-count ABI of S795/S800,
# and canonical six-ratio ABI of S704. No model/PyTorch import in auditors.
COUNT_KEYS=("tp","fp","fn","frames","k_correct","active_frames",
            "active_k_correct","frame_pitch_tp","frame_pitch_fp",
            "frame_pitch_fn","offset_eligible","offset_correct")
SCORES=("onset_f1","exact_k","active_exact_k","offset_accuracy",
        "full_note_f1","frame_pitch_f1")
ROLES=("data","acoustics","architecture","training","metrics","live")
def audit(role,r):
    def need(q,why):
        if not q:raise AssertionError(why)
    need(r.get("schema")=="S803_FIT80_CAUSAL_VETO_SIX_METRIC_V1" and
         r.get("complete"),"real S803 experiment not complete")
    need(r["tracks"]==80 and len(r["compositions"])==5,"missing FIT partitions")
    need(set(r["policies"])=={"rise_attack","parent_conflict","agreement_attack"},
         "unexpected experiments")
    need(r["source_counts"]["frames"]==306536 and
         r["source_counts"]["k_correct"]==189124,"S795 signed FIT count drift")
    need(set(r["baseline_fit"])==set(SCORES),"six baseline metrics absent")
    need(not r["official_promotion"] and r["official_reference"]=="S708",
         "unauthorized promotion")
    if role=="data":
        need(not any(r[z] for z in ("CAL_opened","HELD_opened","TEST_opened")),
             "forbidden heldout use")
        need(r["source_runs"]["S800"]==39352880035 and
             r["source_runs"]["S802"]==39359182007,"unverified source run")
        return dict(real_fit80=True,signed_sources=True)
    if role=="acoustics":
        for name,group in r["retrospection_only_auc"].items():
            need(group["tp"]+group["fp"]==group["n"],"TP/FP accounting")
            need(set(group["features"])=={"on","sustain","off","on_rise",
                "sustain_rise","harmonic_parent","general_p","bass_p",
                "polyphonic_p","transient_p","harmonic_p"},"missing feature")
            for x in group["features"].values():
                need(0<=x["auc"]<=1,"invalid AUC")
                need(len(x["by_composition"])==5,"missing by-composition acoustic result")
        return dict(roc_true_false_per_composition=True)
    if role=="architecture":
        need(r["heads_dynamic_per_frame_pitch"] and
             r["independent_frozen_heads"]==5 and r["native_decoder_unchanged"],
             "dynamic heads/decoder contract lost")
        return dict(five_existing_heads_with_fragment_pitch_gates=True)
    if role=="training":
        need(not r["threshold_search_performed"] and
             not r["feature_oracle_in_decoder"],"hyperparameter search or oracle leak")
        return dict(predeclared_guards=True,retrospective_labels_not_decoder_features=True)
    if role=="metrics":
        for name,cand in r["policy_results"].items():
            need(set(cand["counts"])==set(COUNT_KEYS),"raw metric counters missing")
            need(set(cand["delta"])==set(SCORES),"six-metric comparison incomplete")
            need(set(cand["by_composition_delta"])==set(r["compositions"]),
                 "missing composition comparison")
            safe=all(x>=-1e-12 for x in cand["delta"].values()) and all(
                all(x>=-1e-12 for x in d.values())
                for d in cand["by_composition_delta"].values())
            need(safe==cand["six_metric_no_regression_fit"],"non-regression misreported")
            need(cand["fit_exploratory_eligible"]==
                 (safe and any(x>1e-12 for x in cand["delta"].values())),
                 "eligibility misreported")
        return dict(six_metric_raw_and_composition_veto_verified=True,
                    fit_candidates=r["fit_provisional_eligible"])
    if role=="live":
        need(not r["production_LIVE_modified"] and
             not r["physical_live_latency_measured"],"unproven LIVE claim")
        need(r["synthetic_future_lookahead_samples"]==0 and
             r["extra_hold_samples"]==0,"forbidden extra latency")
        return dict(no_production_mutation=True,physical_test_still_required=True)
    raise ValueError("unknown role")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--role",choices=ROLES,required=True)
    p.add_argument("--report",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    a=p.parse_args()
    output=dict(schema="NOTE_SIX_ROLE_AUDIT_V1",role=a.role,passed=False,
                experimental_promotion_allowed=False,findings={},errors=[])
    try:
        output["findings"]=audit(a.role,json.loads(a.report.read_text()))
        output["passed"]=True
    except Exception as e:output["errors"]=[type(e).__name__+": "+str(e)]
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(output,indent=2,sort_keys=True)+"\n")
    print("S803_AUDIT",a.role,"PASS" if output["passed"] else "FAIL",flush=True)
    if not output["passed"]:sys.exit(1)
if __name__=="__main__":main()
