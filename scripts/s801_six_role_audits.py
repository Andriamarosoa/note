"""Six fail-closed roles reviewing real FIT80 S801 OOF MIDI256 ablations."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path

ROLES=("data","acoustics","architecture","training","metrics","live")
SCORES=("onset_f1","exact_k","active_exact_k",
        "offset_accuracy","full_note_f1","frame_pitch_f1")
COUNT_KEYS=("tp","fp","fn","frames","k_correct","active_frames",
            "active_k_correct","frame_pitch_tp","frame_pitch_fp",
            "frame_pitch_fn","offset_eligible","offset_correct")
MODES=("identity","consensus","harmonic_veto","poly_gap",
       "adaptive","onset_native_guard","offset_birth_guard",
       "harmonic_parent_guard")

def inspect(role,raw):
    r=json.loads(Path(raw).read_text(encoding="utf-8"))
    def require(cond,message):
        if not cond:raise AssertionError(message)
    require(r["schema"]=="S801_FIT80_CAUSAL_DYNAMIC_ABLATION_V1"
            and r["completed"],"S801 FIT actual ablation not completed")
    require(len(r["policy_results"])==len(MODES)
            and set(r["policy_results"])==set(MODES),
            "one declared ablation absent")
    require(set(r["baseline_fit"])==set(SCORES),"all six metrics required")
    require(r["signed_source_counts"]["frames"]==306536 and
            r["signed_source_counts"]["k_correct"]==189124,
            "S800 source FIT parity drift")
    if role=="data":
        require(not any(r[k] for k in (
            "CAL_opened","HELD_opened","TEST_opened",
            "cal_evaluation_performed","cal_policy_selection_performed")),
            "sealed CAL/HELD/TEST opened")
        require(all(len(p["by_composition"])==5
                    for p in r["policy_results"].values()),
                "OOF compositions missing")
        return {"FIT80_only":True,"CAL_unopened":True}
    if role=="acoustics":
        require("harmonic_parent_guard" in r["policy_results"] and
                "onset_native_guard" in r["policy_results"] and
                "offset_birth_guard" in r["policy_results"],
                "required acoustic ablations missing")
        return {"harmonic_and_attack_guards":True,
                "onset_offset_separately_reported":True}
    if role=="architecture":
        require(r["heads_per_fragment_variable"] and
                not r["models_retrained"],"no independent dynamic-head OOF")
        require(all(p["aggregate_new_onsets"]==p["aggregate_new_offsets"]
                    for p in r["policy_results"].values()),
                "orphan supplemental MIDI events")
        return {"five_s795_heads_dynamically_routed":True,
                "native_S795_decoder_unmodified":True}
    if role=="training":
        require(not r["models_retrained"] and
                not r["cal_evaluation_performed"],"test-data selection leakage")
        return {"no_retrain":True,"provisional_FIT_only":True}
    if role=="metrics":
        require(all(p["metrics"].keys()>=set(SCORES) and
                    p["counts"].keys()>=set(COUNT_KEYS)
                    for p in r["policy_results"].values()),
                "six metrics/counts incomplete")
        require(not r["promoted"] and
                (r["provisional_fit_candidate"]=="identity" or
                 r["policy_results"][r["provisional_fit_candidate"]]["eligible_fit_only"]),
                "inadmissible candidate selected")
        return {"all_six_signed_FIT_axes":True,
                "CAL32_scores_not_claimed":True}
    if role=="live":
        require(not r["production_LIVE_modified"] and
                not r["latency_physical_measured"] and
                not r["used_future_frames"] and r["extra_hold_samples"]==0,
                "added latency/unsupported physical timing claim")
        return {"LIVE_unchanged":True,"no_lookahead":True,"physical_latency":"unmeasured"}
    raise AssertionError("unexpected role")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--role",required=True,choices=ROLES)
    p.add_argument("--report",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    out=dict(schema="NOTE_SIX_ROLE_AUDIT_V1",role=a.role,
             passed=False,experimental_promotion_allowed=False,
             findings={},errors=[])
    try:
        out["findings"]=inspect(a.role,a.report)
        out["passed"]=True
    except Exception as e:
        out["errors"]=[type(e).__name__+": "+str(e)]
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
    print("S801_ROLE",a.role,"PASS" if out["passed"] else "FAIL",flush=True)
    if not out["passed"]:sys.exit(1)
if __name__=="__main__":main()
