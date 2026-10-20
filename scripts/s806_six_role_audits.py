"""S806 fail-closed local six-role evidence audit, real FIT80 MIDI256."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROLES=("data","acoustics","architecture","training","metrics","live")
SCORES=("onset_f1","exact_k","active_exact_k",
        "offset_accuracy","full_note_f1","frame_pitch_f1")
COUNTS=("tp","fp","fn","frames","k_correct","active_frames",
        "active_k_correct","frame_pitch_tp","frame_pitch_fp",
        "frame_pitch_fn","offset_eligible","offset_correct")
POLICIES=("fundamental_consensus","fundamental_adaptive",
          "fundamental_strict")
def audit(role,r):
    def need(cond,message):
        if not cond:raise AssertionError(message)
    need(r.get("schema")=="S806_FIT80_FUNDAMENTAL_OCTAVE_CONTRAST_OOF_V1"
         and r.get("complete") is True,"real S806 report missing")
    need(r.get("track_count")==80 and len(r["compositions"])==5,
         "GuitarSet signed five FIT compositions missing")
    need(r["source_provenance"]=={"S800":39352880035,"S803":39363062723},
         "not signed GuitarSet OOF sources")
    need(set(r["source_counts"])==set(COUNTS) and
         r["source_counts"]["frames"]==306536 and
         r["source_counts"]["k_correct"]==189124,"12 FIT source counts drift")
    need(set(r["baseline_fit"])==set(SCORES),"six original scores missing")
    need(r["official_reference"]=="S708" and not r["official_promotion"]
         and not any(r[k] for k in ("CAL_opened","HELD_opened","TEST_opened")),
         "S708 or sealed partition broken")
    if role=="data":
        need(len(r["sampling_stats"])==80 and len({
            x["index"] for x in r["sampling_stats"]})==80,"train provenance incomplete")
        return {"signed80":True,"raw_count_parity":True}
    if role=="acoustics":
        need(r["feature_dimension"]==16 and len(r["features"])==16 and
             r["physical_spectral_proxy_from_frozen_S613_features"]
             and r["full_harmonic_spectrum_available"] is False,
             "full spectrum incorrectly claimed")
        need(r["runtime_truth_or_future_evidence"] is False,
             "future acoustic evidence")
        return {"frequency_octave_proxy_valid":True,
                "genuine_full_spectrum":"unavailable in S800"}
    if role=="architecture":
        need(r["heads_unchanged"]==[
            "general","bass","polyphonic","transient","harmonic"] and
            r["independent_fit80_acoustic_heads"]==5,
            "frozen five specialized heads modified")
        need(set(r["policies"])==set(POLICIES),
             "unauthorized candidate or grid tuning")
        return {"frozen_original_heads":True,
                "physical_head_independent":True}
    if role=="training":
        need(not r["input_truth_on_target_composition_at_training"]
             and not r["threshold_search_performed"],
             "target label leakage or post-hoc thresholds")
        need(set(r["training"])==set(r["compositions"]),"missing fold models")
        for c,row in r["training"].items():
            need(row["eval_composition"]==c and
                 len(row["training_compositions"])==4 and
                 c not in row["training_compositions"] and
                 row["positive"]>0 and row["negative"]>0 and
                 row["train_rows"]==row["positive"]+row["negative"] and
                 row["threshold_frozen_before_scoring"],
                 "bad composition disjointness")
        return {"compositions_heldout_from_fit":5}
    if role=="metrics":
        for policy,row in r["policy_results"].items():
            need(policy in POLICIES and
                set(row["metrics"])==set(SCORES) and
                set(row["delta"])==set(SCORES) and
                set(row["counts"])==set(COUNTS) and
                set(row["by_composition_delta"])==set(r["compositions"]),
                "six emitted MIDI measures or per-comp missing")
            safe=all(v>=-1e-12 for v in row["delta"].values()) and all(
                 all(v>=-1e-12 for v in entry.values()) for entry in
                    row["by_composition_delta"].values())
            need(safe==row["no_regression"],"non-regression verdict invalid")
            need((safe and any(v>1e-12 for v in row["delta"].values()))
                 ==row["fit_exploratory_eligible"],"candidate veto invalid")
        return {"global_and_per_comp_six_metric_veto_checked":True,
                "fit_eligible":r["fit_provisional_eligible"]}
    if role=="live":
        need(not r["production_LIVE_modified"] and
            not r["physical_live_latency_measured"] and
            r["added_latency_hold_samples"]==0 and
            r["added_lookahead_samples"]==0,
            "unproven LIVE latency/production claim")
        return {"zero_algorithmic_hold":True,
                "physical_iRig_untested":True}
    raise ValueError("unknown role")
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--report",type=Path,required=True)
    p.add_argument("--role",choices=ROLES,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    o=dict(schema="NOTE_SIX_ROLE_AUDIT_V1",role=args.role,
           passed=False,findings={},errors=[],
           experimental_promotion_allowed=False)
    try:
        o["findings"]=audit(args.role,json.loads(args.report.read_text()))
        o["passed"]=True
    except Exception as e:
        o["errors"]=[type(e).__name__+": "+str(e)]
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(o,indent=2)+"\n")
    print("S806_LOCAL",args.role,"PASS" if o["passed"] else "FAIL")
    if not o["passed"]:sys.exit(1)
if __name__=="__main__":main()
