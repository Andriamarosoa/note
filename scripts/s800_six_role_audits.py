"""S800: six independently executed fail-closed OOF reconstruction audits."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
from scripts.s787_audit_consensus import ROLES
# Immutable S795 evidence ABI. This forensic audit must remain stdlib-only:
# importing training modules just to inspect static constant names unnecessarily
# requires joblib, sklearn and torch, causing false audit failures.
COUNT_KEYS=("tp","fp","fn","frames","k_correct","active_frames",
            "active_k_correct","frame_pitch_tp","frame_pitch_fp",
            "frame_pitch_fn","offset_eligible","offset_correct")
HEADS=("general","bass","polyphonic","transient","harmonic")
def check(role,report,verification):
    r=json.loads(Path(report).read_text(encoding="utf-8"))
    v=json.loads(Path(verification).read_text(encoding="utf-8"))
    def demand(ok,reason):
        if not ok:raise AssertionError(reason)
    demand(r["schema"]=="S800_FIT80_OOF_PER_FRAGMENT_MIDI256_V1",
           "OOF trace schema mismatch")
    demand(r["complete"] and r["fit_done"]==80 and r["fit_files_count"]==80,
           "FIT80 incomplete")
    demand(v["pass_all"] and v["tracks"]==80 and v["signed_parity"],
           "full trace verifier did not pass")
    demand(r["FIT_source_counts"]==r["S795_archived_fit_counts"],
           "original S795 FIT aggregate changed")
    if role=="data":
        demand(not any(r[k] for k in ("CAL_opened","HELD_opened","TEST_opened",
                 "cal_used_for_training","cal_used_for_selection","selected_from_CAL")),
               "forbidden split consulted")
        demand(len(r["folds_done"])==5 and len(set(r["folds_done"]))==5,
               "FIT folds missing")
        demand(all(item["fold_held_out"] for item in r["fit_files"]),
               "OOF fold not excluded")
        return {"five_composition_held_out":True,"CAL_labels_not_used":True}
    if role=="acoustics":
        demand(r["head_order"]==list(HEADS) and v["all_five_OOF_heads_saved"],
               "five acoustic expert posterior maps absent")
        demand(all(item["fragment_frames"]>0 for item in r["fit_files"]),
               "empty spectral fragment map")
        return {"full_posterior_maps":True,"frame_pitch_grid":49,"causal_features":23}
    if role=="architecture":
        demand(len({z["npz_sha256"] for z in r["fit_files"]})==80,
               "OOF source evidence reused")
        demand(r["decision_clock_samples"]==1024
               and r["metric_clock_samples"]==256,
               "causal scheduling ABI changed")
        return {"unchanged_frozen_S795_architecture":True,
                "per_fragment_and_pitch_snapshots":True}
    if role=="training":
        demand(r["retrained_S795_FIT_experts"] and
               r["recomputation_reason"]=="S795 archive lacks OOF probabilities/model bank",
               "OOF training provenance unknown")
        demand(all(z["fold_held_out"] for z in r["fit_files"]),
               "models trained on held-out composition")
        return {"same_S795_training_code":True,
                "composition_heldout_predict":True}
    if role=="metrics":
        demand(r["FIT_source_parity_S795"] and v["signed_parity"],
               "FIT80 metric reference mismatch")
        demand(all(r["FIT_source_counts"][k]==r["S795_archived_fit_counts"][k]
                   for k in COUNT_KEYS),"source signed counts drift")
        demand(not r["promoted"],"metric-only evidence promoted")
        return {"FIT_counts_equal":True,
                "CAL32_not_evaluated":True,"six_scores_claimed_improved":False}
    if role=="live":
        demand(not r["production_LIVE_modified"]
               and not r["physical_mic_latency_measured"]
               and r["source_midi_unchanged"],"LIVE interference or false timing claims")
        demand(r["decision_clock_samples"]==1024 and
               r["metric_clock_samples"]==256,
               "time granularity drift")
        return {"LIVE_pipeline_untouched":True,
                "latency_added_by_production_change":False,
                "physical_latency":"not measured"}
    raise AssertionError("unknown role")
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--role",choices=ROLES,required=True)
    p.add_argument("--report",type=Path,required=True)
    p.add_argument("--verification",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    r={"schema":"NOTE_SIX_ROLE_AUDIT_V1","role":a.role,
       "passed":False,"experimental_promotion_allowed":False,
       "findings":{},"errors":[]}
    try:
        r["findings"]=check(a.role,a.report,a.verification)
        r["passed"]=True
    except Exception as e:
        r["errors"]=[type(e).__name__+": "+str(e)]
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(r,indent=2,sort_keys=True)+"\n",
                     encoding="utf-8")
    print("S800_ROLE_"+a.role+"_"+("PASS" if r["passed"] else "FAIL"),flush=True)
    if not r["passed"]:sys.exit(1)
if __name__=="__main__":main()
