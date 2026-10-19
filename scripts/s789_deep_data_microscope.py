"""S789 deep data-layer audit: six independent roles on immutable GitHub artifacts.

All values are descriptive from previously exposed S708 DEV/HELD/CAL or
S781/S782 diagnostics. No inference, hyperparameter choice, training or LIVE
modification. Absence of CAL32 per-track S708 final-MIDI events is itself
reported as an audit coverage limitation.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import json
import math
from pathlib import Path
import random
import statistics
import sys
from scripts.s787_audit_consensus import ROLES

SCORES=("onset_f1","exact_k","active_exact_k","offset_accuracy",
        "full_note_f1","frame_pitch_f1")
RUNS={"S708":"39094682419","S781":"39256936602","S782":"39257378677"}
NO_PROMOTION=True

def ensure(condition,message):
    if not condition:raise AssertionError(message)

def close(value,expected,what,tol=1e-9):
    ensure(math.isclose(value,expected,abs_tol=tol,rel_tol=0),f"{what}: {value} != {expected}")

def ratio(n,d):
    return n/max(d,1)

def f1(tp,fp,fn):
    return (2*tp)/max(2*tp+fp+fn,1)

def min_correct_to_exceed_80(correct,n):
    return max(0,math.floor(.8*n)+1-correct)

def min_recovered_f1(tp,fp,fn):
    """How many existing false negatives converted into TP, no extra FP?"""
    for added in range(fn+1):
        if f1(tp+added,fp,fn-added)>.8:return added
    return None

def wilson(x,n,z=1.96):
    p=ratio(x,n)
    denom=1+z*z/max(n,1)
    center=(p+z*z/(2*max(n,1)))/denom
    half=z*math.sqrt((p*(1-p)+z*z/(4*max(n,1)))/max(n,1))/denom
    return [max(0.,center-half),min(1.,center+half)]

def track_texture(name):
    if name.endswith("_comp"):return "comp"
    if name.endswith("_solo"):return "solo"
    raise AssertionError(f"unknown texture: {name}")

def source(args):
    s=json.loads((args.s708/"report.json").read_text())
    o=json.loads((args.s781/"report.json").read_text())
    a=json.loads((args.s782/"report.json").read_text())
    ensure(s["schema"]=="S708_K_GUIDED_S706_CAL_LOOPS","S708 source mismatch")
    ensure(o["schema"]=="S781_KTAIL_COMPOSITION_AND_TEXTURE_AUDIT","S781 mismatch")
    ensure(a["schema"]=="S782_MISSED_ONSET_SPECTRAL_TAXONOMY","S782 mismatch")
    ensure(s["checkpoint_sha256"]==__import__("hashlib").sha256(
        (args.s708/"s708_selected.pt").read_bytes()).hexdigest(),
        "S708 signed checkpoint SHA mismatch")
    ensure(not o["promoted"] and not a["promoted"],"experimental data promoted")
    dev=json.loads((args.s708/"dev_audit.json").read_text())
    held=json.loads((args.s708/"held_audit.json").read_text())
    return s,o,a,dev,held

def records_by(rows,key):
    groups=defaultdict(list)
    for r in rows:groups[key(r)].append(r)
    return groups

def aggregate(rows):
    m=[r["metrics"]["candidate"] for r in rows]
    tp=sum(x["tp"] for x in m);fp=sum(x["fp"] for x in m);fn=sum(x["fn"] for x in m)
    off=sum(x["offset_correct"] for x in m)
    elig=sum(x["offset_eligible"] for x in m)
    pitch_tp=sum(x["frame_pitch_tp"] for x in m)
    pitch_fp=sum(x["frame_pitch_fp"] for x in m)
    pitch_fn=sum(x["frame_pitch_fn"] for x in m)
    kc=sum(x["k_correct"] for x in m)
    frames=sum(x["frames"] for x in m)
    ac=sum(x["active_k_correct"] for x in m)
    af=sum(x["active_frames"] for x in m)
    return {"n_tracks":len(rows),"TP":tp,"FP":fp,"FN":fn,
            "offset_correct":off,"offset_eligible":elig,
            "offset_accuracy":ratio(off,elig),
            "onset_f1":f1(tp,fp,fn),
            "frame_pitch_f1":f1(pitch_tp,pitch_fp,pitch_fn),
            "exact_k":ratio(kc,frames),"active_exact_k":ratio(ac,af),
            "macro_offset_accuracy":statistics.mean(x["offset_accuracy"] for x in m),
            "macro_frame_pitch_f1":statistics.mean(x["frame_pitch_f1"] for x in m),
            "macro_onset_f1":statistics.mean(x["onset_f1"] for x in m)}
def bootstrap_track_offset(rows,seed):
    """Descriptive track bootstrap; tracks within composition NOT independent."""
    rng=random.Random(seed)
    n=len(rows)
    ratios=[]
    for _ in range(500):
        sample=[rows[rng.randrange(n)] for _ in range(n)]
        n_=sum(x["metrics"]["candidate"]["offset_correct"] for x in sample)
        d_=sum(x["metrics"]["candidate"]["offset_eligible"] for x in sample)
        ratios.append(ratio(n_,d_))
    ratios.sort()
    return [ratios[12],ratios[487]]

def audit_data(s,o,a,dev,held):
    ensure(len(dev)==12 and len(held)==32,"per-recording count drift")
    out={}
    for split,rows in (("DEV",dev),("HELD",held)):
        names=[x["name"] for x in rows]
        ensure(len(set(names))==len(names),f"{split}: duplicate track names")
        for x in rows:
            track_texture(x["name"])
            m=x["metrics"]["candidate"]
            close(m["onset_f1"],f1(m["tp"],m["fp"],m["fn"]),x["name"]+" onset")
            close(m["offset_accuracy"],ratio(m["offset_correct"],m["offset_eligible"]),
                  x["name"]+" offset")
            close(m["frame_pitch_f1"],f1(m["frame_pitch_tp"],
                    m["frame_pitch_fp"],m["frame_pitch_fn"]),x["name"]+" frame pitch")
            close(m["exact_k"],ratio(m["k_correct"],m["frames"]),x["name"]+" exact K")
        bt=records_by(rows,lambda r:track_texture(r["name"]))
        bc=records_by(rows,lambda r:r["composition"])
        agg=aggregate(rows)
        signed=s[split]["candidate"]
        for key in ("onset_f1","exact_k","active_exact_k",
                    "offset_accuracy","frame_pitch_f1"):
            close(agg[key],signed[key],f"{split} global {key}")
        out[split]=dict(global_metrics=agg,
             by_texture={k:aggregate(v) for k,v in sorted(bt.items())},
             by_composition={k:aggregate(v) for k,v in sorted(bc.items())},
             offset_track_bootstrap95=bootstrap_track_offset(rows,789),
             worst_five_offset=[dict(track=r["name"],
                           offset_accuracy=r["metrics"]["candidate"]["offset_accuracy"],
                           eligible=r["metrics"]["candidate"]["offset_eligible"])
                          for r in sorted(rows,key=lambda r:
                           r["metrics"]["candidate"]["offset_accuracy"])[:5]],
             best_five_frame_pitch=[dict(track=r["name"],
                           f1=r["metrics"]["candidate"]["frame_pitch_f1"])
                          for r in sorted(rows,key=lambda r:
                           -r["metrics"]["candidate"]["frame_pitch_f1"])[:5]])
    ensure(s["DEV_previously_exposed"] and s["HELD_previously_exposed"],
           "previous exposure flags changed")
    ensure(not {x["name"] for x in dev}&{x["name"] for x in held},
           "DEV/HELD share track filenames")
    return dict(splits=out,cal32_individual_final_midi_events_available=False,
        cal32_coverage_warning="S708 artifact has aggregated CAL scores but lacks CAL32 per-track final-MIDI audit: cannot attribute offset failure by CAL track",
        held_previously_exposed=True,
        data_quality_findings="Between-composition transfer is not established by 32 HELD tracks from only two compositions")

def audit_acoustics(s,o,a,dev,held):
    x=a["results"]
    overall=x["all"]
    ensure(overall["total"]==4083 and overall["matched"]==2913 and overall["missed"]==1170,
           "S782 event counts deviate from S708")
    for category in ("by_texture","by_register","by_poly","by_harmonic_overlap",
                     "by_composition"):
        ensure(sum(v["total"] for v in x[category].values())==4083,
               f"{category}: not a partition")
        ensure(sum(v["missed"] for v in x[category].values())==1170,
               f"{category}: missed mismatch")
    ensure(a["diagnostic_uses_three_post_attack_frames"],
           "S782 onset window timing metadata disappeared")
    ensure(a["labels_used_only_after_prediction"],
           "S782 acoustic diagnostic cannot see truth at inference")
    groups={}
    for c in ("by_texture","by_register","by_poly","by_harmonic_overlap"):
        groups[c]={k:dict(total=v["total"],missed=v["missed"],
                     recall=v["recall"],
                     recall_wilson95=wilson(v["matched"],v["total"]),
                     missed_strong_both=v["missed_strong_both"],
                     missed_onset_posterior_above_0_5=v["missed_onset_above"]["0.5"])
                   for k,v in x[c].items()}
    low=x["by_register"]["E2_B2"]
    high=x["by_register"]["C4_E5"]
    return dict(by_group=groups,
                low_string_recall_deficit_pts=(high["recall"]-low["recall"])*100,
                harmonic_neighbor_recall_deficit_pts=(
                    x["by_harmonic_overlap"]["no_harmonic_neighbor"]["recall"]-
                    x["by_harmonic_overlap"]["harmonic_neighbor"]["recall"])*100,
                diagnostic_lookahead_risk="S782 uses 3 post-attack decision frames, FORBIDDEN as online onset rescue evidence",
                no_proven_safe_rescue_count=overall["missed_strong_both"])

def audit_architecture(s,o,a,dev,held):
    cal=o["CAL"]
    for key in ("baseline","selected"):
        m=cal[key]
        matrix=m["confusion"]
        ensure(len(matrix)==7 and all(len(row)==7 for row in matrix),
               "invalid S781 K confusion")
        ensure(sum(sum(row) for row in matrix)==m["frames"],"confusion support mismatch")
        ensure(sum(matrix[k][k] for k in range(7))/m["frames"]==m["exact_accuracy"],
               "K accuracy computed incorrectly")
    h=cal["tail_histogram_specialist"]
    for k in (5,6):
        ensure(sum(h[str(k)])==cal["selected"]["per_k"][k]["n"],
               "tail histogram mismatch")
    ensure(o["decision_clock_samples"]==1024 and not o["final_MIDI_changed"],
           "isolated head is not allowed to replace MIDI")
    return dict(cal_base_shadow=cal["baseline"]["exact_accuracy"],
                cal_trial_shadow=cal["selected"]["exact_accuracy"],
                delta_pts=(cal["selected"]["exact_accuracy"]-
                           cal["baseline"]["exact_accuracy"])*100,
                per_k5=cal["selected"]["per_k"][5],
                per_k6=cal["selected"]["per_k"][6],
                k6_assigned_k5=h["6"][5],
                k6_assigned_k6=h["6"][6],
                cal_compositions={
                    key:{c:dict(k5=m["overall"]["per_k"][5],
                                k6=m["overall"]["per_k"][6])
                         for c,m in cal[key].items()}
                    for key in ("by_composition_base","by_composition_specialist")},
                warning="1024-sample head data, not 256-sample official MIDI")

def audit_training(s,o,a,dev,held):
    fit=o["FIT"];cal=o["CAL"]
    ensure(o["fit_recordings_done"]==80 and o["cal_recordings_done"]==32,
           "FIT/CAL incomplete")
    ensure(len(fit["folds"])==5 and set(fit["folds"])==set(fit["baseline_by_composition"]),
           "not composition held-out FIT")
    ensure(o["no_CAL_training_or_selection"],"CAL tuning detected")
    best=fit["selected"]
    ensure(best==o["selected_tail_weight"],"selected training weight inconsistent")
    fit_candidate=fit["trial_ood"][str(best)]
    out={}
    for k in (5,6):
        ff=fit_candidate["per_k"][k]
        cc=cal["selected"]["per_k"][k]
        out[f"K{k}"]=dict(FIT_n=ff["n"],FIT_recall=ff["accuracy"],
                        CAL_n=cc["n"],CAL_recall=cc["accuracy"],
                        drop_points=(ff["accuracy"]-cc["accuracy"])*100)
    return dict(fit_compositions=sorted(fit["folds"]),
                cal_compositions=sorted(cal["by_composition_specialist"]),
                oof_selected_tail_weight=best,
                fit_exact_k=fit_candidate["exact_accuracy"],
                cal_exact_k=cal["selected"]["exact_accuracy"],
                k5_k6_cross_composition_shift=out,
                warning="FIT composition OOF is not proof CAL transfer; prior CAL exposure prohibits fitting gates there")

def audit_metrics(s,o,a,dev,held):
    c=s["frozen_candidate"]["CAL"]
    ensure(not c["all_targets_above_80"] and not s["promoted"],
           "unverified official promotion")
    close(c["offset_accuracy"],ratio(c["offset_correct"],c["offset_eligible"]),
          "CAL offset conditional")
    close(c["onset_f1"],f1(c["tp"],c["fp"],c["fn"]),"CAL onset")
    close(c["exact_k"],ratio(c["k_correct"],c["frames"]),"CAL exact K")
    close(c["active_exact_k"],ratio(c["active_k_correct"],c["active_frames"]),
          "CAL active exact K")
    close(c["frame_pitch_f1"],f1(c["frame_pitch_tp"],
          c["frame_pitch_fp"],c["frame_pitch_fn"]),"CAL frame pitch")
    vals={metric:c[metric] for metric in SCORES}
    gap={metric:(.8-c[metric])*100 for metric in SCORES}
    devc=s["DEV"]["candidate"];heldc=s["HELD"]["candidate"]
    cal_dev_held={metric:dict(CAL32=c[metric],DEV=devc[metric],
                        HELD=heldc[metric]) for metric in SCORES}
    return dict(official_S708_six=vals,points_to_80=gap,
          by_split=cal_dev_held,
          metric_nearest_on_CAL=min(SCORES,key=lambda k:gap[k]),
          metric_already_over_80_on_exposed_HELD=[
              metric for metric in SCORES if heldc[metric]>.8],
          offset_additional_correct_at_fixed_denominator=min_correct_to_exceed_80(
              c["offset_correct"],c["offset_eligible"]),
          exact_k_additional_correct_at_fixed_denominator=min_correct_to_exceed_80(
              c["k_correct"],c["frames"]),
          active_k_additional_correct_at_fixed_denominator=min_correct_to_exceed_80(
              c["active_k_correct"],c["active_frames"]),
          onset_recovered_misses_needed_if_FP_unchanged=min_recovered_f1(
              c["tp"],c["fp"],c["fn"]),
          frame_pitch_recovered_misses_needed_if_FP_unchanged=min_recovered_f1(
              c["frame_pitch_tp"],c["frame_pitch_fp"],c["frame_pitch_fn"]),
          note="Offset denominator conditional on already-correct onsets; HELD is previously exposed and not a new blind holdout")

def audit_live(s,o,a,dev,held):
    ensure(s["official_LIVE_changed"] is False,"S708 official LIVE was changed")
    ensure(all(not o[x] for x in ("LIVE_changed","ONSET_OFFSET_changed",
                                  "final_MIDI_changed","promoted")),
           "S781 touched LIVE")
    ensure(not a["LIVE_changed"] and not a["source_MIDI_immutable"] is False,
           "S782 source MIDI changed")
    ensure(a["diagnostic_not_realtime_prediction"],"S782 truth diagnostic misused")
    return dict(unchanged_signed_reference=True,
        no_new_LIVE_runtime_measurements=True,
        no_model_training=True,
        no_code_changes_to_live=True,
        forbidden_to_promote_shadow_K=True,
        forbidden_to_promote_post_attack_window=True,
        warning="No microphone latency or 256-hop integrated MIDI measured by S789")

HANDLERS=dict(data=audit_data,acoustics=audit_acoustics,
              architecture=audit_architecture,training=audit_training,
              metrics=audit_metrics,live=audit_live)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--role",choices=ROLES,required=True)
    parser.add_argument("--s708",type=Path,required=True)
    parser.add_argument("--s781",type=Path,required=True)
    parser.add_argument("--s782",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    result=dict(schema="NOTE_SIX_ROLE_AUDIT_V1",role=args.role,
        passed=False,experimental_promotion_allowed=False,
        findings={},errors=[],source_runs=RUNS)
    try:
        findings=HANDLERS[args.role](*source(args))
        result["findings"]=findings
        result["passed"]=True
    except Exception as error:
        result["errors"]=[f"{type(error).__name__}: {error}"]
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    print("S789_DEEP_"+args.role.upper(),
          json.dumps(result,sort_keys=True),flush=True)
    if not result["passed"]:
        raise SystemExit(1)

if __name__=="__main__":
    main()
