"""S771 exact analytical audit of S770 65 frozen CAL32 onset candidates.

No acoustic inference, no new calibration on CAL, no score promotion.
Uses the existing published S770 diagnostic report and signed S708.
"""
import argparse
import ast
import json
import math
from pathlib import Path

def tp_needed(fp, truth, target=0.8):
    # strictly greater than target: (2-target)*tp > target*(fp+truth)
    return math.floor(target*(fp+truth)/(2-target))+1

def fp_allowed(tp, truth, target=0.8):
    # strictly greater than target: fp < (2/target -1)*tp - truth
    limit=(2/target-1)*tp-truth
    return math.ceil(limit)-1

def audit(r, signed):
    assert r["schema"]=="S770_GROUPED_JOINT_ONSET_SHADOW"
    assert r["fit_recordings_done"]==80 and r["cal_recordings_done"]==32
    assert not r["promoted"] and r["source_midi_immutable"]
    assert r["ONSET_only_not_MIDI"] and not r["TEST_opened"] and not r["HELD_opened"]
    assert len(r["GRID"])==65 and len(r["CAL"]["onset_shadow_grid"])==65
    assert len(r["FIT"]["onset_oof"])==65 and len(r["FIT"]["folds"])==5
    reference=signed["frozen_candidate"]["CAL"]
    trials=r["CAL"]["onset_shadow_grid"]
    fit=r["FIT"]["onset_oof"]
    truth=r["CAL"]["native_onset"]["TP"]+r["CAL"]["native_onset"]["FN"]
    assert truth==4083
    assert r["CAL"]["native_onset"]["TP"]==reference["tp"]
    assert r["CAL"]["native_onset"]["FP"]==reference["fp"]
    assert r["CAL"]["native_onset"]["FN"]==reference["fn"]
    frozen=r["FIT"]["selected_frozen"]
    assert frozen==r["CAL"]["selected_frozen"]
    assert frozen in fit and frozen in trials
    native_only={}
    all_rows=[]
    for name,score in trials.items():
        q,b,rise=ast.literal_eval(name)
        assert (q,b,rise) in {(float(x),float(y) if y is not None else None,float(z))
                              for x,y,z in (ast.literal_eval(e) for e in r["GRID"])}
        assert score["TP"]+score["FN"]==truth
        if b is None:
            native_only[q]=score
        all_rows.append((q,b,rise,name,score))
    assert len(native_only)==5
    curves=[]
    for q,b,rise,name,score in all_rows:
        tp,fp,fn=(int(score[k]) for k in ("TP","FP","FN"))
        entry=dict(choice=name,native_quality_threshold=q,birth_threshold=b,
            spectral_rise_floor=rise,TP=tp,FP=fp,FN=fn,
            precision=float(score["precision"]),recall=float(score["recall"]),
            f1=float(score["f1"]),fit_oof_f1=float(fit[name]["f1"]),
            additional_TP_vs_native_at_same_q=None,additional_FP_vs_native_at_same_q=None,
            incremental_precision_vs_same_q=None,
            new_TP_needed_at_fixed_FP=max(0,tp_needed(fp,truth)-tp),
            FP_to_remove_at_fixed_TP=max(0,fp-fp_allowed(tp,truth)))
        if b is not None:
            base=native_only[q]
            dtp=tp-base["TP"];dfp=fp-base["FP"]
            entry.update(additional_TP_vs_native_at_same_q=dtp,
                         additional_FP_vs_native_at_same_q=dfp,
                         incremental_precision_vs_same_q=(
                         dtp/(dtp+dfp) if dtp>=0 and dfp>=0 and dtp+dfp>0 else None))
        curves.append(entry)
    by_q=[]
    for q,base in sorted(native_only.items()):
        extra=[e for e in curves if e["native_quality_threshold"]==q and e["birth_threshold"] is not None]
        top=max(extra,key=lambda e:e["f1"])
        by_q.append(dict(quality_threshold=q,native_only_f1=base["f1"],
                         best_birth_f1=top["f1"],best_birth_choice=top["choice"],
                         best_birth_gain=top["f1"]-base["f1"],
                         birth_reaches_base=top["f1"]>=base["f1"],
                         added_TP=top["additional_TP_vs_native_at_same_q"],
                         added_FP=top["additional_FP_vs_native_at_same_q"],
                         birth_incremental_precision=top["incremental_precision_vs_same_q"],
                         necessary_incremental_precision_for_gain=base["f1"]/2))
    parity={k:r["source"][k]-reference[k] for k in
            ("onset_f1","exact_k","active_exact_k","offset_accuracy","full_note_f1","frame_pitch_f1")}
    # Signed parity is intentionally NOT rounded or relaxed.
    return dict(schema="S771_S770_ONSET_COMBINATION_FORENSICS",
        complete=True,source_run=39245797266,source_attempt=1,
        diagnostic_only=True,heavily_exposed_CAL=True,promoted=False,
        S708_unchanged=True,TEST_opened=False,HELD_opened=False,
        frozen_choice=frozen,selected_CAL_f1=trials[frozen]["f1"],
        official_S708_onset_f1=reference["onset_f1"],
        selected_delta=trials[frozen]["f1"]-reference["onset_f1"],
        signed_source_parity_valid=all(abs(x)<1e-8 for x in parity),
        signed_source_parity_delta=parity,
        signed_source_exactk_correct_delta=round(parity["exact_k"]*reference["frames"]),
        signed_source_activek_correct_delta=round(parity["active_exact_k"]*reference["active_frames"]),
        true_onsets=truth,number_of_combinations=len(curves),
        selected=next(e for e in curves if e["choice"]==frozen),
        best_CAL_diagnostic=max(curves,key=lambda e:e["f1"]),
        by_native_quality=by_q,number_of_q_where_additions_beat_no_birth=sum(
            x["birth_reaches_base"] for x in by_q),
        combinations=curves,
        decision=("REJECT_S770_AND_NEW_HEAD_GENERATOR_REQUIRED"
                  if not all(abs(x)<1e-8 for x in parity)
                  or trials[frozen]["f1"]<=reference["onset_f1"] else
                  "ONSET_ONLY_SHADOW_GAIN_NOT_MIDI_PROMOTION"))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--s770-report",required=True,type=Path)
    ap.add_argument("--s708-report",required=True,type=Path)
    ap.add_argument("--out",required=True,type=Path)
    args=ap.parse_args()
    result=audit(json.loads(args.s770_report.read_text()),
                 json.loads(args.s708_report.read_text()))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print("S771_SUMMARY",json.dumps({k:result[k] for k in
        ("decision","official_S708_onset_f1","selected_CAL_f1",
         "signed_source_exactk_correct_delta",
         "signed_source_activek_correct_delta",
         "number_of_q_where_additions_beat_no_birth")}),flush=True)

if __name__=="__main__":
    main()
