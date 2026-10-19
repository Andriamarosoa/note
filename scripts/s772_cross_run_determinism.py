"""S772: post-hoc reproducibility audit of two unmodified S770 GitHub runs.

No calibration, no training, no CAL selection, no model promotion.
Input: the exact published report.json files for attempts 1 and 2.
"""
import argparse
import json
from pathlib import Path

SIX=("onset_f1","exact_k","active_exact_k","offset_accuracy",
     "full_note_f1","frame_pitch_f1")
COUNTS=("TP","FP","FN")

def compare(a,b):
    for r in (a,b):
        assert r["schema"]=="S770_GROUPED_JOINT_ONSET_SHADOW"
        assert len(r["GRID"])==65 and r["fit_recordings_done"]==80
        assert r["cal_recordings_done"]==32 and len(r["FIT"]["folds"])==5
        assert not r["promoted"] and r["selected"]=="S708_frozen"
    fit1=a["FIT"]["onset_oof"];fit2=b["FIT"]["onset_oof"]
    cal1=a["CAL"]["onset_shadow_grid"];cal2=b["CAL"]["onset_shadow_grid"]
    assert set(fit1)==set(fit2)==set(cal1)==set(cal2)
    assert a["FIT"]["selected_frozen"]==b["FIT"]["selected_frozen"]
    frozen=a["FIT"]["selected_frozen"]
    selection_delta={k:int(b["CAL"]["frozen_score"][k]-a["CAL"]["frozen_score"][k])
                     for k in COUNTS}
    diffs={}
    for key in cal1:
        old,new=cal1[key],cal2[key]
        diffs[key]={k:int(new[k]-old[k]) for k in COUNTS}
    fitted={}
    for key in fit1:
        old,new=fit1[key],fit2[key]
        fitted[key]={k:int(new[k]-old[k]) for k in COUNTS}
    drift1=a["source_reference_parity"]
    drift2=b["source_reference_parity"]
    assert a["source"]["onset_f1"]==b["source"]["onset_f1"]
    return dict(
        schema="S772_CROSS_RUN_DETERMINISM_AUDIT",
        complete=True,promoted=False,TEST_opened=False,HELD_opened=False,
        training_changed=False,source_sha_unchanged=True,
        comparison="S770 attempts 1 vs 2; same code commit and 65-fold grid",
        reference_parity_first=drift1,reference_parity_second=drift2,
        first_selected=a["CAL"]["frozen_score"],
        second_selected=b["CAL"]["frozen_score"],
        frozen_config=frozen,frozen_config_identical=True,
        fit_training_counts_equal=a["FIT"]["counts"]==b["FIT"]["counts"],
        fit_scores_changed=sum(fit1[k]!=fit2[k] for k in fit1),
        cal_scores_changed=sum(cal1[k]!=cal2[k] for k in cal1),
        max_abs_fit_f1_change=max(abs(fit1[k]["f1"]-fit2[k]["f1"]) for k in fit1),
        max_abs_cal_f1_change=max(abs(cal1[k]["f1"]-cal2[k]["f1"]) for k in cal1),
        native_source_six_metric_delta={k:float(b["source"][k]-a["source"][k]) for k in SIX},
        frozen_counts_change=selection_delta,
        exact_changed_fit=fitted,exact_changed_cal=diffs,
        next_actions=[
            "Record per-track MIDI stream digests and event-level differences BEFORE labels.",
            "Repeat frozen S708 acoustic inference multiple times on same track and compare logits, timestamps and count frames.",
            "Compare float16 feature caches, PyTorch CPU threads and decision margins near threshold; do not guess numerical root cause.",
            "Block promotion on bitwise or signed reference mismatch and validate all six final MIDI metrics on unseen recordings."
        ])

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--first",type=Path,required=True)
    ap.add_argument("--second",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    opt=ap.parse_args()
    r=compare(json.loads(opt.first.read_text()),
              json.loads(opt.second.read_text()))
    opt.out.parent.mkdir(parents=True,exist_ok=True)
    opt.out.write_text(json.dumps(r,indent=2,sort_keys=True)+"\n")
    print("S772_CROSS_RUN",json.dumps({k:r[k] for k in
        ("frozen_config_identical","fit_scores_changed",
         "cal_scores_changed","frozen_counts_change",
         "max_abs_cal_f1_change")}),flush=True)

if __name__=="__main__":
    main()
