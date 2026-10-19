"""S780: evaluate distribution shift and class-tail transfer in signed S779 audit.

Consumes only the immutable S779 report; no refit or CAL threshold tuning.
The 1024-sample shadow metrics are not final-MIDI 256-sample Exact-K.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

TAIL=(5,6)

def dist_summary(metrics):
    return {
        "frames":int(metrics["frames"]),
        "exact_k":float(metrics["exact_accuracy"]),
        "per_k":{
            str(k):{
                "n":int(metrics["per_k"][k]["n"]),
                "correct":int(metrics["per_k"][k]["correct"]),
                "recall":metrics["per_k"][k]["accuracy"],
                "under":int(metrics["per_k"][k]["under"]),
                "major_under":int(metrics["per_k"][k]["major_under"]),
                "over":int(metrics["per_k"][k]["over"])
            } for k in TAIL
        }
    }

def audit(report):
    if report.get("schema")!="S779_CLASS_AWARE_EXACTK_TAIL_SPECIALIST_FIT80":
        raise ValueError("S780 input is not S779 signed audit")
    if not report.get("complete") or report.get("promoted") or report.get("final_MIDI_changed"):
        raise AssertionError("S780 incomplete/nonbaseline S779 input")
    if report["fit_recordings_done"]!=80 or report["cal_recordings_done"]!=32:
        raise AssertionError("S780 FIT80/CAL32 record count mismatch")
    if report["decision_clock_samples"]!=1024 or not report["no_CAL_training_or_selection"]:
        raise AssertionError("S780 invalid independent K selection")
    weight=report["selected_tail_weight"]
    fit=report["FIT"]["trial_ood"][str(weight)]
    fit_base=report["FIT"]["baseline"]
    cal=report["CAL"]["selected"]
    cal_base=report["CAL"]["baseline"]
    comp=report["CAL"]["comp"];solo=report["CAL"]["solo"]
    if comp["frames"]+solo["frames"]!=cal["frames"]:
        raise AssertionError("S780 CAL texture split incomplete")
    class_gaps={}
    for k in TAIL:
        train=fit["per_k"][k];held=cal["per_k"][k]
        if train["n"]==0 or held["n"]==0:
            raise AssertionError(f"S780 missing tail class K{k} in FIT or CAL")
        class_gaps[str(k)]=dict(
            fit_support=int(train["n"]),cal_support=int(held["n"]),
            fit_recall=float(train["accuracy"]),
            cal_recall=float(held["accuracy"]),
            recall_drop_points=100*(float(train["accuracy"])-float(held["accuracy"])),
            fit_correct=int(train["correct"]),cal_correct=int(held["correct"]),
            cal_delta_correct=int(held["correct"]-cal_base["per_k"][k]["correct"]),
            cal_under=int(held["under"]),
            cal_major_under=int(held["major_under"]))
    return dict(
        schema="S780_S779_CLASS_TAIL_DOMAIN_SHIFT_AUDIT",
        complete=True,promoted=False,selected="S708_frozen",
        source_run=39253787743,source_artifact_id=12187441565,
        evaluation_cadence_samples=1024,
        caveat="Shadow K on already exposed CAL32, NOT official 256-sample final-MIDI Exact-K",
        selected_FIT_weight=weight,fit_compositions=list(report["FIT"]["folds"]),
        selected_FIT_oof=dist_summary(fit),
        selected_CAL=dist_summary(cal),
        FIT_baseline=dist_summary(fit_base),
        CAL_baseline=dist_summary(cal_base),
        CAL_comp=dist_summary(comp),
        CAL_solo=dist_summary(solo),
        generalization_gap_points=100*(fit["exact_accuracy"]-cal["exact_accuracy"]),
        baseline_generalization_gap_points=100*(fit_base["exact_accuracy"]-cal_base["exact_accuracy"]),
        CAL_gain_points=100*(cal["exact_accuracy"]-cal_base["exact_accuracy"]),
        tail=class_gaps,
        K6_failure="CAL K6 is heavily undercounted despite high FIT OOF recall",
        next_protocol=[
            "Inspect held-out composition-by-composition K5/K6 errors, not FIT pooled only",
            "Freeze FIT composition-wise worst-tail recall and overall Exact-K floor before CAL",
            "Use class-conditional spectral acoustic evidence for K5/K6; do not calibrate on CAL32",
            "Require evaluation on new untouched compositions before claiming generalization",
            "No promotion until one causal MIDI stream clears all six official non-regression checks"
        ])
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--s779-report",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    d=audit(json.loads(a.s779_report.read_text()))
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(d,indent=2,sort_keys=True)+"\n")
    print("S780_RESULT",json.dumps(dict(weight=d["selected_FIT_weight"],
        fit=d["selected_FIT_oof"]["exact_k"],cal=d["selected_CAL"]["exact_k"],
        cal_gain_points=d["CAL_gain_points"],tail=d["tail"])),flush=True)

if __name__=="__main__":
    main()
