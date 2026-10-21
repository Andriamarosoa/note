"""S833 cross-fitted, local CAL-only risk veto for Exact-K native PCM256.

Every CAL track is scored exactly once by rules trained on *other* CAL
tracks. The chosen risk settings must improve 3/4 heldout CAL partitions,
then final rules are trained on all CAL. OUTER ground truth is unavailable
at rule fit time. Designed for research after seeing S831/S832 FIT80;
therefore its FIT80 result is exploratory, NOT external validation.
"""
from __future__ import annotations
import numpy as np
from causal_note.s832_guarded_private_selector256 import (
    learn_rules,calibration_score,choose
)

SCHEMA="S833_EXACTK_CAL_CROSSFIT_4FOLD_RISK_PCM256_V1"
PARAMETERS=(
    (96,1.40,.57,12),
    (128,1.50,.58,15),
    (160,1.70,.59,18),
    (224,1.90,.60,20),
    (320,2.10,.62,30)
)

def fit_guard(cal_rows):
    """Select with 4 separate track-fold tests; fallback bit-exact S829.

    NOT a fully heldout experiment because global R&D already inspected
    GuitarSet FIT. This is internal repeated-CAL verification only.
    """
    if not cal_rows:raise ValueError("empty CAL")
    names=sorted(str(row["name"]) for row in cal_rows)
    if len(set(names))!=len(names) or len(names)<8:
        raise ValueError("need >=8 unique CAL tracks")
    n=min(4,len(names))
    assignment={name:i % n for i,name in enumerate(names)}
    attempts=[]
    winning=None
    for params in PARAMETERS:
        validated=[]
        for k in range(n):
            train=[r for r in cal_rows if assignment[r["name"]]!=k]
            test=[r for r in cal_rows if assignment[r["name"]]==k]
            if not train or not test:raise AssertionError("bad CAL partition")
            rules,_=learn_rules(train,params=params)
            metrics=calibration_score(test,rules)
            improvement=metrics["correct"]-metrics["reference_correct"]
            pertrack=metrics["track_comparisons"]
            negative_tracks=sum(z["improvement"]<0 for z in pertrack.values())
            positive_tracks=sum(z["improvement"]>0 for z in pertrack.values())
            validated.append(dict(fold=k,frames=metrics["frames"],
               corrected_net=improvement,rule_count=len(rules),
               negative_tracks=negative_tracks,positive_tracks=positive_tracks,
               worst_track_fraction=min(
                 z["improvement"]/max(1,z["frames"])
                 for z in pertrack.values())))
        wins=sum(r["corrected_net"]>0 for r in validated)
        net=sum(r["corrected_net"] for r in validated)
        negative=sum(r["negative_tracks"] for r in validated)
        checked_frames=sum(r["frames"] for r in validated)
        worst=min(r["worst_track_fraction"] for r in validated)
        # Depend on across-CAL generalization, not on a single 4-track
        # split; arbitrary rule changes are not allowed just to pass.
        accepted=(
            wins>=3 and net>max(15,int(checked_frames*.0003))
            and worst>=-.018 and negative<=len(names)//3
        )
        details=dict(params=list(params),folds=validated,
             positive_folds=wins,negative_tracks=negative,
             validation_delta=net,validation_frames=checked_frames,
             worst_track_fraction=worst,accepted=accepted)
        attempts.append(details)
        if accepted and (winning is None or
            (net,-negative,-sum(z["rule_count"] for z in validated))>
            winning[0]):
            winning=((net,-negative,
                 -sum(z["rule_count"] for z in validated)),params)
    if winning is None:
        return {},dict(schema=SCHEMA,reason="4fold_CAL_veto",
            guard_enabled=False,rule_count=0,tried=attempts)
    params=winning[1]
    final,all_rules=learn_rules(cal_rows,params=params)
    # Only accept a substantive policy after fit; an empty set is a
    # necessary and legitimate conservative fallback.
    if not final:
        return {},dict(schema=SCHEMA,reason="no_rule_after_full_CAL",
             guard_enabled=False,rule_count=0,tried=attempts)
    return final,dict(schema=SCHEMA,reason="4fold_CAL_positive",
         guard_enabled=True,rule_count=len(final),params=list(params),
         tried=attempts,rules=[v.record() for v in final.values()])
