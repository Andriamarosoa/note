"""Development-only onset Exact-K diagnostic, distinct from S80 and active K.

Grouping onsets within 30ms is a *declared* approximation for simultaneously
attacked chords, not a claim of exact physical guitar-strum segmentation.
Each matched onset group is counted K-correct iff the numbers of distinct
heights are equal. Exact pitch-set equality is reported separately.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment


def group_onsets(events, *, window_s=.030):
    notes = sorted(
        (float(n["time"]), int(n["pitch"]))
        for n in events if n.get("type", "note_on") == "note_on"
    )
    groups = []
    for t, p in notes:
        if groups and t - groups[-1]["time_s"] <= window_s:
            groups[-1]["pitches"].add(p)
        else:
            groups.append(dict(time_s=t, pitches={p}))
    return groups


def exact_k_onset_audit(predicted, truth, *, group_window_s=.030, match_window_s=.080):
    refs = group_onsets(truth, window_s=group_window_s)
    outputs = group_onsets(predicted, window_s=group_window_s)
    paired = {}
    if refs and outputs:
        dist = np.abs(np.asarray([r["time_s"] for r in refs])[:,None] -
                      np.asarray([p["time_s"] for p in outputs])[None,:])
        rids, pids = linear_sum_assignment(np.where(dist <= match_window_s, dist, 1e6))
        for i,j in zip(rids,pids):
            if dist[i,j] <= match_window_s:
                paired[int(i)] = int(j)
    cohorts = defaultdict(lambda: dict(total=0, exact_k=0, exact_pitch_set=0,
                                      undercount=0, overcount=0, missing=0))
    for i,r in enumerate(refs):
        target=len(r["pitches"])
        bucket=cohorts[str(target if target <= 6 else "7plus")]
        bucket["total"]+=1
        if i not in paired:
            bucket["missing"]+=1
            bucket["undercount"]+=1
            continue
        got=outputs[paired[i]]
        predicted_k=len(got["pitches"])
        if predicted_k==target:
            bucket["exact_k"]+=1
        if got["pitches"]==r["pitches"]:
            bucket["exact_pitch_set"]+=1
        if predicted_k<target:
            bucket["undercount"]+=1
        elif predicted_k>target:
            bucket["overcount"]+=1
    return dict(
        source_true_groups=len(refs), source_predicted_groups=len(outputs),
        group_window_ms=group_window_s*1000.,
        match_window_ms=match_window_s*1000.,
        groups_matched=len(paired),
        by_true_k=dict(cohorts),
        definition="Onset groups by proximity, not S80 Exact-K protocol",
    )


def merge_k_reports(reports):
    agg=defaultdict(lambda: defaultdict(int))
    n=0
    for r in reports:
        n+=1
        for group,record in r["by_true_k"].items():
            for key,amount in record.items():
                agg[group][key]+=int(amount)
    by=dict(agg)
    total=sum(row["total"] for row in by.values())
    exact=sum(row["exact_k"] for row in by.values())
    correct_pitches=sum(row["exact_pitch_set"] for row in by.values())
    return dict(
        recordings=n,
        total_groups=total,
        exact_k_groups=exact,
        onset_exact_k_rate=exact/max(1,total),
        exact_pitch_set_groups=correct_pitches,
        pitch_set_rate=correct_pitches/max(1,total),
        by_true_k=by,
        warning="DEVELOPMENT ONLY; grouping-based onset Exact-K differs from historical S80 and K-active",
    )
