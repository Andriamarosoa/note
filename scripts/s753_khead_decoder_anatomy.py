"""S753: why are true K2..K6 chords undercounted? Shadow-only audit.

Evaluate at ACTUAL S708 four-hop decision times, not the native hop256
frames of the official six metrics. The 7-class head and the count of
currently emitted MIDI pitches are separately inspected. Ground-truth
note intervals are opened ONLY after the complete predicted MIDI stream
has been generated in the S750 runner.

This is not an oracle correction, not a seventh metric and not a new
Exact-K score. The signed S704 six metrics are always unchanged.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts.s750_six_axis_cal32_audit import evaluate
from causal_note.strict_live import SR

def decision_anatomy(ends,k,events,truth):
    ends=np.asarray(ends,np.int64)
    k=np.asarray(k,np.float64)
    if (ends.ndim!=1 or len(ends)==0
        or np.any(np.diff(ends)<=0)
        or k.shape!=(len(ends),7)
        or not np.isfinite(k).all()
        or np.any(k < -1e-6)
        or np.max(np.abs(k.sum(1)-1))>1e-3):
        raise ValueError("S753 head/count decision ABI must be finite, aligned and normalized")
    # Native event stream, available/decision checked by S704 metric.
    # Use a stable availability ordering; do not inspect future events
    # before the current decision time or consult labels for inference.
    ordered=sorted(enumerate(events),
                   key=lambda v:(int(v[1]["available_sample"]),v[0]))
    predicted_count=np.zeros(len(ends),np.int32)
    state=set();at=0
    for i,end in enumerate(ends):
        while at<len(ordered) and int(ordered[at][1]["available_sample"])<=int(end):
            e=ordered[at][1];pitch=int(e["pitch"])
            if not 40<=pitch<=88:
                raise ValueError("S753 emitted pitch outside signed MIDI range")
            if e["type"]=="note_on":
                if pitch in state:
                    raise AssertionError("S753 overlapping emitted same pitch")
                state.add(pitch)
            elif e["type"]=="note_off":
                if pitch not in state:
                    raise AssertionError("S753 orphan signed NoteOff")
                state.remove(pitch)
            else:
                raise ValueError("S753 unexpected MIDI event")
            at+=1
        predicted_count[i]=min(6,len(state))
    true_count=np.zeros(len(ends),np.int32)
    times=ends/SR
    for n in truth:
        true_count+=(times>=float(n["time"]))&(
            times<float(n["time"])+float(n["duration"]))
    truth_clipped=np.minimum(6,true_count)
    model_choice=np.argmax(k,axis=1).astype(np.int32)
    rows={}
    for label in range(7):
        subset=truth_clipped==label
        count=int(np.count_nonzero(subset))
        if not count:continue
        head_under=model_choice[subset]<label
        midi_under=predicted_count[subset]<label
        head_exact=model_choice[subset]==label
        midi_exact=predicted_count[subset]==label
        # The two groups below partition all observed MIDI undercounts:
        # head also predicts too few OR head predicts enough but decoder
        # still emits too few. Correlation does not establish causality.
        head_less_and_midi_less=int(np.count_nonzero(head_under&midi_under))
        head_enough_midi_less=int(np.count_nonzero(~head_under&midi_under))
        rows[str(label)]=dict(
            frames=count,
            model_k_under=int(np.count_nonzero(head_under)),
            model_k_exact=int(np.count_nonzero(head_exact)),
            model_k_over=int(np.count_nonzero(model_choice[subset]>label)),
            emitted_k_under=int(np.count_nonzero(midi_under)),
            emitted_k_exact=int(np.count_nonzero(midi_exact)),
            emitted_k_over=int(np.count_nonzero(predicted_count[subset]>label)),
            both_under=head_less_and_midi_less,
            head_enough_but_midi_under=head_enough_midi_less,
            head_exact_but_midi_under=int(np.count_nonzero(head_exact&midi_under)),
            head_under_but_midi_exact=int(np.count_nonzero(head_under&midi_exact)),
            sum_true_k_probability=float(np.sum(k[subset,label])),
            sum_k_prob_at_least_truth=float(np.sum(k[subset,label:],axis=1).sum()))
        if rows[str(label)]["emitted_k_under"]!=(
            head_less_and_midi_less+head_enough_midi_less):
            raise AssertionError("S753 failed exhaustive partition of undercounts")
        if sum(rows[str(label)][z] for z in (
                "model_k_under","model_k_exact","model_k_over"))!=count:
            raise AssertionError("S753 model K coverage mismatch")
        if sum(rows[str(label)][z] for z in (
                "emitted_k_under","emitted_k_exact","emitted_k_over"))!=count:
            raise AssertionError("S753 emitted K coverage mismatch")
    return dict(decision_frames=len(ends),
                decisions_by_true_k=rows,
                true_gt6_decisions=int(np.count_nonzero(true_count>6)),
                native_six_metrics_are_unchanged=True)

SUM_FIELDS=("frames","model_k_under","model_k_exact","model_k_over",
            "emitted_k_under","emitted_k_exact","emitted_k_over",
            "both_under","head_enough_but_midi_under",
            "head_exact_but_midi_under","head_under_but_midi_exact",
            "sum_true_k_probability","sum_k_prob_at_least_truth")

def aggregate_decision_audit(records):
    if not records:
        raise AssertionError("S753 no CAL decisions audited")
    counts=defaultdict(lambda:{key:0 for key in SUM_FIELDS})
    for rec in records:
        for cls,data in rec["decisions_by_true_k"].items():
            for name in SUM_FIELDS:
                counts[cls][name]+=data[name]
    result={}
    for cls,data in sorted(counts.items(),key=lambda t:int(t[0])):
        count=data["frames"]
        result[cls]={**data,
           "model_k_exact_rate":data["model_k_exact"]/max(count,1),
           "emitted_k_exact_rate":data["emitted_k_exact"]/max(count,1),
           "model_under_rate":data["model_k_under"]/max(count,1),
           "decoder_under_rate":data["emitted_k_under"]/max(count,1),
           "head_enough_share_of_decoder_undercount":(
                data["head_enough_but_midi_under"]/max(data["emitted_k_under"],1)),
           "mean_true_k_probability":data["sum_true_k_probability"]/max(count,1)}
        if data["emitted_k_under"]!=(
            data["both_under"]+data["head_enough_but_midi_under"]):
            raise AssertionError("S753 aggregate undercount partition invalid")
    return dict(
        evaluation_cadence="S708 real four-hop decisions (1024 samples, not S704 256-sample score)",
        sample_rate=SR,
        tracks=len(records),
        decision_frames=sum(r["decision_frames"] for r in records),
        true_gt6_decisions=sum(r["true_gt6_decisions"] for r in records),
        per_true_k=result,
        native_six_metrics_are_unchanged=True,
        diagnostic_only=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--s708-dir",type=Path,required=True)
    p.add_argument("--offset-checkpoint",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    evaluate(p.parse_args(),decision_audit=decision_anatomy,
             schema="S753_CAUSAL_KHEAD_VS_EMITTED_DECODER_GUITARSET_CAL32")

if __name__=="__main__":
    main()
