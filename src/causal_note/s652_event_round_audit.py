"""S652 event-by-event change audit callable INSIDE each experimental loop.

Unlike a net TP delta, this exposes exact ground-truth reference notes
newly matched, lost, preserved, and missed. All predictions and labels
must originate from the same immutable WAV/source. K herein is local
simultaneous-onset multiplicity, NOT the S80 global Exact-K metric.
Diagnostic harmonic/nearby relationships never prove causal error source.
"""
from __future__ import annotations
from collections import Counter,defaultdict
from dataclasses import dataclass
from typing import Sequence


def _reference_class(truth,index,neighborhood_seconds=.020):
    now=truth[index]
    t=float(now["time"])
    simultaneous={int(x["pitch"]) for x in truth
                  if abs(float(x["time"])-t)<=neighborhood_seconds}
    return "K1" if len(simultaneous)==1 else "K2" if len(simultaneous)==2 else "K3" if len(simultaneous)==3 else "K4plus"


def _pitch_register(pitch):
    if pitch<=52:return "grave"
    if pitch<=64:return "medium"
    if pitch<=76:return "aigu"
    return "tres_aigu"


def _validate_events(events):
    on=[e for e in events if e["type"]=="note_on"]
    off=[e for e in events if e["type"]=="note_off"]
    if len(on)!=len(off):
        raise AssertionError("Unbalanced MIDI NoteOn/NoteOff")
    if any(e["available_sample"]<e["decision_sample"] for e in events):
        raise AssertionError("Anti-causal timestamp")
    if [e["available_sample"] for e in events]!=sorted(e["available_sample"] for e in events):
        raise AssertionError("Events not time-ordered")
    return on


def _unmatched_fps(on,matched_indices,truth):
    rows=[]
    for index,e in enumerate(on):
        if index in matched_indices:continue
        t=float(e["time"]);pitch=int(e["pitch"])
        nearby=[r for r in truth if abs(float(r["time"])-t)<=.08]
        same=[r for r in nearby if int(r["pitch"])==pitch]
        harmonic=[r for r in nearby if abs(int(r["pitch"])-pitch) in (12,19,24)]
        if same:kind="same_pitch_unmatched"
        elif harmonic:kind="possible_harmonic_confusion"
        elif nearby:kind="nearby_wrong_pitch"
        else:kind="no_annotated_onset_nearby"
        rows.append(dict(prediction_index=index,time=t,pitch=pitch,
                         possible_mechanism=kind))
    return rows


def audit_event_round(source,corpus,truth,baseline_events,candidate_events):
    """One source; call after each proposed policy, before the next one."""
    from scripts.s613_train_strict_live import event_score
    base_on=_validate_events(baseline_events)
    new_on=_validate_events(candidate_events)
    if any(not (40<=int(n["pitch"])<=88) for n in truth):
        raise AssertionError("Non-native reference pitch")
    b,bpairs=event_score(baseline_events,truth)
    n,npairs=event_score(candidate_events,truth)
    bset={int(j) for i,j in bpairs}
    nset={int(j) for i,j in npairs}
    if len(bset)!=b["TP"] or len(nset)!=n["TP"]:
        raise AssertionError("Reference events matched more than once")
    oldpred={int(i) for i,j in bpairs}
    newpred={int(i) for i,j in npairs}
    gained=nset-bset
    lost=bset-nset
    both=bset&nset
    missed=set(range(len(truth)))-(bset|nset)
    if sum(map(len,(gained,lost,both,missed)))!=len(truth):
        raise AssertionError("Truth partition not complete")
    if len(gained)-len(lost)!=n["TP"]-b["TP"]:
        raise AssertionError("TP delta cannot be justified by true-note identities")
    original_false=_unmatched_fps(base_on,oldpred,truth)
    newly_false=_unmatched_fps(new_on,newpred,truth)
    if len(newly_false)-len(original_false)!=n["FP"]-b["FP"]:
        raise AssertionError("FP delta is not supported by actual unpaired events")
    classes=defaultdict(Counter)
    rows=[]
    for refidx,item in enumerate(truth):
        status=("newly_correct" if refidx in gained else
                "regressed_true_note" if refidx in lost else
                "retained_correct" if refidx in both else
                "still_missed")
        k=_reference_class(truth,refidx)
        reg=_pitch_register(int(item["pitch"]))
        classes[k][status]+=1
        classes[reg][status]+=1
        rows.append(dict(reference_index=refidx,time_s=float(item["time"]),
                         pitch=int(item["pitch"]),status=status,
                         local_simultaneous_group=k,register=reg))
    delta=dict(TP=n["TP"]-b["TP"],FP=n["FP"]-b["FP"],FN=n["FN"]-b["FN"],
               F1=n["F1"]-b["F1"])
    return dict(source=source,corpus=corpus,delta=delta,
        matched_reference_partition=dict(corrected=len(gained),lost=len(lost),
            retained=len(both),still_missed=len(missed)),
        exact_reference_labels=rows,
        classifications={k:dict(v) for k,v in classes.items()},
        baseline_false_notes=original_false,
        candidate_false_notes=newly_false,
        false_note_diagnostic_counts=dict(Counter(x["possible_mechanism"]
                                                 for x in newly_false)),
        per_pitch_time_event_proof=True,
        local_k_is_not_S80_exact_K=True,
        causal_onset_times_not_backdated=True,
        no_source_provenance_inferred_from_audio=True)


def require_event_audit_before_next_round(audits,expected_sources):
    expected=set(expected_sources)
    got={x["source"] for x in audits}
    if expected!=got or len(audits)!=len(got):
        raise AssertionError("No next-loop candidate without a complete per-source event audit")
    for audit in audits:
        if not audit["per_pitch_time_event_proof"]:
            raise AssertionError("Missing one-to-one true note matches")
    return True
