"""S730: strictly causal additive NOTE rescue, preserving original S708 notes.

A specialized harmonic model may recover a higher new note while the old
S708 event stream was already correct elsewhere. Replacing the entire S708
joint prediction causes avoidable loss of correct attacks/releases. Instead,
stream-merge only novel pitch events, gated by the learned Exact-K of the
specialist and by actual emitted source state, without labels or future PCM.

This module ONLY merges already emitted causal streams chronologically.
It does not make an offline-future choice and does not require a new model
architecture. It keeps every original S708 event unchanged as a subsequence.
"""
from __future__ import annotations
import numpy as np
from causal_note.strict_live import SR


def sample(e):
    return int(e.get("available_sample",e["decision_sample"]))


def additive_rescue(reference,proposal,decision_ends,prob_k,
                    *,confidence=.50,refractory_ms=120.):
    """Merge two causal S707 NoteOn/Off streams under cardinality K0..6.

    K is from the candidate joint network, measured at the latest received
    decision <= actual emission (never future). NoteOn can be emitted only
    if it is not already active in the original stream, was not just emitted
    on that pitch, and the model K target is higher than current count.
    Candidate NoteOff is forwarded only for accepted candidate NoteOns.
    An original NoteOn preempts an experimental duplicate, but never moves,
    drops or edits any original event.
    """
    if not 0<=confidence<=1 or refractory_ms<0:
        raise ValueError("S730 invalid causal K/Onset rescue policy")
    ends=np.asarray(decision_ends)
    probs=np.asarray(prob_k)
    if ends.ndim!=1 or probs.shape!=(len(ends),7):
        raise ValueError("S730 expects decision times and 7-way Exact-K scores")
    if np.any(np.diff(ends)<0) or not np.isfinite(probs).all():
        raise ValueError("S730 requires finite ordered K")
    item=[]
    for priority,events in enumerate((reference,proposal)):
        last=-1
        for ix,e in enumerate(events):
            clock=sample(e)
            if clock<last or clock<0:
                raise ValueError("S730 negative/noncausal or unsorted emitted event")
            if e["type"] not in ("note_on","note_off"):
                raise ValueError("S730 unknown emitted event")
            item.append((clock,priority,ix,e))
            last=clock
    item.sort(key=lambda x:(x[0],x[1],x[2]))
    ref_active=set();addon_active=set()
    last_on={}
    result=[];added=0;rejected=0
    minimum=int(np.ceil(refractory_ms*SR/1000))
    for clock,priority,_,e in item:
        p=int(e["pitch"])
        if priority==0:
            if e["type"]=="note_on":
                if p in addon_active:
                    off=dict(type="note_off",pitch=p,decision_sample=clock,
                        available_sample=clock,time=clock/SR,velocity=0,
                        reason="s730_preempted_by_original_onset")
                    result.append(off);addon_active.remove(p)
                ref_active.add(p);last_on[p]=clock
            else:
                ref_active.discard(p)
            result.append(dict(e))
            continue
        if e["type"]=="note_off":
            if p in addon_active:
                result.append({**e,"reason":"s730_added_note_off"})
                addon_active.remove(p)
            continue
        # Original event and any earlier add-on have priority, always.
        if p in ref_active or p in addon_active or clock-last_on.get(p,-10**15)<minimum:
            rejected+=1;continue
        t=int(np.searchsorted(ends,clock,side="right")-1)
        if t<0:
            rejected+=1;continue
        likelihood=probs[t]
        k=int(np.argmax(likelihood))
        if not (float(np.max(likelihood))>=confidence and
                len(ref_active)+len(addon_active)<min(k,6)):
            rejected+=1;continue
        out=dict(e);out["reason"]="s730_added_harmonic_independent_attack"
        out["decision_sample"]=clock;out["available_sample"]=clock;out["time"]=clock/SR
        result.append(out)
        addon_active.add(p);last_on[p]=clock;added+=1
    # The received candidate source must close its own notes. Event merging
    # cannot invent a future NoteOff or retroactively suppress existing notes.
    if addon_active:raise AssertionError("S730 extra notes did not close causally")
    original=[e for e in result if not str(e.get("reason","")).startswith("s730_")]
    if original!=[dict(e) for e in reference]:
        raise AssertionError("S730 failed to preserve every original S708 note event")
    return result,dict(added=added,rejected=rejected,
                       retained_reference_count=len(reference))
