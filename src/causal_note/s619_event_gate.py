"""S619 — learned event-level gate for *original NOTE S614* emitted NoteOn.

This is a causal post-selection NOTE head: instead of inventing an attack from
a high per-frame score, classify a specific proposed attack, then consistently
relay (or suppress) its NoteOff. Original head states continue unchanged.
"""
from __future__ import annotations

import numpy as np
from causal_note.strict_live import HOP
from causal_note.s618_attack_head import PICK

# Original p3 + past difference3 + native acoustic65 + K7
# + before-onset active-cardinality1, lower-harmonic3, upper-harmonic3,
# + same-pitch reattack indication1.
DIM = 3+3+len(PICK)+7+1+3+3+1


def noteon_event_features(events,ends,x,p,k):
    """One evidence row per NoteOn in the *exact* original output order.

    Uses no annotations and no future audio. 'active' refers only to NOTE
    decisions already made before this note-on (including earlier chords).
    """
    ends=np.asarray(ends)
    if x.shape!=(len(ends),49,126) or p.shape!=(len(ends),49,3) or k.shape!=(len(ends),7):
        raise ValueError("S614 native feature/heads ABI required")
    active=set(); features=[]; noteon_indices=[]
    for i,e in enumerate(events):
        pitch=e["pitch"]
        if e["type"]=="note_off":
            active.discard(pitch)
            continue
        if e["type"]!="note_on":
            raise ValueError("Unknown live MIDI event")
        idx=int(e["decision_sample"])//HOP - 1
        if idx<0 or idx>=len(ends) or int(ends[idx])!=e["decision_sample"]:
            raise AssertionError("Events must come from actual right-aligned S614 frames")
        col=pitch-40
        if col<0 or col>=49:raise ValueError("Out-of-scope original NOTE pitch")
        prev=p[idx-1,col] if idx else np.zeros(3,np.float32)
        # -12/-19/-24 and +12/+19/+24 harmonic coincidence are *features*,
        # not hard-coded proof that a note is false.
        neighbor=np.array([int(pitch-d in active) for d in (12,19,24)]+
                          [int(pitch+d in active) for d in (12,19,24)],np.float32)
        stats=np.asarray([len(active), *neighbor, int(pitch in active)],np.float32)
        row=np.concatenate((p[idx,col],p[idx,col]-prev,x[idx,col,PICK],
                            k[idx],stats))
        assert row.shape==(DIM,)
        if not np.isfinite(row).all():raise ValueError("Nonfinite event-level head inputs")
        features.append(row)
        noteon_indices.append(i)
        active.add(pitch)
    if not features:
        return np.empty((0,DIM),np.float32),[]
    return np.stack(features).astype(np.float32),noteon_indices


def relay_original_events(events,accepted):
    """Maintain paired NoteOn/NoteOff, while leaving original S614 state alone."""
    count=sum(e["type"]=="note_on" for e in events)
    keep=np.asarray(accepted,dtype=bool)
    if keep.shape!=(count,):raise ValueError("One verdict per original NoteOn")
    currently_forwarded=set()
    result=[]; n=0
    for e in events:
        pitch=e["pitch"]
        if e["type"]=="note_on":
            if keep[n]:
                if pitch in currently_forwarded:raise AssertionError("Unpaired duplicate NoteOn")
                result.append(dict(e))
                currently_forwarded.add(pitch)
            n+=1
        elif e["type"]=="note_off":
            if pitch in currently_forwarded:
                result.append(dict(e))
                currently_forwarded.remove(pitch)
        else:
            raise ValueError("Unknown NOTE event")
    if currently_forwarded:
        raise AssertionError("Unclosed forwarded note")
    assert n==count
    return result
