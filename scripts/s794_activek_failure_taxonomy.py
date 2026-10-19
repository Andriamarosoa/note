"""S794 strict-zero-wait Active Exact-K forensics on REAL S708 MIDI256.

Read actual source NoteOn/NoteOff; compute six S704 scores and per-true-K
under/over counts. After source events have been generated, open guitar
annotations strictly FOR OFFLINE AUDIT, never for MIDI inference.

Diagnostic expert opportunity signatures are *not trained proposals* and
MUST NOT be called F1 or promoted to microphone LIVE. Investigate whether
missing active pitches have contemporaneous evidence in bass, harmonic
overlap, sustained weak onsets or already emitted MIDI state.

This module never changes the MIDI output or delays any event.
"""
from __future__ import annotations
from collections import Counter
import numpy as np

from causal_note.strict_live import SR
from causal_note.s793_zero_added_latency import (
    SchedulingContract,mark_reference,veto_added_event_delay
)
from scripts.s714_audit_s708_cardinality import (
    events_to_frames,annotation_frames
)
from scripts.s742_activek_forensics import exact_active_breakdown
from scripts.s704_joint_metrics import SCORES,metric

REGISTERS={"E2_B2":(0,8),"C3_B3":(8,20),"C4_E6":(20,49)}
HARMONIC_OFFSETS=(12,19,24,28,31)

def prior_audio_lookups(ends,decision_ends):
    """Index most recent FULLY received 1024-sample decision, no future."""
    e=np.asarray(ends,np.int64)
    d=np.asarray(decision_ends,np.int64)
    if e.ndim!=1 or d.ndim!=1 or np.any(np.diff(e)<=0) or np.any(np.diff(d)<=0):
        raise ValueError("strict causal decision clock required")
    at=np.searchsorted(d,e,side="right")-1
    good=at>=0
    return at,good

def diagnostic(track):
    """Six scores and 256-sample actual note-state failure taxonomy."""
    item,post,events,truth=(
        track["item"],track["post"],track["events"],track["truth"])
    ends=np.asarray(item["ends"],np.int64)
    decision_ends=np.asarray(item["decision_ends"],np.int64)
    samples=int(item["samples"])
    if ends.ndim!=1 or post.shape!=(len(decision_ends),49,3):
        raise ValueError("S794 S708 acoustic ABI changed")
    score=metric(events,truth,ends)
    breakdown=exact_active_breakdown(events,truth,ends)
    if not np.isclose(score["active_exact_k"],breakdown["active_exact_k"],
                      rtol=0,atol=1e-12):
        raise AssertionError("S794 S704 Active K parity failure")
    pred=events_to_frames(events,ends)
    target,counts=annotation_frames(truth,ends)
    delta=np.minimum(pred.sum(1),6)-np.minimum(counts,6)
    active=counts>0
    missed=target&~pred
    spurious=pred&~target
    idx,good=prior_audio_lookups(ends,decision_ends)
    causal=post[np.maximum(0,idx)] if len(post) else np.zeros((len(ends),49,3))
    onset=np.where(good[:,None],causal[:,:,0],0)
    sustain=np.where(good[:,None],causal[:,:,1],0)
    release=np.where(good[:,None],causal[:,:,2],1)
    available=good[:,None]
    # Physical expertise candidates are descriptive COVERAGE COUNTS only.
    # A positive acoustic cue does not imply a correct new NoteOn.
    acoustic=(sustain>=.30)&(onset>=.10)&(release<.78)&available
    bass=np.zeros((len(ends),49),bool);bass[:,:8]=acoustic[:,:8]
    poly=np.broadcast_to((pred.sum(1)>=1)[:,None],pred.shape)&acoustic
    rise=np.zeros_like(acoustic)
    if len(post):
        prev=np.concatenate((post[:1,:,1],post[:-1,:,1]),axis=0)
        slope=np.maximum(post[:,:,1]-prev,0)
        rise=available&(slope[np.maximum(idx,0)]>.06)&(sustain>=.25)
    harmonic=np.zeros_like(acoustic)
    for interval in HARMONIC_OFFSETS:
        harmonic[:,interval:] |= acoustic[:,:-interval]
    signatures={
        "bass_fundamentals":bass,
        "poly_new_voice":poly,
        "sustain_transition":rise,
        "harmonic_collision":harmonic&acoustic
    }
    # 256-hop active K true undercount / overcount denominators.
    under=(active&(delta<0))
    over=(active&(delta>0))
    misses_by_register={}
    for label,(lo,hi) in REGISTERS.items():
        misses_by_register[label]=dict(
            target_pitch_frames=int(target[:,lo:hi].sum()),
            missed_pitch_frames=int(missed[:,lo:hi].sum()),
            extra_pitch_frames=int(spurious[:,lo:hi].sum()))
    opportunity={}
    for name,mask in signatures.items():
        opportunity[name]=dict(
            missed_pitch_frames_with_cue=int(np.count_nonzero(missed&mask)),
            false_pitch_frames_with_cue=int(np.count_nonzero(~target&mask)),
            undercount_frames_with_missing_pitch_cue=int(np.count_nonzero(
                under&np.any(missed&mask,axis=1))),
            total_cue_pitch_frames=int(mask.sum()))
    # A diagnostic upper bound if instantaneous perfect corrections could
    # somehow be made; NOT an achieved score or a real candidate.
    potential_frames=np.any(missed&acoustic,axis=1)
    protected=mark_reference(events)
    identity=veto_added_event_delay(protected,[dict(e) for e in protected])
    SchedulingContract().validate()
    return dict(scores={k:score[k] for k in SCORES},
        original_counts={k:int(score[k]) for k in
            ("tp","fp","fn","frames","k_correct","active_frames",
             "active_k_correct","frame_pitch_tp","frame_pitch_fp",
             "frame_pitch_fn","offset_eligible","offset_correct")},
        breakdown=breakdown,under_active_frames=int(under.sum()),
        over_active_frames=int(over.sum()),
        missed_pitch_frames=int(missed.sum()),
        spurious_pitch_frames=int(spurious.sum()),
        acoustic_evidence_available_undercount_frames=int(np.count_nonzero(
            under&potential_frames)),
        no_acoustic_evidence_undercount_frames=int(np.count_nonzero(
            under&~potential_frames)),
        by_register=misses_by_register,expert_opportunity=opportunity,
        causal_audio_frames=int(good.sum()),
        future_audio_frames_used=0,
        source_event_identity=identity,
        additional_scheduling_samples=0,
        latency_physical_measured=False)

def _sum(rows,path):
    return int(sum(int(row[path]) for row in rows))

def summarize(rows):
    if not rows:raise ValueError("S794 empty corpus")
    by_k={}
    for cls in range(7):
        records=[r["breakdown"]["by_true_k"][str(cls)] for r in rows]
        by_k[str(cls)]={key:_sum(records,key)
            for key in ("frames","correct","under","over")}
        if sum(by_k[str(cls)][k] for k in ("correct","under","over"))!=by_k[str(cls)]["frames"]:
            raise AssertionError("active K confusion does not add up")
    by_reg={name:{field:sum(r["by_register"][name][field] for r in rows)
                      for field in ("target_pitch_frames","missed_pitch_frames",
                                    "extra_pitch_frames")}
            for name in REGISTERS}
    experts={name:{field:sum(r["expert_opportunity"][name][field] for r in rows)
                   for field in ("missed_pitch_frames_with_cue",
                     "false_pitch_frames_with_cue",
                     "undercount_frames_with_missing_pitch_cue",
                     "total_cue_pitch_frames")}
             for name in ("bass_fundamentals","poly_new_voice",
                          "sustain_transition","harmonic_collision")}
    active_frames=sum(r["breakdown"]["active_frames"] for r in rows)
    active_correct=sum(r["breakdown"]["active_correct"] for r in rows)
    under=sum(r["under_active_frames"] for r in rows)
    over=sum(r["over_active_frames"] for r in rows)
    if under+over+active_correct!=active_frames:
        raise AssertionError("S794 Active-K deficit accounting inconsistent")
    return dict(recordings=len(rows),active_frames=active_frames,
        active_correct=active_correct,active_exact_k=active_correct/max(active_frames,1),
        under_active_frames=under,over_active_frames=over,by_true_k=by_k,
        by_register=by_reg,expert_opportunity=experts,
        missed_pitch_frames=sum(r["missed_pitch_frames"] for r in rows),
        spurious_pitch_frames=sum(r["spurious_pitch_frames"] for r in rows),
        acoustic_evidence_available_undercount_frames=sum(
            r["acoustic_evidence_available_undercount_frames"] for r in rows),
        no_acoustic_evidence_undercount_frames=sum(
            r["no_acoustic_evidence_undercount_frames"] for r in rows))
