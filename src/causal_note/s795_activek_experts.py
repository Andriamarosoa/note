"""S795 heterogeneous independent Active-K note-on experts, strict causal MIDI.

Five *candidate* specialist roles, not five mandatory retained models.
Decisions use only received 1024-sample S708 acoustic frames and emitted
MIDI state. No confirmation/lookahead/hold and no source event is delayed.
New NoteOn/NoteOff pairs are independent and restricted to 6 total voices.
Source events retain immutable S793 lineage for exact event timing checks.

This is an OFFLINE scientific candidate only. More CPU inference requires
paired microphone measurements before ANY production promotion.
"""
from __future__ import annotations
from collections import defaultdict
import numpy as np

from causal_note.strict_live import SR
from causal_note.s793_zero_added_latency import (
    SchedulingContract,mark_reference,veto_added_event_delay
)
from scripts.s790_independent_framepitch_expert import occupancy_from_emitted

HEADS=("general","bass","polyphonic","transient","harmonic")
MASKS=(0,1,2,4,8,16,3,5,6,10,12,24,31)
CUTS=(.85,.93)
PITCHES=49
MIDI_MIN=40
FEATURES=23
HARMONICS=(12,19,24,28,31)

def causal_features(item,post,k,original):
    """Current or earlier audio/posterior only; [T,49,23]."""
    p=np.asarray(post,np.float32)
    k=np.asarray(k,np.float32)
    x=np.asarray(item["native_features"],np.float32)
    d=np.asarray(item["decision_ends"],np.int64)
    if p.shape!=(len(d),49,3) or k.shape!=(len(d),7) or x.shape!=(len(d),49,126):
        raise ValueError("S795 signed S708 feature dimensions changed")
    if len(d)==0 or np.any(np.diff(d)<=0):
        raise ValueError("S795 empty/non-monotonic causal clock")
    source=occupancy_from_emitted(original,d)
    oldon=np.concatenate((p[:1,:,0],p[:-1,:,0]),axis=0)
    olds=np.concatenate((p[:1,:,1],p[:-1,:,1]),axis=0)
    parents=[];upper=[]
    for span in HARMONICS:
        y=np.zeros((len(d),49),np.float32)
        y[:,span:]=p[:,:-span,1]
        parents.append(y)
    for span in (12,19):
        y=np.zeros((len(d),49),np.float32)
        y[:,:-span]=p[:,span:,1]
        upper.append(y)
    parts=[
        p[:,:,0],p[:,:,1],p[:,:,2],oldon,olds,
        np.maximum(p[:,:,0]-oldon,0),
        np.maximum(p[:,:,1]-olds,0),
        *[item["native_post"][:,:,i] for i in range(3)],
        np.max(np.stack(parents,axis=-1),axis=-1),
        np.max(np.stack(upper,axis=-1),axis=-1),
        source.astype(np.float32),
        np.broadcast_to(k.argmax(-1)[:,None]/6.,source.shape),
        np.broadcast_to((-np.sum(k*np.log(np.maximum(k,1.e-7)),axis=1)
                         /np.log(7))[:,None],source.shape),
        np.broadcast_to(source.sum(1)[:,None]/6.,source.shape),
        *[x[:,:,j] for j in (0,8,24,64,96,125)],
        np.broadcast_to(np.arange(PITCHES,dtype=np.float32)[None,:]/48.,source.shape),
    ]
    result=np.stack(parts,axis=-1).astype(np.float32)
    if result.shape!=(len(d),49,FEATURES) or not np.isfinite(result).all():
        raise AssertionError(f"S795 causal feature ABI {result.shape}")
    return result,source

def regions(features,source):
    """Different failure modes, defined without annotation/oracle."""
    X=features
    onset=X[:,:,0];sustain=X[:,:,1];release=X[:,:,2]
    onset_rise=X[:,:,5];sustain_rise=X[:,:,6]
    evidence=(~source)&(onset>.035)&(sustain>.16)&(release<.85)&(
        (onset_rise>.025)|(sustain_rise>.045))
    bass=np.broadcast_to((np.arange(49)<8)[None,:],source.shape)
    poly=np.broadcast_to((source.sum(1)>=1)[:,None],source.shape)
    transient=(onset_rise>.085)|(sustain_rise>.12)
    harmonic=X[:,:,10]>.3
    return dict(general=evidence,
       bass=evidence&bass,
       polyphonic=evidence&poly,
       transient=evidence&transient,
       harmonic=evidence&harmonic)

def onset_targets(truth,decision_ends,source):
    """Only supervised FIT labels: onsets available at or before decision."""
    ends=np.asarray(decision_ends,np.int64)
    labels=np.zeros(source.shape,bool)
    for note in truth:
        pitch=int(note["pitch"])-MIDI_MIN
        if not 0<=pitch<PITCHES:continue
        t=float(note["time"])*SR
        frame=int(np.searchsorted(ends,t,side="left"))
        if 0<=frame<len(ends) and t>ends[frame]-1024-1.e-6 and t<=ends[frame]+1.e-6:
            labels[frame,pitch]=True
    return labels&~source

def sample_head(X,labels,eligible,seed,nmax=1800):
    rng=np.random.default_rng(seed)
    good=np.flatnonzero((eligible&labels).ravel())
    bad=np.flatnonzero((eligible&~labels).ravel())
    good=rng.choice(good,min(len(good),nmax//2),replace=False)
    bad=rng.choice(bad,min(len(bad),nmax-len(good)),replace=False)
    ind=np.sort(np.concatenate((good,bad)))
    return X.reshape(-1,FEATURES)[ind].astype(np.float32),\
           labels.ravel()[ind].astype(np.int8)

def midi(kind,pitch,t,reason):
    return dict(type=kind,pitch=int(pitch),decision_sample=int(t),
        available_sample=int(t),time=float(t/SR),
        reason=reason,velocity=96 if kind=="note_on" else 0,
        first_eligible_sample=int(t))

def decode(original,decision_ends,X,probabilities,mask,cut,samples):
    """Immediate extra MIDI notes while preserving every native emission.

    Events are processed chronologically at their actual availability time.
    Synthetic notes may close when same-frame evidence disappears; there is
    NO fixed hold/persistence/retiming. No later original MIDI emission.
    """
    SchedulingContract().validate()
    if mask not in MASKS or (cut not in CUTS and not(mask==0 and cut==0.)):
        raise ValueError("undeclared FIT selection")
    ends=np.asarray(decision_ends,np.int64)
    if X.shape!=(len(ends),49,FEATURES) or samples<ends[-1]:
        raise ValueError("invalid frozen causal evidence")
    eligibility=regions(X,X[:,:,12]>.5)
    # Eligibility for new notes is re-filtered by CURRENT source and extras.
    for name in HEADS:
        if name not in probabilities or np.shape(probabilities[name])!=(len(ends),49):
            raise ValueError(f"missing head predictions {name}")
    originals=mark_reference(original)
    sorted_original=sorted(enumerate(originals),key=lambda z:(
        z[1]["available_sample"],z[0]))
    if [i for i,_ in sorted_original]!=list(range(len(originals))):
        raise AssertionError("S708 source MIDI stream out of time order")
    source=set();extra=set();out=[];added=0;closed=0
    at=0;frame=0
    times=sorted(set(map(int,ends))|
                {int(e["available_sample"]) for e in originals}|{int(samples)})
    for t in times:
        if t<0 or t>samples:raise AssertionError("event outside audio clock")
        # For every native event, dispatch at the unchanged native timestamp.
        while at<len(originals) and originals[at]["available_sample"]==t:
            e=originals[at];pitch=int(e["pitch"])
            if e["type"]=="note_on":
                if pitch in extra:
                    out.append(midi("note_off",pitch,t,"s795_source_retrigger"))
                    extra.remove(pitch);closed+=1
                if pitch in source:
                    raise AssertionError("duplicate S708 native NoteOn")
                # Original source always takes priority over an extra voice.
                while len(source)+len(extra)>=6 and extra:
                    remove=min(extra)
                    out.append(midi("note_off",remove,t,"s795_source_capacity"))
                    extra.remove(remove);closed+=1
                source.add(pitch)
            else:
                if pitch not in source:
                    raise AssertionError("orphan original NoteOff")
                source.remove(pitch)
            out.append(e);at+=1
        if frame<len(ends) and t==int(ends[frame]):
            closed_this_frame=set()
            alive=X[frame,:,1];released=X[frame,:,2]
            onset=X[frame,:,0]
            # All added notes stop at the first AVAILABLE evidence of end,
            # never after an extra confirmation/hold window.
            for pitch in sorted(list(extra)):
                ix=pitch-MIDI_MIN
                if alive[ix]<.30 or released[ix]>.65:
                    out.append(midi("note_off",pitch,t,"s795_first_release_evidence"))
                    extra.remove(pitch);closed+=1
                    closed_this_frame.add(pitch)
            if mask:
                offers=[]
                for ix in range(49):
                    pitch=ix+MIDI_MIN
                    if pitch in source or pitch in extra or pitch in closed_this_frame:continue
                    chosen=[float(probabilities[name][frame,ix])
                        for h,name in enumerate(HEADS)
                        if mask&(1<<h) and eligibility[name][frame,ix]]
                    if chosen and max(chosen)>=cut:
                        offers.append((max(chosen),ix))
                for prob,ix in sorted(offers,key=lambda pair:(-pair[0],pair[1])):
                    pitch=ix+MIDI_MIN
                    if len(source)+len(extra)>=6:break
                    out.append(midi("note_on",pitch,t,"s795_multihead_attack"))
                    extra.add(pitch);added+=1
            frame+=1
    for pitch in sorted(extra):
        out.append(midi("note_off",pitch,int(samples),"stream_end"))
        closed+=1
    if source:raise AssertionError("S708 source notes not closed by file end")
    if added!=closed:raise AssertionError("S795 unpaired synthetic notes")
    if at!=len(originals):raise AssertionError("native events were lost")
    # Zero-modification control must reproduce source EXACTLY (excluding tags).
    if mask==0:
        if [(e["type"],e["pitch"],e["available_sample"]) for e in out]!=[
            (e["type"],e["pitch"],e["available_sample"]) for e in originals]:
            raise AssertionError("source identity control drift")
    # Every native event has its own immutable lineage, incl. timestamps.
    veto_added_event_delay(originals,out)
    return out,dict(new_onsets=added,new_offsets=closed,
                    max_added_native_delay_samples=0,
                    extra_lookahead_samples=0,extra_buffer_samples=0,
                    extra_confirmation_samples=0,extra_hold_samples=0)
