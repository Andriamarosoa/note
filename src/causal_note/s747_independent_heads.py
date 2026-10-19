"""S747: independent causal acoustic proposal heads, never shared S708 state.

One frozen signed S708 event stream is read-only. H1..H5 each have their own
49-pitch refractory counters, and one fragment-local 32-mask router selects
which experts may propose a new note. No extra proposal can modify the
original Onset, Exact-K, Offset, sustain or pitch states. The downstream
S746 priority mux checks the actual complete MIDI metrics.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from causal_note.strict_live import SR
from causal_note.s743_fundamental_router import (
    MultiHeadBirthPolicy, select_fragment_subset)


@dataclass(frozen=True)
class IsolatedBirthPolicy:
    min_sustain: float = .38
    min_acoustic: float = .19
    min_gain: float = .028
    min_rise: float = .012
    min_tail: float = .12
    refractory_seconds: float = .09
    max_per_fragment: int = 2
    mask_delta: float = .04
    proposal_refractory_seconds: float = .10
    native_release_guard_seconds: float = .07

    def validate(self):
        vals=(self.min_sustain,self.min_acoustic,self.min_gain,
              self.min_rise,self.min_tail,self.mask_delta)
        if any(not 0 <= x <= 1 for x in vals) or not (
                .01 <= self.refractory_seconds <= 1.5
                and .01 <= self.proposal_refractory_seconds <= 1.5
                and 0 <= self.native_release_guard_seconds <= .5
                and 1 <= self.max_per_fragment <= 6):
            raise ValueError("Invalid independent H1..H5 policy")
        return self


def isolated_onsets(ends, post, k, heads, context, reference, config=None):
    """Returns a causal list of new candidates; reference is NEVER mutated.

    Source active pitches are replayed from *available* signed MIDI events,
    not inferred using labels or looking ahead. Each proposal head has a
    separate [49] refractory vector. No candidate onset changes the state
    of any other head or the source. The only shared operation is selecting
    non-duplicate proposals at this one received fragment.
    """
    c=(config or IsolatedBirthPolicy()).validate()
    ends=np.asarray(ends,np.int64)
    post=np.asarray(post,np.float32)
    k=np.asarray(k,np.float32)
    heads=np.asarray(heads,np.float32)
    context=np.asarray(context,np.float32)
    n=len(ends)
    if (ends.shape!=(n,) or post.shape!=(n,49,3)
            or k.shape!=(n,7) or heads.shape!=(n,49,6)
            or context.shape!=(n,49,16)
            or np.any(np.diff(ends)<=0)
            or not all(np.isfinite(v).all() for v in (post,k,heads,context))):
        raise ValueError("S747 causal acoustic schema mismatch")
    original_signature=[(e["type"],int(e["pitch"]),
                         int(e["available_sample"]),e["reason"])
                        for e in reference]
    source_events=sorted(enumerate(reference),
                         key=lambda z:(int(z[1]["available_sample"]),z[0]))
    source_active=set()
    cursor=0
    last=np.full((5,49),-10**12,dtype=np.int64)
    last_proposal=np.full(49,-10**12,dtype=np.int64)
    last_native_release=np.full(49,-10**12,dtype=np.int64)
    previous=np.zeros((49,3),np.float32)
    result=[]
    diag=dict(mask_histogram={},per_head={},proposed=0,
              rejected_source=0,rejected_k=0,rejected_acoustic=0,
              rejected_cross_head_restrike=0,rejected_native_tail=0)
    choose=MultiHeadBirthPolicy(k_tail=c.min_tail,min_onset=c.min_acoustic,
        min_sustain=c.min_sustain,mask_delta=c.mask_delta,
        mask_strategy="adaptive",min_rise=c.min_rise).validate()
    for t,end in enumerate(ends):
        # Only events available by the current audio block may alter the
        # READ-ONLY SOURCE SHADOW. No S708 selector state is ever written.
        while cursor<len(source_events) and (
                source_events[cursor][1]["available_sample"]<=int(end)):
            e=source_events[cursor][1]
            pitch=int(e["pitch"])
            if e["type"]=="note_on":
                source_active.add(pitch)
            elif e["type"]=="note_off":
                source_active.discard(pitch)
                if 40 <= pitch <= 88:
                    last_native_release[pitch-40]=int(e["available_sample"])
            cursor+=1
        mask=select_fragment_subset(heads[t],post[t],context[t],choose)
        key=str(mask)
        diag["mask_histogram"][key]=diag["mask_histogram"].get(key,0)+1
        allowed=[j for j in range(1,6) if mask & (1<<j)]
        if not allowed:
            previous=post[t].copy()
            continue
        active=min(6,len(source_active))
        if active>=6 or float(k[t,active+1:].sum())<c.min_tail:
            diag["rejected_k"]+=1
            previous=post[t].copy()
            continue
        rise=np.maximum(post[t,:,0]-previous[:,0],0)
        sustain_rise=np.maximum(post[t,:,1]-previous[:,1],0)
        independent_rise=((rise>=c.min_rise)|
            (sustain_rise>=2*c.min_rise)|
            ((context[t,:,4]>.05)&(context[t,:,7]>.25)))
        keep=((post[t,:,1]>=c.min_sustain)&(post[t,:,2]<.65)
              & independent_rise)
        proposals=[]
        for head in allowed:
            gain=heads[t,:,head]-heads[t,:,0]
            candidates=np.flatnonzero(
                keep & (heads[t,:,head]>=c.min_acoustic)
                & (gain>=c.min_gain)
                & (int(end)-last[head-1]>=c.refractory_seconds*SR))
            for ix in candidates:
                pitch=int(ix+40)
                if pitch in source_active:
                    diag["rejected_source"]+=1
                    continue
                # Two independently trained heads may propose the SAME
                # pitch on successive 23ms fragments. This is an output
                # deduplication gate, NOT shared expert learning state.
                if int(end)-last_proposal[ix] < c.proposal_refractory_seconds*SR:
                    diag["rejected_cross_head_restrike"]+=1
                    continue
                if int(end)-last_native_release[ix] < c.native_release_guard_seconds*SR:
                    diag["rejected_native_tail"]+=1
                    continue
                quality=float(heads[t,ix,head]*post[t,ix,1]*gain[ix])
                proposals.append((quality,int(ix),head))
        if not proposals:
            diag["rejected_acoustic"]+=1
            previous=post[t].copy()
            continue
        # At most max_per_fragment different pitches; a head's state
        # cannot be modified by another head's candidate proposal.
        seen=set()
        for quality,ix,head in sorted(
                proposals,key=lambda z:(-z[0],z[1],z[2])):
            if ix in seen:
                continue
            if len(seen)>=c.max_per_fragment:
                break
            seen.add(ix)
            last[head-1,ix]=int(end)
            last_proposal[ix]=int(end)
            # Per-head single proposal at fragment; all other experts
            # remain eligible at later acoustic observations.
            result.append(dict(type="note_on",pitch=ix+40,
                decision_sample=int(end),available_sample=int(end),
                time=float(end/SR),velocity=96,
                reason="s743_multifundamental",
                s747_head=int(head),s747_mask=int(mask)))
            diag["per_head"][str(head)]=diag["per_head"].get(str(head),0)+1
        diag["proposed"]+=len(seen)
        previous=post[t].copy()
    final_signature=[(e["type"],int(e["pitch"]),
                      int(e["available_sample"]),e["reason"])
                     for e in reference]
    if original_signature!=final_signature:
        raise AssertionError("S747 mutated frozen reference events")
    return result,diag
