"""S743: six causal fundamental-birth evidence heads, 32 dynamic fragment masks.

Research-only additive NoteOn proposals. H0 = frozen S708 (exact); H1-H5
are distinct measured acoustic evidence paths, not duplicated predictions.
Every four-hop fragment chooses a DIFFERENT H0-inclusive subset based only
on observed audio/context (never annotations), then the trained S708/S707
decoder emits all MIDI events. S708 frozen, no LIVE change or claim >80%.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from causal_note.strict_live import SR
from causal_note.s708_cardinality_selection import GuidedSelector
from causal_note.s736_fragment_mask import MASKS

MASK_LABELS=("S708","native","novelty","unshared","onset_rise","multi_partial")


def candidate_heads(post, native, context, sharp, scene):
    """Generate six truly different causal pitch-specific Onset offers [T,49,6].

    `sharp`: original signed S708 causal native 288D features. The first
    24 channels at offset 192 are three resolutions of eight partials.
    """
    p=np.asarray(post,np.float32)
    native=np.asarray(native,np.float32)
    ctx=np.asarray(context,np.float32)
    raw=np.asarray(sharp,np.float32)
    n=len(p)
    if p.shape!=(n,49,3) or native.shape!=(n,49,3) or \
       ctx.shape!=(n,49,16) or raw.shape!=(n,49,288) or len(scene)!=n \
       or not all(np.isfinite(z).all() for z in (p,native,ctx,raw)):
        raise ValueError("S743 native acoustic input violates the frozen S708 ABI")
    onset=np.clip(p[:,:,0],0,1)
    nat=np.clip(native[:,:,0],0,1)
    sustain=np.clip(p[:,:,1],0,1)
    novelty=np.clip(ctx[:,:,4],0,1)
    unshared=np.clip(np.stack([x.novelty_unshared for x in scene]),0,1)
    supports=np.stack([x.supports for x in scene])
    explained=np.clip(ctx[:,:,5],0,1)
    onrise=np.maximum(ctx[:,:,8],0.)
    rising_s=np.maximum(ctx[:,:,9],0.)

    # Independent fundamental partial growth is a causal *current vs prior*
    # difference, never a future FFT or a transposed copy of the candidate.
    partials=raw[:,:,192:216].reshape(n,49,3,8)
    prev=np.concatenate((partials[:1],partials[:-1]),axis=0)
    flux=np.maximum(partials-prev,0)
    first=flux[:,:,:,0].mean(axis=-1)
    second=flux[:,:,:,1].mean(axis=-1)
    third=flux[:,:,:,2].mean(axis=-1)
    two=(np.minimum(first,second)+np.minimum(first,third)+
         np.minimum(second,third))/3.
    # 0.2 scale is the empirically interpretable per-partial log rise; a
    # sigmoid/learned weight can be trained later on FIT only.
    fscore=np.clip(two/.20,0,1)
    credible=(supports>=2)&(sustain>.20)
    novelty_offer=np.where(credible,
                           np.maximum(onset,onset+.65*novelty),onset)
    unshared_offer=np.where(credible & (explained<.20),
                            np.maximum(onset,onset+.80*unshared),onset)
    rise_offer=np.where((onrise>.008)|(rising_s>.02),
                        np.maximum(onset,onset+.55*np.clip(
                            onrise+2*rising_s,0,1)),onset)
    partial_offer=np.where((fscore>.06)&(sustain>.15),
                           np.maximum(onset,onset+.70*fscore),onset)
    result=np.stack((onset,np.maximum(onset,nat),
                     novelty_offer,unshared_offer,rise_offer,
                     partial_offer),axis=-1)
    return np.asarray(np.clip(result,0,1),dtype=np.float32)


@dataclass(frozen=True)
class MultiHeadBirthPolicy:
    """Preset controls for controlled CAL ablations, no oracle at runtime."""
    k_tail: float=.35
    min_onset: float=.12
    min_sustain: float=.40
    mask_delta: float=.06
    mask_strategy: str="adaptive"
    min_rise: float=.00
    max_births: int=1

    def validate(self):
        if self.mask_strategy not in ("adaptive","physical","conservative"):
            raise ValueError("Unknown one-mask-per-fragment strategy")
        if not (0<=self.k_tail<=1 and 0<self.min_onset<1 and
                0<self.min_sustain<1 and 0<=self.mask_delta<=1 and
                0<=self.min_rise<=1 and 1<=self.max_births<=6):
            raise ValueError("Invalid S743 independent evidence policy")
        return self


def select_fragment_subset(head, posterior, context, policy):
    """Choose ONE of 32 H0-inclusive combinations for ONE fragment.

    Selection uses only head differences & acoustic physical evidence. The
    32-mask search space belongs to the per-fragment architecture; this
    S743 iteration audits deterministic selectors before learned FIT routing.
    """
    if head.shape!=(49,6) or posterior.shape!=(49,3) or context.shape!=(49,16):
        raise ValueError("Missing current physical 49-pitch fragment")
    gains=head[:,1:]-head[:,:1]
    m=1
    if policy.mask_strategy=="adaptive":
        for i in range(5):
            if float(np.max(gains[:,i]))>policy.mask_delta:
                m|=1<<(i+1)
    elif policy.mask_strategy=="physical":
        # H2/H3/H5 are acoustic multi-harmonic measurements. Permit H1
        # only when novel rather than copying native head unconditionally.
        for index in (2,3,5):
            if float(np.max(gains[:,index-1]))>policy.mask_delta:
                m|=1<<index
        if float(np.max(context[:,4]))>.12 and \
           float(np.max(gains[:,0]))>policy.mask_delta:
            m|=1<<1
    else:
        # Strict independent novelty + two-partial hypothesis only.
        for index in (3,5):
            if float(np.max(gains[:,index-1]))>policy.mask_delta:
                m|=1<<index
    if m not in MASKS:
        raise AssertionError("Expected one H0-anchored fragment mask")
    return m


class DynamicFundamentalSelector(GuidedSelector):
    """Original S708 decoder + causal physically confirmed missing-note births.

    The original source emits NoteOns first. Afterward S743 can add 1..N
    further pitch-distinct notes only when source-active count and learned
    S708 K posterior tail allow, with the chosen per-fragment head evidence.
    No audio or state beyond this decision is consulted.
    """
    def __init__(self,source_policy,birth_policy):
        super().__init__(source_policy)
        self.birth=birth_policy.validate()
        self.added=0
        self.masks=[]
        self.skipped_no_k=0
        self.skipped_no_evidence=0

    def step_extra(self,post,k,end,head,context):
        old=self.past[0] if self.past else np.zeros_like(post)
        rising=np.maximum(post[:,0]-old[:,0],0)
        rising_s=np.maximum(post[:,1]-old[:,1],0)
        source=super().step(post,k,end)
        mask=select_fragment_subset(head,post,context,self.birth)
        self.masks.append(int(mask))
        count=int(self.active.sum())
        if count>=6 or float(k[count+1:].sum())<self.birth.k_tail:
            self.skipped_no_k+=1
            return source
        chosen=[i for i in range(6) if mask&(1<<i)]
        candidate=np.max(head[:,chosen],axis=-1)
        # Require two separate acoustic signals for an absent note:
        # a pitch-specific onset proposal AND sustain evidence; candidate
        # is never inferred just from K or a transposed harmonic.
        good=(~self.active & self.armed &
              (candidate>=self.birth.min_onset) &
              (post[:,1]>=self.birth.min_sustain) &
              (post[:,2]<.65) &
              ((rising>=self.birth.min_rise) |
               (rising_s>=max(.008,self.birth.min_rise))) &
              (end-self.last_on>=.085*SR))
        if not np.any(good):
            self.skipped_no_evidence+=1
            return source
        pitches=sorted(np.flatnonzero(good),key=lambda i:(
            -float(candidate[i]*post[i,1]),int(i)))
        for pitch in pitches:
            if count>=6 or int(self.birth.max_births)<=0:
                break
            if float(k[count+1:].sum())<self.birth.k_tail:
                break
            source.append(dict(type="note_on",pitch=int(pitch+40),
                decision_sample=int(end),available_sample=int(end),
                time=float(end/SR),reason="s743_multifundamental",
                velocity=96))
            self.active[pitch]=True
            self.armed[pitch]=False
            self.low[pitch]=0
            self.last_on[pitch]=end
            self.added+=1
            count+=1
            if count>=6 or sum(e.get("reason")=="s743_multifundamental"
                               for e in source)>=self.birth.max_births:
                break
        return source


def decode_s743(ends,post,k,head,context,source_policy,birth_policy,samples):
    ends=np.asarray(ends,np.int64)
    n=len(ends)
    if (post.shape!=(n,49,3) or k.shape!=(n,7) or
        head.shape!=(n,49,6) or context.shape!=(n,49,16) or
        np.any(np.diff(ends)!=1024) or (n and ends[0]!=1024)):
        raise ValueError("Expected aligned causally received S708 four-hop blocks")
    selector=DynamicFundamentalSelector(source_policy,birth_policy)
    events=[]
    for t,end in enumerate(ends):
        events.extend(selector.step_extra(post[t],k[t],int(end),
                                          head[t],context[t]))
    events.extend(selector.finish(samples))
    return events,dict(
        added_births=selector.added,
        masks={str(m):selector.masks.count(m)
               for m in sorted(set(selector.masks))},
        blocked_K=selector.skipped_no_k,
        blocked_acoustic=selector.skipped_no_evidence)
