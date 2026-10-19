"""S745: FIT-prior-corrected causal specialist for missing note births.

S744 trained on an oversampled, 4x-weighted attack dataset and produced
~1400-1800 new attacks per CAL32. Here FIT-only natural vs weighted
sampling priors adjust positive *odds*, not the raw acoustic head logits.
This is an approximate case-control correction, not an assertion of
perfect calibration; hard-negative sampling is conditional on acoustics.

All new NoteOns still require independent harmonic/partial evidence, K
posterior mass, sustain, and a causal source state. S708 and S707 frozen.
The masks are selected per four-hop fragment from 32 H0-inclusive offers;
this is shadow-only, no automatic LIVE, distillation, or model promotion.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from causal_note.strict_live import SR
from causal_note.s708_cardinality_selection import GuidedSelector
from causal_note.s743_fundamental_router import (
    MultiHeadBirthPolicy,select_fragment_subset)

def fit_prior_correction(natural_positive,total_eligible,
                         sampled_positive,sampled_total,positive_weight=4.):
    """Return positive-odds multiplier using only genuinely FIT counts."""
    a,b,c,d=map(int,(natural_positive,total_eligible,
                     sampled_positive,sampled_total))
    if not (0<a<b and 0<c<d and positive_weight>=1):
        raise ValueError("Unusable FIT-only prevalence")
    population_prior=a/b
    model_weighted_prior=(c*positive_weight)/(
        c*positive_weight + d-c)
    factor=(population_prior/(1-population_prior))/(
        model_weighted_prior/(1-model_weighted_prior))
    if not np.isfinite(factor) or factor<=0:
        raise ValueError("Unusable prior ratio")
    return float(factor),dict(
        natural_prior=float(population_prior),
        model_weighted_prior=float(model_weighted_prior),
        prior_odds_multiplier=float(factor),
        estimated_calibration=True,truth_source="FIT80 ONLY")


def corrected_positive_probability(prob,odds_ratio):
    p=np.asarray(prob,np.float32)
    if (not np.isfinite(p).all() or np.any(p<0) or np.any(p>1) or
        not np.isfinite(odds_ratio) or odds_ratio<=0):
        raise ValueError("Invalid positive prior calibration")
    numerator=p*float(odds_ratio)
    result=numerator/(1-p+numerator).clip(min=1e-12)
    return np.clip(result,0.,1.).astype(np.float32)


def corrected_specialist_track(model,features,odds_ratio,*,batch=65536):
    x=np.asarray(features,np.float32)
    if x.ndim!=3 or x.shape[1]!=49:
        raise ValueError("Expected [fragments,49,acoustic features]")
    flat=x.reshape(-1,x.shape[-1])
    p=np.empty(len(flat),np.float32)
    for i in range(0,len(flat),batch):
        predicted=model.predict_proba(flat[i:i+batch])
        if predicted.shape!=(min(batch,len(flat)-i),2):
            raise ValueError("Incorrect learned specialist ABI")
        p[i:i+len(predicted)]=predicted[:,1]
    return corrected_positive_probability(p.reshape(len(x),49),odds_ratio)


@dataclass(frozen=True)
class CalibratedBirthPolicy:
    min_probability: float=.02
    k_tail: float=.35
    min_acoustic: float=.13
    min_sustain: float=.35
    min_rise: float=.015
    mask_delta: float=.04
    mode: str="adaptive"
    max_new: int=1

    def validate(self):
        for value in (self.min_probability,self.k_tail,
                      self.min_acoustic,self.min_sustain,
                      self.min_rise,self.mask_delta):
            if not np.isfinite(value) or value<0 or value>1:
                raise ValueError("Invalid S745 physical probability")
        if self.mode not in ("adaptive","physical","conservative"):
            raise ValueError("Invalid mask router choice")
        if not 1<=self.max_new<=6:
            raise ValueError("S745 per-fragment active-note budget")
        return self


class CalibratedBirthSelector(GuidedSelector):
    def __init__(self,policy,extras):
        super().__init__(policy)
        self.extras=extras.validate()
        self.mask_hist={}
        self.births=0
        self.no_K=0
        self.no_acoustic=0
        self.no_risk=0

    def step_checked(self,post,k,end,heads,context,p_positive):
        if (post.shape!=(49,3) or k.shape!=(7,) or
            heads.shape!=(49,6) or context.shape!=(49,16) or
            p_positive.shape!=(49,) or
            not np.isfinite(p_positive).all() or
            not np.isfinite(heads).all()):
            raise ValueError("Invalid causal trained-note head")
        # Only current or previous observed state can support an attack.
        old=self.past[0] if self.past else np.zeros_like(post)
        rise=np.maximum(post[:,0]-old[:,0],0)
        rise_s=np.maximum(post[:,1]-old[:,1],0)
        events=super().step(post,k,end)
        e=self.extras
        mask=select_fragment_subset(
            heads,post,context,MultiHeadBirthPolicy(
                k_tail=e.k_tail,min_onset=e.min_acoustic,
                min_sustain=e.min_sustain,mask_delta=e.mask_delta,
                mask_strategy=e.mode,min_rise=e.min_rise))
        self.mask_hist[str(mask)]=self.mask_hist.get(str(mask),0)+1
        active=int(self.active.sum())
        if active>=6 or float(k[active+1:].sum())<e.k_tail:
            self.no_K+=1
            return events
        selected=[j for j in range(6) if mask&(1<<j)]
        acoustic=heads[:,selected].max(-1)
        physical=(~self.active & self.armed &
            (post[:,1]>=e.min_sustain)&(post[:,2]<.65)&
            (acoustic>=e.min_acoustic)&
            ((rise>=e.min_rise)|(rise_s>=2*e.min_rise)|
             ((context[:,4]>.04)&(context[:,7]>=.25)))&
            (end-self.last_on>=.085*SR))
        if not np.any(physical):
            self.no_acoustic+=1
            return events
        candidates=physical & (p_positive>=e.min_probability)
        if not np.any(candidates):
            self.no_risk+=1
            return events
        order=sorted(np.flatnonzero(candidates),key=lambda ix:(
            -float(p_positive[ix]*acoustic[ix]*post[ix,1]),int(ix)))
        for ix in order[:e.max_new]:
            if active>=6 or float(k[active+1:].sum())<e.k_tail:
                break
            events.append(dict(type="note_on",pitch=int(ix+40),
                decision_sample=int(end),available_sample=int(end),
                time=float(end/SR),reason="s745_calibrated_true_fundamental",
                velocity=96))
            self.active[ix]=True;self.armed[ix]=False
            self.low[ix]=0;self.last_on[ix]=end
            self.births+=1;active+=1
        return events


def decode_s745(ends,post,k,heads,context,p_positive,
                source_policy,extra_policy,samples):
    n=len(ends)
    if (post.shape!=(n,49,3) or k.shape!=(n,7) or
        heads.shape!=(n,49,6) or context.shape!=(n,49,16) or
        p_positive.shape!=(n,49) or not np.isfinite(p_positive).all()):
        raise ValueError("Incompatible active K specialist inputs")
    select=CalibratedBirthSelector(source_policy,extra_policy)
    events=[]
    for t,end in enumerate(ends):
        events.extend(select.step_checked(
            post[t],k[t],int(end),heads[t],context[t],p_positive[t]))
    events.extend(select.finish(samples))
    return events,dict(
        added_births=select.births,
        blocked_K=select.no_K,
        blocked_acoustic=select.no_acoustic,
        blocked_specialist_risk=select.no_risk,
        masks=dict(select.mask_hist))
