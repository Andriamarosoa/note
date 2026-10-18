"""S719 shadow-only S708 selector: conditional evidence-backed retrigger arm.

Do not use this selector in LIVE by default. The original GuidedSelector and
S708 CAL-frozen policy are unchanged; only an explicit CAL-frozen ablation
may consider promotion after six same-protocol emitted-event metrics.
"""
from __future__ import annotations
import numpy as np

from causal_note.strict_live import SR
from causal_note.s708_cardinality_selection import GuidedSelector


class ShadowReattackSelector(GuidedSelector):
    def __init__(self, policy, *, rise=.20, min_gap_ms=120):
        super().__init__(policy)
        if not (.04 <= rise <= .50 and 80 <= min_gap_ms <= 300):
            raise ValueError("Unfrozen S719 retrigger variant out of trial bounds")
        self.reattack_rise=float(rise)
        self.min_gap_ms=int(min_gap_ms)
        self.trial_rearm_proposals=0
        self.trial_reattack_emissions=0

    def step(self, post, count, end):
        # Sound-derived S708 posterior only; no labels or future PCM.
        x=np.asarray(post,dtype=np.float32)
        if x.shape != (49,3) or not np.isfinite(x).all():
            raise ValueError("S719 frozen 49x3 acoustic posterior ABI mismatch")
        old=self.past[-1] if self.past else np.zeros_like(x)
        on,sustain,release=x.T
        extra=(on>=self.policy.onset) & (sustain>=self.policy.sustain_on)
        extra &= (release<.50) & (on-old[:,0]>=self.reattack_rise)
        extra &= (end-self.last_on)>=self.min_gap_ms*SR/1000
        # Do not count the ordinary rearm as a "new" S719 decision.
        extra &= ~self.armed & ~(on < self.policy.onset*.4)
        rearmed=np.flatnonzero(extra)
        self.trial_rearm_proposals+=len(rearmed)
        self.armed[rearmed]=True
        events=super().step(post,count,end)
        self.trial_reattack_emissions+=sum(
            e["type"]=="note_on" and int(e["pitch"])-40 in rearmed for e in events)
        return events


def decode_shadow(ends,post,count,policy,samples,*,rise,min_gap_ms):
    selector=ShadowReattackSelector(policy,rise=rise,min_gap_ms=min_gap_ms)
    events=[]
    for end,p,k in zip(ends,post,count):
        events.extend(selector.step(p,k,int(end)))
    events.extend(selector.finish(samples))
    return events,dict(rearm_proposals=selector.trial_rearm_proposals,
                       actual_retriggers=selector.trial_reattack_emissions)
