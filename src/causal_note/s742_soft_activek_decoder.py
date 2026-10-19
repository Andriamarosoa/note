"""S742 experimental causal *soft-K tail* weak note-birth rescue.

S741 mostly changed K posteriors without changing emitted active note counts.
This shadow decoder preserves S708's full strong onsets and original S707
NoteOff policy, but allows extra *independently acoustic-confirmed* NoteOn
when the total probability of larger K supports an additional note.
No K posterior alone may create a note; no oracle truth enters inference.
Production LIVE remains completely untouched until paired six-score tests.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from causal_note.strict_live import SR
from causal_note.s708_cardinality_selection import GuidedSelector, GuidedPolicy

@dataclass(frozen=True)
class SoftTailPolicy:
    tail_threshold: float = .65
    onset: float = .22
    sustain: float = .65
    release_max: float = .50
    rise: float = .04
    max_new_per_fragment: int = 1

    def validate(self):
        fields=(self.tail_threshold,self.onset,self.sustain,
                self.release_max,self.rise)
        if (not all(np.isfinite(v) for v in fields)
            or not 0. <= self.tail_threshold <= 1.
            or not 0. <= self.onset <= 1.
            or not 0. <= self.sustain <= 1.
            or not 0. <= self.release_max <= 1.
            or not 0. <= self.rise <= 1.
            or not 1 <= self.max_new_per_fragment <= 6):
            raise ValueError("Invalid causal weak-K rescue thresholds")
        return self


class SoftTailGuidedSelector(GuidedSelector):
    """Separate optional NOTE research decoder; S708 GuidedSelector unchanged."""

    def __init__(self, policy: GuidedPolicy, extras: SoftTailPolicy):
        super().__init__(policy)
        self.extras=extras.validate()
        self.rescued=0
        self.veto_low_tail=0
        self.veto_no_acoustic=0

    def step(self,post,k,end):
        post=np.asarray(post,dtype=np.float32)
        k=np.asarray(k,dtype=np.float32)
        if (post.shape!=(49,3) or k.shape!=(7,) or
            not np.isfinite(post).all() or not np.isfinite(k).all() or
            not np.allclose(k.sum(),1.,atol=1e-3) or (k<0).any()):
            raise ValueError("Frozen S708 source probability ABI")
        # Measure novelty BEFORE the original fixed decoder advances state.
        old=self.past[0] if self.past else np.zeros_like(post)
        e=self.extras
        prev_on=post[:,0]-old[:,0]
        prev_sustain=post[:,1]-old[:,1]
        rise=(prev_on>=e.rise)|(prev_sustain>=2.5*e.rise)
        original=super().step(post,k,end)
        active=int(self.active.sum())
        if active>=6:
            return original
        tail=float(k[active+1:].sum())
        if tail<e.tail_threshold:
            self.veto_low_tail+=1
            return original
        candidates=(~self.active&self.armed&rise&
                    (post[:,0]>=e.onset)&(post[:,1]>=e.sustain)&
                    (post[:,2]<e.release_max)&
                    (end-self.last_on>=.12*SR))
        if not np.any(candidates):
            self.veto_no_acoustic+=1
            return original
        # Weighted acoustic strength. Never use oracle pitch labels.
        order=sorted(np.flatnonzero(candidates),key=lambda ix:(
            -float(post[ix,0]*post[ix,1]),int(ix)))
        remaining=e.max_new_per_fragment
        # Upper bound for expected K isn't a hard target. A probability
        # tail is recalculated as each note is added.
        for ix in order:
            if remaining<=0 or self.active.sum()>=6:
                break
            now=int(self.active.sum())
            if float(k[now+1:].sum())<e.tail_threshold:
                break
            original.append(dict(
                type="note_on",pitch=int(ix+40),
                decision_sample=int(end),available_sample=int(end),
                time=float(end/SR),reason="s742_soft_K_plus_acoustics",
                velocity=96))
            self.active[ix]=True
            self.armed[ix]=False
            self.low[ix]=0
            self.last_on[ix]=end
            self.rescued+=1
            remaining-=1
        return original


def decode_soft_tail(ends,post,k,policy,samples,extras):
    """One complete causal MIDI source, original S707 NoteOff comes later."""
    ends=np.asarray(ends,dtype=np.int64)
    post=np.asarray(post,dtype=np.float32)
    k=np.asarray(k,dtype=np.float32)
    if (ends.ndim!=1 or len(ends)!=len(post) or
        len(ends)!=len(k)):
        raise ValueError("Expected one K/post for each causal four-hop block")
    decoder=SoftTailGuidedSelector(policy,extras)
    emitted=[]
    for end,p,counts in zip(ends,post,k):
        emitted.extend(decoder.step(p,counts,int(end)))
    emitted.extend(decoder.finish(samples))
    return emitted,dict(
        s742_soft_births=decoder.rescued,
        low_K_tail_veto=decoder.veto_low_tail,
        no_acoustic_veto=decoder.veto_no_acoustic)
