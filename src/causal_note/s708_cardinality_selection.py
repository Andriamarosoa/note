"""Causal K-supported weak-attack/release selection, with acoustic evidence.

K never forces a note into existence: the attack, sustain and rising evidence
tests must all pass. The one active-note state still determines every event.
"""
from collections import deque
from dataclasses import dataclass
import numpy as np
from causal_note.strict_live import SR
from causal_note.s704_joint_heads import JointPolicy,JointSelector


@dataclass(frozen=True)
class GuidedPolicy(JointPolicy):
    guide_confidence: float = .70
    weak_onset: float = .25
    weak_sustain: float = .65
    rise: float = .04
    trim: bool = False


class GuidedSelector(JointSelector):
    def __init__(self,policy, *, decision_hop_samples=4*256):
        super().__init__(policy, decision_hop_samples=decision_hop_samples)
        self.past=deque(maxlen=2)
        self.lower_count_steps=0

    def step(self,post,count,end):
        events=super().step(post,count,end)
        c=self.policy;on,sustain,release=post.T
        old=self.past[0] if self.past else np.zeros_like(post)
        rising=(on-old[:,0]>=c.rise) | (sustain-old[:,1]>=2.5*c.rise)
        self.past.append(np.asarray(post).copy())
        target=int(count.argmax())
        confidence=float(count.max())

        def emit(kind,i,reason):
            events.append(dict(type=kind,pitch=int(i+40),decision_sample=int(end),
                available_sample=int(end),time=float(end/SR),reason=reason,
                velocity=96 if kind=='note_on' else 0))

        lower=confidence>=.85 and target<int(self.active.sum())
        self.lower_count_steps=self.lower_count_steps+1 if lower else 0
        if c.trim and self.lower_count_steps>=2:
            weak=self.active & (sustain<.55) & (release>.30) & (end-self.last_on>=.075*SR)
            for i in sorted(np.flatnonzero(weak),key=lambda j:float(sustain[j])):
                if int(self.active.sum())<=target:break
                emit('note_off',i,'count_supported_release')
                self.active[i]=False

        if confidence>=c.guide_confidence and int(self.active.sum())<target:
            ready=(~self.active & self.armed & rising & (on>=c.weak_onset)
                   & (sustain>=c.weak_sustain) & (release<.50)
                   & (end-self.last_on>=.12*SR))
            for i in sorted(np.flatnonzero(ready),key=lambda j:(-float(on[j]*sustain[j]),int(j))):
                if int(self.active.sum())>=min(6,target):break
                emit('note_on',i,'count_supported_acoustic_attack')
                self.active[i]=True;self.armed[i]=False;self.low[i]=0;self.last_on[i]=end
        return events


def decode_guided(ends,post,count,policy,samples):
    selector=GuidedSelector(policy);events=[]
    for end,p,k in zip(ends,post,count):
        events.extend(selector.step(p,k,int(end)))
    events.extend(selector.finish(samples))
    return events
