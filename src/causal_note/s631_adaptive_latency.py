"""S631: learned adaptive post-attack decision head for native NOTE.

Candidate's observed attack starts a causal episode; each subsequent
right-aligned received PCM frame provides evidence to a TRAINED
WAIT/EMIT/REJECT head. NO future frames, oracle truth, or hindsight timestamps
are used at inference. The original S614/S80 weights remain byte-identical.

This is an experimental separate stream; not a promoted LIVE reference.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib
import time
from collections import deque
import numpy as np

from causal_note.strict_live import HOP,SR,CausalSelection,CausalPitchFeatures,Selection
from causal_note.s618_attack_head import PICK

WAIT,EMIT,REJECT=0,1,2
MAX_FRAMES=28                       # approximately 162.5 ms upper bound
FLOOR=.16
MIN_SUSTAIN=.13
GAP_FRAMES=12
FEATURE_DIM=len(PICK)+6+5+7+3


@dataclass(frozen=True)
class DecisionPolicy:
    emit_threshold:float=.55
    reject_threshold:float=.70
    max_frames:int=MAX_FRAMES

    def __post_init__(self):
        if not 0.<self.emit_threshold<1. or not 0.<self.reject_threshold<1.:
            raise ValueError("Probability thresholds must be in (0,1)")
        if not isinstance(self.max_frames,int) or not 1<=self.max_frames<=MAX_FRAMES:
            raise ValueError("Invalid maximum causal observation budget")


def candidate_features(current,previous,acoustic,k,birth,frame,peak):
    """Use only current/past observations; time is elapsed since CANDIDATE birth."""
    idx=birth["pitch"]
    neighbors=np.array([current[idx-d,1] if idx>=d else 0.
                        for d in (12,19,24)],np.float32)
    summary=np.asarray([min(1.,(frame-birth["frame"])*HOP/(.160*SR)),
                        peak[0],peak[1],current[idx,0]-birth["onset"],
                        float(np.max(np.delete(current[:,0],idx)))],np.float32)
    z=np.concatenate((acoustic[idx,PICK],current[idx],
                      current[idx]-previous[idx],summary,k,neighbors)).astype(np.float32)
    assert z.shape==(FEATURE_DIM,)
    if not np.isfinite(z).all():raise ValueError("Invalid causal candidate feature")
    return z


class CandidateTracker:
    """Causal proposals; identical generator for training and actual runtime."""
    def __init__(self,policy=DecisionPolicy()):
        self.policy=policy
        self.pending={}
        self.cooldown=np.full(49,-1000000,np.int32)
        self.previous=np.zeros((49,3),np.float32)
        self.frame=-1

    def step(self,prob,acoustic,k):
        p=np.asarray(prob,np.float32)
        x=np.asarray(acoustic,np.float32)
        k=np.asarray(k,np.float32)
        if p.shape!=(49,3) or x.shape!=(49,126) or k.shape!=(7,):
            raise ValueError("Native NOTE frame shapes required")
        self.frame+=1
        # Rising attack evidence, confirmed by any sustain evidence.
        # This is a fixed proposal strategy; the subsequent WAIT/EMIT/REJECT
        # decision (including its latency) is learned from real audio.
        start=(p[:,0]>=FLOOR)&(p[:,1]>=MIN_SUSTAIN)&(
            (p[:,0]-self.previous[:,0]>.045)|
            ((self.previous[:,0]<FLOOR)&(p[:,0]>=FLOOR)))
        for i in np.flatnonzero(start):
            i=int(i)
            if i in self.pending or self.frame-self.cooldown[i]<GAP_FRAMES:
                continue
            self.pending[i]={"pitch":i,"frame":self.frame,
                             "onset":float(p[i,0]),"peak":[float(p[i,0]),float(p[i,1])]}
        rows=[]
        for idx,episode in list(self.pending.items()):
            episode["peak"][0]=max(episode["peak"][0],float(p[idx,0]))
            episode["peak"][1]=max(episode["peak"][1],float(p[idx,1]))
            age=self.frame-episode["frame"]
            if age>=self.policy.max_frames:
                self.finish(idx)
                continue
            f=candidate_features(p,self.previous,x,k,episode,self.frame,episode["peak"])
            rows.append((idx,age,f,episode["frame"]))
        self.previous=p.copy()
        return rows

    def finish(self,pitch):
        self.pending.pop(pitch,None)
        self.cooldown[pitch]=self.frame


def trained_probabilities(model,rows):
    if not rows:return np.zeros((0,3),np.float32)
    features=np.stack([row[2] for row in rows]).astype(np.float32)
    probs=model.predict_proba(features)
    # Explicit class coverage check prevents accidental class remapping.
    if tuple(int(x) for x in model.classes_) != (WAIT,EMIT,REJECT):
        raise ValueError("Trained decision head missing an action class")
    return np.asarray(probs,np.float32)


class LearnedAdaptiveSelector:
    def __init__(self,model,config:Selection,policy=DecisionPolicy()):
        self.model=model
        self.selection=CausalSelection(config)
        self.tracker=CandidateTracker(policy)
        self.policy=policy
        self.latencies=[]
        self.decision_counts=np.zeros(3,np.int64)
        self.birth={}
        self.processed=0

    def step(self,p,k,x,end,available=None):
        if available is None:available=end
        rows=self.tracker.step(p,x,k)
        probabilities=trained_probabilities(self.model,rows)
        edited=np.asarray(p,np.float32).copy()
        # Keep original release/restrike for active notes; every new note
        # must be emitted on a learned decision for its native proposal.
        edited[~self.selection.active,0]=0.
        emit=[]
        for row,q in zip(rows,probabilities):
            pitch,age,_,born=row
            if self.selection.active[pitch]:
                self.tracker.finish(pitch)
                continue
            if q[EMIT]>=self.policy.emit_threshold and q[EMIT]>q[REJECT]:
                edited[pitch,0]=self.selection.config.onset
                edited[pitch,1]=max(edited[pitch,1],self.selection.config.active_on)
                emit.append((pitch,age,born))
                self.decision_counts[EMIT]+=1
            elif q[REJECT]>=self.policy.reject_threshold and q[REJECT]>q[EMIT]:
                self.tracker.finish(pitch)
                self.decision_counts[REJECT]+=1
            else:
                self.decision_counts[WAIT]+=1
        events=self.selection.step(edited,k,int(end),int(available))
        accepted={e["pitch"]-40 for e in events if e["type"]=="note_on"}
        for pitch,age,born in emit:
            if pitch in accepted:
                self.tracker.finish(pitch)
                self.latencies.append(float(age*HOP/SR*1000))
        self.processed+=1
        return events

    def finish(self,total):
        return self.selection.panic(int(total))


def eval_stream(ends,x,p,k,model,selection,policy,total):
    decoder=LearnedAdaptiveSelector(model,selection,policy)
    events=[]
    for end,ac,prob,dist in zip(ends,x,p,k):
        events.extend(decoder.step(prob,dist,ac,int(end),int(end)))
    events.extend(decoder.finish(total))
    return events,decoder


class S631Live:
    def __init__(self,original_heads,trained_path,expected_sha256,policy=DecisionPolicy()):
        data=Path(trained_path).read_bytes()
        if hashlib.sha256(data).hexdigest()!=expected_sha256:
            raise ValueError("Learned action model integrity mismatch")
        import joblib
        self.frontend=CausalPitchFeatures()
        self.original=original_heads
        self.decoder=LearnedAdaptiveSelector(joblib.load(trained_path),
                         Selection(**original_heads.metadata["selection"]),policy)
        self.closed=False
        self.processing_ms=deque(maxlen=8192)

    def push(self,pcm,*,start_sample):
        if self.closed:raise ValueError("Closed NOTE stream")
        tick=time.perf_counter()
        frames=self.frontend.push(pcm,start_sample=start_sample)
        events=[]
        for end,features in frames:
            p,k=self.original.predict(features)
            events.extend(self.decoder.step(p,k,features,end,
                                           self.frontend.received))
        self.processing_ms.append((time.perf_counter()-tick)*1000)
        return events

    def finish(self):
        if self.closed:return []
        self.closed=True
        return self.decoder.finish(self.frontend.received)
