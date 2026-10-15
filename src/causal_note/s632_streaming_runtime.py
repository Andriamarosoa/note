"""S632 strict-live risk-to-go inference from real received PCM.

The original S614/S80 checkpoint is read unchanged. Models are loaded only
with verified hashes from the S632 TRAIN-only artifact. Each new PCM block
triggers a *causal* decision, and the model is evaluated on candidate
features available at this block only.

The batched DEV comparison may precompute costs for speed, but this module
is the real-time reference implementation and emits identical events at
per-HOP reception.
"""
from __future__ import annotations
from collections import deque,defaultdict
import hashlib
from pathlib import Path
import time
import joblib
import numpy as np

from causal_note.strict_live import CausalPitchFeatures,CausalSelection,Selection,SR
from causal_note.s631_adaptive_latency import CandidateTracker,DecisionPolicy
from causal_note.s632_risk_policy import RiskPolicy


class StreamingCostSelector:
    def __init__(self,models,selection,policy:RiskPolicy):
        if len(models)!=3:raise ValueError("Require trained EMIT, REJECT, WAIT costs")
        self.models=models
        self.policy=policy
        self.state=CausalSelection(selection)
        self.tracker=CandidateTracker(DecisionPolicy(max_frames=28))
        self.decided=set()
        self.latencies=[]
        self.actions=defaultdict(int)

    def step(self,p,k,features,end,available=None):
        if available is None:available=end
        rows=self.tracker.step(p,features,k)
        changed=np.asarray(p,np.float32).copy()
        if self.policy.mode=="replace":
            changed[~self.state.active,0]=0.
        proposals=[]
        if rows:
            x=np.stack([r[2] for r in rows]).astype(np.float32)
            risks=np.clip(np.column_stack([m.predict(x) for m in self.models]),0.,10.)
            for (pitch,age,vec,birth),scores in zip(rows,risks):
                episode=(int(pitch),int(birth))
                if episode in self.decided:continue
                if self.state.active[pitch]:
                    self.decided.add(episode)
                    continue
                e,r,w=(float(v) for v in scores)
                w+=self.policy.extra_wait_cost
                if age>=self.policy.max_frames:
                    w=float("inf")
                if e+self.policy.emit_margin<min(r,w) and e<=self.policy.max_emit_cost:
                    if p[pitch,0]>=.11 and p[pitch,1]>=.19:
                        changed[pitch,0]=max(changed[pitch,0],self.state.config.onset)
                        changed[pitch,1]=max(changed[pitch,1],self.state.config.active_on)
                        proposals.append((episode,int(pitch),int(age)))
                    else:self.actions["waits"]+=1
                elif r<w and r<e:
                    self.decided.add(episode)
                    self.actions["rejects"]+=1
                else:self.actions["waits"]+=1
        events=self.state.step(changed,k,int(end),int(available))
        emitted={int(e["pitch"])-40 for e in events if e["type"]=="note_on"}
        for episode,pitch,age in proposals:
            if pitch in emitted:
                self.decided.add(episode)
                self.actions["emits"]+=1
                self.latencies.append(float(age*256/SR*1000))
        for pitch,age,vec,birth in rows:
            if pitch in emitted:self.decided.add((int(pitch),int(birth)))
        # bounded completed-state memory; only currently pending episodes
        # can be revisited, so older decisions never matter again.
        pending={(p,int(info["frame"])) for p,info in self.tracker.pending.items()}
        self.decided.intersection_update(pending)
        return events

    def finish(self,total):
        return self.state.panic(int(total))


class S632Live:
    def __init__(self,native_heads,cost_paths,sha256s,policy=RiskPolicy()):
        if len(cost_paths)!=3 or len(sha256s)!=3:
            raise ValueError("Three cost models with exact SHA256 required")
        models=[]
        for path,digest in zip(cost_paths,sha256s):
            source=Path(path)
            if hashlib.sha256(source.read_bytes()).hexdigest()!=digest:
                raise ValueError("Unverified cost-head model")
            models.append(joblib.load(source))
        self.frontend=CausalPitchFeatures()
        self.original=native_heads
        self.selector=StreamingCostSelector(
            models,Selection(**native_heads.metadata["selection"]),policy)
        self.closed=False
        self.timings_ms=deque(maxlen=4096)

    def push(self,pcm,*,start_sample):
        if self.closed:raise ValueError("Closed NOTE stream")
        before=time.perf_counter()
        frames=self.frontend.push(pcm,start_sample=start_sample)
        events=[]
        for end,feature in frames:
            p,k=self.original.predict(feature)
            events.extend(self.selector.step(p,k,feature,int(end),
                                             int(self.frontend.received)))
        self.timings_ms.append((time.perf_counter()-before)*1000)
        return events

    def finish(self):
        if self.closed:return []
        self.closed=True
        return self.selector.finish(self.frontend.received)
