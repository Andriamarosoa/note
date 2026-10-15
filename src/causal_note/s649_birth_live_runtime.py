"""S649: real block-by-block causal NOTE S648 experimental runtime.

Original shadow S614 events are retained as an independently observable
control. The candidate head is evaluated only on current PCM and past
information; the generated NOTE MIDI events are never backdated.
Production Windows/loopback latency is NOT measured by these tests.
"""
from __future__ import annotations
from collections import deque
import time
import hashlib
from pathlib import Path
import joblib
import numpy as np
from causal_note.strict_live import CausalPitchFeatures,CausalSelection,Selection,HOP,SR
from causal_note.s648_event_birth import frame_features,BirthPolicy,HISTORY_COOLDOWN_HOPS


class S649CausalBirthStream:
    def __init__(self,native_heads,event_head,policy:BirthPolicy=BirthPolicy()):
        self.native=native_heads
        self.model=event_head
        self.policy=policy
        cfg=Selection(**native_heads.metadata["selection"])
        self.front=CausalPitchFeatures()
        self.shadow=CausalSelection(cfg)
        self.augmented=CausalSelection(cfg)
        self.past_x=np.zeros((49,126),np.float32)
        self.past_p=np.zeros((49,3),np.float32)
        self.past_q=np.zeros(49,np.float32)
        self.timings_ms=deque(maxlen=10000)
        self.shadow_events=[]
        self.assisted_noteons=0
        self.closed=False

    @classmethod
    def from_verified_checkpoint(cls,native_heads,model_path,sha256,policy):
        path=Path(model_path)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=sha256:
            raise ValueError("Experimental S648 checkpoint SHA mismatch")
        return cls(native_heads,joblib.load(path),policy)

    def _frame(self,end,x):
        p,k=self.native.predict(x)
        features=frame_features(x,p,k,int(end),self.shadow,self.past_x,self.past_p)
        changed=p.copy()
        chosen=-1
        if self.policy.threshold < 1.:
            q=np.asarray(self.model.predict_proba(features)[:,1],np.float32)
            candidate=(~self.augmented.active)&(q>=self.policy.threshold)&(
                q-self.past_q>=self.policy.min_rise)&(
                end-self.augmented.last_on>=HISTORY_COOLDOWN_HOPS*HOP)
            picks=np.flatnonzero(candidate)
            if len(picks):
                chosen=int(picks[np.argmax(q[picks])])
                changed[chosen,0]=self.augmented.config.onset
                changed[chosen,1]=max(float(changed[chosen,1]),
                                       self.augmented.config.active_on)
            self.past_q=q
        original=self.shadow.step(p,k,int(end),int(self.front.received))
        self.shadow_events.extend(original)
        assisted=self.augmented.step(changed,k,int(end),int(self.front.received))
        if chosen>=0 and any(e["type"]=="note_on" and e["pitch"]==chosen+40
                            for e in assisted):
            self.assisted_noteons+=1
        self.past_x=x.copy()
        self.past_p=p.copy()
        return assisted

    def push(self,samples,*,start_sample):
        if self.closed:raise ValueError("S649 birth stream is closed")
        start=time.perf_counter()
        output=[]
        for end,x in self.front.push(samples,start_sample=start_sample):
            output.extend(self._frame(end,x))
        self.timings_ms.append((time.perf_counter()-start)*1000)
        if any(e["available_sample"]>self.front.received for e in output):
            raise AssertionError("Emitting not-yet-seen audio event")
        return output

    def finish(self):
        if self.closed:return []
        self.closed=True
        self.shadow_events.extend(self.shadow.panic(self.front.received))
        return self.augmented.panic(self.front.received)
