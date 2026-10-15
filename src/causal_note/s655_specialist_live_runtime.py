"""S655: exact strict LIVE implementation of S654 event risk selector.

The action ACCEPT/ABSTAIN is chosen when the S648 candidate first
appears, using the *same* S654 risk_vector. Both selections see only
PCM that has already arrived; original S614 remains a read-only shadow.
CPU processing latency on an actual microphone is NOT measured here.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import joblib
import numpy as np
from causal_note.s649_birth_live_runtime import S649CausalBirthStream
from causal_note.s648_event_birth import BirthPolicy,frame_features,HISTORY_COOLDOWN_HOPS
from causal_note.s654_pitch_risk_selector import (
    BASE_BIRTH,streaming_specialist_decision
)
from causal_note.strict_live import HOP


class S655CausalRiskStream(S649CausalBirthStream):
    def __init__(self,native_heads,birth_head,risk_head,risk_threshold=.75):
        if not 0<=risk_threshold<=1:
            raise ValueError("Risk selector must be in [0,1]")
        super().__init__(native_heads,birth_head,BASE_BIRTH)
        self.risk_head=risk_head
        self.risk_threshold=risk_threshold
        self.candidate_vetoes=0
        self.risk_acceptances=0

    @classmethod
    def from_verified_models(cls,native_heads,birth_path,birth_sha,
                             risk_path,risk_sha,threshold):
        models=[]
        for path,sha in ((birth_path,birth_sha),(risk_path,risk_sha)):
            content=Path(path).read_bytes()
            if hashlib.sha256(content).hexdigest()!=sha:
                raise ValueError("Original birth/risk checkpoint SHA mismatch")
            models.append(joblib.load(path))
        return cls(native_heads,*models,risk_threshold=threshold)

    def _frame(self,end,x):
        p,k=self.native.predict(x)
        features=frame_features(x,p,k,int(end),self.shadow,
                                self.past_x,self.past_p)
        q=np.asarray(self.model.predict_proba(features)[:,1],np.float32)
        amended=p.copy()
        possible=(~self.augmented.active)&(q>=self.policy.threshold)&(
            q-self.past_q>=self.policy.min_rise)&(
            end-self.augmented.last_on>=HISTORY_COOLDOWN_HOPS*HOP)
        candidates=np.flatnonzero(possible)
        chosen=-1
        if len(candidates):
            i=int(candidates[np.argmax(q[candidates])])
            if streaming_specialist_decision(self.risk_head,features,p,k,q,
                    self.past_q,i+40,self.risk_threshold):
                chosen=i
                amended[i,0]=self.augmented.config.onset
                amended[i,1]=max(float(amended[i,1]),
                                  self.augmented.config.active_on)
                self.risk_acceptances+=1
            else:self.candidate_vetoes+=1
        original=self.shadow.step(p,k,int(end),int(self.front.received))
        self.shadow_events.extend(original)
        assisted=self.augmented.step(amended,k,int(end),int(self.front.received))
        if chosen>=0 and any(e["type"]=="note_on" and int(e["pitch"])==chosen+40
                             for e in assisted):
            self.assisted_noteons+=1
        self.past_q=q.copy(); self.past_x=x.copy(); self.past_p=p.copy()
        return assisted
