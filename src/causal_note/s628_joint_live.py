"""S628 isolated *strict LIVE* adapter for S627 joint attack/event K head.

Neither the original NOTE S614 stream nor historical S80 changes.
New event K is a soft new-onset cue, not an active-note budget or an
oracle count. Every NOTE event uses only PCM already received.
"""
from __future__ import annotations
from collections import deque
import hashlib
from pathlib import Path
import time
import numpy as np
from causal_note.strict_live import CausalPitchFeatures,CausalSelection,Selection,SR
from causal_note.s627_joint_heads import JointNPZ,Fusion,fuse


class S628JointLiveStream:
    def __init__(self,original_heads,joint_npz,expected_sha256,policy=Fusion(),
                 selection=None):
        path=Path(joint_npz)
        actual=hashlib.sha256(path.read_bytes()).hexdigest()
        if expected_sha256 and actual != expected_sha256:
            raise ValueError("Unexpected S627 original auxiliary checkpoint")
        self.checkpoint_sha256=actual
        self.heads=original_heads
        self.joint=JointNPZ(path)
        self.frontend=CausalPitchFeatures()
        self.selector=CausalSelection(
            selection if selection is not None else
            Selection(**original_heads.metadata["selection"]))
        self.policy=policy
        self.closed=False
        self.time_ms=deque(maxlen=8192)

    def push(self,samples,*,start_sample):
        if self.closed:raise ValueError("Stream closed")
        begin=time.perf_counter()
        frames=self.frontend.push(samples,start_sample=start_sample)
        events=[]
        for end,acoustic in frames:
            p,k=self.heads.predict(acoustic)
            q,new_event_k=self.joint.frame(acoustic,p)
            revised=fuse(p,q,new_event_k,self.policy)
            # Actual time of audio availability: never fabricate lookahead.
            events.extend(self.selector.step(revised,k,int(end),
                                             int(self.frontend.received)))
        self.time_ms.append((time.perf_counter()-begin)*1000)
        return events

    def finish(self,reason="stream_end"):
        if self.closed:return []
        self.closed=True
        return self.selector.panic(self.frontend.received,reason)

    def compute_p95_ms(self):
        return float(np.percentile(self.time_ms,95)) if self.time_ms else 0.
