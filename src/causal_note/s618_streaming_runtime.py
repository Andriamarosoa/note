"""S618 strict causal *streaming adapter* for original NOTE + trained attack head.

Separate experimental mode. Original StrictLiveStream, original S614 and
historical S80 remain untouched. Contains no post-attack timer, lookahead,
Basic Pitch, other repository model, or precomputed note/K sidecar.
"""
from __future__ import annotations
from collections import deque
import hashlib
from pathlib import Path
import time

import joblib
import numpy as np
from causal_note.strict_live import CausalPitchFeatures,Selection,SR
from causal_note.s618_attack_head import PICK,DIM,TrainedPolicy,NativeLearnedSelection


class CausalFeatureState:
    """Exact feature parity with batched input_features, O(49) bounded state."""
    def __init__(self):
        self.previous=np.zeros((49,3),dtype=np.float32)

    def frame(self,acoustics,prob):
        x=np.asarray(acoustics,np.float32)
        p=np.asarray(prob,np.float32)
        if x.shape!=(49,126) or p.shape!=(49,3):
            raise ValueError("Need native current NOTE features and heads")
        features=np.concatenate((p,p-self.previous,x[:,PICK]),axis=-1)
        assert features.shape==(49,DIM)
        self.previous=p.copy()
        return features


class ExperimentalS618LiveStream:
    """Converts received PCM to original heads + supplementary learned gate."""
    def __init__(self,original_heads,trained_head_path,trained_head_sha256,
                 policy:TrainedPolicy,selection=None):
        path=Path(trained_head_path)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=trained_head_sha256:
            raise ValueError("Wrong experimentally trained S618 checkpoint")
        self.head=joblib.load(path)
        self.original=original_heads
        self.front=CausalPitchFeatures()
        config=selection or Selection(**original_heads.metadata["selection"])
        self.sel=NativeLearnedSelection(config,policy)
        self.state=CausalFeatureState()
        self.closed=False
        self.inference_ms=deque(maxlen=4096)

    def push(self,samples,*,start_sample):
        if self.closed:
            raise ValueError("Stream is closed")
        frames=self.front.push(samples,start_sample=start_sample)
        output=[]
        for end,x in frames:
            start=time.perf_counter()
            prob,k=self.original.predict(x)
            row=self.state.frame(x,prob)
            q=self.head.predict_proba(row)[:,1]
            events=self.sel.step(prob,k,q,x,int(end))
            # Real availability is end of the received block, not retroactive.
            for event in events:
                event["available_sample"]=int(self.front.received)
                event["time"]=self.front.received/SR
            output.extend(events)
            self.inference_ms.append((time.perf_counter()-start)*1000)
        return output

    def finish(self):
        if self.closed:
            return []
        self.closed=True
        return self.sel.finish(self.front.received)
