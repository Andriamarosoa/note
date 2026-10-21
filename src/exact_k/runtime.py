"""Exact-K ONLY — S834's proven K0..K6 signal at a true PCM256 cadence.

All five other musical characteristics have NO detector, selector, decoder,
state, output or tests in this branch. The S708 pretrained acoustic backbone
is retained only because the 75.780659% Exact-K result depends on it.
Its original training clock was 1024; all current decisions are fresh PCM256.
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
import time
import numpy as np

from .acoustics import CausalPitchFeatures,HOP,SR,sharp_track
from .temporal import IncrementalS708TemporalHead,latest_s708_acoustic_features

PAST_AUDIO=8192+HOP
SCHEMA="EXACT_K_ONLY_S834_TRUE_PCM256_V1"

@dataclass(frozen=True)
class ExactKDecision:
    sample: int
    count: int
    confidence: float
    def __post_init__(self):
        if self.sample < HOP or self.sample % HOP != 0 or self.count not in range(7):
            raise ValueError("Only causal aligned Exact-K decisions K0..K6 are valid")
        if not 0 <= self.confidence <= 1:
            raise ValueError("Invalid confidence")

class ExactKOnly256:
    """One S708 K posterior per completed 256 PCM samples; no foreign axes."""
    def __init__(self,native_heads,frozen_model):
        self.native_heads=native_heads
        self.frontend=CausalPitchFeatures(hop=HOP)
        self.temporal=IncrementalS708TemporalHead(frozen_model)
        self.native_feat=deque(maxlen=65)
        self.native_post=deque(maxlen=65)
        self.native_k=deque(maxlen=65)
        self.audio=np.empty(0,np.float32)
        self.received=0
        self.model_calls=0
        self.closed=False
        self.timings_ms=deque(maxlen=8192)
        self.last_decision=0
    def _step(self,end,f):
        native_post,native_k=self.native_heads.predict(f)
        self.native_feat.append(np.asarray(f,np.float32))
        self.native_post.append(np.asarray(native_post,np.float32))
        self.native_k.append(np.asarray(native_k,np.float32))
        acoustics=latest_s708_acoustic_features(
            self.native_feat,self.native_post,self.native_k)
        n=len(self.audio)
        sharp_ends=(np.array([n],np.int64) if end==HOP else
                    np.array([n-HOP,n],np.int64))
        spectral=sharp_track(self.audio,sharp_ends)[-1]
        features=np.concatenate((acoustics,spectral),axis=1)
        if features.shape!=(49,288) or not np.isfinite(features).all():
            raise AssertionError("S708 acoustic feature ABI changed")
        k=np.asarray(self.temporal.step(features),np.float32)
        self.model_calls+=1
        if k.shape!=(7,) or not np.isfinite(k).all() or abs(float(k.sum())-1)>1e-3:
            raise AssertionError("S708 K posterior ABI changed")
        if end!=self.last_decision+HOP:
            raise AssertionError("Exact-K did not emit at next native256 decision")
        self.last_decision=end
        index=int(np.argmax(k))
        return ExactKDecision(end,index,float(k[index]))
    def push(self,samples,*,start_sample):
        if self.closed:raise ValueError("Exact-K stream is already closed")
        if type(start_sample) is not int or start_sample!=self.received:
            raise ValueError("PCM input gap/reorder; reset stream")
        x=np.asarray(samples)
        if x.ndim!=1 or x.dtype.kind!="f" or len(x)>4096 or not np.isfinite(x).all():
            raise ValueError("Finite mono float PCM <=4096 samples required")
        begin=time.perf_counter()
        results=[]
        at=0
        while at<len(x):
            step=min(len(x)-at,HOP-self.frontend.received%HOP)
            block=np.asarray(x[at:at+step],np.float32)
            self.audio=np.concatenate((self.audio,block))[-PAST_AUDIO:]
            rows=self.frontend.push(block,start_sample=self.received)
            self.received+=step;at+=step
            for end,feat in rows:
                if end!=self.received or end%HOP:
                    raise AssertionError("Noncausal Exact-K feature timestamp")
                results.append(self._step(end,feat))
        self.timings_ms.append(1000*(time.perf_counter()-begin))
        return tuple(results)
    def finish(self):
        self.closed=True
        return ()
    def diagnostics(self):
        samples=np.asarray(self.timings_ms,np.float64)
        return dict(schema=SCHEMA,hop_samples=HOP,sampling_rate=SR,
            emitted_decisions=self.model_calls,added_future_samples=0,
            added_hold_samples=0,only_output="exact_k",
            public_midi_modified=False,source_retrained_256=False,
            p95_cpu_chunk_ms=float(np.percentile(samples,95)) if len(samples) else None,
            p99_cpu_chunk_ms=float(np.percentile(samples,99)) if len(samples) else None,
            hardware_iRig_p99_verified=False)
