
"""Active Exact-K only: private original S708+native256 acoustic K source."""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
import time
import numpy as np
from .acoustics import CausalPitchFeatures,HOP,SR,sharp_track
from .temporal import IncrementalS708TemporalHead,latest_s708_acoustic_features
SCHEMA="ACTIVE_K_ONLY_S828_NATIVE256_V1"
PITCHES=49
PAST_AUDIO=8192+HOP
class ActiveKDecision:
    decision_sample:int
    count:int
    model_count:int
    model_confidence:float
    source_abi:str=SCHEMA


class PrivateS708GuidedActiveK:
    """Original S708 JointSelector+GuidedSelector copied into private state.

    Original behavior is verifiable against GuidedSelector on the 1024
    source grid. Only the strict frame-cadence assertion is parametrized.
    Public S708 classes and their global 4x256 constant are UNCHANGED.
    """

    def __init__(self,policy,*,decision_hop=HOP):
        if decision_hop not in (HOP,4*HOP):
            raise ValueError("only exact MIDI256 or S708 source1024 supported")
        self.policy=policy
        self.decision_hop=decision_hop
        self.active=np.zeros(PITCHES,bool)
        self.armed=np.ones(PITCHES,bool)
        self.low=np.zeros(PITCHES,np.int32)
        self.last_on=np.full(PITCHES,-10**12,np.int64)
        self.last_end=0
        self.past=deque(maxlen=2)
        self.lower_count_steps=0
        self.step_calls=0

    def step(self,post,count,end):
        if end!=self.last_end+self.decision_hop:
            raise ValueError("non-contiguous Active-K 256 decision grid")
        p=np.asarray(post)
        k=np.asarray(count)
        if (p.shape!=(PITCHES,3) or k.shape!=(7,) or not
            np.isfinite(p).all() or not np.isfinite(k).all()):
            raise ValueError("must use original finite S708 49x3 and K0..6")
        self.last_end=int(end)
        c=self.policy
        on,sustain,release=p.T

        # JointSelector's original 1024-code path, isolated as Active-K state.
        self.armed|=on<c.onset*.4
        self.low=np.where((sustain<c.sustain_off)|(release>=c.release),
                          self.low+1,0)
        old_enough=end-self.last_on>=.035*SR
        dying=np.flatnonzero(self.active&old_enough&(self.low>=c.release_steps))
        self.active[dying]=False
        ready=(on>=c.onset)&(sustain>=c.sustain_on)&self.armed
        ready&=end-self.last_on>=.065*SR
        budget=6
        if k.max()>=c.count_confidence:
            budget=max(int(self.active.sum()),min(6,int(k.argmax())))
        for pitch in sorted(np.flatnonzero(ready),
                  key=lambda j:(-float(on[j]*sustain[j]),int(j))):
            if not self.active[pitch] and int(self.active.sum())>=budget:
                continue
            self.active[pitch]=True
            self.armed[pitch]=False
            self.low[pitch]=0
            self.last_on[pitch]=int(end)

        # GuidedSelector's exact S708 weak attack/trim state transitions.
        prev=self.past[0] if self.past else np.zeros_like(p)
        rising=(on-prev[:,0]>=c.rise)|(sustain-prev[:,1]>=2.5*c.rise)
        self.past.append(np.asarray(p).copy())
        target=int(k.argmax())
        confidence=float(k.max())
        lower=confidence>=.85 and target<int(self.active.sum())
        self.lower_count_steps=self.lower_count_steps+1 if lower else 0
        if c.trim and self.lower_count_steps>=2:
            weak=(self.active&(sustain<.55)&(release>.30)&
                  (end-self.last_on>=.075*SR))
            for pitch in sorted(np.flatnonzero(weak),
                                key=lambda j:float(sustain[j])):
                if int(self.active.sum())<=target:break
                self.active[pitch]=False

        if confidence>=c.guide_confidence and int(self.active.sum())<target:
            possible=((~self.active)&self.armed&rising&
                     (on>=c.weak_onset)&(sustain>=c.weak_sustain)&
                     (release<.50)&(end-self.last_on>=.12*SR))
            for pitch in sorted(np.flatnonzero(possible),
                          key=lambda j:(-float(on[j]*sustain[j]),int(j))):
                if int(self.active.sum())>=min(6,target):break
                self.active[pitch]=True
                self.armed[pitch]=False
                self.low[pitch]=0
                self.last_on[pitch]=int(end)
        size=int(self.active.sum())
        if size>6:
            raise AssertionError("S708 Active-K voice ceiling exceeded")
        self.step_calls+=1
        return ActiveKDecision(int(end),size,target,confidence)


class S827S708ActiveKOnly:
    """Read-only private S708 model; 1 acoustic/neural/count decision per 256.

    Never emits a NoteOn/Off event to the public MIDI or any other axis.
    Nothing in this module imports or changes S826 or its altered shared code.
    """
    def __init__(self,native_heads,original_s708_model,frozen_policy,
                 *,capture_features=False):
        self.native_heads=native_heads
        self.frontend=CausalPitchFeatures(hop=HOP)
        self.temporal=IncrementalS708TemporalHead(original_s708_model)
        self.decoder=PrivateS708GuidedActiveK(frozen_policy,decision_hop=HOP)
        self.native_feat=deque(maxlen=65)
        self.native_post=deque(maxlen=65)
        self.native_k=deque(maxlen=65)
        self.audio=np.empty(0,dtype=np.float32)
        self.received=0
        self.model_calls=0
        self.closed=False
        self.capture_features=bool(capture_features)
        self.feature_debug=[]
        self.timings_ms=deque(maxlen=8192)

    def _step(self,end,f):
        p,k=self.native_heads.predict(f)
        self.native_feat.append(np.asarray(f,np.float32))
        self.native_post.append(np.asarray(p,np.float32))
        self.native_k.append(np.asarray(k,np.float32))
        acoustics=latest_s708_acoustic_features(
            self.native_feat,self.native_post,self.native_k)
        n=len(self.audio)
        sharp_ends=(np.array([n],np.int64) if end==HOP else
                    np.array([n-HOP,n],np.int64))
        spectral=sharp_track(self.audio,sharp_ends)[-1]
        X=np.concatenate((acoustics,spectral),axis=1)
        if X.shape!=(49,288) or not np.isfinite(X).all():
            raise AssertionError("frozen S708 [49,288] acoustic ABI must be intact")
        if self.capture_features:
            self.feature_debug.append(X.copy())
        nn_post,nn_count=self.temporal.step(X)
        self.model_calls+=1
        blended=self.decoder.policy.blend*nn_post+(
            1-self.decoder.policy.blend)*self.native_post[-1]
        return self.decoder.step(blended,nn_count,int(end))

    def push(self,samples,*,start_sample):
        if self.closed:
            raise ValueError("private Active-K stream already closed")
        if type(start_sample) is not int or start_sample!=self.received:
            raise ValueError("discontinuous source audio, reset required")
        x=np.asarray(samples)
        if (x.ndim!=1 or x.dtype.kind!="f" or
            len(x)>4096 or not np.isfinite(x).all()):
            raise ValueError("finite floating mono PCM <=4096 per callback required")
        begin=time.perf_counter()
        results=[]
        at=0
        while at<len(x):
            chunk_size=min(len(x)-at,HOP-self.frontend.received%HOP)
            block=np.asarray(x[at:at+chunk_size],np.float32)
            self.audio=np.concatenate((self.audio,block))[-PAST_AUDIO:]
            features=self.frontend.push(block,start_sample=self.received)
            self.received+=chunk_size
            at+=chunk_size
            for end,f in features:
                if end!=self.received or end%HOP:
                    raise AssertionError("feature derived after decision or not at 256")
                results.append(self._step(end,f))
        self.timings_ms.append((time.perf_counter()-begin)*1000)
        return tuple(results)

    def finish(self):
        self.closed=True
        return ()

    def diagnostics(self):
        return dict(schema=SCHEMA,decision_hop_samples=HOP,
            acoustic_feature_hop_samples=HOP,
            acoustic_S708_feature_dim=288,model_calls=self.model_calls,
            added_future_samples=0,added_hold_samples=0,
            teacher_original_training_decision_hop_samples=4*HOP,
            output_only="active_exact_k",own_private_decoder=True,
            public_midi_emission=False,other_five_axes_mutated=False,
            source_model_retrained_to_256=False)
