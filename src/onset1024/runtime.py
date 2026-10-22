"""Native PCM1024 isolated onset ONLY. True PCM1024, no five foreign decoders."""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
import time
import numpy as np
from .acoustics import CausalPitchFeatures,HOP,SR,sharp_track
from .temporal import IncrementalS708TemporalHead,latest_s708_acoustic_features
SCHEMA="ONSET_ONLY_EXPERIMENTAL_PCM1024_V1"
PAST_AUDIO=8192+HOP
@dataclass(frozen=True)
class Decision:
    sample:int
    pitches:tuple[int,...]=()
    intents:tuple=()
@dataclass(frozen=True)
class Intent:
    kind:str
    pitch:int
    sample:int
    confidence:float
    reason:str

class OnsetDecoder:
    """Independent causal onset state; release is internal rearming, never another branch."""
    def __init__(self,attack=.75,sustain=.5,release=.8,low=.30):
        self.attack,self.sustain,self.release,self.low=attack,sustain,release,low
        self.held={}
        self.armed=np.ones(49,bool)
    def step(self,post,sample):
        attack,sustain,release=post.T
        self.armed|=attack<self.attack*.4
        for pitch in sorted(self.held):
            i=pitch-40
            if release[i]>=self.release or sustain[i]<self.low:
                del self.held[pitch]
        ready=sorted((int(i) for i in np.flatnonzero(
            (attack>=self.attack)&(sustain>=self.sustain)&self.armed)),
            key=lambda i:(-float(attack[i]*sustain[i]),i))
        onsets=[]
        for i in ready:
            pitch=40+i
            if pitch in self.held or len(self.held)>=6:continue
            onsets.append(Intent("note_on",pitch,sample,float(attack[i]*sustain[i]),"native1024_onset"))
            self.held[pitch]=sample
            self.armed[i]=False
        return Decision(sample,tuple(sorted(self.held)),tuple(onsets))
    def finish(self,sample):
        self.held.clear()
        return ()

class Stream:
    def __init__(self,native,model,blend):
        self.native=native
        self.temporal=IncrementalS708TemporalHead(model)
        self.frontend=CausalPitchFeatures(hop=HOP)
        self.blend=float(blend)
        self.decoder=OnsetDecoder()
        self.native_feat=deque(maxlen=65);self.native_post=deque(maxlen=65)
        self.native_k=deque(maxlen=65)
        self.audio=np.empty(0,np.float32)
        self.received=0;self.last_decision=0;self.calls=0;self.closed=False
        self.times=[]
    def _step(self,end,feat):
        native_post,native_k=self.native.predict(feat)
        self.native_feat.append(np.asarray(feat,np.float32))
        self.native_post.append(np.asarray(native_post,np.float32))
        self.native_k.append(np.asarray(native_k,np.float32))
        acoustic=latest_s708_acoustic_features(
           self.native_feat,self.native_post,self.native_k)
        n=len(self.audio)
        sharp_ends=(np.array([n],np.int64) if end==HOP else
                    np.array([n-HOP,n],np.int64))
        spectral=sharp_track(self.audio,sharp_ends)[-1]
        X=np.concatenate((acoustic,spectral),axis=1)
        assert X.shape==(49,288)
        p,k=self.temporal.step(X)
        post=np.clip(self.blend*p+(1-self.blend)*self.native_post[-1],0,1)
        self.calls+=1
        if end!=self.last_decision+HOP:raise AssertionError("native1024 clock")
        self.last_decision=end
        return self.decoder.step(post,end)
    def push(self,samples,*,start_sample):
        if self.closed or type(start_sample)is not int or start_sample!=self.received:
            raise ValueError("invalid/discontinuous PCM")
        x=np.asarray(samples)
        if x.ndim!=1 or x.dtype.kind!="f" or not np.isfinite(x).all() or len(x)>4096:
            raise ValueError("finite float mono <=4096")
        t=time.perf_counter();out=[];at=0
        while at<len(x):
            take=min(len(x)-at,HOP-self.frontend.received%HOP)
            block=np.asarray(x[at:at+take],np.float32)
            self.audio=np.concatenate((self.audio,block))[-PAST_AUDIO:]
            rows=self.frontend.push(block,start_sample=self.received)
            self.received+=take;at+=take
            for end,f in rows:
                assert end==self.received and end%HOP==0
                out.append(self._step(end,f))
        self.times.append(1000*(time.perf_counter()-t))
        return tuple(out)
    def finish(self,sample=None):
        self.closed=True
        return self.decoder.finish(self.received if sample is None else sample)
