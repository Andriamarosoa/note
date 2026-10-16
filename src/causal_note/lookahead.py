"""S628 causal delayed future-acoustic inference, default reference unchanged.

An adjustable L milliseconds of *future relative to a nominal decision
frame* are obtained ONLY by delaying that decision until PCM through
t+L has physically arrived. A second, phase-offset native frontend
extracts genuine additional acoustics at t+L; no labeling oracle,
predictive sample access, backdating, noncausal STFT, or new model.

Reference S613 heads, K predictor and selector remain frozen. The only
experimental change is combining current and observed-future per-pitch
onset/sustain scores before the original selector.

Original `decision_sample` is the nominated baseline t, whereas
`available_sample` and emitted `time` are the true physical arrival
time including lookahead AND callback delivery granularity.
NOTE-on and NOTE-off both require extra latency. True hardware response
also includes PortAudio input/output and FluidSynth delay.
"""
from __future__ import annotations

from collections import deque,Counter
from dataclasses import dataclass
import math
import time

import numpy as np

from causal_note.strict_live import CausalPitchFeatures,CausalSelection,Selection,HOP,SR


@dataclass(frozen=True)
class LookaheadConfig:
    milliseconds: float=5.0
    mode: str="average"  # delay / average / confirm / max_onset / future

    def __post_init__(self):
        if not math.isfinite(self.milliseconds) or not 0<self.milliseconds<=50:
            raise ValueError("Lookahead milliseconds must be >0 and <=50")
        if self.mode not in ("delay","average","confirm","max_onset","future"):
            raise ValueError("Unknown S628 acoustic lookahead mode")

    @property
    def samples(self):
        return max(1,int(round(self.milliseconds*SR/1000)))


def combine_reference_and_future(reference,future,mode):
    """Only onset/sustain change; frozen release and independent K untouched.

    Signal from the future was actually captured by t+L.
    Mode='delay' is an essential timing-only control. No mode has
    established accuracy benefits until independent evaluation.
    """
    original=np.asarray(reference,dtype=np.float32)
    ahead=np.asarray(future,dtype=np.float32)
    if original.shape!=(49,3) or ahead.shape!=(49,3):
        raise ValueError("Expected original 49x3 onset/sustain/release probabilities")
    result=original.copy()
    if mode=="delay":
        return result
    if mode=="average":
        result[:,:2]=.5*(original[:,:2]+ahead[:,:2])
    elif mode=="confirm":
        result[:,0]=.5*(original[:,0]+ahead[:,0])
        result[:,1]=np.minimum(original[:,1],ahead[:,1])
    elif mode=="max_onset":
        result[:,0]=np.maximum(original[:,0],ahead[:,0])
        result[:,1]=np.maximum(original[:,1],ahead[:,1])
    elif mode=="future":
        result[:,:2]=ahead[:,:2]
    else:
        raise ValueError(mode)
    return result


class LookaheadLiveStream:
    """Full frozen S613 NOTE with physical, configurable post-frame future.

    0ms is handled by the untouched original StrictLiveStream, not this
    experimental path. This stream requires fixed 256-hop base (to protect
    original-trained decision/selector semantics) but handles ANY callback
    block size including split >4096.

    First L captured samples are fed to the reference path only.
    The future path is shifted L input samples; future phase-frame
    `end=t` means audio physically received through `t+L`.
    Both paths retain native 8192/4096/2048-sample trailing windows
    and correct physical 2/4/8/16-hop causal history.
    """
    def __init__(self,heads,*,config=LookaheadConfig()):
        self.heads=heads
        self.config=config
        self.lookahead_samples=config.samples
        self.reference_front=CausalPitchFeatures(hop=HOP)
        self.future_front=CausalPitchFeatures(hop=HOP)
        self.selector=CausalSelection(Selection(**heads.metadata["selection"]),hop=HOP)
        self.harmonic_selector=None
        self.received=0
        self.future_received=0
        self.pending=deque()
        self.closed=False
        self.stats=Counter()
        self.worker_ms=deque(maxlen=4096)

    def push(self,samples,*,start_sample):
        if self.closed or start_sample!=self.received:
            raise ValueError("Closed or discontinuous live audio")
        x=np.asarray(samples)
        if x.ndim!=1 or x.dtype.kind!="f" or not np.isfinite(x).all():
            raise ValueError("Expected finite float mono PCM")
        end=start_sample+len(x)
        begin_time=time.perf_counter()
        emitted=[]
        # Fixed-size transport slices prevent exceeding CausalPitchFeatures
        # max 4096, while accepting any external mic callback size.
        for offset in range(0,len(x),4096):
            segment=x[offset:offset+4096]
            absolute=start_sample+offset
            arrival=end # physically all samples of this callback available
            for t,features in self.reference_front.push(segment,start_sample=absolute):
                p,k=self.heads.predict(features)
                self.pending.append((int(t),p,k))
                self.stats["reference_frames"]+=1
            # Skip exactly L samples at beginning, NOT whenever silence.
            omit=max(0,min(len(segment),self.lookahead_samples-absolute))
            part=segment[omit:]
            if len(part):
                future_begin=absolute+omit-self.lookahead_samples
                if future_begin!=self.future_received:
                    raise ValueError("Lookahead audio phase gap")
                future_rows=self.future_front.push(
                    part,start_sample=future_begin)
                self.future_received+=len(part)
                for shifted_t,features in future_rows:
                    real_end=shifted_t+self.lookahead_samples
                    if real_end>arrival:
                        raise AssertionError("Lookahead used unavailable samples")
                    if not self.pending:
                        raise AssertionError("Future features outran reference predictions")
                    t,p,k=self.pending.popleft()
                    if t!=shifted_t:
                        raise AssertionError("Original/future decision times misaligned")
                    pf,_=self.heads.predict(features)
                    fused=combine_reference_and_future(p,pf,self.config.mode)
                    emitted.extend(self.selector.step(
                        fused,k,t,arrival))
                    self.stats["lookahead_confirmed_frames"]+=1
                    self.stats["future_audio_samples_seen"]+=self.lookahead_samples
        self.received=end
        self.worker_ms.append((time.perf_counter()-begin_time)*1000)
        return emitted

    def finish(self,reason="stream_end"):
        if self.closed:return []
        self.closed=True
        # Last L samples worth of pending reference frames have no REAL
        # future samples and cannot be future-confirmed. Use exact old
        # probabilities with the actual EOF as arrival; no fake padding.
        out=[]
        for t,p,k in self.pending:
            residual=self.selector.step(p,k,t,self.received)
            for event in residual:
                event["reason"]="lookahead_eof_unconfirmed_"+event["reason"]
            out.extend(residual)
            self.stats["EOF_baseline_unconfirmed_frames"]+=1
        self.pending.clear()
        out.extend(self.selector.panic(self.received,reason=reason))
        return out

    def diagnostics(self):
        return dict(
            mode=self.config.mode,configured_ms=self.config.milliseconds,
            actual_lookahead_samples=self.lookahead_samples,
            actual_lookahead_ms=self.lookahead_samples*1000/SR,
            original_hop_samples=HOP,
            source=self.stats.copy(),total_received_samples=self.received,
            max_worker_callback_ms=float(max(self.worker_ms)) if self.worker_ms else None,
            p95_worker_callback_ms=float(np.percentile(self.worker_ms,95))
                                   if self.worker_ms else None,
            warning=("Actual output latency includes capture callback batching, "
                     "processing and FluidSynth; this is true observed audio "
                     "future, not free model access to next 5ms."))


def replay_lookahead(audio,heads,config,*,block_size=256):
    if block_size<=0:raise ValueError("positive block required")
    stream=LookaheadLiveStream(heads,config=config)
    out=[]
    pcm=np.asarray(audio,np.float32)
    for start in range(0,len(pcm),block_size):
        out.extend(stream.push(pcm[start:start+block_size],start_sample=start))
    out.extend(stream.finish())
    return out,stream.diagnostics()
