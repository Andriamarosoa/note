"""Experimental strict-causal NOTE dual-cadence event stream (S620).

S613 full hybrid is the 256-sample decision baseline. A secondary 128-hop
branch proposes acoustic onset events, gated by independent per-pitch
positive spectral novelty. A two-source note ownership merger preserves
the shared MIDI ON/OFF state and prevents double NOTE-on/off when both
branches see the same sound.

This is NOT a retrained 128-hop model: the frozen S613 weights only have a
validated 256-hop distribution. Do not enable guard for production or
promote based on development tuning.

No future samples, no post-onset waiting, no new FFT lengths. The extra
128-hop inference/FFT work CAN increase CPU latency and must be measured.
For pure reference parity use mode='off'. For diagnostics use 'shadow'.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
import time

import numpy as np

from causal_note.strict_live import (
    CausalPitchFeatures, CausalSelection, HOP, SR, PITCHES, Selection,
)


@dataclass(frozen=True)
class DualHopConfig:
    mode: str = "shadow"  # off (reference) / shadow / guard
    fast_onset: float = .50
    novelty_min: float = .04
    novelty_ratio: float = .55
    novelty_mode: str = "lag"  # lag / recent

    def __post_init__(self):
        if self.mode not in ("off", "shadow", "guard"):
            raise ValueError("mode must be off, shadow, guard")
        if not 0 < self.fast_onset < 1:
            raise ValueError("fast onset must be (0,1)")
        if not 0 <= self.novelty_min or not 0 <= self.novelty_ratio <= 1:
            raise ValueError("novelty criteria outside valid range")
        if self.novelty_mode not in ("lag", "recent"):
            raise ValueError("unknown novelty metric")


def novelty_gated_fast_heads(features, pitch_probs, cfg, *,
                             reference_onset=.60, reference_sustain=.50):
    """Gate all fast NoteOn proposals with CURRENT-frame causal evidence.

    Returns a private probability copy. The original frozen heads are NEVER
    mutated; reference predictions are computed in their own 256-hop path.
    """
    x=np.asarray(features,dtype=np.float32)
    p=np.asarray(pitch_probs,dtype=np.float32)
    if x.shape != (49,126) or p.shape != (49,3):
        raise ValueError("S613 per-pitch features/probability shape required")
    if cfg.novelty_mode=="lag":
        diff=x[:,60:63]
    else:
        diff=x[:,114:117]
    novelty=np.maximum(diff,0).max(axis=1)
    relative=novelty/max(float(novelty.max()),1e-5)
    qualified=((p[:,0]>=cfg.fast_onset)&(p[:,1]>=reference_sustain)&
               (novelty>=cfg.novelty_min)&(relative>=cfg.novelty_ratio))
    # For the untrained hop128 candidate, every fast attack must have a
    # *separate* acoustic novelty. Even strong network onset alone does not
    # authorize an additional fast event.
    filtered=p.copy()
    filtered[~qualified,0]=0
    return filtered, qualified


class DualSourceEventMerger:
    """Maintain a single audible NOTE per pitch for asynchronous sources.

    Sources retain their own full causal on/off lifecycle. The first source
    that turns a pitch on emits physical NoteOn, the last one off emits
    physical NoteOff. Baseline events never produce a second NOTE-on for a
    pitch held already by the fast branch. No future sample is consulted.
    """
    def __init__(self):
        self.source_active={"reference":np.zeros(49,bool),
                            "fast":np.zeros(49,bool)}
        self.current=np.zeros(49,bool)
        self.emitted=Counter()
        self.suppressed_duplicates=0

    def step(self, source, event):
        if source not in self.source_active:
            raise ValueError("Unknown head source")
        pitch=int(event["pitch"])
        if not 40<=pitch<=88 or event["type"] not in ("note_on","note_off"):
            raise ValueError("Invalid pitch/on-off event")
        idx=pitch-40
        first=bool(self.current[idx])
        if event["type"]=="note_on":
            if self.source_active[source][idx]:
                raise ValueError("Overlapping identical source NOTE-on")
            self.source_active[source][idx]=True
        else:
            if not self.source_active[source][idx]:
                raise ValueError("Orphan source NOTE-off")
            self.source_active[source][idx]=False
        now=bool(self.source_active["reference"][idx] or self.source_active["fast"][idx])
        self.current[idx]=now
        if now==first:
            if event["type"]=="note_on":
                self.suppressed_duplicates+=1
            return []
        result=dict(event)
        result["reason"]="dual_"+source+"_"+event.get("reason","")
        self.emitted[result["type"]]+=1
        return [result]

    def finish(self, sample):
        events=[]
        for idx in np.flatnonzero(self.current):
            events.append(dict(
                type="note_off",pitch=int(PITCHES[idx]),decision_sample=int(sample),
                available_sample=int(sample),time=sample/SR,velocity=0,
                reason="dual_stream_end"))
        self.current[:]=False
        for val in self.source_active.values():
            val[:]=False
        return events


class DualHopLiveStream:
    """Native S613 hybrid at hop256 + independent causal hop128 proposals.

    The default is shadow mode: returned events are exactly baseline
    NOTE on/off; the fast expert is computed only to diagnose candidates.
    In guard mode, fast events are added via a two-source owner merger.

    Sampling/callback size is independent of decision cadence. All outputs
    preserve the real callback's available_sample, even if the callback
    contains many internal 128-hop frames.
    """
    def __init__(self, frozen_heads, *, config=DualHopConfig()):
        self.heads=frozen_heads
        self.config=config
        self.reference_front=CausalPitchFeatures(hop=HOP)
        self.reference_selector=CausalSelection(
            Selection(**frozen_heads.metadata["selection"]),hop=HOP)
        self.fast_front=(CausalPitchFeatures(hop=128) if config.mode!="off" else None)
        self.fast_selector=(CausalSelection(
            Selection(**{**frozen_heads.metadata["selection"],
                         "onset":config.fast_onset}),hop=128)
                       if config.mode!="off" else None)
        self.merger=DualSourceEventMerger() if config.mode=="guard" else None
        self.received=0
        self.closed=False
        self.fast_candidates=0
        self.fast_note_ons=0
        self.events_reference=Counter()
        self.timings_ms=deque(maxlen=4096)
        self.last_reference_probs=None
        self.last_reference_sample=0

    def push(self, samples, *, start_sample):
        if self.closed or start_sample!=self.received:
            raise ValueError("Closed or discontinuous audio stream")
        x=np.asarray(samples)
        if x.ndim!=1 or x.dtype.kind!="f" or not np.isfinite(x).all():
            raise ValueError("Finite mono float PCM required")
        t0=time.perf_counter()
        available=start_sample+len(x)
        emitted=[]
        # Feed both streams from precisely the same past-only samples. The
        # prediction cadence is not changed by PortAudio callback sizes.
        for offset in range(0,len(x),128):
            part=x[offset:offset+128]
            position=start_sample+offset
            reference_frames=self.reference_front.push(part,start_sample=position)
            fast_frames=(self.fast_front.push(part,start_sample=position)
                         if self.fast_front is not None else [])
            # Merge by decision time; on a shared 256 boundary give the
            # validated reference predictor priority over experimental fast.
            staged=[]
            for end,features in reference_frames:
                pitch,k=self.heads.predict(features)
                self.last_reference_probs=pitch
                self.last_reference_sample=end
                ev=self.reference_selector.step(pitch,k,end,available)
                for e in ev:
                    self.events_reference[e["type"]]+=1
                    staged.append((end,0,"reference",e))
            for end,features in fast_frames:
                pitch,k=self.heads.predict(features)
                gated,qualified=novelty_gated_fast_heads(features,pitch,self.config,
                        reference_onset=self.reference_selector.config.onset,
                        reference_sustain=self.reference_selector.config.active_on)
                self.fast_candidates+=int(qualified.sum())
                ev=self.fast_selector.step(gated,k,end,available)
                for e in ev:
                    if e["type"]=="note_on":
                        self.fast_note_ons+=1
                    staged.append((end,1,"fast",e))
            for _,_,source,event in sorted(staged,key=lambda row:(row[0],row[1])):
                if source=="reference" and self.config.mode!="guard":
                    emitted.append(event)
                elif self.config.mode=="guard":
                    emitted.extend(self.merger.step(source,event))
        self.received=available
        self.timings_ms.append((time.perf_counter()-t0)*1000)
        return emitted

    def finish(self):
        if self.closed:
            return []
        self.closed=True
        slow=self.reference_selector.panic(self.received)
        if self.config.mode!="guard":
            return slow
        # Both sources are reset together. The shared NOTE voice is ended
        # ONCE even when both internal heads have the same active pitch.
        if self.fast_selector is not None:
            self.fast_selector.panic(self.received)
        return self.merger.finish(self.received)

    def diagnostics(self):
        result=dict(mode=self.config.mode,hop_fast=128 if self.fast_front else None,
                    hop_reference=HOP,fast_pitch_frames_qualified=self.fast_candidates,
                    fast_note_ons=self.fast_note_ons,
                    reference_events=dict(self.events_reference),
                    cpu_packet_p95_ms=(
                        float(np.percentile(self.timings_ms,95)) if self.timings_ms else 0.))
        if self.merger is not None:
            result["merged_events"]=dict(self.merger.emitted)
            result["duplicate_note_on_suppressed"]=self.merger.suppressed_duplicates
        return result
