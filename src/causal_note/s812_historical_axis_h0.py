"""S812 historical H0: six independent causal projections of S708/S707 MIDI.

This reproduces the SIGNED historical MIDI output, not six separately
reconstructed historical neural checkpoints. H0 stays read-only and never
participates in S810 training or an automatic H1 promotion.
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
from typing import Mapping

AXES=("exact_k","active_exact_k","onset","offset","frame_pitch","full_note")
SR=44100

def signature(events):
    return [(e["type"],int(e["pitch"]),int(e["available_sample"]),
             int(e["decision_sample"]),float(e["time"]),e.get("reason",""))
            for e in events]

@dataclass(frozen=True)
class AxisResult:
    axis: str
    sample: int
    count: int|None = None
    pitches: tuple[int,...] = ()
    emitted: tuple[tuple,...] = ()

class HistoricalH0Axis:
    """A private projection of signed native MIDI; no other axis is consulted."""
    def __init__(self,axis):
        if axis not in AXES:raise ValueError("invalid axis")
        self.axis=axis
        self.last_sample=-1
        self.active=set()
        self.events=[]
        self.born=0
        self.ended=0
    def step(self,sample,arrived):
        sample=int(sample)
        if sample<=self.last_sample:raise ValueError("nonmonotone H0 clock")
        self.last_sample=sample
        emitted=[]
        for e in arrived:
            if not isinstance(e,Mapping):raise TypeError("event must be mapping")
            if e["available_sample"]>sample:raise ValueError("future event leaked")
            if e["available_sample"]<e["decision_sample"]:raise ValueError("backdated event")
            if abs(e["time"]-e["available_sample"]/SR)>1e-10:raise ValueError("bad event timestamp")
            pitch=int(e["pitch"])
            if not 40<=pitch<=88:raise ValueError("unexpected pitch")
            if e["type"]=="note_on":
                if pitch in self.active:raise ValueError("duplicate NoteOn")
                self.active.add(pitch);self.born+=1
            elif e["type"]=="note_off":
                if pitch not in self.active:raise ValueError("orphan NoteOff")
                self.active.remove(pitch);self.ended+=1
            else:raise ValueError("invalid event type")
            self.events.append(deepcopy(e))
            if (self.axis=="full_note" or
                (self.axis=="onset" and e["type"]=="note_on") or
                (self.axis=="offset" and e["type"]=="note_off")):
                emitted.append(tuple(sorted(e.items())))
        return AxisResult(self.axis,sample,
            min(6,len(self.active)) if self.axis in ("exact_k","active_exact_k") else None,
            tuple(sorted(self.active)) if self.axis=="frame_pitch" else (),
            tuple(emitted))

class SixHistoricalH0:
    """Six separate states and six per-axis copies; source signed MIDI untouched."""
    def __init__(self,lanes=None,order=AXES):
        self.lanes=({a:HistoricalH0Axis(a) for a in AXES}
                    if lanes is None else dict(lanes))
        if set(self.lanes)!=set(AXES) or set(order)!=set(AXES):
            raise ValueError("six axes required")
        if len(set(map(id,self.lanes.values())))!=6 or any(
              lane.axis!=axis for axis,lane in self.lanes.items()):
            raise ValueError("shared or alien state")
        self.order=tuple(order);self.last_sample=-1
    def step(self,sample,events):
        if int(sample)<=self.last_sample:raise ValueError("bad common clock")
        before=signature(events);out={}
        for axis in self.order:
            out[axis]=self.lanes[axis].step(sample,deepcopy(events))
        if signature(events)!=before:raise AssertionError("native input mutated")
        self.last_sample=int(sample)
        return out
    @staticmethod
    def replay(events,metric_clock,samples):
        src=deepcopy(events);before=signature(src)
        clock=tuple(int(t) for t in metric_clock)
        if not clock or any(a>=b for a,b in zip(clock,clock[1:])):
            raise ValueError("bad 256-sample clock")
        if int(samples)<clock[-1]:raise ValueError("bad stream length")
        system=SixHistoricalH0()
        cursor=0;out={a:[] for a in AXES}
        for t in (*clock,*([int(samples)] if int(samples)>clock[-1] else [])):
            arrived=[]
            while cursor<len(src) and int(src[cursor]["available_sample"])<=t:
                arrived.append(src[cursor]);cursor+=1
            for axis,answer in system.step(t,arrived).items():
                out[axis].append(answer)
        if cursor!=len(src) or signature(src)!=before:
            raise AssertionError("original MIDI changed/unconsumed")
        for axis,lane in system.lanes.items():
            if signature(lane.events)!=before:
                raise AssertionError("private historical MIDI not identical: "+axis)
        return out,system

def checked_h1_view(h0,alternative,axis):
    """Private shadow overlay, never writes into protected H0."""
    if set(h0)!=set(AXES) or axis not in AXES:raise ValueError("wrong axes")
    before=deepcopy(h0);out=deepcopy(h0);out[axis]=deepcopy(alternative)
    if h0!=before or any(out[k]!=h0[k] for k in AXES if k!=axis):
        raise AssertionError("H1 contamination of H0/foreign axis")
    return out
