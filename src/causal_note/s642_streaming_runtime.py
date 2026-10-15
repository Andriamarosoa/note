"""S642 genuinely causal per-HOP learned weak onset / delayed MIDI event stream.

This tests runtime parity with the DEV batch replay; it does NOT establish
hardware latency. A caller MUST schedule each output event at its
available_sample and may not transmit future-due NoteOns immediately.

The original S614/S80 checkpoint is unchanged. No future PCM,
ground truth, or backward timestamp changes are read during push().
"""
from __future__ import annotations
from collections import deque
import hashlib
import heapq
from pathlib import Path
import time

import joblib
import numpy as np

from causal_note.strict_live import (
    CausalPitchFeatures,CausalSelection,Selection,PITCHES,PITCHES as _PITCHES,
    SR,HOP,
)
from causal_note.s618_attack_head import PICK,candidate_mask,DIM
from causal_note.s642_dual_learned import S642Policy,MAX_DELAY_HOPS


class S642EventScheduler:
    """Bounded pending message queue, preserving same-pitch NoteOn/NoteOff."""

    def __init__(self,delay_head,policy:S642Policy):
        self.delay_head=delay_head
        self.policy=policy
        self._queue=[]
        self._serial=0
        self._active_shift={}
        self._previous_off={}
        self.histogram=[0,0,0,0]
        self.delivered=0

    def add(self,events,features):
        for event in events:
            pitch=int(event["pitch"])
            if event["type"]=="note_on":
                shift=0
                if self.policy.use_delay:
                    shift=int(self.delay_head.predict(features[pitch-40][None])[0])
                    if not 0<=shift<=MAX_DELAY_HOPS:
                        raise ValueError("Invalid learned holding time")
                    prior=self._previous_off.get(pitch)
                    if prior is not None:
                        needed=(prior-int(event["available_sample"])+HOP-1)//HOP
                        shift=min(MAX_DELAY_HOPS,max(shift,int(needed)))
                        if int(event["available_sample"])+shift*HOP<prior:
                            raise RuntimeError("Cannot ensure causal same-pitch noteoff/next-on order")
                if pitch in self._active_shift:
                    raise ValueError("S642 original decoder emitted duplicate unpaired NoteOn")
                self._active_shift[pitch]=shift
                self.histogram[shift]+=1
            elif event["type"]=="note_off":
                shift=self._active_shift.pop(pitch,0)
            else:
                raise ValueError("Unsupported event type")
            due=int(event["available_sample"])+int(shift)*HOP
            item=dict(event)
            item["available_sample"]=due
            item["time"]=due/SR
            if event["type"]=="note_off":
                self._previous_off[pitch]=due
            priority=0 if event["type"]=="note_off" else 1
            heapq.heappush(self._queue,(due,priority,self._serial,item))
            self._serial+=1

    def drain_ready(self,received_sample):
        ready=[]
        while self._queue and self._queue[0][0]<=received_sample:
            _,_,_,event=heapq.heappop(self._queue)
            if event["decision_sample"]>event["available_sample"]:
                raise AssertionError("Future audio was required for this event")
            ready.append(event)
        self.delivered+=len(ready)
        return ready

    def drain_all_as_scheduled(self):
        """Offline replay ONLY: returns future-due events for clock scheduling.

        Physical callers must wait until each event's available_sample,
        *not* send all returned messages immediately on microphone shutdown.
        """
        pending=[]
        while self._queue:
            _,_,_,event=heapq.heappop(self._queue)
            pending.append(event)
        self.delivered+=len(pending)
        return pending


class S642CausalLive:
    def __init__(self,heads,candidate_head,delay_head,
                 policy:S642Policy=S642Policy()):
        self.heads=heads
        self.candidate_head=candidate_head
        self.delay_head=delay_head
        self.policy=policy
        self.frontend=CausalPitchFeatures()
        self.selector=CausalSelection(Selection(**heads.metadata["selection"]))
        self.scheduler=S642EventScheduler(delay_head,policy)
        self.previous=np.zeros((49,3),np.float32)
        self.closed=False
        self.processing_ms=deque(maxlen=8192)
        self.rescue_attempts=0

    @classmethod
    def from_verified_checkpoints(cls,heads,attack_path,attack_sha,
                                  delay_path,delay_sha,
                                  policy:S642Policy=S642Policy()):
        refs=[(attack_path,attack_sha),(delay_path,delay_sha)]
        models=[]
        for path,sha in refs:
            path=Path(path)
            if hashlib.sha256(path.read_bytes()).hexdigest()!=sha:
                raise ValueError("S642 trained model SHA256 mismatch")
            models.append(joblib.load(path))
        return cls(heads,*models,policy=policy)

    def _step(self,end,x):
        p,k=self.heads.predict(x)
        z=np.concatenate((p,p-self.previous,x[:,PICK]),axis=-1).astype(np.float32)
        if z.shape!=(49,DIM):raise AssertionError("Original native S618 ABI mismatch")
        edited=p.copy()
        if self.policy.use_candidate:
            confidence=self.candidate_head.predict_proba(z)[:,1]
            eligible=(~self.selector.active)&candidate_mask(x[None],p[None])[0]
            viable=eligible&(p[:,0]<self.selector.config.onset)&(p[:,0]>=.13)&(
                p[:,1]>=.26)&(confidence>=self.policy.rescue_min_confidence)
            hits=np.flatnonzero(viable)
            if len(hits):
                best=int(hits[np.argmax(confidence[hits])])
                edited[best,0]=self.selector.config.onset
                edited[best,1]=max(float(edited[best,1]),self.selector.config.active_on)
                self.rescue_attempts+=1
        events=self.selector.step(edited,k,int(end),self.frontend.received)
        self.scheduler.add(events,z)
        self.previous=p.copy()
        return self.scheduler.drain_ready(self.frontend.received)

    def push(self,pcm,*,start_sample:int):
        if self.closed:raise ValueError("S642 stream already finished")
        before=time.perf_counter()
        chunks=self.frontend.push(pcm,start_sample=start_sample)
        events=[]
        for end,x in chunks:
            events.extend(self._step(end,x))
        # If the last block ended before another hop was reached, some
        # pending messages can still become due at the received boundary.
        events.extend(self.scheduler.drain_ready(self.frontend.received))
        self.processing_ms.append((time.perf_counter()-before)*1000)
        return events

    def finish(self):
        if self.closed:return []
        self.closed=True
        terminal=self.selector.panic(self.frontend.received,"stream_end")
        z=np.zeros((49,DIM),np.float32)
        self.scheduler.add(terminal,z)
        return (self.scheduler.drain_ready(self.frontend.received)
                +self.scheduler.drain_all_as_scheduled())
