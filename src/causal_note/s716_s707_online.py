"""S716 opt-in causal S707-style NoteOff for the genuine S708 decision stream.

Uses the independently signed S675 75D hazard classifier and the S708
CAL-frozen S707 release policy (gate=.80, persistence=1, grace=24 native hops).
This class is a source-only ONLINE realization, not an oracle/offline replay;
especially the fallback cannot be backdated. It cannot claim S708 DEV Offset
73.73% or complete-note F1 50.31% until independent emitted-event validation.

Source S708 GuidedSelector retains its original frozen note-on proposals;
this bridge owns the ONE actual emitted-note state, max six active notes.
"""
from __future__ import annotations
from collections import Counter, deque

import numpy as np

from causal_note.strict_live import HOP, SR
from scripts.s672_causal_offset_timing_grid import (
    MIN_AGE_FRAMES, MAX_AGE_FRAMES, frame_matrix,
)

DECISION = 4 * HOP


class S716S707OnlineRelease:
    def __init__(self, model, *, gate=.80, persistence=1, grace=24):
        if (not 0 < gate < 1 or persistence not in (1, 2, 3, 4)
                or not 0 <= grace <= 128):
            raise ValueError("Invalid S708 CAL-frozen S707 release policy")
        if getattr(model, "n_features_in_", None) != 75:
            raise ValueError("S707/S675 online release requires signed 75D hazard head")
        self.model = model
        self.gate, self.persistence, self.grace = float(gate), int(persistence), int(grace)
        self.active = {}
        self.source_onset_frames = deque(maxlen=9)
        self.virtual_features = deque(maxlen=65)
        self.virtual_post = deque(maxlen=65)
        self.virtual_k = deque(maxlen=65)
        self.previous_block = None
        self.last_end = 0
        self.stats = Counter()
        self.finished = False

    @classmethod
    def load(cls, checkpoint, original_heads, frozen_release_policy):
        # The existing S680 loader verifies expected checkpoint filename,
        # trusted export metadata and SHA256 against original S613 heads.
        # joblib loading is unsafe for untrusted files: accept only the
        # original pinned GitHub Actions model artifact.
        from causal_note.s672_live_offset import S672TrialNoteOff
        specialist = S672TrialNoteOff.load(checkpoint, original_heads)
        if frozen_release_policy != dict(gate=.80,persistence=1,grace=24):
            raise ValueError("S707 selected release policy does not match S708 freeze")
        return cls(specialist.model, **frozen_release_policy)

    def _clock(self, frame):
        if not self.source_onset_frames:
            return np.zeros(4, dtype=np.float32)
        latest = self.source_onset_frames[-1]
        gaps = np.diff(np.asarray(self.source_onset_frames, dtype=np.float64))
        tempo = max(4., float(np.median(gaps)) if len(gaps) else 50.)
        since = max(0, frame - latest)
        return np.asarray((np.log1p(since),np.log1p(tempo),
                           min(6.,since/tempo),np.cos(2*np.pi*since/tempo)),
                          dtype=np.float32)

    def _append_virtual(self, features, post, count):
        # The offline S707 feature ABI samples at hops 3,7,11,.., with
        # received_history(): the prior 4-hop value persists through the
        # following three native hops before the latest value arrives.
        if self.previous_block is None:
            old=(np.zeros_like(features),np.zeros_like(post),np.zeros_like(count))
        else:
            old=self.previous_block
        for _ in range(3):
            for q,x in zip((self.virtual_features,self.virtual_post,self.virtual_k),old):
                q.append(x)
        new=(features.copy(),post.copy(),count.copy())
        for q,x in zip((self.virtual_features,self.virtual_post,self.virtual_k),new):
            q.append(x)
        self.previous_block=new

    def _emit_off(self, pitch, end, available, reason):
        if pitch not in self.active:
            raise AssertionError("Cannot NoteOff inactive S716 pitch")
        del self.active[pitch]
        self.stats[reason]+=1
        return dict(type="note_off",pitch=int(pitch),
                    decision_sample=int(end),available_sample=int(available),
                    time=float(available/SR),reason=reason,velocity=0,
                    s707_causal_live_trial=True,s708_shadow_trial=True)

    def step(self, end, source_events, native_features, native_post, native_k, available):
        if self.finished:
            raise ValueError("S716 closed")
        if int(end) != self.last_end + DECISION or available < end:
            raise ValueError("S716 expected continuous four-hop decisions")
        feature = np.asarray(native_features,dtype=np.float32)
        post = np.asarray(native_post,dtype=np.float32)
        count = np.asarray(native_k,dtype=np.float32)
        if feature.shape!=(49,126) or post.shape!=(49,3) or count.shape!=(7,) or (
                not np.isfinite(feature).all() or not np.isfinite(post).all()
                or not np.isfinite(count).all()):
            raise ValueError("S716 original native NOTE feature ABI changed")
        self.last_end=int(end)
        idx=int(end//HOP)-1
        self._append_virtual(feature,post,count)
        outs=[]

        # Observe frozen S708 decisions, but do not prematurely emit its
        # native NoteOff; use it as the S707 known-forward fallback timer.
        for e in source_events:
            if e["decision_sample"]!=end or e["available_sample"]>available:
                raise ValueError("Noncausal S708 source event")
            pitch=int(e["pitch"])
            if e["type"]=="note_off" and pitch in self.active:
                if e["reason"] not in ("joint_restrike","stream_end"):
                    n=self.active[pitch]
                    if n["native_off"] is None:
                        n["native_off"]=idx
            elif e["type"]!="note_on" and e["type"]!="note_off":
                raise ValueError("Unknown source event type")
        if any(e["type"]=="note_on" for e in source_events):
            # The native offline clock uses unique decision frames only.
            if not self.source_onset_frames or self.source_onset_frames[-1]!=idx:
                self.source_onset_frames.append(idx)

        matching=[]; rows=[]
        local=len(self.virtual_features)-1
        clock=np.zeros((len(self.virtual_features),4),np.float32)
        clock[-1]=self._clock(idx)
        feat=np.stack(self.virtual_features)
        posterior=np.stack(self.virtual_post)
        k_values=np.stack(self.virtual_k)
        for pitch,item in sorted(self.active.items()):
            age=idx-item["start"]
            if age>=MAX_AGE_FRAMES:
                outs.append(self._emit_off(pitch,end,available,"s716_max_age"))
                continue
            if age>=MIN_AGE_FRAMES:
                features = frame_matrix(feat,posterior,k_values,None,pitch,
                                        local-age,np.asarray([local]),clock)[0]
                rows.append(features)
                matching.append(pitch)
        if matching:
            probs=self.model.predict_proba(np.asarray(rows,np.float32))[:,1]
            for pitch,prob in zip(matching,probs):
                if pitch not in self.active:
                    continue
                note=self.active[pitch]
                note["consecutive"]=(note["consecutive"]+1 if prob>=self.gate else 0)
                if note["consecutive"]>=self.persistence:
                    outs.append(self._emit_off(pitch,end,available,"s716_causal_hazard"))
        for pitch,note in sorted(list(self.active.items())):
            native_off=note["native_off"]
            if native_off is not None and idx-native_off>=self.grace:
                outs.append(self._emit_off(pitch,end,available,"s716_forward_fallback"))

        # Admit native NoteOns only AFTER due releases. If S707's held note
        # consumes the sixth slot, skip new onsets instead of overbooking.
        # Source selector predictions remain untouched.
        for e in source_events:
            if e["type"]!="note_on":
                continue
            pitch=int(e["pitch"])
            if not 40<=pitch<=88:
                raise ValueError("S708 pitch outside MIDI 40..88")
            if pitch in self.active:
                outs.append(self._emit_off(pitch,end,available,"s716_restrike"))
            if len(self.active)>=6:
                self.stats["note_on_skipped_due_to_six_note_cap"]+=1
                continue
            new=dict(e)
            new["s707_causal_live_trial"]=True
            self.active[pitch]=dict(start=idx,consecutive=0,native_off=None)
            outs.append(new)
            self.stats["source_s708_note_on_emitted"]+=1
        return outs

    def finish(self, sample, reason="stream_end"):
        if self.finished:
            return []
        if sample < self.last_end:
            raise ValueError("Shutdown before last S716 decision")
        self.finished=True
        return [self._emit_off(pitch,sample,sample,reason) for pitch in sorted(self.active)]

    def diagnostics(self):
        return dict(s716_s707_online=True,gate=self.gate,
                    persistence=self.persistence,grace_hops=self.grace,
                    original_S708_note_on_head_unchanged=True,
                    true_midi_events_source_only=True,max_polyphony=6,
                    note_off_fallback_not_backdated=True,
                    source_audio_decisions=self.last_end//DECISION,
                    active_notes=len(self.active),counts=dict(self.stats),
                    caveat="S708 offline S707 Offset score is NOT this microphone score")
