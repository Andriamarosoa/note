"""S680 opt-in microphone trial of the S672 TRAIN-frozen NoteOff specialist.

The specialist uses the same 75 acoustic+native-posterior+past-onset-rhythm
features as the S672 and S675 genuine GuitarSet heads. The immutable native NOTE selector owns original NoteOn
and Exact-K decisions. This wrapper NEVER changes them, and exposes
new NoteOff events from a separate active-note state.

IMPORTANT: research S672 (770/990 DEV batch4) made predictions offline and
could choose the native fallback NoteOff retroactively if no hazard fired.
A live stream cannot retroactively emit that fallback. Here fallback occurs
after 24 hops (~139ms) of grace following original release, if no hazard
has fired. Consequently DO NOT present offline 77.78% as a measured score of
this LIVE adapter. Test raw WAV and microphone playback independently.
"""
from __future__ import annotations
from collections import Counter,deque
import hashlib,json
from pathlib import Path

import numpy as np

from causal_note.strict_live import HOP,SR
from scripts.s672_causal_offset_timing_grid import frame_matrix,MIN_AGE_FRAMES,MAX_AGE_FRAMES

class S672TrialNoteOff:
    def __init__(self, model, *, gate=.60, persistence=1,
                 native_fallback_grace_hops=24):
        if not 0 < gate < 1 or not 1<=persistence<=4:
            raise ValueError("Invalid frozen S672 hazard gate")
        self.model=model
        self.gate=float(gate)
        self.persistence=persistence
        self.grace=int(native_fallback_grace_hops)
        self.rows=deque(maxlen=65)
        self.posteriors=deque(maxlen=65)
        self.k_values=deque(maxlen=65)
        self.active={}
        self.onsets=[]
        self.hops=0
        self.stats=Counter()
        self.last_available_sample=0

    @classmethod
    def load(cls, checkpoint, heads):
        import joblib
        checkpoint=Path(checkpoint)
        if checkpoint.name not in ("s672_hazard.joblib","s675_hazard.joblib"):
            raise ValueError("Unrecognized frozen NOTE specialist filename")
        family="s675" if checkpoint.name.startswith("s675_") else "s672"
        meta_path=checkpoint.with_name(f"{family}_checkpoint.json")
        info=json.loads(meta_path.read_text())
        expected_schema=("S681_LIVE_S675_HAZARD_EXPORT_V1"
                         if family=="s675" else "S680_LIVE_S672_HAZARD_EXPORT_V1")
        if info.get("schema")!=expected_schema:
            raise ValueError("NOTE specialist checkpoint metadata schema mismatch")
        if info.get("TRAIN_selected_policy")!="hazard_0.60_persist_1":
            raise ValueError("Expected the TRAIN-selected causal NOTE policy")
        if checkpoint.name!=info.get("checkpoint_filename"):
            raise ValueError("S680 checkpoint filename mismatch")
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=info["model_checkpoint_sha256"]:
            raise ValueError("S680 checkpoint SHA256 mismatch: file is corrupt or untrusted")
        if heads.sha256!=info["model_original_sha256"]:
            raise ValueError("S680 specialist was trained on a different original NOTE model")
        if info["feature_dimensions"]!=75:
            raise ValueError("S680 feature ABI mismatch")
        # joblib uses pickle: ONLY load the ZIP artifact published by this
        # project's pinned GitHub Actions run; never arbitrary *.joblib.
        model=joblib.load(checkpoint)
        if model.n_features_in_!=75:
            raise ValueError("S680 trained classifier dimensions mismatch")
        return cls(model,gate=float(info["gate"]),
                   persistence=int(info["persistence"]))

    def _clock(self,idx):
        a=np.zeros(4,np.float32)
        if not self.onsets:return a
        since=max(0,idx-self.onsets[-1])
        recent=self.onsets[-9:]
        periods=np.diff(recent)
        tempo=float(np.median(periods)) if len(periods) else 50.
        tempo=max(4.,tempo)
        a[:]=np.log1p(since),np.log1p(tempo),min(6.,since/tempo),np.cos(2*np.pi*since/tempo)
        return a

    def _emit(self,pitch,end,available,reason):
        self.stats[reason]+=1
        return dict(type="note_off",pitch=int(pitch),
                    decision_sample=int(end),
                    available_sample=int(available),
                    time=float(available/SR),reason=reason,velocity=0)

    def step(self,end,features,p,k,native_events,available):
        idx=self.hops
        if int(end)!=(idx+1)*HOP or available<end:
            raise AssertionError("S680 requires contiguous native 256-hop PCM")
        self.last_available_sample=int(available)
        self.rows.append(np.asarray(features,np.float32).copy())
        self.posteriors.append(np.asarray(p,np.float32).copy())
        self.k_values.append(np.asarray(k,np.float32).copy())
        self.hops+=1
        out=[]
        for ev in native_events:
            pitch=int(ev["pitch"])
            if ev["type"]=="note_on":
                if pitch in self.active:
                    out.append(self._emit(pitch,end,available,"s680_restrike"))
                    del self.active[pitch]
                self.active[pitch]=dict(start=idx,consecutive=0,
                                        native_off=None)
                self.onsets.append(idx)
                out.append(ev)
                self.stats["native_note_on_preserved"]+=1
            elif pitch in self.active and ev["type"]=="note_off":
                if ev["reason"]!="restrike":
                    self.active[pitch]["native_off"]=idx
                    self.stats["native_release_deferred"]+=1
                # Original restrike NoteOff is paired with a native NoteOn
                # later in the same causal step, handled by the branch above.

        candidates=[]
        matching=[]
        count=len(self.rows)
        feat=np.stack(self.rows)
        posterior=np.stack(self.posteriors)
        probs_k=np.stack(self.k_values)
        clock=np.zeros((count,4),np.float32)
        clock[-1]=self._clock(idx)
        for pitch,item in list(self.active.items()):
            age=idx-item["start"]
            if age < MIN_AGE_FRAMES:
                continue
            if age>=MAX_AGE_FRAMES:
                out.append(self._emit(pitch,end,available,"s680_max_age_fallback"))
                del self.active[pitch]
                continue
            # Local index is only the last 65 PCM hops, while note age
            # remains an absolute elapsed number. All posterior lags are
            # now clamped to max(t-64, 0) exactly as in S672.
            local=count-1
            local_start=local-age
            x=frame_matrix(feat,posterior,probs_k,None,pitch,local_start,
                           np.asarray([local],np.int64),clock)
            candidates.append(x[0]);matching.append(pitch)
        if matching:
            confidence=self.model.predict_proba(np.asarray(candidates,np.float32))[:,1]
            for pitch,prob in zip(matching,confidence):
                if pitch not in self.active:
                    continue
                item=self.active[pitch]
                item["consecutive"]=(item["consecutive"]+1
                                     if prob>=self.gate else 0)
                if item["consecutive"]>=self.persistence:
                    out.append(self._emit(pitch,end,available,"s680_causal_hazard"))
                    del self.active[pitch]
        for pitch,item in list(self.active.items()):
            native_off=item["native_off"]
            if native_off is not None and idx-native_off>=self.grace:
                out.append(self._emit(pitch,end,available,"s680_native_fallback_grace"))
                del self.active[pitch]
        return out

    def finish(self,available,reason):
        out=[self._emit(pitch,available,available,reason)
             for pitch in sorted(self.active)]
        self.active.clear()
        return out

    def diagnostics(self):
        return dict(s680_release_trial=True,gate=self.gate,
                    persistence=self.persistence,
                    total_hops=self.hops,
                    active_at_shutdown=len(self.active),
                    counts=dict(self.stats),
                    caveat="LIVE event timing/fallback not identical to S672 offline DEV benchmark 770/990")
