"""S826: unchanged S708 neural/harmonic/K/NoteOn/NoteOff architecture at 256.

This is the *actual original* S708 selected 288-D architecture, not an
Active-K replacement or a S795 aggregation and NOT a distillation.
Only the decision clock is changed: acquire 256 PCM, compute native 126D,
derive exactly S708 acoustic 192D plus 96D sharp phase/contrast from
past samples, execute existing S705 causal temporal net, feed the SAME
guided K selector and emit notes at that current 256 sample boundary.

The original frozen s708_selected.pt was FIT at 1024 decision cadence.
Its weights are REUSED FOR DIAGNOSTICS ONLY, not silently described as
a retrained/validated 256 neural model. Future S826 phase must retrain
the SAME TemporalHeads with native256 sequence labels and refreeze CAL.

Right-aligned past 8192 FFT window is NOT a deferred audio decision:
historical signal is available at the current 256 decision boundary.
"""
from __future__ import annotations

import numpy as np

from causal_note.strict_live import HOP,SR
from causal_note.s712_s708_live import (
    S708StrictLiveStream, latest_s708_acoustic_features
)
from causal_note.s708_cardinality_selection import GuidedSelector
from causal_note.s706_sharp_spectra import sharp_track

SCHEMA="S826_EXACT_S708_WHOLE_ARCH_NATIVE256_REPLAY_V1"
NATIVE_DECISION_HOP=256

class S826S708Native256Stream(S708StrictLiveStream):
    """S708's identical architecture, called once for every received 256.

    No S795 features, no replacement K head and no S708 peer to be distilled.
    Replay is opt-in, not installed in production LIVE.
    """
    def __init__(self,heads,model,policy,*,temporal_engine="incremental"):
        super().__init__(heads,model,policy,temporal_engine=temporal_engine,
                         release_adapter=None,batch_hops=1)
        self.selector=GuidedSelector(policy,decision_hop_samples=HOP)
        self.original_weights_fit_hop_samples=4*HOP
        self.s826_source_288d=[]
        self.s826_decision_ends=[]

    def _decide(self,end,available):
        if end!=self.received or end%HOP:
            raise AssertionError("S826 must decide only at last acquired PCM256 end")
        if self.frames!=len(self.s826_decision_ends)+1:
            raise AssertionError("S826 exactly one sequential 256 decision per frame")
        raw=latest_s708_acoustic_features(
            self.native_feat,self.native_post,self.native_k)
        n=len(self.audio)
        # sharp_track is a PAST-only FFT with previous phase coherence.
        # At hop 256, prior FFT is at t-256, never t+future.
        ends=np.asarray([n],np.int64) if end==HOP else np.asarray([n-HOP,n],np.int64)
        if ends[0]<1:
            raise AssertionError("non-positive sharp frame end")
        sharp=sharp_track(self.audio,ends)[-1]
        features=np.concatenate((raw,sharp),axis=1)
        if features.shape!=(49,288) or not np.isfinite(features).all():
            raise AssertionError("S708 original 288D ABI or finite input changed")
        p,k=self.temporal.step(features)
        if np.asarray(p).shape!=(49,3) or np.asarray(k).shape!=(7,):
            raise AssertionError("S708 exact original Onset/Sustain/Release/K outputs missing")
        trial=self.selector.policy.blend*p+(1.-self.selector.policy.blend)*self.native_post[-1]
        emitted=self.selector.step(trial,k,int(end))
        self.model_calls+=1
        self.s826_source_288d.append(features.copy())
        self.s826_decision_ends.append(int(end))
        for e in emitted:
            e["available_sample"]=int(available)
            e["time"]=float(available/SR)
            e["s826_strict256_shadow"]=True
        return emitted

    def diagnostics(self):
        meta=super().diagnostics()
        meta.update(dict(schema=SCHEMA,decision_hop=HOP,
            sample_rate_hz=SR,s708_architecture_reused=True,
            original_frozen_checkpoint_training_hop=4*HOP,
            source_features_288d_every_256=True,
            updated_legacy_s708_production=False,
            pretrained_1024_weights_revalidated_at_256=False,
            source_S708_guided_selector_kept=True,
            added_lookahead_samples=0,added_hold_samples=0))
        return meta
