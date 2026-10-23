"""S834: genuine S809 HEAD -> SELECTOR -> DECODER six-axis private audio.
Feeds six immutable independent S809-native256 pathways from freshly computed
S708 posteriors and K distribution at *each* received PCM256 boundary.

Frozen S708 weights were trained with 1024 decision grid, NOT retrained to 256.
No current or past S708 source MIDI is ever consumed by inference. The protected
public MIDI is not touched. This is a SHADOW scientific adapter.
"""
from __future__ import annotations
from collections import deque
import time
import numpy as np
from causal_note.strict_live import HOP
from causal_note.s706_sharp_spectra import sharp_track
from causal_note.s836_incremental_sharp256 import IncrementalSharp256
from causal_note.s712_s708_live import latest_s708_acoustic_features
from causal_note.s827_s708_activek_only import (
    S827S708ActiveKOnly,PAST_AUDIO
)
from causal_note.s834_native256_decoders import AXES,Fragment
from causal_note.s834_s809_six_axis_pcm256 import (
    SixIndependentAxes,default_pipelines,AxisSelector
)

SCHEMA="S834_S809_NATIVE256_S708_SIX_SHADOW_REALPCM_V1"

class S834S809Native256(S827S708ActiveKOnly):
    """Six READ-ONLY shadow outputs and one fresh temporal S708 call per PCM256."""
    def __init__(self,native_heads,original_s708_model,frozen_policy,*,modes=None,
                 sharp_engine="reference"):
        super().__init__(native_heads,original_s708_model,frozen_policy,
                         capture_features=False)
        if sharp_engine not in ("reference","incremental"):
            raise ValueError("S836 sharp engine must be reference or incremental")
        self.sharp_engine=sharp_engine
        self.sharp_cache=IncrementalSharp256() if sharp_engine=="incremental" else None
        self.six=SixIndependentAxes(default_pipelines(modes))
        self.last_decision=0
        self.decisions={}
        self.selection={}
        self.source_k=[]
        self.source_post=[]
        self.model_call_ms=deque(maxlen=4096)
        self.frontend_call_ms=deque(maxlen=4096)

    def _step(self,end,f):
        # The source computation is the exact S827 native256 S708 acoustic
        # path. S809 architecture is its *consumer*, never the other way.
        before=time.perf_counter()
        p,k=self.native_heads.predict(f)
        self.native_feat.append(np.asarray(f,np.float32))
        self.native_post.append(np.asarray(p,np.float32))
        self.native_k.append(np.asarray(k,np.float32))
        acoustic=latest_s708_acoustic_features(
            self.native_feat,self.native_post,self.native_k)
        n=len(self.audio)
        if self.sharp_cache is None:
            sharp_ends=(np.array([n],np.int64) if end==HOP else
                        np.array([n-HOP,n],np.int64))
            spectral=sharp_track(self.audio,sharp_ends)[-1]
        else:
            spectral=self.sharp_cache.step(self.audio,end=int(end))
        X=np.concatenate((acoustic,spectral),axis=1)
        if X.shape!=(49,288) or not np.isfinite(X).all():
            raise AssertionError("S834 must have fresh 288D S708 PCM256 acoustics")
        # Audit-only pointer for the replay process to hash genuine 288D ABI\n        # after push() returns, outside the measured model CPU interval.\n        self.s836_last_input_288d=X\n        nn_post,nn_count=self.temporal.step(X)
        self.model_calls+=1
        self.model_call_ms.append(1000*(time.perf_counter()-before))
        blend=float(self.decoder.policy.blend)
        post=np.clip(blend*nn_post+(1-blend)*self.native_post[-1],0,1)
        count=np.asarray(nn_count,np.float32)
        if count.shape!=(7,) or abs(float(count.sum())-1)>1e-3:
            raise AssertionError("fresh K0..K6 S708 posterior required")
        snapshot=Fragment(int(end),post,count)
        if end!=self.last_decision+HOP:
            raise AssertionError("S834 must decide on EVERY fresh PCM256 boundary")
        decisions,selections=self.six.step(snapshot)
        if any(x.sample!=end for x in decisions.values()):
            raise AssertionError("S834 output timestamp drift")
        self.last_decision=end
        self.source_k.append(int(count.argmax()))
        self.source_post.append(post)
        self.decisions=end,decisions
        self.selection=end,selections
        return end,decisions,selections,snapshot

    def push(self,samples,*,start_sample):
        before=time.perf_counter()
        items=super().push(samples,start_sample=start_sample)
        self.frontend_call_ms.append(1000*(time.perf_counter()-before))
        return items

    def diagnostics(self):
        d=super().diagnostics()
        d.update(schema=SCHEMA,decision_hop_samples=HOP,
            s836_sharp_engine=self.sharp_engine,
            current_source_S708_nn_calls=self.model_calls,
            six_independent_axes=True,all_six_private=True,
            share_only_immutable_S708_acoustic_snapshot=True,
            original_teacher_training_grid=4*HOP,
            source_s708_retrained_to_256=False,
            public_midi_modified=False,added_lookahead_samples=0,
            added_future_hold_samples=0,private_axis_record_count=self.last_decision//HOP,
            live_physical_latency_verified=False,
            p95_full_chunk_cpu_ms=float(np.percentile(
                self.frontend_call_ms,95)) if self.frontend_call_ms else None,
            p99_full_chunk_cpu_ms=float(np.percentile(
                self.frontend_call_ms,99)) if self.frontend_call_ms else None)
        return d
