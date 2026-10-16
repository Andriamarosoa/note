"""S630: batch frozen NOTE MLP+ONNX across consecutive CAUSAL 256-hop frames.

This is a workload throughput optimisation, NOT a new NOTE acoustic
model/selection. The original CausalPitchFeatures computes the EXACT
original 126-D feature row for every 256-hop; `heads.predict` already
supports batched frames (B,49,126). This performs ONE neural+3-tree
ONNX inference for B neighboring original frames and then feeds all
original `CausalSelection` steps in chronological order.

Real-time cost: to use B frames, model decisions for early frames WAIT
until B successive real frames have arrived. Never backdate emitted
events. At 44.1k, 256-hop=5.805ms, batch 4 adds up to 17.415ms
intentional lookahead WAIT (plus callback batching and computation);
however onset probabilities use only their own prior audio, so this
is batching latency, not a trained future-conditioned classifier.

Optional S629 future evidence reuses shifted already-predicted frozen
hop frames. Both batch and actual future delays are reported distinctly.

No discarded audio, changed thresholds, new MIDI owner, or hidden queue.
"""
from __future__ import annotations

from collections import Counter,deque
import math,time

import numpy as np

from causal_note.strict_live import CausalPitchFeatures,CausalSelection,Selection,HOP,SR
from causal_note.lookahead import LookaheadConfig,combine_reference_and_future


class BatchedLiveStream:
    def __init__(self,heads,*,batch_hops=4,lookahead=None,register_rescue_mode="off",register_rescue_policy=None):
        if not isinstance(batch_hops,int) or not 1<=batch_hops<=32:
            raise ValueError("batch_hops must be 1..32")
        if register_rescue_mode not in ("off","shadow","guard"):
            raise ValueError("Unsupported S633 rescue mode")
        if register_rescue_mode!="off" and register_rescue_policy is None:
            raise ValueError("S633 rescue requires a fixed explicit policy")
        if lookahead is not None and register_rescue_mode=="guard":
            raise ValueError("S633 guard was only validated with zero acoustic lookahead")
        self.register_rescue_mode=register_rescue_mode
        self.register_rescue_policy=register_rescue_policy
        self.heads=heads
        self.batch_hops=batch_hops
        self.config=lookahead
        self.future_hops=(math.ceil(lookahead.samples/HOP) if lookahead is not None else 0)
        self.actual_lookahead_samples=self.future_hops*HOP
        self.frontend=CausalPitchFeatures(hop=HOP)
        self.selector=CausalSelection(Selection(**heads.metadata["selection"]),hop=HOP)
        self.harmonic_selector=None
        self.received=0
        self.closed=False
        self.pending_features=[]
        self.pending_predictions=deque()
        self.stats=Counter()
        self.worker_ms=deque(maxlen=4096)

    def _predict_pending(self,available_sample):
        if not self.pending_features:
            return []
        frames=self.pending_features
        self.pending_features=[]
        # Treat a single row as a 3-D batch to enforce the *same*
        # inference output shapes in every case.
        features=np.stack([row for _,row in frames],axis=0)
        predictions,counts=self.heads.predict(features)
        if predictions.shape!=(len(frames),49,3) or counts.shape!=(len(frames),7):
            raise ValueError("Frozen NOTE batched head returned unexpected shapes")
        self.stats["batch_inference_calls"]+=1
        self.stats["frames_inferred"]+=len(frames)
        self.stats["largest_batch"]=max(self.stats["largest_batch"],len(frames))
        out=[]
        for (t,source_features),p,k in zip(frames,predictions,counts):
            if t>available_sample:
                raise AssertionError("Inference before source frames captured")
            self.pending_predictions.append((t,p,k,source_features))
            if len(self.pending_predictions)>self.future_hops:
                original_t,original_p,original_k,original_features=self.pending_predictions.popleft()
                if self.future_hops:
                    if t-original_t!=self.actual_lookahead_samples:
                        raise AssertionError("Lookahead future hop misalignment")
                    fused=combine_reference_and_future(original_p,p,self.config.mode)
                else:
                    fused=original_p
                if self.register_rescue_mode!="off":
                    from causal_note.register_rescue import live_apply
                    fused=live_apply(
                        original_features,fused,original_k,self.selector,
                        mode=self.register_rescue_mode,
                        policy=self.register_rescue_policy,stats=self.stats)
                out.extend(self.selector.step(fused,original_k,original_t,available_sample))
                self.stats["frames_selected"]+=1
        return out

    def push(self,samples,*,start_sample):
        if self.closed or start_sample!=self.received:
            raise ValueError("Closed or discontinuous NOTE microphone stream")
        x=np.asarray(samples)
        if x.ndim!=1 or x.dtype.kind!="f" or not np.isfinite(x).all():
            raise ValueError("Expected contiguous finite float mono PCM")
        available_sample=start_sample+len(x)
        before=time.perf_counter()
        out=[]
        for offset in range(0,len(x),4096):
            segment=x[offset:offset+4096]
            for t,features in self.frontend.push(segment,start_sample=start_sample+offset):
                self.pending_features.append((int(t),features))
                self.stats["frames_acoustically_computed"]+=1
                if len(self.pending_features)>=self.batch_hops:
                    out.extend(self._predict_pending(available_sample))
        self.received=available_sample
        self.worker_ms.append((time.perf_counter()-before)*1000)
        return out

    def finish(self,reason="stream_end"):
        if self.closed:
            return []
        self.closed=True
        out=self._predict_pending(self.received)
        # Original predictions with unobserved future kept at EOF only,
        # tagged; no padding/fabricated attack evidence.
        for t,p,k,features in self.pending_predictions:
            if self.register_rescue_mode!="off":
                from causal_note.register_rescue import live_apply
                p=live_apply(features,p,k,self.selector,
                    mode=self.register_rescue_mode,policy=self.register_rescue_policy,
                    stats=self.stats)
            events=self.selector.step(p,k,t,self.received)
            if self.future_hops:
                for e in events:
                    e["reason"]="lookahead_eof_unconfirmed_"+e["reason"]
            out.extend(events)
            self.stats["EOF_unconfirmed_frames"]+=1
        self.pending_predictions.clear()
        out.extend(self.selector.panic(self.received,reason=reason))
        return out

    def diagnostics(self):
        ms=self.actual_lookahead_samples*1000/SR
        worker=np.asarray(self.worker_ms,dtype=np.float64)
        return dict(
            engine="S630_single_batched_frozen_onnx",
            model_inference_batch_hops=self.batch_hops,
            native_decision_hop=HOP,
            native_decision_hop_ms=HOP*1000/SR,
            max_intentional_batch_wait_ms=(self.batch_hops-1)*HOP*1000/SR,
            requested_lookahead_ms=(self.config.milliseconds if self.config else 0.),
            effective_shared_future_lookahead_ms=ms,
            combined_max_additional_wait_before_processing_ms=(
                (self.batch_hops-1)*HOP*1000/SR+ms),
            future_mode=(self.config.mode if self.config else "off"),
            original_note_heads_frozen=True,
            selected_note_K_from_original_frame=True,
            s633_register_rescue_mode=self.register_rescue_mode,
            s633_guard_changes_selected_pitch_onsets=self.register_rescue_mode=="guard",
            source=dict(self.stats),
            samples_received=self.received,
            worker_callback_count=len(worker),
            worker_p95_ms=(float(np.percentile(worker,95)) if len(worker) else None),
            worker_max_ms=(float(worker.max()) if len(worker) else None),
            worker_total_ms=float(worker.sum()),
            worker_over_audio_ratio=float(worker.sum())/
                 max(1,self.received*1000/SR),
            warning=(
              "Batching amortizes ONNX overhead at the COST of waiting"
              " for multiple future captured hops, even when"
              " lookahead-ms=0. Events not backdated. An observed"
              " Windows CPU ratio>=1 means queue will eventually fill."
            ),
        )
