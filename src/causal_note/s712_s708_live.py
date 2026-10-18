"""S712 opt-in S708 GuitarSet causal heads -> actual microphone NoteOn/NoteOff.

The selected S708 model has 4x256-sample decision blocks, 65 prior native
frames for 371ms source-only acoustic context, extra 96D sharp spectra and
a left-padded causal temporal net (30 decision blocks). No future samples,
JAMS, oracle K, MIDI sidecars, retrospective timestamps or stream buffering
beyond past history. The live S708 NoteOff is the JOINT selector release,
not S707's separately trained offline hazard policy: never claim offline
S708 full-note/offset scores as actual microphone scores.
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from causal_note.strict_live import CausalPitchFeatures, HOP, SR
from causal_note.s704_joint_heads import acoustic_features
from causal_note.s705_temporal_heads import TemporalHeadStream, TemporalHeads
from causal_note.s706_sharp_spectra import sharp_track
from causal_note.s708_cardinality_selection import GuidedPolicy, GuidedSelector

BATCH = 4
DECISION = BATCH * HOP
PAST_AUDIO = 8192 + DECISION


def load_s708_checkpoint(checkpoint, report):
    """Verify the actual selected S708 model, family and frozen CAL policy."""
    import torch
    checkpoint, report = Path(checkpoint), Path(report)
    data = json.loads(report.read_text(encoding="utf-8"))
    if data.get("schema") != "S708_K_GUIDED_S706_CAL_LOOPS":
        raise ValueError("Expected original S708 family-706 benchmark report")
    if data.get("source_family") != 706 or data.get("promoted") is not False:
        raise ValueError("S708 family or shadow promotion flag mismatch")
    sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if sha != data.get("checkpoint_sha256"):
        raise ValueError("S708 selected model SHA256 mismatch; use workflow artifact")
    original_name = data["frozen_candidate"]["choice"]["model_file"]
    if original_name != "s706_epoch10.pt" or checkpoint.name != "s708_selected.pt":
        raise ValueError("S708 source checkpoint does not match the measured candidate")
    policy_data = data["frozen_candidate"]["choice"]["policy"]
    policy = GuidedPolicy(**policy_data)
    bundle = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = TemporalHeads(bundle["model"]["mean"], bundle["model"]["scale"],
                          width=int(bundle["width"]))
    model.load_state_dict(bundle["model"], strict=True)
    model.eval()
    if len(model.mean) != 288:
        raise ValueError("S708 288D input ABI mismatch")
    # Deterministic CPU thread budget is important for a callback-free worker.
    torch.set_num_threads(1)
    return model, policy, sha


class S708StrictLiveStream:
    """Four-hop S708 stream with source-only past data and fail-fast continuity."""
    def __init__(self, heads, model, policy):
        self.heads = heads
        self.frontend = CausalPitchFeatures(hop=HOP)
        self.temporal = TemporalHeadStream(model)
        self.selector = GuidedSelector(policy)
        self.native_feat = deque(maxlen=65)
        self.native_post = deque(maxlen=65)
        self.native_k = deque(maxlen=65)
        self.audio = np.empty(0, dtype=np.float32)
        self.frames = 0
        self.received = 0
        self.closed = False
        self.timings_ms = deque(maxlen=4096)
        self.model_calls = 0

    def _decide(self, end, available):
        feat = np.stack(self.native_feat)
        post = np.stack(self.native_post)
        count = np.stack(self.native_k)
        x = acoustic_features(feat, post, count)[-1]
        # Exactly two consecutive four-hop FFT instants are required for
        # phase-coherence evidence. First block has no previous FFT.
        ne = len(self.audio)
        if end == DECISION:
            ends = np.array([ne], np.int64)
        else:
            ends = np.array([ne - DECISION, ne], np.int64)
        sharp = sharp_track(self.audio, ends)[-1]
        features = np.concatenate((x, sharp), axis=1)
        if features.shape != (49, 288):
            raise AssertionError("S708 acoustic/temporal feature ABI mismatch")
        p, k = self.temporal.step(features)
        trial = self.selector.policy.blend*p + (
            1-self.selector.policy.blend)*post[-1]
        emitted = self.selector.step(trial, k, int(end))
        self.model_calls += 1
        for e in emitted:
            # An external input callback may already have received more PCM
            # than the nominal decision boundary; NEVER timestamp in past.
            e["available_sample"] = int(available)
            e["time"] = float(available/SR)
            e["s708_shadow_trial"] = True
        return emitted

    def push(self, samples, *, start_sample):
        if self.closed:
            raise ValueError("Closed S708 stream")
        if type(start_sample) is not int or start_sample != self.received:
            raise ValueError("S708 discontinuous microphone stream")
        x = np.asarray(samples)
        if x.ndim != 1 or x.dtype.kind != "f" or not np.isfinite(x).all():
            raise ValueError("S708 expects finite mono floating-point PCM")
        begin = time.perf_counter()
        available = start_sample + len(x)
        emitted = []
        at = 0
        while at < len(x):
            # Advance precisely to next native frame so extra FFTs NEVER see
            # future PCM accidentally present later in the callback block.
            length = min(len(x)-at, HOP-(self.frontend.received % HOP))
            block = np.asarray(x[at:at+length], np.float32)
            self.audio = np.concatenate((self.audio, block))[-PAST_AUDIO:]
            frames = self.frontend.push(block, start_sample=self.received)
            self.received += length
            at += length
            for end, f in frames:
                p, k = self.heads.predict(f)
                self.native_feat.append(np.asarray(f, np.float32))
                self.native_post.append(np.asarray(p, np.float32))
                self.native_k.append(np.asarray(k, np.float32))
                self.frames += 1
                if self.frames % BATCH == 0:
                    emitted.extend(self._decide(end, available))
        self.timings_ms.append((time.perf_counter()-begin)*1000)
        return emitted

    def finish(self, reason="stream_end"):
        if self.closed:
            return []
        self.closed = True
        end = self.received
        events = self.selector.finish(end)
        for e in events:
            e["reason"] = reason
            e["s708_shadow_trial"] = True
        return events

    def diagnostics(self):
        return dict(s708=True, frames=self.frames,
                    decision_hop=DECISION, model_calls=self.model_calls,
                    source_past_only=True, torch_cpu_inference=True,
                    offset_source="S708_joint_selector_not_S707",
                    native_reference_unmodified=True)
