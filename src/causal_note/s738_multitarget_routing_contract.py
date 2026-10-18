"""S738: Shared Exact-K-style neural selection contract for NOTE targets.

The SAME architecture chooses a different one-of-32 H0-anchored subset
for each 4-hop fragment, separately for each trained target.

Do NOT imply a released checkpoint: only S737 Onset is currently trained.
Sustain/Release targets can be trained next via disjoint FIT data. Offset
requires an independent S707 hazard adapter; exact note identity needs
proper ADD/REMOVE/CHANGE_PITCH events; Exact-K needs seven-class proposals.
Those are not disguised as binary posteriors or reported as 80% achieved.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from causal_note.s736_fragment_mask import (
    MASKS, all_mask_offers, apply_fragment_masks, BLOCK_MS)

METRICS = (
    "onset_f1", "exact_k", "active_exact_k", "offset_accuracy",
    "full_note_f1", "frame_pitch_f1")
TRAIN_ORDER = ("onset", "sustain", "release", "offset",
               "pitch_identity", "exact_k")
POST_CHANNELS = {"onset": 0, "sustain": 1, "release": 2}
NON_POST_TARGETS = ("offset", "pitch_identity", "exact_k")

@dataclass(frozen=True)
class StageState:
    target: str
    source_sha256: str
    trained_on_fit: bool
    cal_metrics: dict
    cal_baseline: dict
    preserves_correct_offsets: bool
    preserves_long_notes: bool
    promotion: bool = False

    def eligible(self):
        if self.target not in TRAIN_ORDER:
            raise ValueError("Unsupported NOTE training target")
        if len(self.source_sha256) != 64:
            raise ValueError("Signed source checkpoint required")
        if not self.trained_on_fit or self.promotion:
            return False
        if any(key not in self.cal_metrics or key not in self.cal_baseline
               for key in METRICS):
            raise ValueError("Must measure every one of the six official metrics")
        return (all(self.cal_metrics[k] >= self.cal_baseline[k] - 1e-12
                    for k in METRICS) and
                self.preserves_correct_offsets and
                self.preserves_long_notes)

    def target_achieved(self):
        return self.eligible() and all(self.cal_metrics[k] > .80
                                       for k in METRICS)


def acoustic_other_target_heads(target, post, native_post, context, k):
    """Proposals for Sustain or Release, each independently scored by router.

    H0 frozen temporal head (required). H1 native acoustic head.
    H2 causal previous block. H3 acoustic novelty/fade.
    H4 independent Onset/Sustain cues. H5 soft conditional K evidence.
    No future observations, labels, hard K budget, or oracle notes.
    """
    if target not in ("sustain", "release"):
        raise ValueError("This adapter supports Sustain or Release ONLY; "
                         "S707 Offset and pitch-identity are separate domains")
    t=len(post)
    if (post.shape!=(t,49,3) or native_post.shape!=(t,49,3)
            or context.shape!=(t,49,16) or k.shape!=(t,7)):
        raise ValueError("Unmatched frozen NOTE target features")
    if any(not np.isfinite(a).all()
           for a in (post,native_post,context,k)):
        raise ValueError("Invalid acoustic features")
    col=POST_CHANNELS[target]
    base=post[:,:,col]
    native=native_post[:,:,col]
    prev=np.concatenate((base[:1],base[:-1]),axis=0)
    onset=post[:,:,0]
    sustain=post[:,:,1]
    novelty=np.clip(context[:,:,4],0,1)
    source_count=np.clip(context[:,:,15],0,1)
    k_active=np.sum(k[:,1:],axis=1)[:,None]
    if target=="sustain":
        offers=np.stack((
            base,
            native,
            .75*prev+.25*base,
            np.clip(base+.30*novelty,0,1),
            np.clip(np.maximum(base,.5*onset*novelty),0,1),
            np.clip(.60*base+.20*source_count+.20*k_active,0,1)
        ),axis=-1)
    else:
        falling=np.maximum(prev-base,0.)
        offers=np.stack((
            base,
            native,
            np.clip(base+.5*falling,0,1),
            np.clip(base+.3*(1-sustain)*(1-novelty),0,1),
            np.clip(base+.30*(1-source_count)*(1-sustain),0,1),
            np.clip(base+.25*(1-k_active)*(1-sustain),0,1)
        ),axis=-1)
    if offers.shape!=(t,49,6):
        raise AssertionError("Each target needs six true acoustic proposals")
    return np.asarray(np.clip(offers,0.,1.),dtype=np.float32)


def apply_one_target(post, target, heads, eligible, masks):
    """Apply *one* one-mask-per-fragment decision, bit-preserving other heads.

    The generic 32-subset implementation is reused without refitting Exact-K.
    """
    if target not in POST_CHANNELS:
        raise ValueError("Offset/S707 and note identity cannot be modified "
                         "through a binary posterior-channel substitution")
    channel=POST_CHANNELS[target]
    # Universal 32-mask implementation is defined for post channel 0.
    permuted=post.copy()
    permuted[:,:,0]=post[:,:,channel]
    adapted=apply_fragment_masks(permuted,heads,eligible,masks)
    out=post.copy()
    out[:,:,channel]=adapted[:,:,0]
    keep=[i for i in range(3) if i!=channel]
    if not np.array_equal(out[:,:,keep],post[:,:,keep]):
        raise AssertionError("Target selector altered frozen other functions")
    return out


def require_sequential_training(target, completed):
    """Only advance after previous target's safe CAL freeze; never joint FIT."""
    if target not in TRAIN_ORDER:
        raise ValueError("Unknown target")
    index=TRAIN_ORDER.index(target)
    missing=[s for s in TRAIN_ORDER[:index] if s not in completed]
    if missing:
        raise RuntimeError("First qualify prior target(s): "+", ".join(missing))
    return True


def ensure_valid_adapters(target):
    if target in NON_POST_TARGETS:
        raise NotImplementedError(
            target+" must have a separately trained action-specific adapter "
            "before its scores can be claimed")
    return True
