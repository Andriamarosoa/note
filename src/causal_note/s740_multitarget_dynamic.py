"""S740: six independent Exact-K-architecture fragment selectors (research only).

Each target has its OWN FragmentMaskRouter parameters.  The shared central
architecture is S739's faithful K0..K6 attention/KEEP/risk decoder, not
one global mask for all targets.  Every decision is per causal four-hop
fragment.  No trained S740 checkpoints, official scores or Live promotion
are implied by constructing this module.

Offset operates on S707's *per-note instance* hazards. Pitch identity
operates on five explicit event actions, not on Sustain/Release aliases.
"""
from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
import numpy as np
import torch
from torch import nn

from causal_note.s736_fragment_mask import (
    MASKS, FragmentMaskRouter, choose_masks, all_mask_offers,
)
from causal_note.s738_multitarget_routing_contract import (
    METRICS, acoustic_other_target_heads,
)

TARGETS = ("onset", "sustain", "release", "offset", "pitch_identity", "exact_k")
ACTIONS = ("KEEP", "ADD", "REMOVE", "CHANGE_PITCH", "REATTACK")


class SixDomainFragmentRouters(nn.Module):
    """One independent learned network per domain, 32 subsets per fragment.

    Training and audit are performed separately by domain on FIT with
    composition-disjoint OOF statistics.  Never call a randomly initialized
    model at inference: select_masks rejects unapproved checkpoints.
    """

    def __init__(self):
        super().__init__()
        self.routers = nn.ModuleDict({
            target: FragmentMaskRouter() for target in TARGETS
        })

    def forward(self, target, features, descriptors, audits, frozen_k, compatible):
        if target not in TARGETS:
            raise ValueError("Unknown target")
        return self.routers[target](
            features, descriptors, audits, frozen_k, compatible
        )

    def select_masks(self, target, features, descriptors, audits, frozen_k,
                     compatible, mean, std, *, checkpoint_verified=False,
                     fit_oof_verified=False, min_confidence=0.):
        if target not in TARGETS:
            raise ValueError("Unknown target")
        if not checkpoint_verified or not fit_oof_verified:
            raise RuntimeError("Untrained or unaudited target must KEEP S708")
        return choose_masks(
            self.routers[target], features, descriptors, audits, frozen_k,
            compatible, mean, std, min_confidence=min_confidence
        )


def _shape_checks(base, heads, eligible):
    if base.ndim < 2 or heads.shape != (*base.shape, 6):
        raise ValueError("Expected base [fragments,...] and six real heads")
    if eligible.shape != base.shape:
        raise ValueError("Expected target-specific availability matching base")
    if not all(np.isfinite(x).all() for x in (base, heads)):
        raise ValueError("Non-finite acoustic proposals")
    if np.any(base < 0) or np.any(base > 1) or np.any(heads < 0) or np.any(heads > 1):
        raise ValueError("All six heads must be probabilities")
    if not np.allclose(base, heads[..., 0], atol=1e-6):
        raise ValueError("H0 MUST be the frozen baseline, never a proxy")


def build_offers(base, heads, eligible, *, fusion="max"):
    """Return every one of the 32 *fragment-specific* counterfactuals.

    Monotonic max is appropriate for Onset/Sustain/Release proposals.
    Mean permits suppression AND augmentation for Offset, identity, Exact-K.
    """
    base = np.asarray(base, dtype=np.float32)
    heads = np.asarray(heads, dtype=np.float32)
    eligible = np.asarray(eligible, dtype=bool)
    _shape_checks(base, heads, eligible)
    if fusion not in ("max", "mean"):
        raise ValueError("Unsupported fusion")
    out = np.empty((len(base), 32, *base.shape[1:]), np.float32)
    for i, mask in enumerate(MASKS):
        members = [h for h in range(6) if mask & (1 << h)]
        proposal = heads[..., members].max(axis=-1) if fusion == "max" \
                   else heads[..., members].mean(axis=-1)
        out[:, i] = np.where(eligible, proposal, base)
    # Numerically exact baseline for mask H0, and original frozen non-eligible cells.
    out[:, 0] = base
    return out


def select_offers(base, heads, eligible, masks, *, fusion="max",
                  normalization=None):
    """Use exactly one subset per fragment, independently for each target."""
    base = np.asarray(base, dtype=np.float32)
    masks = np.asarray(masks)
    if masks.shape != (len(base),) or any(int(m) not in MASKS for m in masks):
        raise ValueError("One H0-inclusive mask per fragment required")
    options = build_offers(base, heads, eligible, fusion=fusion)
    result = options[np.arange(len(base)), masks.astype(np.int64) // 2].copy()
    if normalization is not None:
        if normalization != "last":
            raise ValueError("Unknown normalization")
        # H0 must remain bit-identical even in probability-simplex targets.
        changed = masks != 1
        denom = result[changed].sum(axis=-1, keepdims=True)
        if np.any(denom <= 0):
            raise ValueError("Zero-sum class distribution")
        result[changed] /= denom
    return result


def target_descriptors(base, heads, eligible, *, novelty=None, fusion="max"):
    """Twelve real acoustic/behavior descriptors per subset.

    Flexible target shape: [B,49], [B,49,5] or [B,7].
    The FIT OOF per-K audit remains a separate mandatory input.
    """
    options = build_offers(base, heads, eligible, fusion=fusion)
    base = np.asarray(base, dtype=np.float32)
    if novelty is None:
        novelty = np.zeros_like(base)
    else:
        novelty = np.asarray(novelty, dtype=np.float32)
        if novelty.shape != base.shape or not np.isfinite(novelty).all():
            raise ValueError("Novelty must match target distribution")
    flat = options.reshape(len(base), 32, -1)
    origin = base.reshape(len(base), 1, -1)
    diff = np.abs(flat - origin)
    eligible_flat = np.asarray(eligible, bool).reshape(len(base), 1, -1)
    nov = novelty.reshape(len(base), 1, -1)
    d = np.empty((len(base), 32, 12), np.float32)
    d[:, :, 0] = diff.mean(axis=-1)
    d[:, :, 1] = diff.max(axis=-1)
    d[:, :, 2] = (diff > .02).mean(axis=-1)
    d[:, :, 3] = (diff * np.clip(nov, 0, 1)).mean(axis=-1)
    d[:, :, 4] = (flat >= .45).mean(axis=-1)
    d[:, :, 5] = ((flat >= .45) & eligible_flat).mean(axis=-1)
    for i, mask in enumerate(MASKS):
        d[:, i, 6:] = [(mask >> h) & 1 for h in range(6)]
    return d


def posterior_target(target, post, native, context, frozen_k):
    """Actual independent Sustain and Release proposal adapters from S738."""
    if target not in ("sustain", "release"):
        raise ValueError("Use separate adapters for Offset, pitch identity or K")
    heads = acoustic_other_target_heads(
        target, post, native, context, frozen_k)
    column = 1 if target == "sustain" else 2
    return post[:, :, column].copy(), heads


def offset_instance_candidates(plan, post, context, frozen_k):
    """Six real event-hazard candidates on S707's exact NOTE INSTANCES.

    plan is S707 prepare_hazards output; no track-level pitch hazard proxy.
    Frame indices are existing 4-hop decision end hops (3,7,11,...).
    The original S707 fallback timers and emitter are left untouched.
    """
    t = len(post)
    if post.shape != (t, 49, 3) or context.shape != (t, 49, 16) \
       or frozen_k.shape != (t, 7):
        raise ValueError("Offset needs S707 frozen hop-aligned features")
    if len(plan["notes"]) != len(plan["frames"]) \
       or len(plan["notes"]) != len(plan["probabilities"]):
        raise ValueError("Malformed S707 per-instance plan")
    all_heads = []
    for note, frames, original in zip(
            plan["notes"], plan["frames"], plan["probabilities"]):
        frames = np.asarray(frames, dtype=np.int64)
        p = np.asarray(original, dtype=np.float32)
        if len(frames) != len(p) or np.any(frames < 3) \
           or np.any(frames >= t) or np.any(frames % 4 != 3):
            raise ValueError("Non-causal S707 hazard frame grid")
        if not 40 <= int(note["pitch"]) <= 88:
            raise ValueError("Invalid pitch")
        if not len(frames):
            all_heads.append(np.empty((0, 6), np.float32))
            continue
        col = int(note["pitch"]) - 40
        previous = np.concatenate((p[:1], p[:-1]))
        sustain = np.clip(post[frames, col, 1], 0, 1)
        release = np.clip(post[frames, col, 2], 0, 1)
        novelty = np.clip(context[frames, col, 4], 0, 1)
        age = np.clip((frames - int(note["start"])) / 128., 0, 1)
        quiet_k = np.clip(frozen_k[frames, 0], 0, 1)
        offers = np.stack((
            p, release, previous,
            np.clip(p * (1 - .35 * novelty), 0, 1),
            np.clip(p + .2 * age * (1 - sustain), 0, 1),
            np.clip(p + .2 * quiet_k * (1 - sustain), 0, 1),
        ), axis=-1).astype(np.float32)
        _shape_checks(p.reshape(1, -1), offers.reshape(1, -1, 6),
                      np.ones((1, len(p)), bool))
        all_heads.append(offers)
    return all_heads


def apply_offset_masks(plan, per_instance_heads, fragment_masks):
    """Alter only hazard observations, never native fallback timers.

    This is a research candidate. Calling S707 render_release on its output
    and scoring *complete emitted notes* is still mandatory for acceptance.
    """
    masks = np.asarray(fragment_masks)
    if any(int(m) not in MASKS for m in masks):
        raise ValueError("Illegal four-hop mask")
    if len(plan["notes"]) != len(per_instance_heads):
        raise ValueError("One offset proposal series per note instance")
    out = deepcopy(plan)
    for i, (frames, old, offers) in enumerate(zip(
            plan["frames"], plan["probabilities"], per_instance_heads)):
        frames = np.asarray(frames, dtype=np.int64)
        p = np.asarray(old, dtype=np.float32)
        if len(frames) != len(p) or offers.shape != (len(p), 6):
            raise ValueError("Offset proposals must align S707 instance frames")
        if not len(frames):
            continue
        blocks = (frames - 3) // 4
        if np.max(blocks) >= len(masks):
            raise ValueError("Missing causal mask decision for Offset")
        selected = np.empty_like(p)
        for row, block in enumerate(blocks):
            mask = int(masks[block])
            members = [h for h in range(6) if mask & (1 << h)]
            selected[row] = np.mean(offers[row, members])
        out["probabilities"][i] = selected
    return out


def pitch_identity_candidates(base_actions, native_actions, harmonic_actions,
                              post, context, frozen_k):
    """Five explicit action proposals; CHANGE_PITCH is not Sustain or Release.

    Channels: KEEP, ADD, REMOVE, CHANGE_PITCH, REATTACK.
    True CHANGE_PITCH requires a destination in the event decoder; it is not
    silently mapped to a NoteOn.  Sources are required, not simulated.
    """
    base = np.asarray(base_actions, np.float32)
    native = np.asarray(native_actions, np.float32)
    harmonic = np.asarray(harmonic_actions, np.float32)
    t = len(base)
    if base.shape != (t, 49, 5) or native.shape != base.shape \
       or harmonic.shape != base.shape or post.shape != (t, 49, 3) \
       or context.shape != (t, 49, 16) or frozen_k.shape != (t, 7):
        raise ValueError("Identity requires five independent event channels")
    for x in (base, native, harmonic):
        if not np.isfinite(x).all() or np.any(x < 0) \
           or not np.allclose(x.sum(-1), 1., atol=1e-4):
            raise ValueError("Each event head must be a class distribution")
    previous = np.concatenate((base[:1], base[:-1]), axis=0)
    novelty = np.clip(context[:, :, 4], 0, 1)
    k_active = (1 - frozen_k[:, 0])[:, None]
    h3 = base.copy()
    h3[:, :, 1] += .20 * post[:, :, 0] * novelty
    h3[:, :, 4] += .15 * post[:, :, 0] * novelty
    h4 = base.copy()
    h4[:, :, 1] += .15 * post[:, :, 0] * k_active
    h4[:, :, 2] += .15 * post[:, :, 2] * (1 - k_active)
    def norm(x):
        return x / x.sum(axis=-1, keepdims=True).clip(min=1e-8)
    heads = np.stack((base, native, previous, norm(h3),
                      norm(h4), harmonic), axis=-1).astype(np.float32)
    return base, heads


def exact_k_candidates(frozen_k, independent_k_heads):
    """Keep official S708 as H0; other five K sources must be supplied.

    Independent sources are 5 independently computed [B,7] distributions.
    No repetition of H0 masquerading as independent evidence.
    """
    k = np.asarray(frozen_k, np.float32)
    alt = np.asarray(independent_k_heads, np.float32)
    if k.ndim != 2 or k.shape[1] != 7 \
       or alt.shape != (len(k), 7, 5):
        raise ValueError("Exact-K requires five independently sourced K heads")
    heads = np.concatenate((k[..., None], alt), axis=-1)
    if not np.isfinite(heads).all() or np.any(heads < 0) \
       or not np.allclose(heads.sum(axis=1), 1, atol=1e-4):
        raise ValueError("K0..K6 distributions must be normalized")
    return k, heads


def apply_domain(target, base, heads, eligible, masks):
    """A fragment's independent decision is never copied to another target."""
    if target not in TARGETS:
        raise ValueError("Unsupported target")
    if target == "offset":
        raise ValueError("Offset must use S707 per-instance apply_offset_masks")
    fusion = "max" if target in ("onset", "sustain", "release") else "mean"
    normalization = "last" if target in ("pitch_identity", "exact_k") else None
    return select_offers(base, heads, eligible, masks, fusion=fusion,
                         normalization=normalization)


@dataclass(frozen=True)
class PromotionGate:
    """All six complete-MIDI CAL/HELD/DEV metrics; no masked regression."""
    original: dict
    candidate: dict
    lost_correct_offsets: int
    lost_long_onsets: int
    evaluated_complete_decoder: bool
    signed_fit_checkpoint: bool
    fit_composition_oof_audit: bool

    def safe(self):
        if not (self.evaluated_complete_decoder and self.signed_fit_checkpoint
                and self.fit_composition_oof_audit):
            return False
        if any(k not in self.original or k not in self.candidate for k in METRICS):
            raise ValueError("All six final MIDI metrics are mandatory")
        return (self.lost_correct_offsets == 0 and
                self.lost_long_onsets == 0 and
                all(self.candidate[k] >= self.original[k] - 1e-12
                    for k in METRICS))

    def above_80(self):
        return self.safe() and all(self.candidate[k] > .80 for k in METRICS)
