"""S736 experimental causal 4-hop Onset router; 32 H0-inclusive masks.

One mask is inferred per decision block, shared by every pitch of that block.
The 49 notes retain separate Onset posteriors. K is frozen evidence, not
a hard ceiling. This module never changes Sustain, Release, or K.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from causal_note.strict_live import HOP, SR

MASKS = tuple(range(1, 64, 2))
BLOCK_HOPS = 4
BLOCK_SAMPLES = BLOCK_HOPS * HOP
BLOCK_MS = 1000.0 * BLOCK_SAMPLES / SR
FEATURE_DIM = 37
assert len(MASKS) == 32 and len(set(MASKS)) == 32

def block_features(heads, context, eligible, k_probs):
    """Causal one-vector-per-block features; no annotations or future frames.

    heads: [T,49,6]; context: [T,49,16] from S733;
    eligible: [T,49]; frozen K0..K6: [T,7].
    """
    t = len(heads)
    if (heads.shape != (t, 49, 6)
            or context.shape != (t, 49, 16)
            or eligible.shape != (t, 49)
            or k_probs.shape != (t, 7)):
        raise ValueError("S736 block inputs violate frozen S708 ABI")
    for arr in (heads, context, k_probs):
        if not np.isfinite(arr).all():
            raise ValueError("Nonfinite audio features")
    if not np.allclose(k_probs.sum(1), 1, atol=1e-3):
        raise ValueError("Frozen Exact-K posterior must sum to one")
    live = np.where(eligible[:, :, None], heads, 0.)
    number = np.maximum(eligible.sum(axis=1, keepdims=True), 1)
    peak = live.max(axis=1)
    mean = live.sum(axis=1) / number
    strong = (live >= .45).sum(axis=1) / 49.
    # Only current and preceding *decision* block; no future leakage.
    previous = np.concatenate((peak[:1], peak[:-1]), axis=0)
    rise = np.maximum(peak - previous, 0.)
    statistics = np.stack((
        eligible.mean(axis=1),
        context[:, :, 4].max(axis=1),   # novelty
        context[:, :, 4].mean(axis=1),
        context[:, :, 8].max(axis=1),   # onset rise
        context[:, :, 15].mean(axis=1), # emitted active note ratio
        context[:, :, 1].max(axis=1),   # sustain evidence
    ), axis=1)
    result = np.concatenate((peak, mean, strong, rise, k_probs,
                             statistics), axis=1).astype(np.float32)
    if result.shape != (t, FEATURE_DIM):
        raise AssertionError("Incorrect S736 feature dimension")
    return result

def all_mask_offers(post, heads, eligible):
    """For each block, 32 potential 49-pitch Onset arrays.

    Includes H0 in EVERY mask. Uses the unchanged S735 OR/max fusion.
    Labels are never used. This is an inference proposal, not an oracle.
    """
    t = len(post)
    if (post.shape != (t, 49, 3)
            or heads.shape != (t, 49, 6)
            or eligible.shape != (t, 49)):
        raise ValueError("S736 proposal shape mismatch")
    base = post[:, :, 0]
    result = np.empty((t, len(MASKS), 49), np.float32)
    for ix, mask in enumerate(MASKS):
        selected = [j for j in range(6) if mask & (1 << j)]
        offer = heads[:, :, selected].max(axis=2)
        result[:, ix, :] = np.where(eligible, np.maximum(base, offer), base)
    return result

def apply_fragment_masks(post, heads, eligible, masks):
    """One selected mask for all pitches of EACH block (not per note/song)."""
    t = len(post)
    if masks.shape != (t,) or any(int(x) not in MASKS for x in masks):
        raise ValueError("Expected one legal H0-inclusive mask per block")
    offers = all_mask_offers(post, heads, eligible)
    indices = np.asarray(masks, dtype=np.int64) // 2
    selected = offers[np.arange(t), indices]
    result = post.copy()
    result[:, :, 0] = selected
    if not np.array_equal(result[:, :, 1:], post[:, :, 1:]):
        raise AssertionError("S736 modified frozen Sustain/Release")
    return result

class FragmentMaskRouter(nn.Module):
    """One neural categorical decision across all 32 masks per 4-hop block."""

    def __init__(self, n_features=FEATURE_DIM, width=64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(n_features, width), nn.LayerNorm(width), nn.GELU(),
            nn.Linear(width, width), nn.GELU(),
        )
        self.mask_logits = nn.Linear(width, len(MASKS))

    def forward(self, features):
        if features.ndim != 2 or features.shape[1] != FEATURE_DIM:
            raise ValueError("One feature vector per decision block required")
        return self.mask_logits(self.features(features))

def choose_masks(model, features, mean, std, min_confidence=0.):
    """Frozen inference only: FIT-trained weights, no reference labels."""
    if not 0 <= min_confidence < 1:
        raise ValueError("confidence must be in [0,1)")
    x = torch.from_numpy(((features - mean) / std).astype(np.float32))
    model.eval()
    with torch.inference_mode():
        distribution = torch.softmax(model(x), axis=-1).cpu().numpy()
    winners = distribution.argmax(axis=1)
    masks = np.asarray(MASKS, dtype=np.uint8)[winners]
    masks[distribution.max(axis=1) < min_confidence] = 1
    return masks, distribution.max(axis=1)

def validate_decision_grid(ends):
    ends = np.asarray(ends, dtype=np.int64)
    if ends.ndim != 1 or not len(ends):
        raise ValueError("Missing S708 decision grid")
    if ends[0] != BLOCK_SAMPLES or not np.all(np.diff(ends) == BLOCK_SAMPLES):
        raise ValueError("S736 requires contiguous four-hop S708 blocks")
