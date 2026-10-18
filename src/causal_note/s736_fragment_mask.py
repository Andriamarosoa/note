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

def subset_descriptors(post, heads, eligible, context):
    """Audio-only descriptors per block/subset [B,32,12]; no labels.

    6 signal descriptors + 6 bits defining which original heads interact.
    FIT OOF correction/regression statistics are a SEPARATE required input.
    """
    t = len(post)
    if context.shape != (t, 49, 16):
        raise ValueError("S736 frozen acoustic context shape mismatch")
    offers = all_mask_offers(post, heads, eligible)
    delta = np.maximum(offers - post[:, None, :, 0], 0.)
    novelty = np.clip(context[:, :, 4], 0., 1.)
    descriptors = np.zeros((t, 32, 12), np.float32)
    descriptors[:, :, 0] = np.mean(delta, axis=-1)
    descriptors[:, :, 1] = np.max(delta, axis=-1)
    descriptors[:, :, 2] = np.mean(delta > .02, axis=-1)
    descriptors[:, :, 3] = np.mean(delta * novelty[:, None, :], axis=-1)
    descriptors[:, :, 4] = np.mean(offers >= .45, axis=-1)
    descriptors[:, :, 5] = np.mean(
        np.logical_and(offers >= .45, eligible[:, None, :]), axis=-1)
    for i, mask in enumerate(MASKS):
        descriptors[:, i, 6:] = np.asarray(
            [(mask >> j) & 1 for j in range(6)], np.float32)
    return descriptors


def subset_class_compatibility(head_available, head_k_mask):
    """Legal subsets per *hypothetical* K; never use ground-truth K.

    A subset is usable for K only if ALL its included heads may legally
    propose evidence for K. H0 is mandatory in all 32 subsets.
    """
    if head_available.ndim != 2 or head_available.shape[1] != 6:
        raise ValueError("Expected [B,6] head availability")
    b = len(head_available)
    if head_k_mask.shape != (b, 6, 7):
        raise ValueError("Expected [B,6,7] structural head/K mask")
    legal = head_k_mask.bool() & head_available.bool()[:, :, None]
    selection = torch.tensor(
        [[bool(mask & (1 << h)) for h in range(6)] for mask in MASKS],
        dtype=torch.bool, device=head_available.device)
    compatible = torch.all(~selection[None, :, :, None] |
                           legal[:, None, :, :], dim=2)
    if not bool(compatible[:, 0, :].all()):
        raise ValueError("H0 fallback required for all hypothetical K0-K6")
    return compatible


class FragmentMaskRouter(nn.Module):
    """Exact-K ClassConditionalAuditSelector architecture, ported to fragments.

    Faithful structural port of
    scripts.learn_v273_neural_head_selector.ClassConditionalAuditSelector:
      head projection, context projection, 3-head self-attention, fuse;
      independent softmax OVER 32 proposals FOR EACH of K0..K6;
      class-specific pooled states/evidence, class decoder K0..K6;
      independent learned KEEP head and risk-per-proposal head.

    Onset-specific *adapters*, not replacements of that central network:
      32 H0-inclusive proposals instead of seven-class expert heads,
      12 measured acoustic subset descriptors added to head projection,
      33-way fragment decision adapted from class logits/conditional weights.
    Inputs and training weights are not shared with the historical K network.
    """

    def __init__(self, n_features=FEATURE_DIM, width=48):
        super().__init__()
        if n_features != FEATURE_DIM or width % 3:
            raise ValueError("Original Exact-K requires 37 context features and 3 attention heads")
        self.context_dim = n_features
        # Source: Dense(8+21+4 -> hidden). Twelve onset subset descriptors
        # are an explicit input adapter; all subsequent Exact-K stages match.
        self.head_proj = nn.Sequential(nn.Linear(8 + 21 + 4 + 12, width), nn.GELU())
        self.context_proj = nn.Sequential(nn.Linear(n_features + 7, width), nn.GELU())
        self.attention = nn.MultiheadAttention(width, num_heads=3, batch_first=True, dropout=0.)
        self.fuse = nn.Sequential(nn.Linear(width, width), nn.GELU())
        self.k_gate = nn.Linear(width, 7)  # Original class_gate.
        self.class_hidden = nn.Sequential(nn.Linear(2 * width + 8, width), nn.GELU())
        self.class_out = nn.Linear(width, 1)
        self.keep_gate = nn.Linear(width, 1)
        self.keep_head = nn.Sequential(
            nn.Linear(2 * width + n_features + 8, width), nn.GELU(),
            nn.Linear(width, 1))
        self.risk = nn.Sequential(nn.Linear(width, 2), nn.Sigmoid())

    def forward(self, features, descriptors, audit, k_probs, compatible):
        b = len(features)
        if (features.shape != (b, FEATURE_DIM)
                or descriptors.shape != (b, 32, 12)
                or audit.shape != (b, 32, 7, 3)
                or k_probs.shape != (b, 7)
                or compatible.shape != (b, 32, 7)):
            raise ValueError("Exact-K port: per-fragment 32-option input mismatch")
        if not all(bool(torch.isfinite(x).all())
                   for x in (features, descriptors, audit, k_probs)):
            raise ValueError("Nonfinite learned selector input")
        if bool((k_probs < 0).any()) or not bool(torch.allclose(
                k_probs.sum(-1), torch.ones_like(k_probs[:, 0]), atol=1e-3)):
            raise ValueError("Exact-K frozen soft posteriors required")
        class_mask = compatible.bool()
        available = class_mask.any(dim=-1)
        if not bool(class_mask[:, 0, :].all()):
            raise ValueError("H0-only option must support all hypothetical K")
        if not bool(class_mask.any(dim=1).all()):
            raise ValueError("Each K class must retain a legal proposal")

        # Exact-K's original head input interface is [K0..K6+KEEP,
        # OOF audit(21), head type(4)]. Option-specific onset measurements
        # are appended as an adapter; no truth K or future audio is supplied.
        k_logits = torch.log(k_probs.clamp_min(1e-7))
        keep_evidence = -descriptors[:, :, 1:2].clamp_min(0.)
        head_logits = torch.cat((
            k_logits[:, None, :].expand(-1, 32, -1), keep_evidence), dim=-1)
        head_post = torch.softmax(head_logits, dim=-1)
        kind = torch.zeros(b, 32, 4, device=features.device, dtype=features.dtype)
        kind[:, 0, 0] = 1.  # H0 baseline
        kind[:, 1:, 1] = 1.  # Other combinations: acoustic predictors
        head_input = torch.cat((head_post, audit.flatten(start_dim=2),
                                kind, descriptors), dim=-1)
        heads = self.head_proj(head_input)
        ctx = self.context_proj(torch.cat((features, k_probs), dim=-1))
        # Source attention: each proposed head interacts with all other
        # available heads; unavailable ones receive zero probability.
        attended, _ = self.attention(
            heads, heads, heads, key_padding_mask=~available,
            need_weights=False)
        state = self.fuse(heads + attended + ctx[:, None, :])

        # The original Exact-K class-conditional gate is preserved, not a
        # single mixture reused across K0..K6.
        by_k_logits = self.k_gate(state).masked_fill(~class_mask, -1e9)
        by_k = torch.softmax(by_k_logits, dim=1) * class_mask
        by_k = by_k / by_k.sum(dim=1, keepdim=True).clamp_min(1e-9)
        class_states = torch.einsum("bhk,bhd->bkd", by_k, state)
        evidence = (by_k * head_post[:, :, :7]).sum(dim=1)
        identities = torch.eye(7, device=features.device,
                               dtype=features.dtype).unsqueeze(0).expand(b, -1, -1)
        class_features = torch.cat((
            class_states, ctx[:, None, :].expand(-1, 7, -1),
            evidence[:, :, None], identities), dim=-1)
        class_logits = self.class_out(self.class_hidden(class_features)).squeeze(-1)

        # Exact-K has an independent KEEP path, not a special combination.
        keep_scores = self.keep_gate(state).squeeze(-1).masked_fill(~available, -1e9)
        keep_weights = torch.softmax(keep_scores, dim=1)
        keep_state = torch.einsum("bh,bhd->bd", keep_weights, state)
        keep_prob = torch.einsum("bh,bh->b", keep_weights, head_post[:, :, 7])
        keep_logit = self.keep_head(torch.cat((
            keep_state, ctx, features, k_probs, keep_prob[:, None]), dim=-1)).squeeze(-1)

        # Adapter from original K0..K6+KEEP logits to 32 actual onset
        # combination actions + KEEP. Each candidate is conditioned on all
        # seven classes and can vary independently for every audio fragment.
        p_fix, p_regress = self.risk(state).clamp(1e-5, 1 - 1e-5).unbind(-1)
        class_prob = torch.softmax(class_logits, dim=-1)
        subset_weights = torch.einsum("bhk,bk->bh", by_k, class_prob)
        action_by_mask = torch.logsumexp(
            torch.log_softmax(class_logits, dim=-1)[:, None, :] +
            torch.log(by_k.clamp_min(1e-12)), dim=-1)
        action_by_mask = action_by_mask + torch.log(p_fix) - torch.log(p_regress)
        action_by_mask = action_by_mask.masked_fill(~available, -1e9)
        action_logits = torch.cat((keep_logit[:, None], action_by_mask), dim=1)
        return dict(
            action_logits=action_logits, class_logits=torch.cat(
                (class_logits, keep_logit[:, None]), dim=-1),
            class_probability=class_prob, class_selection_weights=by_k,
            subset_weights=subset_weights, keep_selection_weights=keep_weights,
            p_correction=p_fix, p_regression=p_regress,
            keep_logit=keep_logit,
        )


def choose_masks(model, features, descriptors, audit, k_probs,
                 compatible, mean, std, min_confidence=0.):
    """Learned 33-way action: KEEP, or one of all 32 possible combos.

    A low-confidence decision is KEEP/H0 rather than hallucinated Onsets.
    """
    if not 0 <= min_confidence < 1:
        raise ValueError("confidence must be in [0,1)")
    x = torch.from_numpy(((features - mean) / std).astype(np.float32))
    d = torch.as_tensor(descriptors, dtype=torch.float32)
    a = torch.as_tensor(audit, dtype=torch.float32)
    k = torch.as_tensor(k_probs, dtype=torch.float32)
    c = torch.as_tensor(compatible, dtype=torch.bool)
    model.eval()
    with torch.inference_mode():
        predictions = model(x, d, a, k, c)
        distribution = torch.softmax(
            predictions["action_logits"], dim=1).cpu().numpy()
    winners = distribution.argmax(axis=1)
    masks = np.ones(len(winners), dtype=np.uint8)
    proposed = winners > 0
    masks[proposed] = np.asarray(
        MASKS, dtype=np.uint8)[winners[proposed] - 1]
    confidence = distribution.max(axis=1)
    masks[confidence < min_confidence] = 1
    return masks, confidence, predictions


def validate_decision_grid(ends):
    ends = np.asarray(ends, dtype=np.int64)
    if ends.ndim != 1 or not len(ends):
        raise ValueError("Missing S708 decision grid")
    if ends[0] != BLOCK_SAMPLES or not np.all(np.diff(ends) == BLOCK_SAMPLES):
        raise ValueError("S736 requires contiguous four-hop S708 blocks")
