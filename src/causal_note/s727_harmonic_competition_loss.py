"""S727 FIT-only hard-harmonic negative training for independent octave onsets.

Models can mistake E2's 2nd harmonic for an independently plucked E3.
When a lower pitch is annotated as sustaining, a higher pitch at +12
or +19 semitones becomes a *hard negative* ONLY if no onset of the
higher pitch actually occurs. TRUE independent higher note attacks
remain positive and are never suppressed.

Ground-truth y is used **only** for FIT supervised losses; no GT active
state, true-K, note labels or future frame is an inference feature.
This is a controlled loss ablation of the S725 320D acoustic memory.
"""
from __future__ import annotations
import torch
from torch.nn import functional as F


def harmonic_parent_mask(y):
    """FIT-only true sustain of lower harmonic-related parent pitches."""
    if y.ndim!=4 or y.shape[-2:]!=(49,3):
        raise ValueError("S727 49x3 supervised onset/offset ABI expected")
    state=y[...,1].float()
    parents=torch.zeros_like(state)
    # Exactly octave (second harmonic) and +19 semitones (third harmonic).
    parents[...,12:]=state[...,:-12]
    parents[...,19:]=torch.maximum(parents[...,19:],state[...,:-19])
    return parents.clamp(0,1)


def onset_competition_weights(y,strength):
    """Harder harmonic negatives, preserve positives (including overlaps).

    strength=1 is exactly old S706 supervised loss. Positive multipliers
    only change at true new independent harmonically-related note starts
    (e.g. E2 already ringing while a real E3 is plucked).
    """
    if not 1<=strength<=4:
        raise ValueError("S727 predeclared penalty must be within [1,4]")
    yy=y.float()
    parent=harmonic_parent_mask(yy)
    neg=parent*(1-yy[...,0])
    pos=parent*yy[...,0]
    factors=torch.ones_like(yy)
    factors[...,0]+=(strength-1)*neg+(.10*(strength-1))*pos
    return factors


def competition_loss(logits,k_logits,y,k,mask,*,strength):
    """Original native BCE+K+consistency, with optional parent weights."""
    yy=y.float();k=k.long()
    loss=F.binary_cross_entropy_with_logits(logits,yy,
                       pos_weight=logits.new_tensor((10.,2.,8.)),
                       reduction="none")
    weights=onset_competition_weights(yy,strength)
    pitch=(loss*weights).mean((-1,-2))
    count=F.cross_entropy(k_logits.flatten(0,1),k.flatten(),
                          reduction="none").reshape_as(k)
    consistency=(logits[:,:,:,1].sigmoid().sum(-1)-k.float()).square()
    return ((pitch+.25*count+.008*consistency)*mask).sum()/mask.sum()
