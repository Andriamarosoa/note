"""S741: five *causal acoustic* seven-class candidate sources for Exact-K.

These are distinct measurements (native K, previous K, acoustic Onset
density, Sustain density, previously emitted active-note count), not
five copies of a frozen reference.  They are deliberately NOT asserted
to be independently pretrained models or calibrated probabilities.
Any change to S708 requires a fitted selector and all-six-score guard.
"""
from __future__ import annotations
import numpy as np
from causal_note.s740_multitarget_dynamic import exact_k_candidates
from scripts.s733_train_onset_only import active_count_from_events


def continuous_count_distribution(expected_count, *, sigma=0.85):
    """Soft class K0..K6 evidence, without oracle count or hard clipping."""
    val = np.asarray(expected_count, np.float32)
    if val.ndim != 1 or not np.isfinite(val).all() or sigma <= 0:
        raise ValueError("Expected finite one-dimensional causal count")
    classes = np.arange(7, dtype=np.float32)
    logits = -(val[:, None] - classes[None, :])**2 / (2*sigma**2)
    logits -= logits.max(axis=-1, keepdims=True)
    raw = np.exp(logits).astype(np.float32)
    return raw / raw.sum(-1, keepdims=True)


def frozen_acoustic_exactk_heads(k, native_k, post, events, decision_ends):
    """Five source distributions for every *current* four-hop block.

    H0 is the untouched official S708 K.
    H1 is the frozen native acoustic Exact-K head.
    H2 is the previous four-hop official prediction (no lookahead).
    H3 and H4 are soft independent-pitch onset/sustain energy-count cues.
    H5 uses only MIDI notes ALREADY emitted by S708 at that decision time.
    """
    k = np.asarray(k, np.float32)
    native_k = np.asarray(native_k, np.float32)
    post = np.asarray(post, np.float32)
    ends = np.asarray(decision_ends, np.int64)
    n = len(k)
    if k.shape != (n, 7) or native_k.shape != (n, 7) \
       or post.shape != (n, 49, 3) or ends.shape != (n,) \
       or (len(ends) > 1 and np.any(np.diff(ends) <= 0)):
        raise ValueError("Frozen S708 4-hop grid mismatch")
    if not all(np.isfinite(z).all() for z in (k, native_k, post)) \
       or not np.allclose(k.sum(1), 1, atol=1e-3) \
       or not np.allclose(native_k.sum(1), 1, atol=1e-3):
        raise ValueError("Unnormalized frozen K source")
    previous = np.concatenate([k[:1], k[:-1]], axis=0)

    # Derived from the current received acoustic block; do not interpret the
    # estimated sum as a hard cardinality cap in the MIDI decoder.
    onset_mass = np.clip((post[:, :, 0] - .20) / .6, 0, 1)
    sustain_mass = np.clip((post[:, :, 1] - .30) / .5, 0, 1)
    onset_count = continuous_count_distribution(onset_mass.sum(axis=1),
                                                sigma=1.15)
    sustain_count = continuous_count_distribution(sustain_mass.sum(axis=1),
                                                  sigma=1.15)
    active_count = active_count_from_events(events, ends) * 6.
    emitted_count = continuous_count_distribution(active_count, sigma=.65)
    five = np.stack([native_k, previous, onset_count,
                     sustain_count, emitted_count], axis=-1)
    base, heads = exact_k_candidates(k, five)
    return base, heads
