"""S60–S64 experimental transition-risk head contracts (NOT PROMOTED).

These helpers define six independent selectors over proposals from another
model. The fitted nonlinear critic and the consensus policy must be evaluated
with fold-excluded targets. This module does not alter the native predictor.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

TRANSITIONS = {
    "H3_under": ((3, 2),),
    "H3_over": ((2, 3),),
    "H4_under": ((4, 3),),
    "H4_over": ((3, 4),),
    "H01": ((0, 1), (1, 0)),
}


def head_masks(baseline, proposal):
    """Disjoint K-specific active regions, defined without true K."""
    b = np.asarray(baseline, dtype=int)
    p = np.asarray(proposal, dtype=int)
    if b.shape != p.shape or b.ndim != 1:
        raise ValueError("baseline/proposal shape mismatch")
    if not (np.isin(b, range(7)).all() and np.isin(p, range(7)).all()):
        raise ValueError("only native K0..K6 permitted")
    changing = b != p
    out = {}
    claimed = np.zeros(len(b), bool)
    for name, edges in TRANSITIONS.items():
        mask = changing & np.logical_or.reduce([(b == a) & (p == c) for a, c in edges])
        if np.any(mask & claimed):
            raise ValueError("selector overlap")
        out[name] = mask
        claimed |= mask
    out["H_other"] = changing & ~claimed
    assert np.array_equal(
        np.sum(np.stack(list(out.values())), axis=0), changing.astype(int))
    return out


def audited_training_targets(truth, baseline, proposal):
    """Regression=+1, correction=-1, wrong-to-wrong/unchanged=0.

    NEVER evaluate this on held-out examples to choose their actions.
    """
    y, b, p = [np.asarray(a) for a in (truth, baseline, proposal)]
    if y.shape != b.shape or y.shape != p.shape:
        raise ValueError("training label shape mismatch")
    changed = b != p
    return ((changed & (b == y)).astype(float)
            - (changed & (p == y)).astype(float))


def new_nonlinear_head():
    """Unfitted, regularized S62 critic; must be cross-fitted before use."""
    return HistGradientBoostingRegressor(
        max_iter=55, max_leaf_nodes=7, min_samples_leaf=35,
        l2_regularization=20., max_bins=80, learning_rate=.05,
        early_stopping=False, random_state=27402)


def countervote_features(policy_predictions, baseline, proposal):
    """Base and candidate evidence from 36 CORRELATED PR16 policies."""
    policies = np.asarray(policy_predictions)
    b, p = np.asarray(baseline), np.asarray(proposal)
    if policies.shape != (len(b), 36) or len(p) != len(b):
        raise ValueError("36 correlated policies and matched actions required")
    if not np.isin(policies, range(7)).all():
        raise ValueError("invalid policy output class")
    counts = np.stack([(policies == k).sum(axis=1) for k in range(7)], axis=1)
    return {
        "baseline_count": counts[np.arange(len(b)), b],
        "baseline_advantage": counts[np.arange(len(b)), b]
                              - counts[np.arange(len(b)), p],
        "candidate_count": counts[np.arange(len(b)), p],
        "policy_count": 36,
    }


def apply_veto(baseline, proposal, veto):
    """Refuse a proposal by returning to the FROZEN baseline (no new K)."""
    b, p, v = [np.asarray(a) for a in (baseline, proposal, veto)]
    if not (b.shape == p.shape == v.shape and v.dtype == bool):
        raise ValueError("veto must be a matched boolean mask")
    if np.any(v & (b == p)):
        raise ValueError("cannot veto an unchanged event")
    return np.where(v, b, p)
