"""127 full-verdict groups, with receiver-independent local outcome audits.

Seven candidates: frozen action, five existing specialists, and their mean.
The seventh candidate reuses specialists; it is not an independent expert.
No constituent is gated before its complete groups have been evaluated.
"""
from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from scripts.v273_selector_contract import require, id_digest

GROUP_MASKS = np.array([[int(bool(g & (1 << j))) for j in range(7)]
                        for g in range(1, 128)], np.float32)
GROUP_SIZES = GROUP_MASKS.sum(1)
CANDIDATES = ("frozen_action", "spectral", "lifecycle", "harmonic_full",
              "fundamentals", "sources", "five_specialist_mean")
NEIGHBORS = 64
AUDIT_STRENGTH = 12.
DIRECT_DIM = 55
AUDIT_DIM = 9


def seven_candidates(p):
    p = np.asarray(p, np.float32)
    require(p.ndim == 3 and p.shape[1:] == (6, 5), "six existing producers required")
    out = np.concatenate([p, p[:, 1:].mean(1, keepdims=True)], axis=1)
    require(np.isfinite(out).all() and (out >= 0).all() and np.allclose(out.sum(2), 1, atol=1e-6), "candidate votes")
    return out


def joint_groups(votes, baseline):
    """Deterministic joint proposals define observable M/N/neutral labels.

Group probability vectors are arithmetic means, with strict KEEP on a tie.
H0 is an optional action vote, not an assertion of perfect correctness.
The nonlinear neural critic is trained separately on these joint outcomes.
"""
    v = np.asarray(votes, np.float32); b = np.asarray(baseline, int)
    require(v.shape == (len(b), 7, 5) and np.isin(b, [2, 3, 4]).all(), "candidate shape")
    require(np.isfinite(v).all() and (v >= 0).all() and np.allclose(v.sum(2), 1, atol=1e-6), "invalid candidate distribution")
    mean = (np.einsum("sh,bhk->bsk", GROUP_MASKS, v)/GROUP_SIZES[None, :, None]).astype(np.float32)
    second = np.einsum("sh,bhk->bsk", GROUP_MASKS, v*v)/GROUP_SIZES[None, :, None]
    std = np.sqrt(np.maximum(second-mean*mean, 0))
    row = np.arange(len(b))[:, None]; group = np.arange(127)[None]
    best = mean.argmax(2)+2
    proposal = np.where(mean[row, group, best-2] > mean[row, group, b[:, None]-2], best, b[:, None])
    identity = np.broadcast_to(GROUP_MASKS[None], (len(b), 127, 7))
    members = (v[:, None]*GROUP_MASKS[None, :, :, None]).reshape(len(b), 127, 35)
    size = np.broadcast_to(GROUP_SIZES[None, :, None]/7, (len(b), 127, 1))
    entropy = -(mean*np.log(np.maximum(mean, 1e-12))).sum(2, keepdims=True)/np.log(5)
    ordered = np.sort(mean, axis=2)
    margin = ordered[:, :, -1:]-ordered[:, :, -2:-1]
    features = np.concatenate([identity, members, mean, std, size, entropy, margin], axis=2).astype(np.float32)
    require(features.shape == (len(b), 127, DIRECT_DIM), "group descriptor size")
    return proposal.astype(np.int32), features, mean


def outcome_labels(proposal, truth, baseline):
    """0=correction, 1=regression, 2=neutral, relative to frozen baseline."""
    p = np.asarray(proposal); y = np.asarray(truth)[:, None]; b = np.asarray(baseline)[:, None]
    change = p != b
    return np.where(change & (p == y), 0, np.where(change & (y == b), 1, 2)).astype(np.int32)


def local_audits(ref_raw, receiver_raw, ref_proposal, ref_y, ref_b, receiver_proposal,
                 receiver_b, ref_ids, receiver_ids):
    """Local similarity is fit only on permitted reference events.

Neighbors share the original K. Each group's outcome counts further require
the same joint destination as the receiver group. The global prior uses the
same group, original class and joint destination, never a sum of head risks.
"""
    require(not set(ref_ids) & set(receiver_ids), "receiver in local audit references")
    order = np.argsort(ref_ids, kind="stable")
    ref_raw, ref_proposal, ref_y, ref_b = (np.asarray(a)[order] for a in
                                         (ref_raw, ref_proposal, ref_y, ref_b))
    scaler = StandardScaler().fit(ref_raw)
    a = np.clip(scaler.transform(ref_raw), -6, 6)
    b = np.clip(scaler.transform(receiver_raw), -6, 6)
    labels = outcome_labels(ref_proposal, ref_y, ref_b)
    output = np.zeros((len(receiver_b), 127, AUDIT_DIM), np.float32)
    for source in (2, 3, 4):
        fit = np.flatnonzero(ref_b == source); receive = np.flatnonzero(receiver_b == source)
        if not len(receive):
            continue
        require(len(fit) > 0, "no independent same-source audit reference")
        count = min(NEIGHBORS, len(fit))
        nn = NearestNeighbors(n_neighbors=count, algorithm="brute", n_jobs=1).fit(a[fit])
        distance, neighbor = nn.kneighbors(b[receive])
        neighbor = fit[neighbor]
        refp = ref_proposal[fit]; refl = labels[fit]
        npred, nlabel = ref_proposal[neighbor], labels[neighbor]
        query = receiver_proposal[receive]
        matching = npred == query[:, None]
        local_n = matching.sum(1)
        local_counts = np.stack([np.sum(matching & (nlabel == c), axis=1) for c in range(3)], 2)
        for target in range(2, 7):
            same_target = refp == target
            used = same_target.sum(0)
            global_counts = np.stack([np.sum(same_target & (refl == c), axis=0) for c in range(3)], 1)
            prior = (global_counts+1)/(used[:, None]+3)
            loc = (local_counts+AUDIT_STRENGTH*prior[None])/(local_n[:, :, None]+AUDIT_STRENGTH)
            at = query == target
            block = np.concatenate([
                np.broadcast_to(prior[None], (len(receive), 127, 3)), loc,
                np.broadcast_to((used/len(fit))[None, :, None], (len(receive), 127, 1)),
                (local_n/count)[:, :, None],
                np.broadcast_to((distance.mean(1)/np.sqrt(a.shape[1]))[:, None, None], (len(receive), 127, 1)),
            ], axis=2)
            chunk = output[receive]; chunk[at] = block[at]; output[receive] = chunk
    require(np.isfinite(output).all(), "invalid local audits")
    require(np.allclose(output[..., :3].sum(2), 1) and np.allclose(output[..., 3:6].sum(2), 1), "audit outcome normalization")
    return output


def assemble_group_outer(pool, raw, outer):
    folds, y, base, ids = pool.fold, pool.y, pool.b, pool.ids
    train = np.flatnonzero(folds != outer); held = np.flatnonzero(folds == outer)
    tp, records = pool.crossfit(train, {outer})
    hp = pool.predict(set(folds[train]), held, {outer})
    tv, hv = seven_candidates(tp), seven_candidates(hp)
    tpred, tdesc, _ = joint_groups(tv, base[train])
    hpred, hdesc, _ = joint_groups(hv, base[held])
    ta = np.zeros((len(train), 127, AUDIT_DIM), np.float32)
    manifests = []
    for receiver in sorted(set(folds[train])):
        local = np.flatnonzero(folds[train] == receiver); at = train[local]
        ref = train[folds[train] != receiver]
        rp, producers = pool.crossfit(ref, {outer, int(receiver)})
        rpred, _, _ = joint_groups(seven_candidates(rp), base[ref])
        ta[local] = local_audits(raw[ref], raw[at], rpred, y[ref], base[ref],
            tpred[local], base[at], ids[ref], ids[at])
        manifests.append(dict(outer_fold=int(outer), receiver_fold=int(receiver),
            reference_folds=sorted(map(int, set(folds[ref]))),
            reference_ids_sha256=id_digest(ids[ref]), reference_rows=len(ref),
            receiver_ids_sha256=id_digest(ids[at]), forbidden_id_overlap=0,
            reference_expert_producers=producers))
    ha = local_audits(raw[train], raw[held], tpred, y[train], base[train],
                      hpred, base[held], ids[train], ids[held])
    scaler = StandardScaler().fit(raw[train])

    def inputs(at, votes, proposals, direct, audit):
        return dict(group_features=np.concatenate([direct, audit], axis=2),
            member_votes=votes, proposal=proposals,
            context=np.clip(scaler.transform(raw[at]), -6, 6).astype(np.float32),
            baseline=np.eye(7, dtype=np.float32)[base[at]])

    manifest = dict(outer_fold=int(outer), train_ids_sha256=id_digest(ids[train]),
        test_ids_sha256=id_digest(ids[held]), forbidden_id_overlap=0,
        train_oof_producers=records, inner_audits=manifests,
        neighbor_features_fit="reference-only scaler; nearest same-source events; no receiver labels",
        raw_context_scaler="unlabeled outer-train preprocessing")
    return inputs(train, tv, tpred, tdesc, ta), inputs(held, hv, hpred, hdesc, ha), train, held, manifest


def choose_groups(gain, proposal, baseline, singletons=False):
    """Every group competes directly with KEEP=0. No constituent veto."""
    gain = np.asarray(gain); proposal = np.asarray(proposal); b = np.asarray(baseline)
    require(gain.shape == proposal.shape == (len(b), 127) and np.isfinite(gain).all(), "group decoding inputs")
    allowed = proposal != b[:, None]
    if singletons:
        allowed &= GROUP_SIZES[None] == 1
    score = np.where(allowed, gain, -np.inf)
    best = score.argmax(1); row = np.arange(len(b))
    accepted = score[row, best] > 0
    return np.where(accepted, proposal[row, best], b), np.where(accepted, best+1, 0)
