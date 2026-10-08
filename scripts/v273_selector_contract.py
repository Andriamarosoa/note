"""Auditable inputs for the repaired experimental Exact-K selector.

No TensorFlow dependency. Specialist and audit producers explicitly record
and check their fit IDs. H0 encodes an available deterministic action, not
an invented estimate of its correctness.
"""
from __future__ import annotations

import hashlib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 27402
KS = np.arange(2, 7)
FOLDS = (0, 1, 2, 4)
CORRECTORS = {(2, 3): 6, (3, 2): 7, (3, 4): 8, (4, 3): 9}
MASKS = np.array([[1] + [int(bool(i & (1 << j))) for j in range(6)]
                  for i in range(64)], np.float32)
FAMILIES = ("spectral__", "birth__", "persistence__", "damping__")
EXPERTS = {
    "spectral": ("features", ("spectral__",)),
    "lifecycle": ("features", ("birth__", "persistence__", "damping__")),
    "harmonic_full": ("features", ()),
    "fundamentals": ("fundamental", ()),
    "sources": ("features", ("source__", "coherence__")),
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def id_digest(ids):
    return hashlib.sha256(np.sort(np.asarray(ids, dtype="<i8")).tobytes()).hexdigest()


def context_columns(names):
    """Keep all four declared families; never truncate alphabetically."""
    chosen = sorted(n for n in names if n.startswith(FAMILIES))
    require(all(any(n.startswith(f) for n in chosen) for f in FAMILIES),
            "missing declared acoustic family")
    return chosen


def matrices(rows):
    names = sorted(rows[0]["features"])
    context_names = context_columns(names)
    x = {}
    for key, (field, prefixes) in EXPERTS.items():
        columns = names if not prefixes else [n for n in names if n.startswith(prefixes)]
        x[key] = np.array([[r[field][n] for n in columns] for r in rows], float)
    raw = np.array([[r["features"][n] for n in context_names] for r in rows], float)
    require(all(np.isfinite(a).all() for a in (*x.values(), raw)), "invalid features")
    return x, raw, context_names


class ProducerPool:
    """Cache experts by exact training folds, with an auditable ID manifest."""

    def __init__(self, x, truth, baseline, folds, ids):
        self.x, self.y, self.b = x, np.asarray(truth), np.asarray(baseline)
        self.fold, self.ids = np.asarray(folds), np.asarray(ids)
        require(len(np.unique(self.ids)) == len(self.ids), "duplicate producer ID")
        self.models = {}
        self.manifests = {}

    def predict(self, train_folds, positions, forbidden_folds):
        key = tuple(sorted(map(int, train_folds)))
        positions = np.asarray(positions, int)
        forbidden = set(map(int, forbidden_folds))
        require(key and not set(key) & forbidden, "forbidden fold in producer training")
        fit = np.isin(self.fold, key) & np.isin(self.y, KS)
        require(not set(self.ids[fit]) & set(self.ids[positions]), "in-sample expert prediction")
        require(np.array_equal(np.unique(self.y[fit]), KS), "missing specialist class")
        if key not in self.models:
            models = []
            for name in EXPERTS:
                estimator = make_pipeline(StandardScaler(), LogisticRegression(
                    C=.1, max_iter=2500, random_state=SEED))
                estimator.fit(self.x[name][fit], self.y[fit])
                require(np.array_equal(estimator[-1].classes_, KS), "specialist class order")
                models.append(estimator)
            self.models[key] = models
            self.manifests[key] = dict(train_folds=list(key), train_rows=int(fit.sum()),
                train_global_ids=list(map(int, self.ids[fit])),
                train_id_sha256=id_digest(self.ids[fit]),
                true_k_counts=np.bincount(self.y[fit], minlength=7).tolist())
        p = np.zeros((len(positions), 6, 5), np.float32)
        # A deterministic fallback vote. Correctness is learned by the arbiter.
        p[np.arange(len(positions)), 0, self.b[positions]-2] = 1.
        for j, (name, model) in enumerate(zip(EXPERTS, self.models[key]), 1):
            q = model.predict_proba(self.x[name][positions])
            p[:, j] = np.maximum(q, 1e-10) / np.maximum(q.sum(axis=1, keepdims=True), 1e-10)
        require(np.isfinite(p).all(), "nonfinite expert predictions")
        return p

    def crossfit(self, positions, excluded):
        positions = np.asarray(positions, int)
        available = set(map(int, np.unique(self.fold[positions])))
        require(not available & set(excluded) and len(available) >= 2, "crossfit split")
        out = np.empty((len(positions), 6, 5), np.float32)
        records = []
        for receiver in sorted(available):
            local = self.fold[positions] == receiver
            fit_folds = available - {receiver}
            out[local] = self.predict(fit_folds, positions[local], set(excluded) | {receiver})
            key = tuple(sorted(fit_folds))
            records.append(dict(excluded_folds=sorted(set(excluded)),
                reference_prediction_fold=receiver, train_folds=list(key),
                train_id_sha256=self.manifests[key]["train_id_sha256"],
                forbidden_train_overlap=0))
        return out, records


def direct_options(p, base):
    """Same anchored subset votes; repair actual Cxy membership in channel 5."""
    p, base = np.asarray(p, np.float32), np.asarray(base, int)
    require(p.shape == (len(base), 6, 5) and np.isin(base, (2, 3, 4)).all(), "input shape")
    opt = np.zeros((len(base), 5, 64, 6), np.float32)
    valid = np.zeros((len(base), 5, 64), bool)
    identity = np.zeros((len(base), 5, 64, 7), np.float32)
    shares = np.array([.78, .93, .93, .93, .93, .93])
    for source in (2, 3, 4):
        at = np.flatnonzero(base == source)
        if not len(at):
            continue
        for j, target in enumerate(KS):
            if source == target:
                continue
            correction = (source, target) in CORRECTORS
            stay = np.zeros((len(at), 7), np.float64)
            offer = np.zeros_like(stay)
            stay[:, :6] = shares*p[at, :, source-2] + (1-shares)
            offer[:, :6] = shares*p[at, :, j]
            if correction:
                q = p[at, 1:, j].mean(axis=1)
                offer[:, 6] = .95*q
                stay[:, 6] = 1-.95*q
            masks = MASKS.copy()
            if not correction:
                masks[:, 6] = 0
            count = masks.sum(axis=1)
            avg_keep = stay@masks.T/count[None, :]
            avg_offer = offer@masks.T/count[None, :]
            margin = avg_offer-avg_keep
            opt[at, j, :, 0] = margin
            opt[at, j, :, 1] = avg_keep
            opt[at, j, :, 2] = avg_offer
            opt[at, j, :, 3] = margin > 0
            opt[at, j, :, 4] = count[None, :]/7.
            opt[at, j, :, 5] = masks[:, 6][None, :]
            valid[at, j, :64 if correction else 32] = True
            identity[at, j] = masks
    identity *= valid[:, :, :, None]
    return opt, valid, identity


def audit_rates(proposals, truth, source, target, prior=None):
    """Train-only proposal frequency and smoothed correct/regress/neutral rates."""
    used = proposals.sum(axis=0).astype(float)
    cor = (proposals*(truth == target)[:, None]).sum(axis=0)
    reg = (proposals*(truth == source)[:, None]).sum(axis=0)
    counts = np.stack([cor, reg, used-cor-reg], axis=1)
    prior = np.full((64, 3), 1/3) if prior is None else prior
    rates = (counts+12*prior)/(used[:, None]+12)
    return np.column_stack([np.log1p(used), rates]).astype(np.float32)


def audit_receivers(reference_options, reference_y, reference_b, receiver_b,
                    reference_groups, receiver_groups):
    global_audit = np.zeros((len(receiver_b), 5, 64, 4), np.float32)
    local_audit = np.zeros_like(global_audit)
    for source in (2, 3, 4):
        ref_source = reference_b == source
        require(ref_source.any(), "missing source in independent audit reference")
        for j, target in enumerate(KS):
            if source == target:
                continue
            proposal = reference_options[ref_source, j, :, 3] > .5
            global_desc = audit_rates(proposal, reference_y[ref_source], source, target)
            used = proposal.sum(axis=0).astype(float)
            cor = (proposal*(reference_y[ref_source] == target)[:, None]).sum(axis=0)
            reg = (proposal*(reference_y[ref_source] == source)[:, None]).sum(axis=0)
            counts = np.stack([cor, reg, used-cor-reg], axis=1)
            # Preserve the original contextual audit's two-stage smoothing.
            weak_prior = (counts+1)/(used[:, None]+3)
            context_prior = audit_rates(proposal, reference_y[ref_source], source,
                                        target, prior=weak_prior)[:, 1:]
            dest = np.flatnonzero(receiver_b == source)
            global_audit[dest, j] = global_desc
            # Context prior and local evidence use exactly the same permitted folds.
            for group in (0, 1, 2):
                ref = ref_source & (reference_groups == group)
                local = audit_rates(reference_options[ref, j, :, 3] > .5,
                    reference_y[ref], source, target, prior=context_prior)
                dest = np.flatnonzero((receiver_b == source) & (receiver_groups == group))
                local_audit[dest, j] = local
    return global_audit, local_audit


def reference_regimes(raw_reference, raw_receiver):
    scaler = StandardScaler().fit(raw_reference)
    a = np.clip(scaler.transform(raw_reference), -6, 6)
    b = np.clip(scaler.transform(raw_receiver), -6, 6)
    cluster = KMeans(n_clusters=3, n_init=10, random_state=SEED)
    return cluster.fit_predict(a), cluster.predict(b)


def fallback_metadata(p, base):
    """14 observable slots: action flags and specialist support, not 14 risks."""
    out = np.zeros((len(base), 14), np.float32)
    out[:, 0] = 1  # H0 fallback available, no correctness claim
    out[:, 1:6] = p[np.arange(len(base))[:, None], np.arange(1, 6)[None, :], (base-2)[:, None]]
    for (source, target), index in CORRECTORS.items():
        out[:, index] = (base == source)*(1-.95*p[:, 1:, target-2].mean(axis=1))
    for index, source in enumerate((2, 3, 4), 10):
        out[:, index] = base == source
    out[:, 13] = 1  # general fallback available
    return out


def assemble_outer(pool, raw, outer):
    """Double cross-fit all producers contributing audit descriptors.

    Outer test descriptors use outer-train references. A training receiver
    fold R uses references whose experts, audit scaler and KMeans all exclude
    R and the outer test fold. Raw-context normalization is ordinary label-free
    outer-train normalization, distinct from the audit-regime producers.
    """
    folds, y, b, ids = pool.fold, pool.y, pool.b, pool.ids
    tr = np.flatnonzero(folds != outer)
    val = np.flatnonzero(folds == outer)
    train_p, records = pool.crossfit(tr, {outer})
    test_p = pool.predict(set(folds[tr]), val, {outer})
    train_opt, train_mask, train_identity = direct_options(train_p, b[tr])
    test_opt, test_mask, test_identity = direct_options(test_p, b[val])
    global_train = np.zeros((len(tr), 5, 64, 4), np.float32)
    local_train = np.zeros_like(global_train)
    audit_manifests = []
    for receiver in sorted(set(folds[tr])):
        receive_local = np.flatnonzero(folds[tr] == receiver)
        receive = tr[receive_local]
        reference = tr[folds[tr] != receiver]
        ref_p, inner_records = pool.crossfit(reference, {outer, int(receiver)})
        ref_opt, _, _ = direct_options(ref_p, b[reference])
        ref_group, receive_group = reference_regimes(raw[reference], raw[receive])
        ga, la = audit_receivers(ref_opt, y[reference], b[reference], b[receive],
                                ref_group, receive_group)
        global_train[receive_local] = ga
        local_train[receive_local] = la
        require(not set(ids[receive]) & set(ids[reference]), "receiver in audit fit IDs")
        audit_manifests.append(dict(outer_fold=int(outer), receiver_fold=int(receiver),
            reference_folds=sorted(map(int, set(folds[reference]))),
            audit_and_regime_train_id_sha256=id_digest(ids[reference]),
            audit_and_regime_train_rows=len(reference),
            receiver_id_sha256=id_digest(ids[receive]), forbidden_train_overlap=0,
            reference_expert_producers=inner_records))
    ref_group, val_group = reference_regimes(raw[tr], raw[val])
    global_test, local_test = audit_receivers(train_opt, y[tr], b[tr], b[val], ref_group, val_group)
    scaler = StandardScaler().fit(raw[tr])

    def inputs(p, positions, opt, mask, identity, ga, la):
        features = np.concatenate([opt, ga, identity, la], axis=-1)
        features *= mask[:, :, :, None]
        require(features.shape[-1] == 21 and np.isfinite(features).all(), "descriptor contract")
        return dict(subset_features=features, subset_mask=mask,
            baseline=np.eye(7, dtype=np.float32)[b[positions]],
            context=np.clip(scaler.transform(raw[positions]), -6, 6).astype(np.float32),
            keep_heads=fallback_metadata(p, b[positions]))

    train = inputs(train_p, tr, train_opt, train_mask, train_identity, global_train, local_train)
    held = inputs(test_p, val, test_opt, test_mask, test_identity, global_test, local_test)
    manifest = dict(outer_fold=int(outer), outer_train_ids_sha256=id_digest(ids[tr]),
        outer_test_ids_sha256=id_digest(ids[val]), outer_test_train_overlap=0,
        outer_train_oof_producers=records, inner_audits=audit_manifests,
        audit_context_scalers_and_clusters_exclude_receiver=True,
        raw_context_scaler="label-free fit on outer train, ordinary model preprocessing")
    return train, held, tr, val, manifest
