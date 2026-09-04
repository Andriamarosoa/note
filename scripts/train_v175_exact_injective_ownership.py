"""V17.5 exact injective candidate ownership.

Starts from canonical V17.3 and replaces the rejected V17.4 continuous-capacity
idea with one structured ownership mechanism:

Training:
  keep the complete V17.3 set loss and add the exact negative log probability
  that all query rows matched to valid true objects choose distinct candidates.
  This probability is computed exactly by a 2^6-state bipartite-matching DP.
  It is query-permutation invariant, zero-cost for <=1 candidate-valid object,
  and adds no trainable parameter.  Presence is not part of this term, so the
  V17.3 Poisson-binomial count objective remains separate.

Runtime candidate ownership:
  K is still sum(presence >= 0.5).  Active queries are assigned to candidates
  by an exact maximum-weight rectangular assignment.  Private null columns make
  every row feasible; real candidate IDs are therefore one-to-one by
  construction.  The count decode itself is unchanged.

No threshold tuning, categorical K head, q-specific boost, Locked12 access, or
candidate-ranker change is introduced.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import sys
from typing import Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for _p in (ROOT, SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from causal_note.guitarset import SLOT_COUNT
from scripts import train_v102_source_time_assignment as v102
from scripts import train_v130_causal_event_set_decoder as v130
from scripts import train_v171_controlled_assignment_ab as v171
from scripts import train_v172_mass_preserving_exchangeable as v172
from scripts import train_v173_poibin_count_consistency as v173

DEFAULT_SEED = v173.DEFAULT_SEED
EVENT_QUERIES = SLOT_COUNT
PRESENCE_THRESHOLD = v173.PRESENCE_THRESHOLD
BASE_ARM = v173.BASE_ARM
MODEL_KEY = "v175_injective"
OWNERSHIP_NLL_WEIGHT = float(v172.SET_CANDIDATE_WEIGHT)
MAX_CANDIDATES = v171.SET_WIDTH - v171.SET_CANDIDATE_OFFSET
OWNERSHIP_EPS = 1e-7
RUNTIME_NULL_SCORE = 1e-12
MASK_COUNT = 1 << EVENT_QUERIES
MASKS = np.arange(MASK_COUNT, dtype=np.int32)
MASK_HAS = np.asarray(
    [[1.0 if (m & (1 << q)) else 0.0 for m in MASKS] for q in range(EVENT_QUERIES)],
    dtype=np.float32,
)
MASK_PREV = np.asarray(
    [[m ^ (1 << q) if (m & (1 << q)) else m for m in MASKS] for q in range(EVENT_QUERIES)],
    dtype=np.int32,
)
MASK_BITS = np.asarray([1 << q for q in range(EVENT_QUERIES)], dtype=np.int32)


class V175Error(RuntimeError):
    pass


def _normalize_candidate_np(candidate: np.ndarray) -> np.ndarray:
    c = np.maximum(np.asarray(candidate, dtype=np.float64), 0.0)
    den = np.sum(c, axis=2, keepdims=True)
    return np.divide(c, np.maximum(den, 1e-15), out=np.zeros_like(c), where=den > 0)


def _injective_probability_np(candidate: np.ndarray, object_mask: np.ndarray) -> np.ndarray:
    """Exact P(candidate choices are distinct for the selected query subset)."""
    c = _normalize_candidate_np(candidate)
    selected = np.asarray(object_mask, dtype=np.float64) > 0.5
    if c.ndim != 3 or c.shape[1] != EVENT_QUERIES:
        raise V175Error(f"candidate shape must be [B,{EVENT_QUERIES},C], got {c.shape}")
    if selected.shape != c.shape[:2]:
        raise V175Error(f"object mask mismatch: {selected.shape} vs {c.shape[:2]}")
    dp = np.zeros((len(c), MASK_COUNT), dtype=np.float64)
    dp[:, 0] = 1.0
    for col in range(c.shape[2]):
        old = dp
        add = np.zeros_like(old)
        for q in range(EVENT_QUERIES):
            add += old[:, MASK_PREV[q]] * c[:, q, col : col + 1] * MASK_HAS[q][None, :]
        dp = old + add
    target = np.sum(selected.astype(np.int32) * MASK_BITS[None, :], axis=1)
    return dp[np.arange(len(dp)), target]


def _injective_probability_tf(tf, candidate, object_mask):
    """TensorFlow equivalent of the exact 2^6 ownership DP."""
    c = tf.maximum(tf.cast(candidate, tf.float32), 0.0)
    c = c / tf.maximum(tf.reduce_sum(c, axis=2, keepdims=True), 1e-12)
    batch = tf.shape(c)[0]
    dp = tf.concat(
        [tf.ones((batch, 1), tf.float32), tf.zeros((batch, MASK_COUNT - 1), tf.float32)],
        axis=1,
    )
    has = tf.constant(MASK_HAS, dtype=tf.float32)
    prev = tf.constant(MASK_PREV, dtype=tf.int32)
    for col in range(MAX_CANDIDATES):
        old = dp
        terms = []
        for q in range(EVENT_QUERIES):
            terms.append(
                tf.gather(old, prev[q], axis=1)
                * c[:, q, col : col + 1]
                * has[q][None, :]
            )
        dp = old + tf.add_n(terms)
    selected = tf.cast(tf.cast(object_mask, tf.float32) > 0.5, tf.int32)
    target = tf.reduce_sum(selected * tf.constant(MASK_BITS[None, :], tf.int32), axis=1)
    return tf.gather_nd(dp, tf.stack([tf.range(batch, dtype=tf.int32), target], axis=1))


def _set_loss(spec: dict):
    """V17.3 exact set/count loss + exact collision-free ownership NLL."""
    import tensorflow as tf
    from tensorflow import keras

    perm_mats = tf.constant(v172.PERMUTATION_MATRICES, dtype=tf.float32)
    perms = tf.constant(v172.PERMUTATIONS, dtype=tf.int32)

    class ExactInjectiveOwnershipSetLoss(keras.losses.Loss):
        def __init__(self):
            super().__init__(name="v175_exact_injective_ownership_set_loss")

        def call(self, y_true, y_pred):
            yt = tf.cast(y_true, tf.float32)
            yp = tf.cast(y_pred, tf.float32)
            truth_present = yt[:, :, 0]
            truth_valid = yt[:, :, v171.SET_VALID_OFFSET]
            truth_time = yt[:, :, v171.SET_TIME_OFFSET : v171.SET_CANDIDATE_OFFSET]
            truth_candidate = yt[:, :, v171.SET_CANDIDATE_OFFSET :]

            pred_present = tf.clip_by_value(yp[:, :, 0], 1e-6, 1.0 - 1e-6)
            pred_time = tf.clip_by_value(
                yp[:, :, v171.SET_TIME_OFFSET : v171.SET_CANDIDATE_OFFSET], 1e-7, 1.0
            )
            pred_candidate_raw = tf.maximum(yp[:, :, v171.SET_CANDIDATE_OFFSET :], 0.0)
            pred_candidate = tf.clip_by_value(pred_candidate_raw, 1e-7, 1.0)

            # Canonical V17.3 / V17.2-C exact 720-permutation matching, unchanged.
            y = truth_present[:, None, :]
            p = pred_present[:, :, None]
            cw = v172._class_weight_tensor(tf, BASE_ARM, truth_present, spec)
            presence_cost = -cw * (
                y * tf.math.log(p) + (1.0 - y) * tf.math.log(1.0 - p)
            )
            time_cost = -tf.einsum("btd,bqd->bqt", truth_time, tf.math.log(pred_time))
            candidate_cost = -tf.einsum(
                "btd,bqd->bqt", truth_candidate, tf.math.log(pred_candidate)
            )
            detail = (truth_present * truth_valid)[:, None, :]
            pair = (
                v172.SET_PRESENCE_WEIGHT * presence_cost
                + v172.SET_TIME_WEIGHT * detail * time_cost
                + v172.SET_CANDIDATE_WEIGHT * detail * candidate_cost
            )
            scores = tf.einsum("bqt,rqt->br", pair, perm_mats)
            best = tf.argmin(scores, axis=1, output_type=tf.int32)
            matching_loss = tf.reduce_min(scores, axis=1)

            # Canonical V17.3 exact Poisson-binomial count objective, unchanged.
            count_dist = v173._poibin_distribution_tf(tf, pred_present)
            true_k = tf.cast(tf.reduce_sum(truth_present, axis=1), tf.int32)
            batch = tf.range(tf.shape(true_k)[0], dtype=tf.int32)
            true_prob = tf.gather_nd(count_dist, tf.stack([batch, true_k], axis=1))
            count_nll = -tf.math.log(tf.clip_by_value(true_prob, 1e-7, 1.0))

            # Select the predicted query rows matched by the exact assignment to
            # candidate-valid true objects, then compute the exact probability
            # that these candidate draws are injective.  K<=1 gives P=1 exactly.
            assignment = tf.gather(perms, best)  # query -> truth slot
            gather = tf.stack(
                [tf.tile(batch[:, None], [1, EVENT_QUERIES]), assignment], axis=-1
            )
            matched_candidate_valid_object = tf.gather_nd(
                truth_present * truth_valid, gather
            )
            injective_prob = _injective_probability_tf(
                tf, pred_candidate_raw, matched_candidate_valid_object
            )
            ownership_nll = -tf.math.log(
                tf.clip_by_value(injective_prob, OWNERSHIP_EPS, 1.0)
            )
            return (
                matching_loss
                + tf.constant(v173.COUNT_NLL_WEIGHT, tf.float32) * count_nll
                + tf.constant(OWNERSHIP_NLL_WEIGHT, tf.float32) * ownership_nll
            )

    return ExactInjectiveOwnershipSetLoss()


def _build_model(spec: dict):
    """Reuse the V17.3 graph bit-for-bit; only the event-set loss changes."""
    try:
        from tensorflow import keras
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required") from exc

    model, old_lw, token_shape = v173._build_model(spec)
    loss = {f"string_{slot}": "binary_crossentropy" for slot in range(SLOT_COUNT)}
    loss.update({f"pitch_{slot}": "mse" for slot in range(SLOT_COUNT)})
    loss.update({f"time_{slot}": keras.losses.KLDivergence() for slot in range(SLOT_COUNT)})
    loss["event_set"] = _set_loss(spec)
    loss["event_count_norm"] = "mse"
    lw = dict(old_lw)
    lw["event_count_norm"] = 0.0
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=2e-4),
        loss=loss,
        loss_weights=lw,
    )
    return model, lw, token_shape


def _hungarian_min(cost: np.ndarray) -> np.ndarray:
    """Exact rectangular Hungarian assignment for n rows <= m columns."""
    a = np.asarray(cost, dtype=np.float64)
    n, m = a.shape
    if n == 0:
        return np.empty(0, dtype=np.int32)
    if n > m:
        raise V175Error(f"Hungarian requires rows<=cols, got {a.shape}")
    u = np.zeros(n + 1, dtype=np.float64)
    v = np.zeros(m + 1, dtype=np.float64)
    p = np.zeros(m + 1, dtype=np.int32)
    way = np.zeros(m + 1, dtype=np.int32)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, np.inf, dtype=np.float64)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = int(p[j0])
            delta = np.inf
            j1 = 0
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = a[i0 - 1, j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = int(way[j0])
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    assignment = np.full(n, -1, dtype=np.int32)
    for j in range(1, m + 1):
        if p[j] > 0:
            assignment[p[j] - 1] = j - 1
    return assignment


def _injective_runtime_ids(
    candidate: np.ndarray, presence: np.ndarray, candidate_mask: np.ndarray
):
    """Exact maximum-weight real-candidate ownership with private null dustbins."""
    c = _normalize_candidate_np(candidate)
    p = np.asarray(presence, dtype=np.float64)
    mask = np.asarray(candidate_mask, dtype=np.float64) > 0.5
    active = p >= PRESENCE_THRESHOLD
    if c.shape[:2] != active.shape or mask.shape != (len(c), c.shape[2]):
        raise V175Error(f"runtime shape mismatch c={c.shape} p={p.shape} mask={mask.shape}")
    ids = np.full(active.shape, -1, dtype=np.int16)
    feasible_real = np.zeros(len(c), dtype=bool)
    changed = np.zeros(len(c), dtype=bool)
    objective_regret = np.full(len(c), np.nan, dtype=np.float64)
    for row in range(len(c)):
        qs = np.flatnonzero(active[row])
        if not len(qs):
            feasible_real[row] = True
            continue
        valid = np.flatnonzero(mask[row])
        feasible_real[row] = len(valid) >= len(qs)
        real_scores = c[row, qs][:, valid] if len(valid) else np.zeros((len(qs), 0))
        # Private null columns guarantee feasibility but are only competitive if
        # there are insufficient real candidates (or all real mass is ~zero).
        n = len(qs)
        null = np.full((n, n), 0.0, dtype=np.float64)
        np.fill_diagonal(null, RUNTIME_NULL_SCORE)
        scores = np.concatenate([real_scores, null], axis=1)
        cost = -np.log(np.clip(scores, RUNTIME_NULL_SCORE, 1.0))
        # Make another query's private null effectively inaccessible.
        if n:
            cost[:, len(valid) :] = -math.log(RUNTIME_NULL_SCORE) + 50.0
            for i in range(n):
                cost[i, len(valid) + i] = -math.log(RUNTIME_NULL_SCORE)
        ass = _hungarian_min(cost)
        for i, q in enumerate(qs):
            col = int(ass[i])
            if col < len(valid):
                ids[row, q] = int(valid[col])
        indep = np.argmax(c[row, qs], axis=1)
        hard = ids[row, qs].astype(np.int32)
        changed[row] = bool(np.any((hard >= 0) & (hard != indep)))
        if np.all(hard >= 0):
            independent_cost = float(np.sum(-np.log(np.clip(c[row, qs, indep], RUNTIME_NULL_SCORE, 1.0))))
            injective_cost = float(np.sum(-np.log(np.clip(c[row, qs, hard], RUNTIME_NULL_SCORE, 1.0))))
            objective_regret[row] = injective_cost - independent_cost
    return ids, {
        "active": active,
        "real_feasible": feasible_real,
        "changed": changed,
        "objective_regret": objective_regret,
    }


def _raw_duplicate_rate(candidate: np.ndarray, active: np.ndarray, rows: np.ndarray):
    values = []
    arg = np.argmax(candidate, axis=2)
    for row in np.flatnonzero(rows):
        qs = np.flatnonzero(active[row])
        if len(qs) < 2:
            continue
        values.append(len(np.unique(arg[row, qs])) < len(qs))
    return float(np.mean(values)) if values else None


def _structured_duplicate_rate(ids: np.ndarray, active: np.ndarray, rows: np.ndarray):
    values = []
    for row in np.flatnonzero(rows):
        qs = np.flatnonzero(active[row])
        if len(qs) < 2:
            continue
        real = ids[row, qs]
        real = real[real >= 0]
        values.append(len(np.unique(real)) < len(real))
    return float(np.mean(values)) if values else None


def _postprocess(args, report, ctx):
    report = v173._postprocess(args, report, ctx)
    v171._rename_report(report, v173.MODEL_KEY, MODEL_KEY)

    report["protocol"].update(
        {
            "v175_exact_injective_candidate_ownership": True,
            "v175_base_version": "V17.3",
            "v175_only_ownership_mechanism_change": True,
            "v173_poisson_binomial_count_objective_unchanged": True,
            "v173_count_nll_weight": v173.COUNT_NLL_WEIGHT,
            "mass_preserving_exchangeable_weights_unchanged": True,
            "exact_720_truth_matching_unchanged": True,
            "model_graph_unchanged_from_v173": True,
            "trainable_parameters_added": 0,
            "injective_ownership_nll_weight": OWNERSHIP_NLL_WEIGHT,
            "injective_ownership_nll_exact_2pow6_dp": True,
            "runtime_count_decode_unchanged_from_v173": True,
            "runtime_presence_threshold": PRESENCE_THRESHOLD,
            "runtime_presence_threshold_tuned": False,
            "runtime_candidate_decode": "exact maximum-weight injective assignment with private null dustbins",
            "runtime_candidate_real_ids_one_to_one_by_construction": True,
            "categorical_cardinality_head_exists": False,
            "historical_validation_or_locked12_indexed_or_evaluated": False,
        }
    )

    inherited = report.pop("v173")
    report["v175"] = {
        **inherited,
        "model_key": MODEL_KEY,
        "base_version": "V17.3",
        "ownership": {
            "training": "exact collision-free candidate ownership NLL",
            "training_dp_states": MASK_COUNT,
            "weight": OWNERSHIP_NLL_WEIGHT,
            "runtime": "exact maximum-weight injective rectangular assignment",
            "runtime_null": "private dustbin only when real assignment is infeasible",
            "trainable_parameters_added": 0,
        },
    }

    npz_path = args.output_dir / f"predictions-fold-{args.outer_fold}.npz"
    with np.load(npz_path, allow_pickle=False) as z:
        data = {key: np.asarray(z[key]) for key in z.files}
    if "pred173_poibin" not in data:
        raise V175Error("missing V17.3-compatible prediction key")
    data["pred175_injective"] = data.pop("pred173_poibin")

    outer_idx = np.asarray(ctx["outer_idx"], dtype=np.int64)
    candidate_mask = np.asarray(ctx["cache"]["mask"], dtype=np.float64)[outer_idx]
    presence = np.asarray(data["presence"], dtype=np.float64)
    candidate = np.asarray(data["event_candidate"], dtype=np.float64)
    ids, runtime = _injective_runtime_ids(candidate, presence, candidate_mask)
    data["ownership_candidate_id"] = ids
    data["ownership_real_feasible"] = runtime["real_feasible"].astype(np.int8)
    np.savez_compressed(npz_path, **data)

    k = np.asarray(data["k"], dtype=np.int32)
    hard = np.sum(presence >= PRESENCE_THRESHOLD, axis=1).astype(np.int32)
    active = presence >= PRESENCE_THRESHOLD
    ownership_diag = {
        "real_assignment_feasible_rate": float(np.mean(runtime["real_feasible"])),
        "runtime_assignment_changed_independent_argmax_rate": float(np.mean(runtime["changed"])),
        "runtime_assignment_mean_logloss_regret_when_real": float(np.nanmean(runtime["objective_regret"]))
        if np.any(np.isfinite(runtime["objective_regret"])) else None,
        "per_true_k": {},
    }
    for value in range(2, EVENT_QUERIES + 1):
        rows = k == value
        exact = rows & (hard == value)
        ownership_diag["per_true_k"][str(value)] = {
            "clusters": int(np.sum(rows)),
            "exact_count_clusters": int(np.sum(exact)),
            "raw_duplicate_argmax_exact_count": _raw_duplicate_rate(candidate, active, exact),
            "structured_duplicate_real_id_exact_count": _structured_duplicate_rate(ids, active, exact),
            "real_assignment_feasible_exact_count": float(np.mean(runtime["real_feasible"][exact]))
            if np.any(exact) else None,
        }
    report["v175"]["ownership_diagnostics"] = ownership_diag

    old_w = args.output_dir / f"v173-poibin-fold-{args.outer_fold}.weights.h5"
    new_w = args.output_dir / f"v175-injective-fold-{args.outer_fold}.weights.h5"
    if old_w.exists():
        old_w.replace(new_w)
    report_path = args.output_dir / f"report-fold-{args.outer_fold}.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def train_fold(args):
    if args.seed != DEFAULT_SEED:
        raise V175Error(f"V17.5 requires seed {DEFAULT_SEED}, got {args.seed}")
    if args.arm != BASE_ARM:
        raise V175Error(f"V17.5 only supports base arm {BASE_ARM!r}")

    ctx = v172._fold_context(args)
    stage_specs = [ctx["meta_spec"], ctx["final_spec"]]
    calls = {"count": 0}

    def builder():
        i = calls["count"]
        if i >= 2:
            raise V175Error(f"unexpected model build call {i + 1}")
        calls["count"] += 1
        return _build_model(stage_specs[i])

    old_build, old_targets, old_weights = v130._build_model, v130._targets, v130._sample_weights
    try:
        v130._build_model = builder
        v130._targets = v171._targets
        v130._sample_weights = v171._sample_weights
        report = v130.train_fold(args)
    finally:
        v130._build_model, v130._targets, v130._sample_weights = old_build, old_targets, old_weights
    if calls["count"] != 2:
        raise V175Error(f"expected exactly two builds, got {calls['count']}")
    report = _postprocess(args, report, ctx)
    g = report["strata"]["aggregate"][MODEL_KEY]["metrics"]["global"]
    d = report["v175"]["ownership_diagnostics"]
    print(json.dumps({
        "outer": args.outer_fold,
        "selected_epochs": report["data"]["selected_epochs"],
        "v104_f1": report["strata"]["aggregate"]["v104"]["metrics"]["global"]["f1"],
        "v175_f1": g["f1"],
        "pred_ref": g["prediction_reference_ratio"],
        "real_assignment_feasible_rate": d["real_assignment_feasible_rate"],
        "raw_duplicate_exact_k3": d["per_true_k"]["3"]["raw_duplicate_argmax_exact_count"],
        "structured_duplicate_exact_k3": d["per_true_k"]["3"]["structured_duplicate_real_id_exact_count"],
        "raw_duplicate_exact_k5": d["per_true_k"]["5"]["raw_duplicate_argmax_exact_count"],
        "structured_duplicate_exact_k5": d["per_true_k"]["5"]["structured_duplicate_real_id_exact_count"],
    }, indent=2, sort_keys=True))
    return report


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("dataset_dir", nargs="?", type=Path, default=Path("data/GuitarSet"))
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--baseline-eval-dir", type=Path, required=True)
    p.add_argument("--outer-fold", type=int, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--arm", choices=[BASE_ARM], default=BASE_ARM)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return p


def main(argv: Optional[Sequence[str]] = None):
    train_fold(parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
