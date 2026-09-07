"""Pre-V23 audit: explicit cardinality and structured dense transport feasibility.

Audit only; no V23 model is trained or promoted. For one canonical outer fold:
- load the saved V19 model and freeze it;
- fit deterministic ridge probes on final-fit rows only;
- reproduce the V21/V22 hard-negative local scorer on conv3 cells;
- test whether the dense score field carries cardinality K=0..6;
- with TRUE K used only as an oracle diagnostic, measure injective recovery of
  true birth centers from fixed top-rank pools K,2K,4K,8K,48;
- quantify the end-to-end ceiling obtained by feeding true K into the existing
  frozen candidate ranking.

No threshold sweep, no model hyperparameter sweep, no Locked12 access.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for p in (ROOT, SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from scripts import train_v100_spectral_string_slots as v100
from scripts import train_v102_source_time_assignment as v102
from scripts import train_v120_integrated_birth_source_time as v120
from scripts import train_v172_mass_preserving_exchangeable as v172
from scripts import train_v190_dense_birth_centers as v190

RIDGE = 1.0
MAX_TRAIN_PER_K = 5000
BATCH = 96
RANK_BUDGETS = ("K", "2K", "4K", "8K", "48")
TOLERANCES = {
    "exact": (0, 0),
    "dt1df1": (1, 1),
    "dt1df2": (1, 2),
    "dt2df4": (2, 4),
}
COUNT_RANKS = (1, 2, 3, 4, 5, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192)
COUNT_QUANTILES = (50.0, 75.0, 90.0, 95.0, 97.5, 99.0, 99.5)


def one(root: Path, pattern: str) -> Path:
    xs = sorted(root.glob(pattern))
    if len(xs) != 1:
        raise RuntimeError(f"expected one {pattern}, got {len(xs)}")
    return xs[0]


def standardize_fit(x):
    x = np.asarray(x, dtype=np.float64)
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    std = np.where(std > 1e-7, std, 1.0)
    return mean, std


def ridge_multiclass_fit(x, y, classes):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y)
    mean, std = standardize_fit(x)
    z = (x - mean) / std
    z = np.concatenate([z, np.ones((len(z), 1), dtype=np.float64)], axis=1)
    class_to_col = {c: i for i, c in enumerate(classes)}
    cols = np.asarray([class_to_col[v] for v in y], dtype=np.int32)
    counts = np.bincount(cols, minlength=len(classes)).astype(np.float64)
    w = np.asarray([len(y) / (len(classes) * max(1.0, counts[c])) for c in cols], dtype=np.float64)
    yy = np.eye(len(classes), dtype=np.float64)[cols]
    a = z.T @ (w[:, None] * z)
    reg = np.eye(a.shape[0], dtype=np.float64) * RIDGE
    reg[-1, -1] = 0.0
    b = z.T @ (w[:, None] * yy)
    coef = np.linalg.solve(a + reg, b)
    return {"mean": mean, "std": std, "coef": coef, "classes": np.asarray(classes)}


def ridge_multiclass_predict(model, x):
    z = (np.asarray(x, dtype=np.float64) - model["mean"]) / model["std"]
    z = np.concatenate([z, np.ones((len(z), 1), dtype=np.float64)], axis=1)
    score = z @ model["coef"]
    return model["classes"][np.argmax(score, axis=1)], score


def ridge_binary_fit(x, y):
    return ridge_multiclass_fit(x, np.asarray(y, dtype=np.int32), [0, 1])


def fold_linear_scorer(model):
    coef = np.asarray(model["coef"], dtype=np.float64)
    mean = np.asarray(model["mean"], dtype=np.float64)
    std = np.asarray(model["std"], dtype=np.float64)
    d = coef[:, 1] - coef[:, 0]
    w = d[:-1] / std
    b = float(d[-1] - np.sum((mean / std) * d[:-1]))
    return w.astype(np.float32), np.float32(b)


def auc_binary(y, score):
    y = np.asarray(y, dtype=np.int8).reshape(-1)
    s = np.asarray(score, dtype=np.float64).reshape(-1)
    pos = int(np.sum(y == 1)); neg = int(np.sum(y == 0))
    if pos == 0 or neg == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and s[order[j]] == s[order[i]]:
            j += 1
        rank = 0.5 * ((i + 1) + j)
        ranks[order[i:j]] = rank
        i = j
    sum_pos = float(np.sum(ranks[y == 1]))
    return float((sum_pos - pos * (pos + 1) / 2.0) / (pos * neg))


def conv_descriptor(feat):
    x = np.asarray(feat, dtype=np.float32)
    return np.concatenate([
        np.mean(x, axis=(1, 2)), np.max(x, axis=(1, 2)), np.std(x, axis=(1, 2))
    ], axis=1).astype(np.float32)


def score_descriptor(score):
    x = np.asarray(score, dtype=np.float32)
    n = x.shape[1]
    topn = max(COUNT_RANKS)
    part = np.partition(x, kth=n - topn, axis=1)[:, n - topn:]
    desc = np.sort(part, axis=1)[:, ::-1]
    rank_values = np.stack([desc[:, r - 1] for r in COUNT_RANKS], axis=1)
    quant = np.percentile(x, COUNT_QUANTILES, axis=1).T.astype(np.float32)
    stats = np.stack([
        np.mean(x, axis=1), np.std(x, axis=1), np.max(x, axis=1), np.min(x, axis=1)
    ], axis=1)
    gaps = desc[:, :6] - desc[:, 1:7]
    return np.concatenate([rank_values, quant, stats, gaps], axis=1).astype(np.float32)


def card_metrics(y, pred):
    y = np.asarray(y, dtype=np.int32); pred = np.asarray(pred, dtype=np.int32)
    vals = []; per = {}
    for value in range(7):
        m = y == value
        exact = float(np.mean(pred[m] == value)) if np.any(m) else None
        if exact is not None:
            vals.append(exact)
        per[str(value)] = {"rows": int(np.sum(m)), "exact": exact}
    poly = y >= 2
    return {
        "exact": float(np.mean(pred == y)),
        "balanced_exact": float(np.mean(vals)),
        "poly_exact": float(np.mean(pred[poly] == y[poly])) if np.any(poly) else None,
        "mae": float(np.mean(np.abs(pred - y))),
        "mean_predicted_k": float(np.mean(pred)),
        "mean_true_k": float(np.mean(y)),
        "per_true_k": per,
    }


def balanced_train_sample(fit_idx, k, fold):
    rng = np.random.default_rng(16061 + 97 * fold)
    fit_idx = np.asarray(fit_idx, dtype=np.int64)
    out = []
    for value in range(7):
        pool = fit_idx[np.asarray(k)[fit_idx] == value]
        if len(pool) > MAX_TRAIN_PER_K:
            pool = rng.choice(pool, size=MAX_TRAIN_PER_K, replace=False)
        out.append(np.asarray(pool, dtype=np.int64))
    return np.sort(np.concatenate(out))


def far_mask(truth):
    mask = np.ones(v190.CENTER_CELLS, dtype=bool)
    for flat in truth:
        t = int(flat // v100.SPECTRAL_BANDS); f = int(flat % v100.SPECTRAL_BANDS)
        for dt in range(-1, 2):
            for df in range(-2, 3):
                tt, ff = t + dt, f + df
                if 0 <= tt < v100.TIME_FRAMES and 0 <= ff < v100.SPECTRAL_BANDS:
                    mask[tt * v100.SPECTRAL_BANDS + ff] = False
    return mask


def extract_train(latent_model, spectral, indices, target, eligible, k):
    desc = []; cell_x = []; cell_y = []
    idx = np.asarray(indices, dtype=np.int64)
    for start in range(0, len(idx), BATCH):
        rows = idx[start:start + BATCH]
        feat, logits = latent_model.predict(np.asarray(spectral[rows], np.float32), batch_size=BATCH, verbose=0)
        feat = np.asarray(feat, dtype=np.float32); logits = np.asarray(logits, dtype=np.float32)
        desc.append(conv_descriptor(feat))
        flat_feat = feat.reshape((len(rows), v190.CENTER_CELLS, feat.shape[-1]))
        for bi, row in enumerate(rows):
            if not eligible[row] or int(k[row]) <= 0:
                continue
            truth = np.flatnonzero(target[row] > 0)
            if len(truth) == 0:
                continue
            pool = np.flatnonzero(far_mask(truth))
            if len(pool) < len(truth):
                continue
            hard = pool[np.argsort(logits[bi, pool])[-len(truth):]]
            for flat in truth:
                cell_x.append(flat_feat[bi, int(flat)]); cell_y.append(1)
            for flat in hard:
                cell_x.append(flat_feat[bi, int(flat)]); cell_y.append(0)
    return np.concatenate(desc, axis=0), np.asarray(cell_x, np.float32), np.asarray(cell_y, np.int8)


def field_descriptors(latent_model, spectral, indices, scorer_w, scorer_b):
    idx = np.asarray(indices, dtype=np.int64); conv_out = []; score_out = []
    for start in range(0, len(idx), BATCH):
        rows = idx[start:start + BATCH]
        feat, _ = latent_model.predict(np.asarray(spectral[rows], np.float32), batch_size=BATCH, verbose=0)
        feat = np.asarray(feat, dtype=np.float32)
        score = np.tensordot(feat, scorer_w, axes=([3], [0])) + scorer_b
        score = score.reshape((len(rows), v190.CENTER_CELLS))
        conv_out.append(conv_descriptor(feat)); score_out.append(score_descriptor(score))
    return np.concatenate(conv_out, axis=0), np.concatenate(score_out, axis=0)


def max_match_size(truth, selected, dt, df):
    truth = [int(x) for x in truth]; selected = [int(x) for x in selected]
    if not truth or not selected:
        return 0
    adj = []
    for tr in truth:
        tt, tf = divmod(tr, v100.SPECTRAL_BANDS); choices = []
        for ci, cell in enumerate(selected):
            ct, cf = divmod(cell, v100.SPECTRAL_BANDS)
            if abs(ct - tt) <= dt and abs(cf - tf) <= df:
                choices.append(ci)
        adj.append(choices)
    owner = [-1] * len(selected)
    def dfs(ti, seen):
        for ci in adj[ti]:
            if seen[ci]:
                continue
            seen[ci] = True
            if owner[ci] < 0 or dfs(owner[ci], seen):
                owner[ci] = ti; return True
        return False
    matched = 0
    for ti in range(len(truth)):
        if dfs(ti, [False] * len(selected)):
            matched += 1
    return matched


def budget_size(name, kval):
    if name == "48":
        return 48
    return min(48, {"K": 1, "2K": 2, "4K": 4, "8K": 8}[name] * int(kval))


def min_prefix_full(truth, ranked48, dt, df, kval):
    if max_match_size(truth, ranked48, dt, df) < int(kval):
        return -1
    lo, hi = int(kval), len(ranked48)
    while lo < hi:
        mid = (lo + hi) // 2
        if max_match_size(truth, ranked48[:mid], dt, df) >= int(kval):
            hi = mid
        else:
            lo = mid + 1
    return int(lo)


def outer_field(latent_model, spectral, indices, target, eligible, distinct, k, scorer_w, scorer_b):
    idx = np.asarray(indices, dtype=np.int64); n = len(idx)
    conv_out = []; score_out = []
    coverage = {f"match_{tol}_{budget}": np.full(n, -1.0, np.float32)
                for tol in TOLERANCES for budget in RANK_BUDGETS}
    min_depth_12 = np.full(n, -1, np.int16); min_depth_24 = np.full(n, -1, np.int16)
    close_pair_fraction = np.full(n, -1.0, np.float32); any_close_pair = np.full(n, -1, np.int8)
    old_auc_y = []; old_auc_score = []; new_auc_score = []
    for start in range(0, n, BATCH):
        rows = idx[start:start + BATCH]
        feat, old_logits = latent_model.predict(np.asarray(spectral[rows], np.float32), batch_size=BATCH, verbose=0)
        feat = np.asarray(feat, np.float32); old_logits = np.asarray(old_logits, np.float32)
        score = np.tensordot(feat, scorer_w, axes=([3], [0])) + scorer_b
        score = score.reshape((len(rows), v190.CENTER_CELLS))
        conv_out.append(conv_descriptor(feat)); score_out.append(score_descriptor(score))
        flat_feat = feat.reshape((len(rows), v190.CENTER_CELLS, feat.shape[-1]))
        part = np.argpartition(-score, kth=47, axis=1)[:, :48]
        pscore = np.take_along_axis(score, part, axis=1)
        ranked = np.take_along_axis(part, np.argsort(-pscore, axis=1), axis=1)
        for bi, row in enumerate(rows):
            local = start + bi; kval = int(k[row])
            if eligible[row] and kval > 0:
                truth = np.flatnonzero(target[row] > 0); pool = np.flatnonzero(far_mask(truth))
                if len(pool) >= len(truth) and len(truth):
                    hard = pool[np.argsort(old_logits[bi, pool])[-len(truth):]]
                    for flat in truth:
                        old_auc_y.append(1); old_auc_score.append(float(old_logits[bi, int(flat)]))
                        new_auc_score.append(float(np.dot(flat_feat[bi, int(flat)], scorer_w) + scorer_b))
                    for flat in hard:
                        old_auc_y.append(0); old_auc_score.append(float(old_logits[bi, int(flat)]))
                        new_auc_score.append(float(np.dot(flat_feat[bi, int(flat)], scorer_w) + scorer_b))
            if not (eligible[row] and kval > 0 and int(distinct[row]) == kval):
                continue
            truth = np.flatnonzero(target[row] > 0); ranked48 = ranked[bi].tolist()
            for tol, (dt, df) in TOLERANCES.items():
                for budget in RANK_BUDGETS:
                    m = budget_size(budget, kval)
                    coverage[f"match_{tol}_{budget}"][local] = max_match_size(truth, ranked48[:m], dt, df) / float(kval)
            min_depth_12[local] = min_prefix_full(truth, ranked48, 1, 2, kval)
            min_depth_24[local] = min_prefix_full(truth, ranked48, 2, 4, kval)
            topk = ranked48[:kval]; pairs = 0; close = 0
            for a in range(len(topk)):
                at, af = divmod(int(topk[a]), v100.SPECTRAL_BANDS)
                for b in range(a + 1, len(topk)):
                    bt, bf = divmod(int(topk[b]), v100.SPECTRAL_BANDS); pairs += 1
                    close += int(abs(at - bt) <= 1 and abs(af - bf) <= 2)
            close_pair_fraction[local] = close / float(pairs) if pairs else 0.0
            any_close_pair[local] = int(close > 0)
    return {
        "conv": np.concatenate(conv_out), "score": np.concatenate(score_out), "coverage": coverage,
        "min_depth_dt1df2": min_depth_12, "min_depth_dt2df4": min_depth_24,
        "close_pair_fraction": close_pair_fraction, "any_close_pair": any_close_pair,
        "old_hard_auc": auc_binary(old_auc_y, old_auc_score),
        "new_hard_auc": auc_binary(old_auc_y, new_auc_score), "auc_examples": int(len(old_auc_y)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_dir", type=Path); ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--v190-fold-dir", type=Path, required=True); ap.add_argument("--outer-fold", type=int, choices=range(5), required=True)
    ap.add_argument("--output-dir", type=Path, required=True); args = ap.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)

    cache = v100._load_spectral_caches(args.cache_dir); spectral = np.asarray(cache["spectral"])
    members = np.asarray([str(x) for x in cache["members"]], dtype="U96")
    _, train_split, _ = v120._dataset_split(args.dataset_dir)
    candidates, recon = v102._reconstruct_candidates(cache)
    pitch, _, time_dist, _, supervision = v102._derive_supervision(members, candidates, args.dataset_dir, expected_slot_targets=cache["slot_targets"])
    k = np.minimum(np.asarray(cache["exact"], np.int32), 6)
    target, eligf, distinct = v190._birth_center_targets(cache, pitch, time_dist, k); eligible = np.asarray(eligf) > 0.5
    fold = int(args.outer_fold)
    ctx = v172._fold_context(SimpleNamespace(dataset_dir=args.dataset_dir, cache_dir=args.cache_dir, outer_fold=fold))
    outer = np.asarray(ctx["outer_idx"], np.int64); train_idx = balanced_train_sample(ctx["final_fit_idx"], k, fold)
    if np.intersect1d(train_idx, outer).size:
        raise RuntimeError("outer leakage in probe training rows")

    import tensorflow as tf
    from tensorflow import keras
    model, _, _ = v190._build_model(ctx["final_spec"])
    model.load_weights(one(args.v190_fold_dir, f"**/v190-dense-birth-centers-fold-{fold}.weights.h5"))
    spectral_input = next(t for t in model.inputs if t.name.split(":", 1)[0] == "spectral_map")
    latent = keras.Model(spectral_input, [model.get_layer("v190_dense_conv3").output, model.get_layer("v190_birth_center_logits_flat").output])

    train_conv, cell_x, cell_y = extract_train(latent, spectral, train_idx, target, eligible, k)
    scorer = ridge_binary_fit(cell_x, cell_y); scorer_w, scorer_b = fold_linear_scorer(scorer)
    train_conv2, train_score = field_descriptors(latent, spectral, train_idx, scorer_w, scorer_b)
    if not np.allclose(train_conv, train_conv2, atol=2e-5, rtol=2e-5):
        raise RuntimeError("conv descriptor changed between frozen passes")
    conv_count = ridge_multiclass_fit(train_conv, k[train_idx], list(range(7)))
    score_count = ridge_multiclass_fit(train_score, k[train_idx], list(range(7)))
    combined_count = ridge_multiclass_fit(np.concatenate([train_conv, train_score], axis=1), k[train_idx], list(range(7)))

    field = outer_field(latent, spectral, outer, target, eligible, distinct, k, scorer_w, scorer_b)
    pred_conv, _ = ridge_multiclass_predict(conv_count, field["conv"]); pred_score, _ = ridge_multiclass_predict(score_count, field["score"])
    pred_combined, _ = ridge_multiclass_predict(combined_count, np.concatenate([field["conv"], field["score"]], axis=1))
    pred_conv = np.asarray(pred_conv, np.int32); pred_score = np.asarray(pred_score, np.int32); pred_combined = np.asarray(pred_combined, np.int32)
    pred_npz = one(args.v190_fold_dir, f"**/predictions-fold-{fold}.npz")
    with np.load(pred_npz, allow_pickle=False) as z:
        gi = np.asarray(z["global_index"], np.int64); pred_v19 = np.asarray(z[v190.PRED_KEY], np.int32)
    if not np.array_equal(gi, outer):
        raise RuntimeError("V19 prediction shard does not match outer fold")

    metrics = {
        "v19": v120._metrics(cache, train_split, outer, pred_v19),
        "conv_global_probe": v120._metrics(cache, train_split, outer, pred_conv),
        "dense_score_probe": v120._metrics(cache, train_split, outer, pred_score),
        "combined_probe": v120._metrics(cache, train_split, outer, pred_combined),
        "true_k_oracle_existing_candidate_ranking": v120._metrics(cache, train_split, outer, k[outer]),
    }
    rep = eligible[outer] & (k[outer] > 0) & (distinct[outer].astype(np.int32) == k[outer]); rep_poly = rep & (k[outer] >= 2)
    eligible_poly = eligible[outer] & (k[outer] >= 2)
    cov_summary = {}
    for tol in TOLERANCES:
        cov_summary[tol] = {}
        for budget in RANK_BUDGETS:
            a = field["coverage"][f"match_{tol}_{budget}"]
            cov_summary[tol][budget] = {
                "representable_positive_rows": int(np.sum(rep)), "representable_poly_rows": int(np.sum(rep_poly)),
                "full_recovery_positive": float(np.mean(a[rep] >= 1.0 - 1e-7)) if np.any(rep) else None,
                "mean_recovered_fraction_positive": float(np.mean(a[rep])) if np.any(rep) else None,
                "full_recovery_poly": float(np.mean(a[rep_poly] >= 1.0 - 1e-7)) if np.any(rep_poly) else None,
                "mean_recovered_fraction_poly": float(np.mean(a[rep_poly])) if np.any(rep_poly) else None,
            }
    def depth_summary(arr):
        arr = np.asarray(arr); valid = rep_poly; solved = valid & (arr >= 0); vals = arr[solved].astype(np.float64)
        return {"representable_poly_rows": int(np.sum(valid)),
                "recoverable_within_top48_rate": float(np.mean(arr[valid] >= 0)) if np.any(valid) else None,
                "median_min_rank_depth_when_recoverable": float(np.median(vals)) if len(vals) else None,
                "p90_min_rank_depth_when_recoverable": float(np.percentile(vals, 90)) if len(vals) else None}

    result = {
        "schema_version": 1, "outer_fold": fold,
        "protocol": {"pre_v23_cardinality_transport_audit": True, "audit_only_no_v23_model": True,
                     "saved_v19_model_frozen": True, "probe_training_rows_final_fit_only": True,
                     "outer_rows_untouched_until_probe_fit": True, "ridge": RIDGE, "ridge_sweep": False,
                     "max_train_rows_per_k": MAX_TRAIN_PER_K,
                     "count_probes_fixed_before_outer": ["conv_global_mean_max_std", "dense_shared_score_rank_distribution", "conv_plus_dense_score"],
                     "rank_budgets_fixed_before_outer": list(RANK_BUDGETS), "rank_budgets_are_diagnostics_not_selected_hyperparameters": True,
                     "center_tolerances_fixed_from_v19_audit": {name: {"dt": dt, "df": df} for name, (dt, df) in TOLERANCES.items()},
                     "true_k_used_only_for_oracle_diagnostics": True, "runtime_threshold_sweep": False,
                     "locked12_indexed_or_evaluated": False},
        "data": {"outer_rows": int(len(outer)), "probe_train_rows": int(len(train_idx)), "probe_cell_examples": int(len(cell_y)),
                 "eligible_outer_positive_rows": int(np.sum(eligible[outer] & (k[outer] > 0))), "eligible_outer_poly_rows": int(np.sum(eligible_poly)),
                 "representable_distinct_outer_positive_rows": int(np.sum(rep)), "representable_distinct_outer_poly_rows": int(np.sum(rep_poly)),
                 "distinct_center_rate_eligible_poly": float(np.mean(distinct[outer][eligible_poly] == k[outer][eligible_poly])) if np.any(eligible_poly) else None},
        "local_scorer": {"old_v19_hard_negative_auc": field["old_hard_auc"], "shared_conv_ridge_hard_negative_auc": field["new_hard_auc"], "outer_auc_examples": field["auc_examples"]},
        "cardinality": {"v19": card_metrics(k[outer], pred_v19), "conv_global_probe": card_metrics(k[outer], pred_conv),
                        "dense_score_probe": card_metrics(k[outer], pred_score), "combined_probe": card_metrics(k[outer], pred_combined)},
        "end_to_end_count_realization": metrics,
        "transport": {"coverage": cov_summary, "min_rank_depth_dt1df2": depth_summary(field["min_depth_dt1df2"]),
                      "min_rank_depth_dt2df4": depth_summary(field["min_depth_dt2df4"]),
                      "topk_redundancy_representable_poly": {
                          "mean_close_pair_fraction_dt1df2": float(np.mean(field["close_pair_fraction"][rep_poly])) if np.any(rep_poly) else None,
                          "rows_with_any_close_pair_dt1df2": float(np.mean(field["any_close_pair"][rep_poly] > 0)) if np.any(rep_poly) else None}},
        "supervision": {"candidate_reconstruction": recon, "source_time_supervision": supervision},
    }
    (args.output_dir / f"report-fold-{fold}.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    payload = {"global_index": outer, "k": k[outer], "eligible": eligible[outer].astype(np.int8), "distinct": distinct[outer],
               "pred_v19": pred_v19, "pred_conv": pred_conv, "pred_score": pred_score, "pred_combined": pred_combined,
               "min_depth_dt1df2": field["min_depth_dt1df2"], "min_depth_dt2df4": field["min_depth_dt2df4"],
               "close_pair_fraction": field["close_pair_fraction"], "any_close_pair": field["any_close_pair"]}
    payload.update(field["coverage"]); np.savez_compressed(args.output_dir / f"predictions-fold-{fold}.npz", **payload)
    print(json.dumps({"fold": fold, "v19_f1": metrics["v19"]["global"]["f1"],
                      "combined_probe_f1": metrics["combined_probe"]["global"]["f1"],
                      "true_k_oracle_f1": metrics["true_k_oracle_existing_candidate_ranking"]["global"]["f1"],
                      "v19_card_exact": result["cardinality"]["v19"]["exact"], "combined_card_exact": result["cardinality"]["combined_probe"]["exact"],
                      "hard_auc_old": result["local_scorer"]["old_v19_hard_negative_auc"], "hard_auc_new": result["local_scorer"]["shared_conv_ridge_hard_negative_auc"],
                      "poly_topk_dt1df2_full": cov_summary["dt1df2"]["K"]["full_recovery_poly"],
                      "poly_top48_dt1df2_full": cov_summary["dt1df2"]["48"]["full_recovery_poly"], "locked12": False}, indent=2, sort_keys=True))
    tf.keras.backend.clear_session()


if __name__ == "__main__":
    main()
