"""Mechanistic audit of V13 event-set over-counting, without retraining.

This audit answers whether V13's extra predicted events are:
1) well-separated event representations with a miscalibrated presence decision,
2) acoustically unsupported activations, or
3) duplicate queries explaining the same acoustic/candidate evidence.

Only the five train-split outer-clean folds are used. Historical validation and
locked12 are never indexed or evaluated. Frozen V13 weights are reloaded only to
expose internal assignment masses; no parameter is updated and no threshold is
selected for deployment.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from scripts import v130_graph_patch
from scripts import train_v102_source_time_assignment as v102
from scripts import train_v104_oof_fold as oofmod
from scripts.train_boundaries import group_stem
from scripts.train_v100_spectral_string_slots import _load_spectral_caches

FOLDS = 5
QUERIES = 6
THRESHOLD = 0.5


def _one(root: Path, pattern: str) -> Path:
    rows = sorted(root.glob(pattern))
    if len(rows) != 1:
        raise RuntimeError(f"expected exactly one {pattern} under {root}, found {len(rows)}")
    return rows[0]


def _entropy(p: np.ndarray, axis: int = -1) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    return -np.sum(np.where(p > 0.0, p * np.log(np.clip(p, 1e-12, 1.0)), 0.0), axis=axis)


def _pair_redundancy(active: np.ndarray, candidate: np.ndarray, time: np.ndarray):
    n = len(active)
    max_c = np.full(n, np.nan, dtype=np.float32)
    max_t = np.full(n, np.nan, dtype=np.float32)
    same = np.full(n, np.nan, dtype=np.float32)
    pairs = np.zeros(n, dtype=np.int16)
    cand_arg = np.argmax(candidate, axis=2)
    for row in range(n):
        qs = np.flatnonzero(active[row])
        if len(qs) < 2:
            continue
        cb, tb, sa = [], [], []
        for i in range(len(qs)):
            for j in range(i + 1, len(qs)):
                a, b = int(qs[i]), int(qs[j])
                cb.append(float(np.sum(np.sqrt(np.clip(candidate[row, a], 0, 1) * np.clip(candidate[row, b], 0, 1)))))
                tb.append(float(np.sum(np.sqrt(np.clip(time[row, a], 0, 1) * np.clip(time[row, b], 0, 1)))))
                sa.append(float(cand_arg[row, a] == cand_arg[row, b]))
        pairs[row] = len(cb)
        max_c[row] = max(cb)
        max_t[row] = max(tb)
        same[row] = float(np.mean(sa))
    return max_c, max_t, same, pairs


def audit_fold(args):
    if not 0 <= args.fold < FOLDS:
        raise RuntimeError("fold outside range")
    try:
        import tensorflow as tf
        from tensorflow import keras
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required for the frozen-weight audit") from exc

    v130 = v130_graph_patch.apply()
    cache = _load_spectral_caches(args.cache_dir)
    _, train_split, validation = v130._dataset_split(args.dataset_dir)
    train_members = {t.annotation_member for t in train_split}
    validation_members = {t.annotation_member for t in validation}
    if train_members & validation_members:
        raise RuntimeError("train/validation overlap")
    if set(cache["track_members"]) != train_members:
        raise RuntimeError("spectral cache does not exactly cover train split")

    assignment, _, _, _ = oofmod._balanced_group_folds(cache, train_split)
    by_member = {t.annotation_member: t for t in train_split}
    members = np.asarray([str(x) for x in cache["members"]], dtype="U96")
    row_fold = np.asarray([assignment[group_stem(by_member[m])] for m in members], dtype=np.int16)
    outer_idx = np.flatnonzero(row_fold == args.fold).astype(np.int64)

    pred_path = _one(args.fold_dir, f"**/predictions-fold-{args.fold}.npz")
    weight_path = _one(args.fold_dir, f"**/v130-fold-{args.fold}.weights.h5")
    with np.load(pred_path, allow_pickle=False) as z:
        saved = {k: np.asarray(z[k]) for k in z.files}
    saved_index = np.asarray(saved["global_index"], dtype=np.int64)
    if not np.array_equal(saved_index, outer_idx):
        raise RuntimeError("saved V13 outer rows do not match reconstructed outer fold")

    k_all = np.minimum(np.asarray(cache["exact"], dtype=np.int32), QUERIES)
    ko = k_all[outer_idx]
    presence = np.asarray(saved["presence"], dtype=np.float64)
    pred = np.asarray(saved["pred130"], dtype=np.int32)
    if not np.array_equal(pred, np.sum(presence >= THRESHOLD, axis=1).astype(np.int32)):
        raise RuntimeError("saved V13 count is not the event-presence count")

    # Reconstruct training-only anonymous event targets for proper time/candidate diagnostics.
    candidate_samples, reconstruction = v102._reconstruct_candidates(cache)
    _, time_mask, string_time_targets, time_sample, supervision = v102._derive_supervision(
        members, candidate_samples, args.dataset_dir, expected_slot_targets=cache["slot_targets"]
    )
    event_present, _, event_candidate_target, event_valid, true_sample, event_diag = v130._ordered_event_supervision(
        cache, time_mask, string_time_targets, time_sample, k_all
    )

    tf.keras.backend.clear_session()
    model, _, _ = v130_graph_patch.build_model_fixed()
    model.load_weights(str(weight_path))
    candidate_mask_tensor = v130._input_by_name(model, "candidate_mask")
    tf_assignment = model.get_layer("event_tf_competition").output
    candidate_assignment = model.get_layer("event_candidate_competition").output

    outputs = {}
    for q in range(QUERIES):
        outputs[f"tf_mass_{q}"] = model.get_layer(f"event_{q}_tf_mass").output
        outputs[f"candidate_mass_{q}"] = model.get_layer(f"event_{q}_candidate_mass").output
    outputs["tf_background_mass"] = keras.layers.Lambda(
        lambda a: tf.reduce_mean(a[:, :, QUERIES], axis=1, keepdims=True),
        name="audit_tf_background_mass",
    )(tf_assignment)
    outputs["candidate_background_mass"] = keras.layers.Lambda(
        lambda z: tf.reduce_sum(z[0][:, :, QUERIES] * z[1], axis=1, keepdims=True)
        / (tf.reduce_sum(z[1], axis=1, keepdims=True) + 1e-6),
        name="audit_candidate_background_mass",
    )([candidate_assignment, candidate_mask_tensor])
    diag_model = keras.Model(model.inputs, outputs, name="v130_frozen_failure_audit")
    raw = diag_model.predict(v102._inputs(cache, outer_idx), batch_size=128, verbose=0)
    tf_mass = np.stack([np.asarray(raw[f"tf_mass_{q}"]).reshape(-1) for q in range(QUERIES)], axis=1)
    candidate_mass = np.stack([np.asarray(raw[f"candidate_mass_{q}"]).reshape(-1) for q in range(QUERIES)], axis=1)
    tf_background = np.asarray(raw["tf_background_mass"]).reshape(-1)
    candidate_background = np.asarray(raw["candidate_background_mass"]).reshape(-1)

    event_time = np.asarray(saved["event_time"], dtype=np.float64)
    event_candidate = np.asarray(saved["event_candidate"], dtype=np.float64)
    valid = np.asarray(event_valid[outer_idx], dtype=np.float32) > 0.5
    true_s = np.asarray(true_sample[outer_idx], dtype=np.float64)
    expected_s = np.sum(event_time * v102.FRAME_CENTER_SAMPLES[None, None, :], axis=2)
    time_error_ms = np.full(valid.shape, np.nan, dtype=np.float32)
    time_error_ms[valid] = (
        np.abs(expected_s[valid] - true_s[valid]) * 1000.0 / float(v102.SAMPLE_RATE)
    ).astype(np.float32)
    target_candidate = np.asarray(event_candidate_target[outer_idx], dtype=np.float64)
    candidate_correct = np.full(valid.shape, -1, dtype=np.int8)
    pred_candidate = np.argmax(event_candidate, axis=2)
    true_candidate = np.argmax(target_candidate, axis=2)
    candidate_correct[valid] = (pred_candidate[valid] == true_candidate[valid]).astype(np.int8)

    active = presence >= THRESHOLD
    max_candidate_bc, max_time_bc, same_argmax_fraction, active_pair_count = _pair_redundancy(
        active, event_candidate, event_time
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / f"diagnostics-fold-{args.fold}.npz",
        outer_fold=np.full(len(outer_idx), args.fold, dtype=np.int16),
        global_index=outer_idx,
        k=ko,
        pred=pred,
        presence=presence.astype(np.float32),
        tf_mass=tf_mass.astype(np.float32),
        candidate_mass=candidate_mass.astype(np.float32),
        tf_background=tf_background.astype(np.float32),
        candidate_background=candidate_background.astype(np.float32),
        event_valid=valid.astype(np.int8),
        time_error_ms=time_error_ms,
        candidate_correct=candidate_correct,
        candidate_entropy=_entropy(event_candidate, axis=2).astype(np.float32),
        time_entropy=_entropy(event_time, axis=2).astype(np.float32),
        max_candidate_bc=max_candidate_bc,
        max_time_bc=max_time_bc,
        same_candidate_argmax_fraction=same_argmax_fraction,
        active_pair_count=active_pair_count,
    )
    report = {
        "schema_version": 1,
        "fold": args.fold,
        "protocol": {
            "historical_validation_or_locked12_indexed_or_evaluated": False,
            "outer_fold_parameters_updated": False,
            "frozen_v130_weights_reloaded_only": True,
            "presence_threshold_tuned": False,
            "presence_threshold": THRESHOLD,
            "runtime_annotations_used": False,
            "annotations_used_for_audit_truth_only": True,
        },
        "data": {
            "outer_clusters": int(len(outer_idx)),
            "saved_outer_rows_match_reconstruction": True,
            "mean_true_count": float(np.mean(ko)),
            "mean_predicted_count": float(np.mean(pred)),
            "prefix_violation_rate": float(np.mean(np.asarray(saved["prefix_violation"]) > 0)),
            "timed_targets": int(np.sum(valid)),
        },
        "reconstruction": reconstruction,
        "supervision": supervision,
        "event_supervision": event_diag,
    }
    (args.output_dir / f"report-fold-{args.fold}.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _auc(y, p):
    y = np.asarray(y, dtype=np.int8)
    p = np.asarray(p, dtype=np.float64)
    n1 = int(np.sum(y == 1)); n0 = int(np.sum(y == 0))
    if n1 == 0 or n0 == 0:
        return None
    order = np.argsort(p, kind="stable")
    ps = p[order]
    ranks = np.empty(len(p), dtype=np.float64)
    i = 0
    while i < len(p):
        j = i + 1
        while j < len(p) and ps[j] == ps[i]:
            j += 1
        rank = ((i + 1) + j) / 2.0
        ranks[order[i:j]] = rank
        i = j
    return float((np.sum(ranks[y == 1]) - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def _ece(y, p, bins=10):
    y = np.asarray(y, dtype=np.float64); p = np.asarray(p, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y); ece = 0.0; rows = []
    for i in range(bins):
        mask = (p >= edges[i]) & ((p < edges[i + 1]) if i < bins - 1 else (p <= edges[i + 1]))
        if not np.any(mask):
            continue
        conf = float(np.mean(p[mask])); rate = float(np.mean(y[mask])); n = int(np.sum(mask))
        ece += n / total * abs(conf - rate)
        rows.append({"lo": float(edges[i]), "hi": float(edges[i + 1]), "n": n, "mean_probability": conf, "positive_rate": rate})
    return float(ece), rows


def _mean(a, mask):
    a = np.asarray(a); mask = np.asarray(mask, dtype=bool)
    return float(np.mean(a[mask])) if np.any(mask) else None


def _finite_summary(a, mask=None):
    a = np.asarray(a, dtype=np.float64)
    if mask is not None:
        a = a[np.asarray(mask, dtype=bool)]
    a = a[np.isfinite(a)]
    if not len(a):
        return {"n": 0, "mean": None, "median": None, "p90": None}
    return {"n": int(len(a)), "mean": float(np.mean(a)), "median": float(np.median(a)), "p90": float(np.percentile(a, 90))}


def _metric_sum(rows):
    rows = [r for r in rows if r is not None]
    tp = sum(int(r["true_positive"]) for r in rows); fp = sum(int(r["false_positive"]) for r in rows); fn = sum(int(r["false_negative"]) for r in rows)
    pred = sum(int(r["prediction_count"]) for r in rows); ref = sum(int(r["reference_count"]) for r in rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"f1": f1, "precision": precision, "recall": recall, "true_positive": tp, "false_positive": fp, "false_negative": fn, "prediction_reference_ratio": pred / ref if ref else None}


def summarize(args):
    reports = []
    pred_parts = []
    diag_parts = []
    for fold in range(FOLDS):
        rp = _one(args.fold_pred_dir, f"**/report-fold-{fold}.json")
        pp = _one(args.fold_pred_dir, f"**/predictions-fold-{fold}.npz")
        ap = _one(args.audit_dir, f"**/diagnostics-fold-{fold}.npz")
        reports.append(json.loads(rp.read_text()))
        with np.load(pp, allow_pickle=False) as z:
            n = len(z["global_index"])
            pred_parts.append({k: np.asarray(z[k]) for k in z.files if np.asarray(z[k]).ndim >= 1 and np.asarray(z[k]).shape[0] == n})
        with np.load(ap, allow_pickle=False) as z:
            diag_parts.append({k: np.asarray(z[k]) for k in z.files})

    predm = {k: np.concatenate([p[k] for p in pred_parts], axis=0) for k in pred_parts[0]}
    diagm = {k: np.concatenate([p[k] for p in diag_parts], axis=0) for k in diag_parts[0]}
    op = np.argsort(predm["global_index"], kind="stable"); od = np.argsort(diagm["global_index"], kind="stable")
    predm = {k: v[op] for k, v in predm.items()}; diagm = {k: v[od] for k, v in diagm.items()}
    if not np.array_equal(predm["global_index"], diagm["global_index"]):
        raise RuntimeError("prediction/audit row mismatch")
    if len(np.unique(predm["global_index"])) != len(predm["global_index"]):
        raise RuntimeError("outer rows overlap")

    k = np.asarray(predm["k"], dtype=np.int32)
    pred = np.asarray(predm["pred130"], dtype=np.int32)
    pred104 = np.asarray(predm["pred104"], dtype=np.int32)
    presence = np.asarray(predm["presence"], dtype=np.float64)
    tf_mass = np.asarray(diagm["tf_mass"], dtype=np.float64)
    candidate_mass = np.asarray(diagm["candidate_mass"], dtype=np.float64)
    active = presence >= THRESHOLD

    query_report = {}
    for q in range(QUERIES):
        y = k > q
        a = active[:, q]
        tp = a & y; fp = a & ~y; fn = ~a & y; tn = ~a & ~y
        ece, calibration = _ece(y.astype(np.int8), presence[:, q])
        by_k = {}
        for value in range(7):
            m = k == value
            by_k[str(value)] = {
                "clusters": int(np.sum(m)),
                "active_rate": _mean(a.astype(np.float64), m),
                "mean_probability": _mean(presence[:, q], m),
                "false_positive_count": int(np.sum(m & fp)),
            }
        states = {}
        for name, mask in (("tp", tp), ("fp", fp), ("fn", fn), ("tn", tn)):
            states[name] = {
                "n": int(np.sum(mask)),
                "presence": _mean(presence[:, q], mask),
                "tf_mass": _mean(tf_mass[:, q], mask),
                "candidate_mass": _mean(candidate_mass[:, q], mask),
                "candidate_entropy": _mean(diagm["candidate_entropy"][:, q], mask),
                "time_entropy": _mean(diagm["time_entropy"][:, q], mask),
            }
        query_report[str(q)] = {
            "target_positive_rate": float(np.mean(y)),
            "active_rate_at_050": float(np.mean(a)),
            "auc": _auc(y.astype(np.int8), presence[:, q]),
            "brier": float(np.mean((presence[:, q] - y.astype(np.float64)) ** 2)),
            "ece_10bin": ece,
            "precision_at_050": float(np.sum(tp) / max(1, np.sum(a))),
            "recall_at_050": float(np.sum(tp) / max(1, np.sum(y))),
            "false_positives": int(np.sum(fp)),
            "false_negatives": int(np.sum(fn)),
            "states": states,
            "by_true_k": by_k,
            "calibration_bins": calibration,
        }

    relation = np.where(pred > k, "over", np.where(pred < k, "under", "exact"))
    relation_report = {}
    for name in ("exact", "over", "under"):
        m = relation == name
        relation_report[name] = {
            "clusters": int(np.sum(m)),
            "mean_true_k": _mean(k, m),
            "mean_predicted_k": _mean(pred, m),
            "tf_background_mass": _mean(diagm["tf_background"], m),
            "candidate_background_mass": _mean(diagm["candidate_background"], m),
            "max_active_candidate_bhattacharyya": _finite_summary(diagm["max_candidate_bc"], m),
            "max_active_time_bhattacharyya": _finite_summary(diagm["max_time_bc"], m),
            "same_candidate_argmax_fraction": _finite_summary(diagm["same_candidate_argmax_fraction"], m),
        }

    event_report = {"overall": {
        "time_error_ms": _finite_summary(diagm["time_error_ms"]),
        "candidate_top1": float(np.mean(diagm["candidate_correct"][diagm["candidate_correct"] >= 0])) if np.any(diagm["candidate_correct"] >= 0) else None,
        "valid_targets": int(np.sum(diagm["event_valid"] > 0)),
    }, "by_query": {}}
    for q in range(QUERIES):
        valid = diagm["event_valid"][:, q] > 0
        cc = diagm["candidate_correct"][:, q]
        event_report["by_query"][str(q)] = {
            "valid_targets": int(np.sum(valid)),
            "time_error_ms": _finite_summary(diagm["time_error_ms"][:, q], valid),
            "candidate_top1": float(np.mean(cc[valid] == 1)) if np.any(valid) else None,
        }

    per_k = {}
    for value in range(7):
        m = k == value
        per_k[str(value)] = {
            "clusters": int(np.sum(m)),
            "v104_exact": float(np.mean(pred104[m] == value)) if np.any(m) else None,
            "v130_exact": float(np.mean(pred[m] == value)) if np.any(m) else None,
            "v130_under_rate": float(np.mean(pred[m] < value)) if np.any(m) else None,
            "v130_over_rate": float(np.mean(pred[m] > value)) if np.any(m) else None,
        }

    threshold_diagnostic = {}
    for threshold in (0.30, 0.40, 0.50, 0.60, 0.70):
        pk = np.sum(presence >= threshold, axis=1).astype(np.int32)
        threshold_diagnostic[f"{threshold:.2f}"] = {
            "diagnostic_only_not_selected": True,
            "exact_k": float(np.mean(pk == k)),
            "mae": float(np.mean(np.abs(pk - k))),
            "mean_predicted_k": float(np.mean(pk)),
            "over_rate": float(np.mean(pk > k)),
            "under_rate": float(np.mean(pk < k)),
        }

    onset = {}
    for stratum in ("aggregate", "comp", "solo", "player00", "player00_comp", "player00_rock_comp"):
        onset[stratum] = {}
        for model in ("v104", "v130"):
            onset[stratum][model] = _metric_sum([
                r["strata"].get(stratum, {}).get(model, {}).get("metrics", {}).get("global")
                for r in reports
            ])

    result = {
        "schema_version": 1,
        "protocol": {
            "five_outer_composition_folds": True,
            "every_train_row_audited_once_outer_clean": True,
            "historical_validation_or_locked12_indexed_or_evaluated": False,
            "v130_parameters_updated_during_audit": False,
            "presence_threshold_tuned": False,
            "threshold_curve_is_diagnostic_only": True,
            "annotations_used_for_audit_truth_only": True,
        },
        "data": {
            "clusters": int(len(k)),
            "mean_true_k": float(np.mean(k)),
            "mean_v130_k": float(np.mean(pred)),
            "mean_v104_k": float(np.mean(pred104)),
            "prefix_violation_rate": float(np.mean(np.asarray(predm["prefix_violation"]) > 0)),
        },
        "onset": onset,
        "per_true_k": per_k,
        "queries": query_report,
        "count_relation": relation_report,
        "event_target_quality": event_report,
        "background": {
            "true_k0_tf_background_mass": _mean(diagm["tf_background"], k == 0),
            "true_positive_tf_background_mass": _mean(diagm["tf_background"], k > 0),
            "true_k0_candidate_background_mass": _mean(diagm["candidate_background"], k == 0),
            "true_positive_candidate_background_mass": _mean(diagm["candidate_background"], k > 0),
        },
        "threshold_diagnostic": threshold_diagnostic,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "global_v104": onset["aggregate"]["v104"],
        "global_v130": onset["aggregate"]["v130"],
        "player00_rock_comp": onset["player00_rock_comp"],
        "mean_counts": result["data"],
        "query_auc": {q: query_report[q]["auc"] for q in query_report},
        "query_fp": {q: query_report[q]["false_positives"] for q in query_report},
        "count_relation": relation_report,
        "event_target_quality": event_report,
    }, indent=2, sort_keys=True))
    return 0


def parser():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    f = sub.add_parser("fold")
    f.add_argument("--cache-dir", type=Path, required=True)
    f.add_argument("--fold-dir", type=Path, required=True)
    f.add_argument("--dataset-dir", type=Path, default=Path("data/GuitarSet"))
    f.add_argument("--fold", type=int, required=True)
    f.add_argument("--output-dir", type=Path, required=True)
    s = sub.add_parser("summarize")
    s.add_argument("--fold-pred-dir", type=Path, required=True)
    s.add_argument("--audit-dir", type=Path, required=True)
    s.add_argument("--output-dir", type=Path, required=True)
    return p


def main(argv: Optional[Sequence[str]] = None):
    args = parser().parse_args(argv)
    return audit_fold(args) if args.command == "fold" else summarize(args)


if __name__ == "__main__":
    raise SystemExit(main())
