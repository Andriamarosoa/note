"""Deep audit of V16 outer-clean cardinality before any next architecture.

Analysis-only: no training, no threshold tuning, no Locked12 access, no change to
candidate ranking. Reconstructs the training-only event targets so V16's saved
outer predictions can be audited correctly.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
from typing import Optional, Sequence

import numpy as np

from scripts import train_v102_source_time_assignment as v102
from scripts import train_v130_causal_event_set_decoder as v130
from scripts import train_v120_integrated_birth_source_time as v120
from scripts.train_v91_ordinal_cardinality import _dataset_split
from scripts.train_v100_spectral_string_slots import _load_spectral_caches

SLOT_COUNT = 6
THRESHOLD = 0.5
TOLERANCE_MS = 50.0


def _load(input_dir: Path):
    reports, parts = [], []
    for fold in range(5):
        rp = sorted(input_dir.glob(f"**/report-fold-{fold}.json"))
        npz = sorted(input_dir.glob(f"**/predictions-fold-{fold}.npz"))
        if len(rp) != 1 or len(npz) != 1:
            raise RuntimeError(f"fold {fold}: expected one report and one prediction shard")
        reports.append(json.loads(rp[0].read_text()))
        with np.load(npz[0], allow_pickle=False) as z:
            n = len(z["global_index"])
            # Ignore metadata arrays such as schema_version=(1,). The old V13
            # summarizer incorrectly tried to reorder them as row arrays.
            parts.append({k: np.asarray(z[k]) for k in z.files if np.asarray(z[k]).ndim and len(z[k]) == n})
    common = set(parts[0])
    for p in parts[1:]:
        common &= set(p)
    merged = {k: np.concatenate([p[k] for p in parts]) for k in common}
    order = np.argsort(merged["global_index"], kind="stable")
    merged = {k: v[order] for k, v in merged.items()}
    if len(np.unique(merged["global_index"])) != len(merged["global_index"]):
        raise RuntimeError("outer rows overlap")
    return reports, merged


def _confusion(k, pred):
    out = np.zeros((7, 7), dtype=np.int64)
    for y, p in zip(k, pred):
        out[int(y), int(p)] += 1
    return out


def _card(k, pred):
    poly = k >= 2
    return {
        "clusters": int(len(k)),
        "accuracy": float(np.mean(pred == k)),
        "poly_accuracy": float(np.mean(pred[poly] == k[poly])) if np.any(poly) else None,
        "mae": float(np.mean(np.abs(pred - k))),
        "under_rate": float(np.mean(pred < k)),
        "over_rate": float(np.mean(pred > k)),
        "mean_true_k": float(np.mean(k)),
        "mean_predicted_k": float(np.mean(pred)),
        "predicted_histogram": {str(v): int(np.sum(pred == v)) for v in range(7)},
        "target_histogram": {str(v): int(np.sum(k == v)) for v in range(7)},
        "confusion_true_rows_pred_columns": _confusion(k, pred).tolist(),
    }


def _auc(y, score):
    y = np.asarray(y, dtype=bool)
    score = np.asarray(score, dtype=np.float64)
    n1, n0 = int(np.sum(y)), int(np.sum(~y))
    if not n1 or not n0:
        return None
    order = np.argsort(score, kind="stable")
    ss = score[order]
    ranks = np.empty(len(score), dtype=np.float64)
    i = 0
    while i < len(score):
        j = i + 1
        while j < len(score) and ss[j] == ss[i]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    return float((np.sum(ranks[y]) - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def _ap(y, score):
    y = np.asarray(y, dtype=bool)
    n = int(np.sum(y))
    if not n:
        return None
    yy = y[np.argsort(-np.asarray(score), kind="stable")]
    precision = np.cumsum(yy) / np.arange(1, len(yy) + 1)
    return float(np.sum(precision[yy]) / n)


def _presence(k, p):
    out = {}
    for q in range(6):
        y = k >= q + 1
        s = p[:, q]
        a = s >= THRESHOLD
        tp, fp, fn = int(np.sum(a & y)), int(np.sum(a & ~y)), int(np.sum(~a & y))
        out[str(q)] = {
            "ordered_target": f"K>={q + 1}",
            "positive_rows": int(np.sum(y)),
            "prevalence": float(np.mean(y)),
            "auc": _auc(y, s),
            "average_precision": _ap(y, s),
            "positive_mean": float(np.mean(s[y])),
            "negative_mean": float(np.mean(s[~y])),
            "precision_at_0_5": tp / (tp + fp) if tp + fp else None,
            "recall_at_0_5": tp / (tp + fn) if tp + fn else None,
            "positive_below_0_5": float(np.mean(s[y] < 0.5)),
            "positive_quantiles": {str(x): float(np.quantile(s[y], x)) for x in (0.1, 0.25, 0.5, 0.75, 0.9)},
        }
    return out


def _per_k(k, p104, p160, presence):
    out = {}
    for value in range(7):
        m = k == value
        row = {"clusters": int(np.sum(m))}
        for name, pred in (("v104", p104), ("v160", p160)):
            pp = pred[m]
            row[name] = {
                "exact": float(np.mean(pp == value)),
                "under_rate": float(np.mean(pp < value)),
                "over_rate": float(np.mean(pp > value)),
                "mae": float(np.mean(np.abs(pp - value))),
                "mean_predicted_k": float(np.mean(pp)),
                "prediction_distribution": {str(v): int(np.sum(pp == v)) for v in range(7)},
            }
        c104, c160 = p104[m] == value, p160[m] == value
        row["transition"] = {
            "both_correct": float(np.mean(c104 & c160)),
            "v104_only_correct": float(np.mean(c104 & ~c160)),
            "v160_only_correct": float(np.mean(~c104 & c160)),
            "both_wrong": float(np.mean(~c104 & ~c160)),
        }
        row["v160_presence"] = {
            "mean": np.mean(presence[m], axis=0).tolist(),
            "active_rate_at_0_5": np.mean(presence[m] >= 0.5, axis=0).tolist(),
        }
        out[str(value)] = row
    return out


def _metric_sum(reports, stratum, model):
    rows = []
    for r in reports:
        sr = r["strata"].get(stratum)
        if sr is not None:
            rows.append(sr[model]["metrics"]["global"])
    tp = sum(int(r["true_positive"]) for r in rows)
    fp = sum(int(r["false_positive"]) for r in rows)
    fn = sum(int(r["false_negative"]) for r in rows)
    pred = sum(int(r["prediction_count"]) for r in rows)
    ref = sum(int(r["reference_count"]) for r in rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"f1": f1, "precision": precision, "recall": recall, "prediction_count": pred,
            "reference_count": ref, "prediction_reference_ratio": pred / ref if ref else None,
            "true_positive": tp, "false_positive": fp, "false_negative": fn}


def _dup(dist, active):
    any_dup, pair_rates, max_cos = [], [], []
    for row in range(len(active)):
        ids = np.flatnonzero(active[row])
        if len(ids) < 2:
            continue
        arg = np.argmax(dist[row, ids], axis=1)
        pairs = dup = 0
        mc = 0.0
        for i in range(len(ids)):
            a = dist[row, ids[i]].astype(np.float64)
            na = np.linalg.norm(a)
            for j in range(i + 1, len(ids)):
                b = dist[row, ids[j]].astype(np.float64)
                pairs += 1
                dup += int(arg[i] == arg[j])
                den = na * np.linalg.norm(b)
                mc = max(mc, float(np.dot(a, b) / den) if den else 0.0)
        any_dup.append(dup > 0)
        pair_rates.append(dup / pairs)
        max_cos.append(mc)
    return {"rows": len(any_dup), "any_duplicate_argmax_rate": float(np.mean(any_dup)) if any_dup else None,
            "mean_duplicate_pair_rate": float(np.mean(pair_rates)) if pair_rates else None,
            "mean_max_pair_cosine": float(np.mean(max_cos)) if max_cos else None}


def _max_match(truth, candidates, tolerance):
    truth = [float(x) for x in truth if math.isfinite(float(x))]
    candidates = [float(x) for x in candidates]
    edges = [[j for j, c in enumerate(candidates) if abs(c - t) <= tolerance] for t in truth]
    owner = {}
    def aug(ti, seen):
        for cj in edges[ti]:
            if cj in seen:
                continue
            seen.add(cj)
            if cj not in owner or aug(owner[cj], seen):
                owner[cj] = ti
                return True
        return False
    return sum(int(aug(ti, set())) for ti in range(len(edges)))


def _correct_event_diag(cache, idx, k, pred_time, pred_candidate, event_valid, true_sample, candidate_target):
    valid = event_valid[idx] > 0.5
    truth = true_sample[idx].astype(np.float64)
    target_c = candidate_target[idx]
    expected = np.sum(pred_time * v102.FRAME_CENTER_SAMPLES[None, None, :], axis=2)
    err = np.abs(expected - truth) * 1000.0 / float(v102.SAMPLE_RATE)
    pc, tc = np.argmax(pred_candidate, axis=2), np.argmax(target_c, axis=2)
    overall = {"targets": int(np.sum(valid)), "time_mae_ms": float(np.mean(err[valid])),
               "time_median_ms": float(np.median(err[valid])), "time_p90_ms": float(np.percentile(err[valid], 90)),
               "candidate_top1": float(np.mean(pc[valid] == tc[valid]))}

    rel = np.asarray(cache["sequence"][:, :, -2], dtype=np.float64) * float(v130.CLUSTER_WINDOW_SAMPLES)
    cmask = np.asarray(cache["mask"]) > 0.5
    tol = TOLERANCE_MS * float(v102.SAMPLE_RATE) / 1000.0
    complete = np.zeros(len(k), dtype=bool)
    recoverable = np.zeros(len(k), dtype=bool)
    recover_n = np.zeros(len(k), dtype=np.int16)
    per_k = {}
    for row, gid in enumerate(idx):
        t = truth[row, valid[row]]
        complete[row] = len(t) >= int(k[row])
        if k[row] == 0:
            recoverable[row] = True
            continue
        n = _max_match(t, rel[gid, cmask[gid]], tol)
        recover_n[row] = n
        recoverable[row] = complete[row] and n >= int(k[row])
    for value in range(7):
        m = k == value
        vm = valid[m]
        ee, pcc, tcc = err[m], pc[m], tc[m]
        per_k[str(value)] = {"clusters": int(np.sum(m)),
            "complete_event_supervision_rate": float(np.mean(complete[m])),
            "fully_candidate_recoverable_rate_50ms": float(np.mean(recoverable[m])),
            "mean_recoverable_births": float(np.mean(recover_n[m])),
            "valid_event_targets": int(np.sum(vm)),
            "time_mae_ms": float(np.mean(ee[vm])) if np.any(vm) else None,
            "candidate_top1": float(np.mean(pcc[vm] == tcc[vm])) if np.any(vm) else None}
    return overall, per_k, recoverable, complete


def audit(args):
    reports, m = _load(args.input_dir)
    idx = np.asarray(m["global_index"], dtype=np.int64)
    k = np.asarray(m["k"], dtype=np.int32)
    p104, p160 = np.asarray(m["pred104"], dtype=np.int32), np.asarray(m["pred160"], dtype=np.int32)
    presence = np.asarray(m["presence"], dtype=np.float64)
    if len(k) != 76768:
        raise RuntimeError(f"expected 76768 outer rows, got {len(k)}")

    cache = _load_spectral_caches(args.cache_dir)
    _, train_split, validation = _dataset_split(args.dataset_dir)
    members_all = np.asarray([str(x) for x in cache["members"]])
    k_all = np.minimum(np.asarray(cache["exact"], dtype=np.int32), 6)
    if not np.array_equal(k, k_all[idx]):
        raise RuntimeError("K/cache mismatch")
    if not np.array_equal(np.asarray(m["member"]).astype(str), members_all[idx]):
        raise RuntimeError("member/cache mismatch")

    candidate_samples, reconstruction = v102._reconstruct_candidates(cache)
    _, time_mask, time_targets, time_sample, supervision = v102._derive_supervision(
        members_all, candidate_samples, args.dataset_dir, expected_slot_targets=cache["slot_targets"])
    ep, et, ec, ev, ts, ordered_diag = v130._ordered_event_supervision(
        cache, time_mask, time_targets, time_sample, k_all)
    event_overall, event_per_k, recoverable, complete = _correct_event_diag(
        cache, idx, k, np.asarray(m["event_time"], dtype=np.float64),
        np.asarray(m["event_candidate"], dtype=np.float64), ev, ts, ec)

    headline_k = {}
    for value in range(7):
        mask = k == value
        ids = idx[mask]
        headline_k[str(value)] = {"clusters": int(np.sum(mask)),
            "v104": v120._metrics(cache, train_split, ids, p104[mask]),
            "v160": v120._metrics(cache, train_split, ids, p160[mask]),
            "oracle_true_k": v120._metrics(cache, train_split, ids, k[mask])}
    oracle_global = v120._metrics(cache, train_split, idx, k)

    active = presence >= 0.5
    duplicates = {}
    for value in range(2, 7):
        mask = k == value
        duplicates[str(value)] = {
            "candidate": _dup(np.asarray(m["event_candidate"])[mask], active[mask]),
            "time": _dup(np.asarray(m["event_time"])[mask], active[mask])}

    qfold = {}
    folds = np.asarray(m["outer_fold"], dtype=np.int32)
    for fold in range(5):
        fm = folds == fold
        qfold[str(fold)] = {}
        for q in (3, 4, 5):
            y, s = k[fm] >= q + 1, presence[fm, q]
            qfold[str(fold)][str(q)] = {"positive_rows": int(np.sum(y)), "positive_mean": float(np.mean(s[y])),
                "auc": _auc(y, s), "recall_at_0_5": float(np.mean(s[y] >= 0.5)),
                "false_positive_rate_at_0_5": float(np.mean(s[~y] >= 0.5))}

    f1 = {}
    for s in ("aggregate", "comp", "solo", "player00", "player00_comp", "player00_rock_comp"):
        f1[s] = {"v104": _metric_sum(reports, s, "v104"), "v160": _metric_sum(reports, s, "v160")}

    under = p160 < k
    result = {"schema_version": 1,
        "protocol": {"audit_only": True, "outer_clean_rows": 76768, "locked12_indexed_or_evaluated": False,
                     "threshold_tuned": False, "presence_threshold": 0.5, "model_trained": False,
                     "candidate_ranking_changed": False},
        "measurement_integrity": {"legacy_fold_event_diagnostics_are_invalid": True,
            "reason": "V13 train_fold passes supervision event_time/event_candidate to _event_diagnostics instead of saved outer predictions.",
            "correct_outer_event_diagnostics": event_overall},
        "cardinality": {"v104": _card(k, p104), "v160": _card(k, p160),
                        "per_true_k": _per_k(k, p104, p160, presence)},
        "headline_event_metrics": {"slices": f1, "per_true_k": headline_k, "oracle_true_k_global": oracle_global},
        "ordered_presence": {"queries": _presence(k, presence), "tail_by_fold": qfold,
            "prefix_violation_rate": float(np.mean(np.asarray(m["prefix_violation"]) > 0)),
            "positive_counts_if_ordered": {str(q): int(np.sum(k >= q + 1)) for q in range(6)},
            "mean_positive_opportunity_if_exchangeable": float(np.sum(k) / 6.0)},
        "proposal_quality": {"correct_event_diagnostics_by_true_k": event_per_k, "duplicates_by_true_k": duplicates,
            "candidate_recoverability": {"all_fully_recoverable_rate_50ms": float(np.mean(recoverable)),
                "undercount_rows": int(np.sum(under)),
                "undercount_fully_recoverable_rate_50ms": float(np.mean(recoverable[under])) if np.any(under) else None,
                "undercount_complete_event_supervision_rate": float(np.mean(complete[under])) if np.any(under) else None}},
        "supervision": {"v102": supervision, "ordered_event": ordered_diag, "candidate_reconstruction": reconstruction,
                        "validation_tracks_not_evaluated": len(validation)}}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    summary = {"v104_global_f1": f1["aggregate"]["v104"]["f1"], "v160_global_f1": f1["aggregate"]["v160"]["f1"],
        "oracle_true_k_global_f1": oracle_global["global"]["f1"], "v104_poly_exact": result["cardinality"]["v104"]["poly_accuracy"],
        "v160_poly_exact": result["cardinality"]["v160"]["poly_accuracy"], "v160_k5_exact": result["cardinality"]["per_true_k"]["5"]["v160"]["exact"],
        "v160_k6_exact": result["cardinality"]["per_true_k"]["6"]["v160"]["exact"], "q5_auc": result["ordered_presence"]["queries"]["5"]["auc"],
        "q5_recall_at_0_5": result["ordered_presence"]["queries"]["5"]["recall_at_0_5"],
        "correct_candidate_top1": event_overall["candidate_top1"], "correct_time_mae_ms": event_overall["time_mae_ms"],
        "undercount_fully_recoverable_rate_50ms": result["proposal_quality"]["candidate_recoverability"]["undercount_fully_recoverable_rate_50ms"]}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return result


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("dataset_dir", nargs="?", type=Path, default=Path("data/GuitarSet"))
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    return p


def main(argv: Optional[Sequence[str]] = None):
    audit(parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
