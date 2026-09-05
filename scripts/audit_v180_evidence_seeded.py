from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np

V104_F1 = 0.8038423365943571
V173_F1 = 0.7920712806963364
THRESHOLD = 0.5
EVENT_QUERIES = 6
CLUSTER_WINDOW_MS = 40.0
FUSED_FEATURE_FROM_END = 3


def _load_cache_light(cache_dir: Path):
    paths = sorted(cache_dir.rglob("v100-spectral-shard-*.npz"))
    if not paths:
        raise RuntimeError(f"no v100 spectral shards under {cache_dir}")
    seq, mask = [], []
    for p in paths:
        with np.load(p, allow_pickle=False) as z:
            seq.append(np.asarray(z["sequence"], dtype=np.float32))
            mask.append(np.asarray(z["mask"], dtype=np.float32))
    return np.concatenate(seq, axis=0), np.concatenate(mask, axis=0)


def _anchor_indices(sequence: np.ndarray, mask: np.ndarray):
    fused = sequence[:, :, -FUSED_FEATURE_FROM_END]
    score = np.where(mask > 0.5, fused, -1e9)
    order = np.argsort(-score, axis=1, kind="stable")[:, :EVENT_QUERIES]
    valid = np.take_along_axis(mask, order, axis=1) > 0.5
    return order.astype(np.int32), valid


def _load_folds(input_dir: Path):
    reports, parts = [], []
    for fold in range(5):
        rps = list(input_dir.glob(f"**/report-fold-{fold}.json"))
        if len(rps) != 1:
            raise RuntimeError(f"fold {fold}: expected one report, got {len(rps)}")
        rp = rps[0]
        report = json.loads(rp.read_text())
        reports.append(report)
        npz = rp.parent / f"predictions-fold-{fold}.npz"
        with np.load(npz, allow_pickle=False) as z:
            parts.append({
                "fold": np.full(len(z["k"]), fold, dtype=np.int16),
                "global_index": np.asarray(z["global_index"], dtype=np.int64),
                "k": np.asarray(z["k"], dtype=np.int16),
                "presence": np.asarray(z["presence"], dtype=np.float32),
                "candidate": np.asarray(z["event_candidate"], dtype=np.float32),
                "pred180": np.asarray(z["pred180_evidence_seeded"], dtype=np.int16),
                "pred104": np.asarray(z["pred104"], dtype=np.int16),
            })
    merged = {k: np.concatenate([p[k] for p in parts], axis=0) for k in parts[0]}
    order = np.argsort(merged["global_index"], kind="stable")
    merged = {k: v[order] for k, v in merged.items()}
    reports = sorted(reports, key=lambda r: int(r["outer_fold"]))
    if len(merged["k"]) != 76768 or len(np.unique(merged["global_index"])) != 76768:
        raise RuntimeError("invalid outer-clean coverage")
    return reports, merged


def _safe_corr(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _row_candidate_stats(candidate, active):
    n = len(active)
    duplicate = np.zeros(n, dtype=bool)
    entropy = np.full(n, np.nan, dtype=np.float64)
    margin = np.full(n, np.nan, dtype=np.float64)
    overlap = np.full(n, np.nan, dtype=np.float64)
    argmax = np.argmax(candidate, axis=2)
    for i in range(n):
        qs = np.flatnonzero(active[i])
        if len(qs) == 0:
            continue
        probs = np.asarray(candidate[i, qs], dtype=np.float64)
        ids = argmax[i, qs]
        duplicate[i] = len(np.unique(ids)) < len(ids)
        ent = -np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0)), axis=1)
        entropy[i] = float(np.mean(ent))
        if probs.shape[1] >= 2:
            top2 = np.partition(probs, -2, axis=1)[:, -2:]
            margin[i] = float(np.mean(np.max(top2, axis=1) - np.min(top2, axis=1)))
        if len(qs) >= 2:
            vals = []
            for a in range(len(qs)):
                for b in range(a + 1, len(qs)):
                    vals.append(float(np.sum(np.minimum(probs[a], probs[b]))))
            overlap[i] = float(np.mean(vals)) if vals else np.nan
    return argmax, duplicate, entropy, margin, overlap


def _pairwise_time_stats(times_ms, valid):
    n = len(valid)
    min_gap = np.full(n, np.nan, dtype=np.float64)
    median_gap = np.full(n, np.nan, dtype=np.float64)
    any_le_2 = np.zeros(n, dtype=bool)
    any_le_5 = np.zeros(n, dtype=bool)
    any_le_10 = np.zeros(n, dtype=bool)
    for i in range(n):
        x = np.asarray(times_ms[i][valid[i]], dtype=np.float64)
        if len(x) < 2:
            continue
        diffs = []
        for a in range(len(x)):
            for b in range(a + 1, len(x)):
                diffs.append(abs(float(x[a] - x[b])))
        d = np.asarray(diffs, dtype=np.float64)
        min_gap[i] = float(np.min(d))
        median_gap[i] = float(np.median(d))
        any_le_2[i] = bool(np.any(d <= 2.0))
        any_le_5[i] = bool(np.any(d <= 5.0))
        any_le_10[i] = bool(np.any(d <= 10.0))
    return min_gap, median_gap, any_le_2, any_le_5, any_le_10


def _slice_stats(mask, *, c, soft173, soft180, hard173, hard180, min_gap, any5, dup, entropy, margin, overlap):
    if not np.any(mask):
        return {"rows": 0}
    def mean(x):
        x = np.asarray(x)[mask]
        x = x[np.isfinite(x)] if np.issubdtype(x.dtype, np.floating) else x
        return float(np.mean(x)) if len(x) else None
    return {
        "rows": int(np.sum(mask)),
        "valid_candidate_count_mean": float(np.mean(c[mask])),
        "soft_k_v173_mean": float(np.mean(soft173[mask])),
        "soft_k_v180_mean": float(np.mean(soft180[mask])),
        "hard_k_v173_mean": float(np.mean(hard173[mask])),
        "hard_k_v180_mean": float(np.mean(hard180[mask])),
        "anchor_min_pair_gap_ms_mean": mean(min_gap),
        "anchor_any_pair_within_5ms_rate": float(np.mean(any5[mask])),
        "candidate_duplicate_rate": float(np.mean(dup[mask])),
        "candidate_entropy_mean": mean(entropy),
        "candidate_top1_margin_mean": mean(margin),
        "candidate_pair_overlap_mean": mean(overlap),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--summary-dir", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    reports, m = _load_folds(args.input_dir)
    summary = json.loads((args.summary_dir / "report.json").read_text())
    with np.load(args.summary_dir / "predictions.npz", allow_pickle=False) as z:
        sg = np.asarray(z["global_index"], dtype=np.int64)
        sk = np.asarray(z["k"], dtype=np.int16)
        p173 = np.asarray(z["presence173"], dtype=np.float64)
        p180 = np.asarray(z["presence180"], dtype=np.float64)
        pred173 = np.asarray(z["pred173"], dtype=np.int16)
        pred180_summary = np.asarray(z["pred180"], dtype=np.int16)
    if not np.array_equal(sg, m["global_index"]) or not np.array_equal(sk, m["k"]):
        raise RuntimeError("summary/fold row mismatch")
    if not np.allclose(p180, m["presence"], atol=1e-7):
        raise RuntimeError("summary/fold presence mismatch")

    sequence_all, mask_all = _load_cache_light(args.cache_dir)
    gi = m["global_index"]
    sequence = sequence_all[gi]
    cmask = mask_all[gi]
    c = np.sum(cmask > 0.5, axis=1).astype(np.int16)
    anchors, anchor_valid = _anchor_indices(sequence, cmask)
    valid_anchor_count = np.sum(anchor_valid, axis=1).astype(np.int16)
    anchor_times = np.take_along_axis(sequence[:, :, -2], anchors, axis=1) * CLUSTER_WINDOW_MS
    min_gap, median_gap, any2, any5, any10 = _pairwise_time_stats(anchor_times, anchor_valid)

    hard173 = np.sum(p173 >= THRESHOLD, axis=1).astype(np.int16)
    hard180 = np.sum(p180 >= THRESHOLD, axis=1).astype(np.int16)
    if not np.array_equal(hard180, m["pred180"]) or not np.array_equal(hard180, pred180_summary):
        raise RuntimeError("V18 hard-count decode mismatch")
    soft173 = np.sum(p173, axis=1)
    soft180 = np.sum(p180, axis=1)
    active = p180 >= THRESHOLD

    argmax, duplicate, entropy, margin, overlap = _row_candidate_stats(np.asarray(m["candidate"], dtype=np.float64), active)

    # Anchor retention and where realized candidate identities come from.
    active_pairs = active & anchor_valid
    own = argmax == anchors
    in_anchor_set = np.zeros_like(active, dtype=bool)
    for q in range(EVENT_QUERIES):
        in_anchor_set[:, q] = np.any(argmax[:, q, None] == anchors, axis=1)
    active_n = int(np.sum(active_pairs))
    own_rate = float(np.sum(active_pairs & own) / active_n) if active_n else None
    any_anchor_rate = float(np.sum(active_pairs & in_anchor_set) / active_n) if active_n else None
    other_anchor_rate = float(np.sum(active_pairs & in_anchor_set & ~own) / active_n) if active_n else None
    off_anchor_rate = float(np.sum(active_pairs & ~in_anchor_set) / active_n) if active_n else None

    k = m["k"].astype(int)
    old_correct = hard173 == k
    new_correct = hard180 == k
    lost = old_correct & ~new_correct
    gained = ~old_correct & new_correct
    both_correct = old_correct & new_correct
    both_wrong = ~old_correct & ~new_correct

    transition_by_k = {}
    per_k = {}
    for value in range(7):
        rows = k == value
        transition_by_k[str(value)] = {
            "rows": int(np.sum(rows)),
            "v173_correct_v180_wrong": int(np.sum(rows & lost)),
            "v173_wrong_v180_correct": int(np.sum(rows & gained)),
            "net_exact_rows_v180_minus_v173": int(np.sum(rows & gained) - np.sum(rows & lost)),
        }
        per_k[str(value)] = {
            "rows": int(np.sum(rows)),
            "valid_candidate_count_mean": float(np.mean(c[rows])),
            "valid_anchor_count_mean": float(np.mean(valid_anchor_count[rows])),
            "soft_k_v173_mean": float(np.mean(soft173[rows])),
            "soft_k_v180_mean": float(np.mean(soft180[rows])),
            "hard_k_v173_mean": float(np.mean(hard173[rows])),
            "hard_k_v180_mean": float(np.mean(hard180[rows])),
            "corr_candidate_count_soft_v173": _safe_corr(c[rows], soft173[rows]),
            "corr_candidate_count_soft_v180": _safe_corr(c[rows], soft180[rows]),
            "corr_candidate_count_hard_v173": _safe_corr(c[rows], hard173[rows]),
            "corr_candidate_count_hard_v180": _safe_corr(c[rows], hard180[rows]),
            "anchor_min_pair_gap_ms_mean": float(np.nanmean(min_gap[rows])) if np.any(np.isfinite(min_gap[rows])) else None,
            "anchor_any_pair_within_2ms_rate": float(np.mean(any2[rows])),
            "anchor_any_pair_within_5ms_rate": float(np.mean(any5[rows])),
            "anchor_any_pair_within_10ms_rate": float(np.mean(any10[rows])),
        }

    # Compare rows where V18 loses/gains exact K relative to V17.3.
    slices = {
        "v173_correct_v180_wrong": _slice_stats(lost, c=c, soft173=soft173, soft180=soft180, hard173=hard173, hard180=hard180, min_gap=min_gap, any5=any5, dup=duplicate, entropy=entropy, margin=margin, overlap=overlap),
        "v173_wrong_v180_correct": _slice_stats(gained, c=c, soft173=soft173, soft180=soft180, hard173=hard173, hard180=hard180, min_gap=min_gap, any5=any5, dup=duplicate, entropy=entropy, margin=margin, overlap=overlap),
        "both_correct": _slice_stats(both_correct, c=c, soft173=soft173, soft180=soft180, hard173=hard173, hard180=hard180, min_gap=min_gap, any5=any5, dup=duplicate, entropy=entropy, margin=margin, overlap=overlap),
        "both_wrong": _slice_stats(both_wrong, c=c, soft173=soft173, soft180=soft180, hard173=hard173, hard180=hard180, min_gap=min_gap, any5=any5, dup=duplicate, entropy=entropy, margin=margin, overlap=overlap),
    }

    # Exact-count poly collision confidence: are collisions uncertain or confident?
    poly_exact = (k >= 2) & new_correct
    dup_exact = poly_exact & duplicate
    clean_exact = poly_exact & ~duplicate
    collision_confidence = {
        "duplicate_exact_poly": _slice_stats(dup_exact, c=c, soft173=soft173, soft180=soft180, hard173=hard173, hard180=hard180, min_gap=min_gap, any5=any5, dup=duplicate, entropy=entropy, margin=margin, overlap=overlap),
        "clean_exact_poly": _slice_stats(clean_exact, c=c, soft173=soft173, soft180=soft180, hard173=hard173, hard180=hard180, min_gap=min_gap, any5=any5, dup=duplicate, entropy=entropy, margin=margin, overlap=overlap),
    }

    # Rank polarity: evidence rank becomes a role, but direction flips across folds.
    rank_polarity = {}
    slopes = []
    for fold in range(5):
        r = reports[fold]["v180"]["architecture"]
        rates = np.asarray(r["outer_active_rate_by_anchor_rank"], dtype=np.float64)
        corr = _safe_corr(np.arange(EVENT_QUERIES), rates)
        slopes.append(corr if corr is not None else 0.0)
        rank_polarity[str(fold)] = {
            "active_rate_by_anchor_rank": rates.tolist(),
            "rank_activity_correlation": corr,
            "gini": float(r["outer_activity_gini"]),
            "effective_slots": float(r["outer_effective_active_slots"]),
        }

    comparison = summary["comparison"]
    report = {
        "schema_version": 1,
        "protocol": {
            "source_v180_run": 33955593836,
            "outer_clean_rows": int(len(k)),
            "same_rows_as_v173": True,
            "threshold": THRESHOLD,
            "threshold_tuned": False,
            "no_retraining": True,
            "locked12_touched": False,
        },
        "headline": comparison,
        "count_transitions": {
            "v173_correct_v180_wrong_rows": int(np.sum(lost)),
            "v173_wrong_v180_correct_rows": int(np.sum(gained)),
            "net_exact_rows_v180_minus_v173": int(np.sum(gained) - np.sum(lost)),
            "by_true_k": transition_by_k,
            "slices": slices,
        },
        "candidate_count_dependence": {
            "per_true_k": per_k,
        },
        "evidence_anchor_diversity": {
            "mean_valid_candidate_count": float(np.mean(c)),
            "mean_valid_anchor_count": float(np.mean(valid_anchor_count)),
            "rows_with_fewer_than_6_valid_anchors": int(np.sum(valid_anchor_count < 6)),
            "rows_candidate_infeasible_c_lt_k": int(np.sum(c < k)),
            "overall_anchor_min_pair_gap_ms_mean": float(np.nanmean(min_gap)),
            "overall_anchor_median_pair_gap_ms_mean": float(np.nanmean(median_gap)),
            "overall_any_anchor_pair_within_2ms_rate": float(np.mean(any2)),
            "overall_any_anchor_pair_within_5ms_rate": float(np.mean(any5)),
            "overall_any_anchor_pair_within_10ms_rate": float(np.mean(any10)),
        },
        "realization_competition": {
            "active_valid_proposal_candidate_pairs": active_n,
            "realized_candidate_is_own_anchor_rate": own_rate,
            "realized_candidate_is_any_anchor_rate": any_anchor_rate,
            "realized_candidate_is_other_anchor_rate": other_anchor_rate,
            "realized_candidate_is_off_anchor_rate": off_anchor_rate,
            "poly_exact_duplicate_rate": float(np.mean(duplicate[poly_exact])) if np.any(poly_exact) else None,
            "collision_confidence": collision_confidence,
            "headline_uses_frozen_historical_realization_not_event_candidate_argmax": True,
        },
        "rank_polarity": {
            "by_fold": rank_polarity,
            "positive_direction_folds": int(np.sum(np.asarray(slopes) > 0)),
            "negative_direction_folds": int(np.sum(np.asarray(slopes) < 0)),
            "mean_abs_rank_activity_correlation": float(np.mean(np.abs(slopes))),
        },
    }

    findings = []
    findings.append("V18 genuinely reduces fixed-slot specialization, but global F1 falls on all five folds; specialization was therefore real but not the dominant headline bottleneck.")
    findings.append("Headline F1 loss is cardinality/recall loss, not raw event_candidate collision directly, because headline realization still uses the frozen historical candidate ranking.")
    findings.append("Evidence rank itself becomes a learned proposal role; the rank-to-activity direction flips across folds, revealing an unstable symmetry-breaking convention rather than stable object formation.")
    findings.append("Soft candidate competition does not guarantee one-to-one realization. Multiple active proposals can normalize around the same candidate even though each candidate distributes assignment mass across proposals plus background.")
    findings.append("The next architecture should form distinct event groups before presence/count, with explicit exclusivity or grouping semantics, instead of relying on top-score anchors plus soft competition.")
    report["findings"] = findings

    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    lines = [
        "# V18 post-audit — evidence-seeded competition",
        "",
        f"- V18 F1: {100*comparison['global_f1_v180']:.6f}%",
        f"- V17.3 F1: {100*comparison['global_f1_v173']:.6f}%",
        f"- delta: {100*comparison['delta_v180_minus_v173_f1']:+.6f} pp",
        f"- activity Gini: {comparison['activity_gini_v173']:.4f} -> {comparison['activity_gini_v180']:.4f}",
        f"- effective slots: {comparison['effective_active_slots_v173']:.3f} -> {comparison['effective_active_slots_v180']:.3f}",
        f"- poly exact: {100*comparison['poly_exact_v173']:.3f}% -> {100*comparison['poly_exact_v180']:.3f}%",
        "",
        "## Exact-K transitions vs V17.3",
        "",
        "| K | lost correct rows | gained correct rows | net |",
        "|---:|---:|---:|---:|",
    ]
    for value in range(7):
        t = transition_by_k[str(value)]
        lines.append(f"| {value} | {t['v173_correct_v180_wrong']} | {t['v173_wrong_v180_correct']} | {t['net_exact_rows_v180_minus_v173']:+d} |")
    lines += [
        "",
        "## Interpretation",
        "",
        *[f"- {x}" for x in findings],
    ]
    (args.output_dir / "audit.md").write_text("\n".join(lines) + "\n")

    print(json.dumps({
        "headline": {
            "v180_f1": comparison["global_f1_v180"],
            "delta_vs_v173_pp": 100*comparison["delta_v180_minus_v173_f1"],
            "gini_v180": comparison["activity_gini_v180"],
            "effective_slots_v180": comparison["effective_active_slots_v180"],
            "poly_exact_v180": comparison["poly_exact_v180"],
        },
        "count_transitions": report["count_transitions"],
        "anchor_diversity": report["evidence_anchor_diversity"],
        "realization": report["realization_competition"],
        "rank_polarity": report["rank_polarity"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
