"""V19 pre-architecture audit: can raw candidate hypotheses be grouped into births cleanly?

Training-only supervision is reconstructed exactly as V13/V17.3 already does.
For each positive cluster, every valid raw candidate is assigned to its nearest
true birth in causal cluster time (Voronoi partition). This introduces no runtime
annotation and no arbitrary distance threshold. The audit asks whether those
candidate groups are sufficiently complete/separable to justify a learned
candidate-affinity graph before implementing V19.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for p in (ROOT, SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from scripts.train_v90_structured_cluster_cardinality import CLUSTER_WINDOW_SAMPLES
from scripts.train_v100_spectral_string_slots import _load_spectral_caches
from scripts.train_v91_ordinal_cardinality import _dataset_split
from scripts import train_v102_source_time_assignment as v102
from scripts import train_v130_causal_event_set_decoder as v130

EVENT_QUERIES = 6


def _auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    pos = int(labels.sum()); neg = int((~labels).sum())
    if not pos or not neg:
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and scores[order[j]] == scores[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + 1 + j)
        i = j
    rank_sum = float(ranks[labels].sum())
    return float((rank_sum - pos * (pos + 1) / 2.0) / (pos * neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_dir", type=Path)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cache = _load_spectral_caches(args.cache_dir)
    _, train_split, validation = _dataset_split(args.dataset_dir)
    members = np.asarray([str(x) for x in cache["members"]], dtype="U96")
    candidate_samples, reconstruction = v102._reconstruct_candidates(cache)
    pitch_targets, time_mask, string_time_targets, time_sample, supervision = v102._derive_supervision(
        members, candidate_samples, args.dataset_dir, expected_slot_targets=cache["slot_targets"]
    )
    k = np.minimum(np.asarray(cache["exact"], dtype=np.int32), EVENT_QUERIES)
    present, event_time, event_candidate, event_valid, true_sample, event_diag = v130._ordered_event_supervision(
        cache, time_mask, string_time_targets, time_sample, k
    )

    seq = np.asarray(cache["sequence"], dtype=np.float32)
    mask = np.asarray(cache["mask"], dtype=np.float32) > 0.5
    rel = np.asarray(seq[:, :, -2], dtype=np.float64)
    fused = np.asarray(seq[:, :, -3], dtype=np.float64)

    rows = len(k)
    exact_group_coverage = np.zeros(rows, dtype=bool)
    distinct_truth_groups = np.zeros(rows, dtype=np.int16)
    nearest_target_collision = np.zeros(rows, dtype=bool)
    top6_distinct_groups = np.zeros(rows, dtype=np.int16)
    top6_covers_all = np.zeros(rows, dtype=bool)
    pair_same_scores = []
    pair_same_labels = []
    per_k = {}

    for r in range(rows):
        kr = int(k[r])
        valid = np.flatnonzero(mask[r])
        if kr <= 0 or len(valid) == 0:
            exact_group_coverage[r] = (kr == 0)
            top6_covers_all[r] = (kr == 0)
            continue
        truth = np.asarray(true_sample[r, :kr], dtype=np.float64)
        truth = truth[np.isfinite(truth)]
        if len(truth) != kr:
            continue
        cand_sample = np.clip(rel[r, valid], 0.0, 1.0) * float(CLUSTER_WINDOW_SAMPLES)
        d = np.abs(cand_sample[:, None] - truth[None, :])
        group = np.argmin(d, axis=1).astype(np.int16)
        distinct = int(len(np.unique(group)))
        distinct_truth_groups[r] = distinct
        exact_group_coverage[r] = (distinct == kr)

        true_nearest = []
        for q in range(kr):
            true_nearest.append(int(valid[np.argmin(np.abs(cand_sample - truth[q]))]))
        nearest_target_collision[r] = len(set(true_nearest)) < len(true_nearest)

        order = valid[np.argsort(-fused[r, valid], kind="stable")[:EVENT_QUERIES]]
        order_samples = np.clip(rel[r, order], 0.0, 1.0) * float(CLUSTER_WINDOW_SAMPLES)
        order_groups = np.argmin(np.abs(order_samples[:, None] - truth[None, :]), axis=1)
        top6_distinct_groups[r] = int(len(np.unique(order_groups)))
        top6_covers_all[r] = (top6_distinct_groups[r] == kr)

        if len(valid) >= 2:
            ii, jj = np.triu_indices(len(valid), k=1)
            if len(ii) > 256:
                sel = np.linspace(0, len(ii)-1, 256, dtype=np.int64)
                ii, jj = ii[sel], jj[sel]
            dt = np.abs(cand_sample[ii] - cand_sample[jj])
            same = group[ii] == group[jj]
            pair_same_scores.append(-dt)
            pair_same_labels.append(same)

    pair_scores = np.concatenate(pair_same_scores) if pair_same_scores else np.empty(0)
    pair_labels = np.concatenate(pair_same_labels) if pair_same_labels else np.empty(0, dtype=bool)

    for value in range(7):
        m = k == value
        per_k[str(value)] = {
            "rows": int(m.sum()),
            "mean_valid_candidates": float(mask[m].sum(axis=1).mean()) if np.any(m) else None,
            "mean_distinct_truth_groups": float(distinct_truth_groups[m].mean()) if np.any(m) else None,
            "exact_group_coverage_rate": float(exact_group_coverage[m].mean()) if np.any(m) else None,
            "nearest_truth_representative_collision_rate": float(nearest_target_collision[m].mean()) if np.any(m) else None,
            "top6_fused_covers_all_truth_groups_rate": float(top6_covers_all[m].mean()) if np.any(m) else None,
            "mean_top6_distinct_groups": float(top6_distinct_groups[m].mean()) if np.any(m) else None,
        }

    positive = k > 0
    poly = k >= 2
    report = {
        "schema_version": 1,
        "protocol": {
            "training_universe_rows": int(rows),
            "training_only_annotations_used_for_audit": True,
            "runtime_annotations_required": False,
            "group_target": "each valid candidate assigned to nearest true birth in causal cluster time; no distance threshold",
            "locked12_touched": False,
            "no_training": True,
        },
        "supervision": {
            "event_target_diag": event_diag,
            "source_supervision": supervision,
            "candidate_reconstruction": reconstruction,
        },
        "headline": {
            "positive_rows": int(positive.sum()),
            "poly_rows": int(poly.sum()),
            "candidate_group_exact_coverage_positive": float(exact_group_coverage[positive].mean()),
            "candidate_group_exact_coverage_poly": float(exact_group_coverage[poly].mean()),
            "nearest_truth_representative_collision_poly": float(nearest_target_collision[poly].mean()),
            "top6_fused_covers_all_truth_groups_poly": float(top6_covers_all[poly].mean()),
            "pair_samples": int(len(pair_labels)),
            "pair_same_fraction": float(pair_labels.mean()) if len(pair_labels) else None,
            "time_only_same_birth_auc": _auc(pair_scores, pair_labels) if len(pair_labels) else None,
        },
        "per_true_k": per_k,
        "findings": [
            "Nearest-truth Voronoi grouping is threshold-free and available only during training/audit.",
            "If exact group coverage is high, raw candidates can support event groups before presence/count.",
            "Top-6 fused group coverage quantifies the structural duplicate-anchor problem measured in V18.",
            "Time-only pair AUC is a lower-complexity baseline; V19 should learn affinity from candidate/acoustic context rather than hard-code a millisecond threshold.",
        ],
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (args.output_dir / "audit.md").write_text(
        "# V19 candidate-group supervision audit\n\n"
        f"- positive exact group coverage: {100*report['headline']['candidate_group_exact_coverage_positive']:.4f}%\n"
        f"- poly exact group coverage: {100*report['headline']['candidate_group_exact_coverage_poly']:.4f}%\n"
        f"- poly nearest-representative collision: {100*report['headline']['nearest_truth_representative_collision_poly']:.4f}%\n"
        f"- poly top-6 fused covers all truth groups: {100*report['headline']['top6_fused_covers_all_truth_groups_poly']:.4f}%\n"
        f"- time-only pair same-birth AUC: {report['headline']['time_only_same_birth_auc']:.6f}\n"
    )
    print(json.dumps(report["headline"], indent=2, sort_keys=True))
    print(json.dumps(per_k, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
