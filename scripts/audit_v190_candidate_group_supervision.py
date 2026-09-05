"""V19 pre-architecture audit.

Part A tests the hypothesis raw candidates -> birth groups using a threshold-free
nearest-truth Voronoi partition available only during training/audit.
Part B tests a stronger alternative: anonymous birth centers on the causal
23 x 64 time-frequency observation. True centers use training-only birth time
and MIDI fundamental; runtime would use only the spectral map.
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
from scripts import train_v100_spectral_string_slots as v100
from scripts.train_v100_spectral_string_slots import _load_spectral_caches
from scripts.train_v91_ordinal_cardinality import _dataset_split
from scripts import train_v101_string_query_attention as v101
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


def _nearest_index(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)
    return np.argmin(np.abs(values[:, None] - grid[None, :]), axis=1)


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

    # Dense center representability. We audit only rows where V10.2 training
    # supervision contains exactly K individual births; the eligible rate is
    # reported separately so no missing-label row is silently treated as clean.
    dense_eligible = np.zeros(rows, dtype=bool)
    dense_23x64_exact = np.zeros(rows, dtype=bool)
    dense_23x8_exact = np.zeros(rows, dtype=bool)
    time23_exact = np.zeros(rows, dtype=bool)
    freq64_exact = np.zeros(rows, dtype=bool)
    fine_freq = np.geomspace(v100.MIN_HZ, v100.MAX_HZ, v100.SPECTRAL_BANDS).astype(np.float64)
    coarse_freq = np.geomspace(v100.MIN_HZ, v100.MAX_HZ, 8).astype(np.float64)
    time_grid = np.asarray(v102.FRAME_CENTER_SAMPLES, dtype=np.float64)

    for r in range(rows):
        kr = int(k[r])
        valid = np.flatnonzero(mask[r])
        if kr <= 0 or len(valid) == 0:
            exact_group_coverage[r] = (kr == 0)
            top6_covers_all[r] = (kr == 0)
        else:
            truth = np.asarray(true_sample[r, :kr], dtype=np.float64)
            truth = truth[np.isfinite(truth)]
            if len(truth) == kr:
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

        active = np.flatnonzero(np.asarray(time_mask[r]) > 0.5)
        if kr == 0:
            dense_eligible[r] = True
            dense_23x64_exact[r] = dense_23x8_exact[r] = True
            time23_exact[r] = freq64_exact[r] = True
        elif len(active) == kr:
            samples = np.asarray(time_sample[r, active], dtype=np.float64)
            midi = np.asarray(pitch_targets[r, active], dtype=np.float64) * float(v101.PITCH_SCALE)
            good = np.all(np.isfinite(samples)) and np.all(np.isfinite(midi))
            if good:
                dense_eligible[r] = True
                hz = 440.0 * np.power(2.0, (midi - 69.0) / 12.0)
                ti = _nearest_index(samples, time_grid)
                fi64 = _nearest_index(np.log(np.clip(hz, 1e-9, None)), np.log(fine_freq))
                fi8 = _nearest_index(np.log(np.clip(hz, 1e-9, None)), np.log(coarse_freq))
                dense_23x64_exact[r] = len(set(zip(ti.tolist(), fi64.tolist()))) == kr
                dense_23x8_exact[r] = len(set(zip(ti.tolist(), fi8.tolist()))) == kr
                time23_exact[r] = len(np.unique(ti)) == kr
                freq64_exact[r] = len(np.unique(fi64)) == kr

    pair_scores = np.concatenate(pair_same_scores) if pair_same_scores else np.empty(0)
    pair_labels = np.concatenate(pair_same_labels) if pair_same_labels else np.empty(0, dtype=bool)

    per_k = {}
    for value in range(7):
        m = k == value
        em = m & dense_eligible
        per_k[str(value)] = {
            "rows": int(m.sum()),
            "mean_valid_candidates": float(mask[m].sum(axis=1).mean()) if np.any(m) else None,
            "mean_distinct_truth_groups": float(distinct_truth_groups[m].mean()) if np.any(m) else None,
            "exact_group_coverage_rate": float(exact_group_coverage[m].mean()) if np.any(m) else None,
            "nearest_truth_representative_collision_rate": float(nearest_target_collision[m].mean()) if np.any(m) else None,
            "top6_fused_covers_all_truth_groups_rate": float(top6_covers_all[m].mean()) if np.any(m) else None,
            "mean_top6_distinct_groups": float(top6_distinct_groups[m].mean()) if np.any(m) else None,
            "dense_supervision_eligible_rate": float(dense_eligible[m].mean()) if np.any(m) else None,
            "dense_23x64_exact_rate_on_eligible": float(dense_23x64_exact[em].mean()) if np.any(em) else None,
            "dense_23x8_exact_rate_on_eligible": float(dense_23x8_exact[em].mean()) if np.any(em) else None,
            "time23_exact_rate_on_eligible": float(time23_exact[em].mean()) if np.any(em) else None,
            "freq64_exact_rate_on_eligible": float(freq64_exact[em].mean()) if np.any(em) else None,
        }

    positive = k > 0
    poly = k >= 2
    eligible_poly = poly & dense_eligible
    report = {
        "schema_version": 2,
        "protocol": {
            "training_universe_rows": int(rows),
            "training_only_annotations_used_for_audit": True,
            "runtime_annotations_required": False,
            "candidate_group_target": "each valid candidate assigned to nearest true birth in causal cluster time; no distance threshold",
            "dense_center_target": "nearest 23-frame time center x nearest log-frequency band of true MIDI fundamental; training-only",
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
            "dense_supervision_eligible_poly_rate": float(dense_eligible[poly].mean()),
            "dense_23x64_exact_poly_on_eligible": float(dense_23x64_exact[eligible_poly].mean()) if np.any(eligible_poly) else None,
            "dense_23x8_exact_poly_on_eligible": float(dense_23x8_exact[eligible_poly].mean()) if np.any(eligible_poly) else None,
            "time23_exact_poly_on_eligible": float(time23_exact[eligible_poly].mean()) if np.any(eligible_poly) else None,
            "freq64_exact_poly_on_eligible": float(freq64_exact[eligible_poly].mean()) if np.any(eligible_poly) else None,
        },
        "per_true_k": per_k,
        "findings": [
            "Raw-candidate Voronoi groups are not a complete event representation at high K when exact coverage falls materially below 1.",
            "A dense time-frequency center field can split births even when candidate representatives collide, provided quantized center collisions remain rare.",
            "Time-only center representability is audited separately because simultaneous different-pitch births require a frequency axis.",
            "The 23x8 coarse grid tests whether V10.2 compressed TF tokens retain enough object separation; 23x64 tests the original spectral map resolution.",
        ],
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (args.output_dir / "audit.md").write_text(
        "# V19 representation audit\n\n"
        f"- candidate-group poly exact coverage: {100*report['headline']['candidate_group_exact_coverage_poly']:.4f}%\n"
        f"- top-6 fused covers all truth groups: {100*report['headline']['top6_fused_covers_all_truth_groups_poly']:.4f}%\n"
        f"- dense supervision eligible poly: {100*report['headline']['dense_supervision_eligible_poly_rate']:.4f}%\n"
        f"- 23x64 center exact on eligible poly: {100*report['headline']['dense_23x64_exact_poly_on_eligible']:.4f}%\n"
        f"- 23x8 center exact on eligible poly: {100*report['headline']['dense_23x8_exact_poly_on_eligible']:.4f}%\n"
        f"- time-only 23-frame exact on eligible poly: {100*report['headline']['time23_exact_poly_on_eligible']:.4f}%\n"
        f"- frequency-only 64-band exact on eligible poly: {100*report['headline']['freq64_exact_poly_on_eligible']:.4f}%\n"
    )
    print(json.dumps(report["headline"], indent=2, sort_keys=True))
    print(json.dumps(per_k, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
