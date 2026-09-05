"""Deep after-audit for V17 versus the frozen V16/V10.4 outer-clean rows.

No training, threshold tuning, candidate-ranking changes or Locked12 access.
The audit intentionally reuses the same cardinality/oracle/recoverability and
duplicate-proposal measurements used before V17.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from scripts import audit_v160_cardinality as a16
from scripts import train_v102_source_time_assignment as v102
from scripts import train_v130_causal_event_set_decoder as v130
from scripts import train_v120_integrated_birth_source_time as v120
from scripts.train_v91_ordinal_cardinality import _dataset_split
from scripts.train_v100_spectral_string_slots import _load_spectral_caches


def _per_k(k, predictions):
    out = {}
    for value in range(7):
        mask = k == value
        row = {"clusters": int(np.sum(mask))}
        for name, pred in predictions.items():
            p = np.asarray(pred)[mask]
            row[name] = {
                "exact": float(np.mean(p == value)) if len(p) else None,
                "under_rate": float(np.mean(p < value)) if len(p) else None,
                "over_rate": float(np.mean(p > value)) if len(p) else None,
                "mae": float(np.mean(np.abs(p - value))) if len(p) else None,
                "mean_predicted_k": float(np.mean(p)) if len(p) else None,
                "prediction_distribution": {str(v): int(np.sum(p == v)) for v in range(7)},
            }
        c16 = np.asarray(predictions["v160"])[mask] == value
        c17 = np.asarray(predictions["v170"])[mask] == value
        row["v16_to_v17_transition"] = {
            "both_correct": float(np.mean(c16 & c17)) if len(c16) else None,
            "v16_only_correct": float(np.mean(c16 & ~c17)) if len(c16) else None,
            "v17_only_correct": float(np.mean(~c16 & c17)) if len(c16) else None,
            "both_wrong": float(np.mean(~c16 & ~c17)) if len(c16) else None,
        }
        out[str(value)] = row
    return out


def _proposal_balance(k, presence, folds):
    presence = np.asarray(presence, dtype=np.float64)
    active = presence >= 0.5
    out = {
        "mean_presence_by_query": np.mean(presence, axis=0).tolist(),
        "active_rate_by_query": np.mean(active, axis=0).tolist(),
        "active_count_mean": float(np.mean(np.sum(active, axis=1))),
        "by_true_k": {},
        "by_fold": {},
    }
    for value in range(7):
        m = k == value
        out["by_true_k"][str(value)] = {
            "clusters": int(np.sum(m)),
            "mean_presence_by_query": np.mean(presence[m], axis=0).tolist(),
            "active_rate_by_query": np.mean(active[m], axis=0).tolist(),
        }
    for fold in range(5):
        m = folds == fold
        out["by_fold"][str(fold)] = {
            "mean_presence_by_query": np.mean(presence[m], axis=0).tolist(),
            "active_rate_by_query": np.mean(active[m], axis=0).tolist(),
        }
    return out


def _duplicates(k, merged):
    active = np.asarray(merged["presence"], dtype=np.float64) >= 0.5
    out = {}
    for value in range(2, 7):
        m = k == value
        out[str(value)] = {
            "candidate": a16._dup(np.asarray(merged["event_candidate"])[m], active[m]),
            "time": a16._dup(np.asarray(merged["event_time"])[m], active[m]),
        }
    return out


def _recoverability(cache, idx, k, dataset_dir):
    members_all = np.asarray([str(x) for x in cache["members"]])
    candidate_samples, reconstruction = v102._reconstruct_candidates(cache)
    _, time_mask, time_targets, time_sample, supervision = v102._derive_supervision(
        members_all, candidate_samples, dataset_dir, expected_slot_targets=cache["slot_targets"]
    )
    _, _, event_candidate, event_valid, true_sample, ordered_diag = v130._ordered_event_supervision(
        cache, time_mask, time_targets, time_sample,
        np.minimum(np.asarray(cache["exact"], dtype=np.int32), 6),
    )
    truth = np.asarray(true_sample[idx], dtype=np.float64)
    valid = np.asarray(event_valid[idx]) > 0.5
    rel = np.asarray(cache["sequence"][:, :, -2], dtype=np.float64) * float(v130.CLUSTER_WINDOW_SAMPLES)
    cmask = np.asarray(cache["mask"]) > 0.5
    tol = a16.TOLERANCE_MS * float(v102.SAMPLE_RATE) / 1000.0
    complete = np.zeros(len(k), dtype=bool)
    recoverable = np.zeros(len(k), dtype=bool)
    for row, gid in enumerate(idx):
        t = truth[row, valid[row]]
        complete[row] = len(t) >= int(k[row])
        if k[row] == 0:
            recoverable[row] = True
            continue
        recoverable[row] = complete[row] and a16._max_match(t, rel[gid, cmask[gid]], tol) >= int(k[row])
    return recoverable, complete, {
        "v102": supervision,
        "ordered_event": ordered_diag,
        "candidate_reconstruction": reconstruction,
        "event_candidate_target": event_candidate,
    }


def audit(args):
    r16, m16 = a16._load(args.v16_input_dir)
    r17, m17 = a16._load(args.v17_input_dir)
    for key in ("global_index", "k", "member"):
        if not np.array_equal(np.asarray(m16[key]).astype(str), np.asarray(m17[key]).astype(str)):
            raise RuntimeError(f"V16/V17 row mismatch for {key}")

    idx = np.asarray(m17["global_index"], dtype=np.int64)
    k = np.asarray(m17["k"], dtype=np.int32)
    if len(k) != 76768:
        raise RuntimeError(f"expected 76768 rows, got {len(k)}")
    predictions = {
        "v104": np.asarray(m17["pred104"], dtype=np.int32),
        "v160": np.asarray(m16["pred160"], dtype=np.int32),
        "v170": np.asarray(m17["pred170"], dtype=np.int32),
    }
    if not np.array_equal(predictions["v104"], np.asarray(m16["pred104"], dtype=np.int32)):
        raise RuntimeError("V10.4 baseline changed between runs")

    cache = _load_spectral_caches(args.cache_dir)
    _, train_split, validation = _dataset_split(args.dataset_dir)
    k_all = np.minimum(np.asarray(cache["exact"], dtype=np.int32), 6)
    if not np.array_equal(k, k_all[idx]):
        raise RuntimeError("K/cache mismatch")

    recoverable, complete, supervision = _recoverability(cache, idx, k, args.dataset_dir)
    per_k = _per_k(k, predictions)
    cards = {name: a16._card(k, pred) for name, pred in predictions.items()}

    slices = {}
    for s in ("aggregate", "comp", "solo", "player00", "player00_comp", "player00_rock_comp"):
        slices[s] = {
            "v104": a16._metric_sum(r17, s, "v104"),
            "v160": a16._metric_sum(r16, s, "v160"),
            "v170": a16._metric_sum(r17, s, "v170"),
        }

    headline_per_k = {}
    for value in range(7):
        mask = k == value
        ids = idx[mask]
        headline_per_k[str(value)] = {
            "clusters": int(np.sum(mask)),
            "v104": v120._metrics(cache, train_split, ids, predictions["v104"][mask]),
            "v160": v120._metrics(cache, train_split, ids, predictions["v160"][mask]),
            "v170": v120._metrics(cache, train_split, ids, predictions["v170"][mask]),
            "oracle_true_k": v120._metrics(cache, train_split, ids, k[mask]),
        }
    oracle = v120._metrics(cache, train_split, idx, k)

    under16 = predictions["v160"] < k
    under17 = predictions["v170"] < k
    over16 = predictions["v160"] > k
    over17 = predictions["v170"] > k
    folds16 = np.asarray(m16["outer_fold"], dtype=np.int32)
    folds17 = np.asarray(m17["outer_fold"], dtype=np.int32)

    result = {
        "schema_version": 1,
        "protocol": {
            "audit_only": True,
            "outer_clean_rows": 76768,
            "locked12_indexed_or_evaluated": False,
            "threshold_tuned": False,
            "presence_threshold": 0.5,
            "model_trained": False,
            "candidate_ranking_changed": False,
            "same_rows_v16_v17": True,
        },
        "cardinality": {"aggregate": cards, "per_true_k": per_k},
        "headline_event_metrics": {
            "slices": slices,
            "per_true_k": headline_per_k,
            "oracle_true_k_global": oracle,
        },
        "proposal_structure": {
            "v160": {
                "balance": _proposal_balance(k, m16["presence"], folds16),
                "duplicates_by_true_k": _duplicates(k, m16),
            },
            "v170": {
                "balance": _proposal_balance(k, m17["presence"], folds17),
                "duplicates_by_true_k": _duplicates(k, m17),
            },
        },
        "candidate_recoverability": {
            "all_fully_recoverable_rate_50ms": float(np.mean(recoverable)),
            "complete_event_supervision_rate": float(np.mean(complete)),
            "v160_undercount_rows": int(np.sum(under16)),
            "v170_undercount_rows": int(np.sum(under17)),
            "v160_undercount_fully_recoverable_rate_50ms": float(np.mean(recoverable[under16])) if np.any(under16) else None,
            "v170_undercount_fully_recoverable_rate_50ms": float(np.mean(recoverable[under17])) if np.any(under17) else None,
            "v160_overcount_rows": int(np.sum(over16)),
            "v170_overcount_rows": int(np.sum(over17)),
        },
        "supervision": {
            **supervision,
            "validation_tracks_not_evaluated": len(validation),
        },
        "comparison": {
            "v170_minus_v160_global_f1": slices["aggregate"]["v170"]["f1"] - slices["aggregate"]["v160"]["f1"],
            "v170_minus_v104_global_f1": slices["aggregate"]["v170"]["f1"] - slices["aggregate"]["v104"]["f1"],
            "v170_minus_v160_poly_exact": cards["v170"]["poly_accuracy"] - cards["v160"]["poly_accuracy"],
            "v170_minus_v104_poly_exact": cards["v170"]["poly_accuracy"] - cards["v104"]["poly_accuracy"],
            "v170_minus_v160_k5_exact": per_k["5"]["v170"]["exact"] - per_k["5"]["v160"]["exact"],
            "v170_minus_v160_k6_exact": per_k["6"]["v170"]["exact"] - per_k["6"]["v160"]["exact"],
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    summary = {
        "v104_global_f1": slices["aggregate"]["v104"]["f1"],
        "v160_global_f1": slices["aggregate"]["v160"]["f1"],
        "v170_global_f1": slices["aggregate"]["v170"]["f1"],
        "v104_poly_exact": cards["v104"]["poly_accuracy"],
        "v160_poly_exact": cards["v160"]["poly_accuracy"],
        "v170_poly_exact": cards["v170"]["poly_accuracy"],
        "v170_k2": per_k["2"]["v170"]["exact"],
        "v170_k3": per_k["3"]["v170"]["exact"],
        "v170_k4": per_k["4"]["v170"]["exact"],
        "v170_k5": per_k["5"]["v170"]["exact"],
        "v170_k6": per_k["6"]["v170"]["exact"],
        "v170_undercount_fully_recoverable_rate_50ms": result["candidate_recoverability"]["v170_undercount_fully_recoverable_rate_50ms"],
        "oracle_true_k_global_f1": oracle["global"]["f1"],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return result


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("dataset_dir", nargs="?", type=Path, default=Path("data/GuitarSet"))
    p.add_argument("--v16-input-dir", type=Path, required=True)
    p.add_argument("--v17-input-dir", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    return p


def main(argv: Optional[Sequence[str]] = None):
    audit(parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
