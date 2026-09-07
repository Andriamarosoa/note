"""V23 pre-audit: explicit cardinality + structured dense transport feasibility.

Analysis-only. No model training, no threshold tuning, no Locked12 access.
Measures whether the next architecture is justified before implementation:
  1) exact-K oracle upside with the existing realization/evaluation path;
  2) dense 23x64 birth-center injectivity, including coarser frequency resolutions;
  3) candidate-memory injective realization within 50 ms;
  4) previously measured frozen V19 conv3 hard-negative signal from V21.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from scripts import train_v100_spectral_string_slots as v100
from scripts import train_v102_source_time_assignment as v102
from scripts import train_v120_integrated_birth_source_time as v120
from scripts import train_v130_causal_event_set_decoder as v130
from scripts import train_v190_dense_birth_centers as v190
from scripts.train_v91_ordinal_cardinality import _dataset_split

OUTER_ROWS = 76768
TOLERANCE_MS = 50.0
FREQ_RESOLUTIONS = (1, 8, 16, 32, 64)


def one(root: Path, pattern: str) -> Path:
    xs = sorted(root.glob(pattern))
    if len(xs) != 1:
        raise RuntimeError(f"expected one {pattern}, got {len(xs)}")
    return xs[0]


def load_v190(root: Path):
    reports, parts = [], []
    for fold in range(5):
        rp = one(root, f"**/report-fold-{fold}.json")
        pp = one(root, f"**/predictions-fold-{fold}.npz")
        reports.append(json.loads(rp.read_text()))
        with np.load(pp, allow_pickle=False) as z:
            n = len(z["global_index"])
            parts.append({k: np.asarray(z[k]) for k in z.files if np.asarray(z[k]).ndim and len(np.asarray(z[k])) == n})
    common = set(parts[0])
    for p in parts[1:]:
        common &= set(p)
    merged = {k: np.concatenate([p[k] for p in parts], axis=0) for k in common}
    order = np.argsort(merged["global_index"], kind="stable")
    merged = {k: v[order] for k, v in merged.items()}
    idx = np.asarray(merged["global_index"], dtype=np.int64)
    if len(idx) != OUTER_ROWS or len(np.unique(idx)) != OUTER_ROWS:
        raise RuntimeError(f"invalid V19 outer coverage: {len(idx)} / {len(np.unique(idx))}")
    return reports, merged


def aggregate_metric(reports, model_key: str):
    rows = [r["strata"]["aggregate"][model_key]["metrics"]["global"] for r in reports]
    tp = sum(int(r["true_positive"]) for r in rows)
    fp = sum(int(r["false_positive"]) for r in rows)
    fn = sum(int(r["false_negative"]) for r in rows)
    pred = sum(int(r["prediction_count"]) for r in rows)
    ref = sum(int(r["reference_count"]) for r in rows)
    p = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * rec / (p + rec) if p + rec else 0.0
    return {
        "f1": f1,
        "precision": p,
        "recall": rec,
        "prediction_count": pred,
        "reference_count": ref,
        "prediction_reference_ratio": pred / ref if ref else None,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
    }


def card_metrics(k, pred):
    k = np.asarray(k, dtype=np.int32); pred = np.asarray(pred, dtype=np.int32)
    poly = k >= 2
    return {
        "exact": float(np.mean(pred == k)),
        "poly_exact": float(np.mean(pred[poly] == k[poly])) if np.any(poly) else None,
        "mae": float(np.mean(np.abs(pred - k))),
        "under_rate": float(np.mean(pred < k)),
        "over_rate": float(np.mean(pred > k)),
        "mean_true_k": float(np.mean(k)),
        "mean_predicted_k": float(np.mean(pred)),
        "per_true_k": {
            str(v): {
                "rows": int(np.sum(k == v)),
                "exact": float(np.mean(pred[k == v] == v)) if np.any(k == v) else None,
                "mean_predicted_k": float(np.mean(pred[k == v])) if np.any(k == v) else None,
            }
            for v in range(7)
        },
    }


def max_match(truth: Iterable[float], candidates: Iterable[float], tolerance: float) -> int:
    truth = [float(x) for x in truth if math.isfinite(float(x))]
    candidates = [float(x) for x in candidates if math.isfinite(float(x))]
    edges = [[j for j, c in enumerate(candidates) if abs(c - t) <= tolerance] for t in truth]
    owner = {}

    def aug(ti: int, seen: set[int]) -> bool:
        for cj in edges[ti]:
            if cj in seen:
                continue
            seen.add(cj)
            if cj not in owner or aug(owner[cj], seen):
                owner[cj] = ti
                return True
        return False

    return sum(int(aug(ti, set())) for ti in range(len(edges)))


def coarse_band(band: np.ndarray, bins: int) -> np.ndarray:
    band = np.asarray(band, dtype=np.int32)
    return np.minimum(bins - 1, (band * bins) // int(v100.SPECTRAL_BANDS))


def dense_resolution_stats(target, eligible, k, idx):
    target = np.asarray(target)
    eligible = np.asarray(eligible) > 0.5
    k = np.asarray(k, dtype=np.int32)
    idx = np.asarray(idx, dtype=np.int64)
    out = {}
    for bins in FREQ_RESOLUTIONS:
        ok = np.zeros(len(idx), dtype=bool)
        distinct_count = np.zeros(len(idx), dtype=np.int16)
        for j, gid in enumerate(idx):
            if not eligible[gid] or k[gid] <= 0:
                continue
            flats = np.flatnonzero(target[gid] > 0)
            t = flats // int(v100.SPECTRAL_BANDS)
            f = flats % int(v100.SPECTRAL_BANDS)
            cell = t * bins + coarse_band(f, bins)
            distinct_count[j] = len(np.unique(cell))
            ok[j] = distinct_count[j] == int(k[gid])
        pos = eligible[idx] & (k[idx] > 0)
        poly = eligible[idx] & (k[idx] >= 2)
        row = {
            "grid": [int(v100.TIME_FRAMES), int(bins)],
            "eligible_positive_rows": int(np.sum(pos)),
            "eligible_poly_rows": int(np.sum(poly)),
            "exact_injective_positive": float(np.mean(ok[pos])) if np.any(pos) else None,
            "exact_injective_poly": float(np.mean(ok[poly])) if np.any(poly) else None,
            "per_true_k": {},
        }
        for value in range(1, 7):
            m = eligible[idx] & (k[idx] == value)
            row["per_true_k"][str(value)] = {
                "rows": int(np.sum(m)),
                "exact_injective": float(np.mean(ok[m])) if np.any(m) else None,
                "mean_distinct_cells": float(np.mean(distinct_count[m])) if np.any(m) else None,
            }
        out[str(bins)] = row
    return out


def dense_geometry(target, eligible, k, idx):
    rows = []
    same_time = adjacent_3x3 = hard_window = 0
    for gid in np.asarray(idx, dtype=np.int64):
        if not eligible[gid] or int(k[gid]) < 2:
            continue
        flats = np.flatnonzero(target[gid] > 0)
        if len(flats) != int(k[gid]):
            continue
        t = flats // int(v100.SPECTRAL_BANDS)
        f = flats % int(v100.SPECTRAL_BANDS)
        pair = []
        st = a3 = hw = False
        for i in range(len(flats)):
            for j in range(i + 1, len(flats)):
                dt = abs(int(t[i]) - int(t[j])); df = abs(int(f[i]) - int(f[j]))
                pair.append((dt, df, max(dt, df), dt + df, math.hypot(dt, df)))
                st |= dt == 0
                a3 |= dt <= 1 and df <= 1
                hw |= dt <= 1 and df <= 2
        if not pair:
            continue
        p = np.asarray(pair, dtype=np.float64)
        rows.append([np.min(p[:, c]) for c in range(5)])
        same_time += int(st); adjacent_3x3 += int(a3); hard_window += int(hw)
    arr = np.asarray(rows, dtype=np.float64)
    def pack(col):
        return {
            "p10": float(np.quantile(arr[:, col], .10)),
            "median": float(np.quantile(arr[:, col], .50)),
            "p90": float(np.quantile(arr[:, col], .90)),
        }
    return {
        "poly_distinct_rows": int(len(arr)),
        "row_rate_any_same_time": same_time / len(arr) if len(arr) else None,
        "row_rate_any_pair_within_3x3": adjacent_3x3 / len(arr) if len(arr) else None,
        "row_rate_any_pair_within_dt1_df2": hard_window / len(arr) if len(arr) else None,
        "minimum_pair_dt_frames": pack(0) if len(arr) else None,
        "minimum_pair_df_bands": pack(1) if len(arr) else None,
        "minimum_pair_chebyshev": pack(2) if len(arr) else None,
        "minimum_pair_manhattan": pack(3) if len(arr) else None,
        "minimum_pair_euclidean": pack(4) if len(arr) else None,
    }


def candidate_recoverability(cache, idx, k, event_valid, true_sample):
    idx = np.asarray(idx, dtype=np.int64); k = np.asarray(k, dtype=np.int32)
    valid = np.asarray(event_valid)[idx] > 0.5
    truth = np.asarray(true_sample, dtype=np.float64)[idx]
    rel = np.asarray(cache["sequence"][:, :, -2], dtype=np.float64) * float(v130.CLUSTER_WINDOW_SAMPLES)
    cmask = np.asarray(cache["mask"]) > 0.5
    tol = TOLERANCE_MS * float(v102.SAMPLE_RATE) / 1000.0
    complete = np.zeros(len(idx), dtype=bool)
    recoverable = np.zeros(len(idx), dtype=bool)
    matched = np.zeros(len(idx), dtype=np.int16)
    for r, gid in enumerate(idx):
        kr = int(k[gid])
        t = truth[r, valid[r]]
        complete[r] = len(t) >= kr
        if kr == 0:
            recoverable[r] = True
            continue
        n = max_match(t, rel[gid, cmask[gid]], tol)
        matched[r] = n
        recoverable[r] = complete[r] and n >= kr
    kk = k[idx]
    poly = kk >= 2
    result = {
        "tolerance_ms": TOLERANCE_MS,
        "complete_supervision_all": float(np.mean(complete)),
        "fully_injective_candidate_recoverable_all": float(np.mean(recoverable)),
        "fully_injective_candidate_recoverable_poly": float(np.mean(recoverable[poly])) if np.any(poly) else None,
        "per_true_k": {},
    }
    for value in range(7):
        m = kk == value
        result["per_true_k"][str(value)] = {
            "rows": int(np.sum(m)),
            "complete_supervision": float(np.mean(complete[m])) if np.any(m) else None,
            "fully_recoverable": float(np.mean(recoverable[m])) if np.any(m) else None,
            "mean_matched_births": float(np.mean(matched[m])) if np.any(m) else None,
        }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_dir", type=Path)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--v190-fold-dir", type=Path, required=True)
    ap.add_argument("--v210-audit-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)

    reports, merged = load_v190(args.v190_fold_dir)
    idx = np.asarray(merged["global_index"], dtype=np.int64)
    cache = v100._load_spectral_caches(args.cache_dir)
    members = np.asarray([str(x) for x in cache["members"]])
    k_all = np.minimum(np.asarray(cache["exact"], dtype=np.int32), 6)
    if not np.array_equal(np.asarray(merged["k"], dtype=np.int32), k_all[idx]):
        raise RuntimeError("V19 K/cache mismatch")
    if not np.array_equal(np.asarray(merged["member"]).astype(str), members[idx].astype(str)):
        raise RuntimeError("V19 member/cache mismatch")

    _, train_split, _ = _dataset_split(args.dataset_dir)
    candidates, candidate_reconstruction = v102._reconstruct_candidates(cache)
    pitch, time_mask, time_targets, time_sample, supervision = v102._derive_supervision(
        members, candidates, args.dataset_dir, expected_slot_targets=cache["slot_targets"]
    )
    target, center_eligible, center_distinct = v190._birth_center_targets(cache, pitch, time_targets, k_all)
    ep, et, ec, ev, true_sample, ordered_diag = v130._ordered_event_supervision(
        cache, time_mask, time_targets, time_sample, k_all
    )

    actual_k = np.sum(np.asarray(merged["presence"], dtype=np.float64) >= 0.5, axis=1).astype(np.int32)
    actual_card = card_metrics(k_all[idx], actual_k)
    actual_v19 = aggregate_metric(reports, v190.MODEL_KEY)
    true_k_oracle = v120._metrics(cache, train_split, idx, k_all[idx])

    resolutions = dense_resolution_stats(target, center_eligible, k_all, idx)
    geometry = dense_geometry(target, center_eligible, k_all, idx)
    candidate = candidate_recoverability(cache, idx, k_all, ev, true_sample)

    v210 = json.loads(one(args.v210_audit_dir, "**/report.json").read_text())
    if not v210.get("protocol", {}).get("v210_v190_latent_conv_audit"):
        raise RuntimeError("wrong V21 audit artifact")
    hard = v210["pooled_outer"]["cell_birth_mean_fold_auc"]["hard"]
    conv_probe = v210["pooled_outer"]["conv_cardinality_probe"]

    oracle_gain = float(true_k_oracle["f1"] - actual_v19["f1"])
    dense_poly = float(resolutions["64"]["exact_injective_poly"])
    candidate_poly = float(candidate["fully_injective_candidate_recoverable_poly"])
    hard_auc = float(hard["conv_probe_auc"])
    gates = {
        "true_k_oracle_material_gain": oracle_gain >= 0.10,
        "dense_23x64_injective_poly_at_least_98pct": dense_poly >= 0.98,
        "candidate_memory_injective_poly_at_least_97pct": candidate_poly >= 0.97,
        "conv3_hard_negative_auc_at_least_0_70": hard_auc >= 0.70,
    }
    gates["proceed_to_v23_architecture"] = bool(all(gates.values()))

    result = {
        "schema_version": 1,
        "protocol": {
            "v230_pre_audit_cardinality_transport": True,
            "analysis_only": True,
            "model_training": False,
            "threshold_tuning": False,
            "outer_clean_rows": int(len(idx)),
            "canonical_v190_run": 33995600250,
            "canonical_v210_run": 34034935120,
            "frozen_nested_spectral_run": 33647694565,
            "locked12_indexed_or_evaluated": False,
        },
        "cardinality": {
            "actual_v19": actual_card,
            "actual_v19_global": actual_v19,
            "true_k_oracle_global": true_k_oracle,
            "true_k_oracle_f1_gain_over_v19": oracle_gain,
            "v21_frozen_conv_linear_probe": conv_probe,
        },
        "dense_transport": {
            "center_eligible_positive_rows_outer": int(np.sum((center_eligible[idx] > 0.5) & (k_all[idx] > 0))),
            "original_distinct_center_poly_rate": float(np.mean(center_distinct[idx][(center_eligible[idx] > 0.5) & (k_all[idx] >= 2)] == k_all[idx][(center_eligible[idx] > 0.5) & (k_all[idx] >= 2)])),
            "resolution_injectivity": resolutions,
            "geometry": geometry,
        },
        "candidate_memory": candidate,
        "latent_signal": {
            "v21_hard_negative_conv_auc": hard_auc,
            "v21_hard_negative_raw_patch_auc": float(hard["raw_patch_probe_auc"]),
            "v19_center_head_hard_negative_auc": float(hard["v19_center_head_auc"]),
        },
        "gates": gates,
        "supervision": {
            "candidate_reconstruction": candidate_reconstruction,
            "source": supervision,
            "ordered_event": ordered_diag,
        },
    }
    (args.output_dir / "report.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "cardinality": result["cardinality"],
        "dense_transport": {
            "original_distinct_center_poly_rate": result["dense_transport"]["original_distinct_center_poly_rate"],
            "resolution_injectivity": resolutions,
            "geometry": geometry,
        },
        "candidate_memory": candidate,
        "latent_signal": result["latent_signal"],
        "gates": gates,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
