"""Deep post-V17.3 audit: count calibration vs proposal-slot specialization.

Analysis only. It consumes the canonical V17.2-C and V17.3 fold/summary
artifacts, touches no training data, does not tune the 0.5 runtime threshold,
and never indexes Locked12.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np

SLOT_COUNT = 6
THRESHOLD = 0.5


def _load_json(path: Path):
    return json.loads(path.read_text())


def _find_one(root: Path, pattern: str) -> Path:
    hits = sorted(root.glob(pattern))
    if len(hits) != 1:
        raise RuntimeError(f"expected one {pattern!r} under {root}, got {len(hits)}")
    return hits[0]


def _load_folds(root: Path, version: str):
    reports, parts = [], []
    for fold in range(5):
        rp = _find_one(root, f"**/report-fold-{fold}.json")
        npz = _find_one(root, f"**/predictions-fold-{fold}.npz")
        report = _load_json(rp)
        if int(report["outer_fold"]) != fold:
            raise RuntimeError(f"fold mismatch in {rp}")
        if version == "v172":
            assert report["protocol"]["assignment_arm"] == "mass_permutation"
            assert report["protocol"]["permutation_invariant_set_matching"] is True
            assert report["protocol"]["presence_threshold"] == THRESHOLD
        elif version == "v173":
            assert report["protocol"]["v173_poibin_count_consistency"] is True
            assert report["protocol"]["v173_base_arm"] == "mass_permutation"
            assert report["protocol"]["runtime_presence_threshold"] == THRESHOLD
            assert report["protocol"]["runtime_decode_unchanged_from_v172_c"] is True
        else:
            raise RuntimeError(version)
        reports.append(report)
        with np.load(npz, allow_pickle=False) as z:
            n = len(z["global_index"])
            part = {
                k: np.asarray(z[k])
                for k in z.files
                if np.asarray(z[k]).ndim and len(np.asarray(z[k])) == n
            }
        parts.append(part)

    common = set(parts[0])
    for part in parts[1:]:
        common &= set(part)
    merged = {k: np.concatenate([p[k] for p in parts], axis=0) for k in common}
    order = np.argsort(merged["global_index"], kind="stable")
    merged = {k: v[order] for k, v in merged.items()}
    if len(merged["global_index"]) != 76768:
        raise RuntimeError(f"{version}: expected 76768 outer rows")
    if len(np.unique(merged["global_index"])) != 76768:
        raise RuntimeError(f"{version}: duplicate outer rows")
    return reports, merged


def _poibin_distribution(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-9, 1.0 - 1e-9)
    dist = np.zeros((len(p), SLOT_COUNT + 1), dtype=np.float64)
    dist[:, 0] = 1.0
    for q in range(SLOT_COUNT):
        pq = p[:, q : q + 1]
        shifted = np.concatenate([np.zeros((len(p), 1)), dist[:, :-1]], axis=1)
        dist = dist * (1.0 - pq) + shifted * pq
    return dist


def _poibin_logit_gradient(p: np.ndarray, k: np.ndarray) -> np.ndarray:
    """d[-log P(N=K)] / d logits for the six independent Bernoullis."""
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-9, 1.0 - 1e-9)
    k = np.asarray(k, dtype=np.int32)
    full = _poibin_distribution(p)
    target = np.clip(full[np.arange(len(k)), k], 1e-12, None)
    out = np.zeros_like(p)
    for i in range(SLOT_COUNT):
        d = np.zeros((len(p), SLOT_COUNT), dtype=np.float64)
        d[:, 0] = 1.0
        for j in range(SLOT_COUNT):
            if j == i:
                continue
            pj = p[:, j : j + 1]
            shifted = np.concatenate([np.zeros((len(p), 1)), d[:, :-1]], axis=1)
            d = d * (1.0 - pj) + shifted * pj
        pk = np.zeros(len(p), dtype=np.float64)
        valid = k <= SLOT_COUNT - 1
        pk[valid] = d[np.arange(len(p))[valid], k[valid]]
        pkm1 = np.zeros(len(p), dtype=np.float64)
        validm = k >= 1
        pkm1[validm] = d[np.arange(len(p))[validm], k[validm] - 1]
        dp = pkm1 - pk
        out[:, i] = -(dp / target) * p[:, i] * (1.0 - p[:, i])
    return out


def _gini(values: Iterable[float]) -> float:
    x = np.sort(np.asarray(list(values), dtype=np.float64))
    if not np.any(x):
        return 0.0
    n = len(x)
    return float(2.0 * np.sum(np.arange(1, n + 1) * x) / (n * np.sum(x)) - (n + 1.0) / n)


def _effective_slots(values: Iterable[float]) -> float:
    x = np.asarray(list(values), dtype=np.float64)
    den = float(np.sum(x * x))
    return float(np.sum(x) ** 2 / den) if den else 0.0


def _dup(dist: np.ndarray, active: np.ndarray) -> dict:
    any_dup, pair_rates, max_cos = [], [], []
    for row in range(len(active)):
        ids = np.flatnonzero(active[row])
        if len(ids) < 2:
            continue
        arg = np.argmax(dist[row, ids], axis=1)
        pairs = dup = 0
        mc = 0.0
        for ii in range(len(ids)):
            a = np.asarray(dist[row, ids[ii]], dtype=np.float64)
            na = np.linalg.norm(a)
            for jj in range(ii + 1, len(ids)):
                b = np.asarray(dist[row, ids[jj]], dtype=np.float64)
                pairs += 1
                dup += int(arg[ii] == arg[jj])
                den = na * np.linalg.norm(b)
                mc = max(mc, float(np.dot(a, b) / den) if den else 0.0)
        any_dup.append(dup > 0)
        pair_rates.append(dup / pairs)
        max_cos.append(mc)
    return {
        "rows_with_2plus_active": int(len(any_dup)),
        "any_duplicate_argmax_rate": float(np.mean(any_dup)) if any_dup else None,
        "mean_duplicate_pair_rate": float(np.mean(pair_rates)) if pair_rates else None,
        "mean_max_pair_cosine": float(np.mean(max_cos)) if max_cos else None,
    }


def _count_confusion(k: np.ndarray, pred: np.ndarray):
    return [
        [int(np.sum((k == y) & (pred == p))) for p in range(SLOT_COUNT + 1)]
        for y in range(SLOT_COUNT + 1)
    ]


def _transition(a: np.ndarray, b: np.ndarray):
    return [
        [int(np.sum((a == x) & (b == y))) for y in range(SLOT_COUNT + 1)]
        for x in range(SLOT_COUNT + 1)
    ]


def _count_diagnostics(k: np.ndarray, p: np.ndarray) -> dict:
    hard = np.sum(p >= THRESHOLD, axis=1).astype(np.int32)
    dist = _poibin_distribution(p)
    mode = np.argmax(dist, axis=1).astype(np.int32)
    grad = _poibin_logit_gradient(p, k)
    grad_l1 = np.sum(np.abs(grad), axis=1)
    sorted_p = np.sort(p, axis=1)[:, ::-1]

    per_k = {}
    total_grad = float(np.sum(grad_l1))
    for value in range(SLOT_COUNT + 1):
        mask = k == value
        target_prob = dist[mask, value]
        row = {
            "clusters": int(np.sum(mask)),
            "prevalence": float(np.mean(mask)),
            "soft_sum_mean": float(np.mean(np.sum(p[mask], axis=1))),
            "hard_mean_k": float(np.mean(hard[mask])),
            "hard_exact": float(np.mean(hard[mask] == value)),
            "poibin_mode_mean_k": float(np.mean(mode[mask])),
            "poibin_mode_exact": float(np.mean(mode[mask] == value)),
            "true_count_probability_mean": float(np.mean(target_prob)),
            "true_count_nll_mean": float(np.mean(-np.log(np.clip(target_prob, 1e-12, 1.0)))),
            "count_gradient_l1_mean": float(np.mean(grad_l1[mask])),
            "count_gradient_l1_share": float(np.sum(grad_l1[mask]) / total_grad) if total_grad else None,
            "sorted_presence_mean": np.mean(sorted_p[mask], axis=0).tolist(),
        }
        if 1 <= value <= 5:
            row["true_k_boundary_gap_mean"] = float(
                np.mean(sorted_p[mask, value - 1] - sorted_p[mask, value])
            )
        elif value == 0:
            row["true_k_boundary_gap_mean"] = float(np.mean(THRESHOLD - sorted_p[mask, 0]))
        else:
            row["true_k_boundary_gap_mean"] = float(np.mean(sorted_p[mask, -1] - THRESHOLD))
        per_k[str(value)] = row
    return {
        "soft_presence_mass": float(np.mean(np.sum(p, axis=1))),
        "hard_mean_k": float(np.mean(hard)),
        "hard_exact": float(np.mean(hard == k)),
        "hard_poly_exact": float(np.mean(hard[k >= 2] == k[k >= 2])),
        "poibin_mode_exact": float(np.mean(mode == k)),
        "poibin_mode_poly_exact": float(np.mean(mode[k >= 2] == k[k >= 2])),
        "hard_confusion": _count_confusion(k, hard),
        "per_true_k": per_k,
    }


def _slot_specialization(reports: list, merged: dict, model_block: str) -> dict:
    out = {}
    all_gini, all_eff, all_corr = [], [], []
    folds = np.asarray(merged["outer_fold"], dtype=np.int32)
    for fold in range(5):
        report = next(r for r in reports if int(r["outer_fold"]) == fold)
        block = report[model_block]
        active = np.asarray(
            [block["outer_presence"][str(q)]["active_rate_at_0p5"] for q in range(SLOT_COUNT)],
            dtype=np.float64,
        )
        meanp = np.asarray(
            [block["outer_presence"][str(q)]["mean"] for q in range(SLOT_COUNT)],
            dtype=np.float64,
        )
        occupancy = np.asarray(
            [block["outer_match_occupancy"][str(q)]["matched_object_rate"] for q in range(SLOT_COUNT)],
            dtype=np.float64,
        )
        fm = folds == fold
        kf = np.asarray(merged["k"], dtype=np.int32)[fm]
        survival = np.asarray([np.mean(kf >= rank) for rank in range(1, SLOT_COUNT + 1)])
        sorted_active = np.sort(active)[::-1]
        corr = float(np.corrcoef(active, occupancy)[0, 1])
        gini = _gini(active)
        eff = _effective_slots(active)
        all_corr.append(corr)
        all_gini.append(gini)
        all_eff.append(eff)
        out[str(fold)] = {
            "active_rate_by_query": active.tolist(),
            "mean_presence_by_query": meanp.tolist(),
            "matched_object_rate_by_query": occupancy.tolist(),
            "sorted_active_rate": sorted_active.tolist(),
            "truth_survival_p_k_ge_rank": survival.tolist(),
            "sorted_activity_vs_truth_survival_mae": float(np.mean(np.abs(sorted_active - survival))),
            "active_occupancy_correlation": corr,
            "active_rate_gini": gini,
            "effective_active_slots": eff,
        }
    return {
        "by_fold": out,
        "mean_active_rate_gini": float(np.mean(all_gini)),
        "mean_effective_active_slots": float(np.mean(all_eff)),
        "min_active_occupancy_correlation": float(np.min(all_corr)),
        "mean_active_occupancy_correlation": float(np.mean(all_corr)),
    }


def _paired_exact_transitions(k: np.ndarray, old: np.ndarray, new: np.ndarray):
    out = {}
    for value in range(SLOT_COUNT + 1):
        mask = k == value
        old_ok = old == value
        new_ok = new == value
        fixed = int(np.sum(mask & ~old_ok & new_ok))
        broken = int(np.sum(mask & old_ok & ~new_ok))
        out[str(value)] = {
            "clusters": int(np.sum(mask)),
            "fixed": fixed,
            "broken": broken,
            "net_exact_rows": fixed - broken,
            "changed_predicted_k": int(np.sum(mask & (old != new))),
        }
    return out


def audit(args):
    _ = _load_json(args.v172_summary / "report.json")
    r173 = _load_json(args.v173_summary / "report.json")
    reports172, m172 = _load_folds(args.v172_folds, "v172")
    reports173, m173 = _load_folds(args.v173_folds, "v173")

    for key in ("global_index", "k", "member", "pred104"):
        if not np.array_equal(np.asarray(m172[key]).astype(str), np.asarray(m173[key]).astype(str)):
            raise RuntimeError(f"V17.2/V17.3 row mismatch: {key}")

    k = np.asarray(m173["k"], dtype=np.int32)
    p172 = np.asarray(m172["presence"], dtype=np.float64)
    p173 = np.asarray(m173["presence"], dtype=np.float64)
    hard172 = np.sum(p172 >= THRESHOLD, axis=1).astype(np.int32)
    hard173 = np.sum(p173 >= THRESHOLD, axis=1).astype(np.int32)

    c172 = _count_diagnostics(k, p172)
    c173 = _count_diagnostics(k, p173)
    spec172 = _slot_specialization(reports172, m172, "v172")
    spec173 = _slot_specialization(reports173, m173, "v173")

    duplicates = {}
    for value in range(2, SLOT_COUNT + 1):
        mask = k == value
        duplicates[str(value)] = {}
        for name, merged, pres in (("v172", m172, p172), ("v173", m173, p173)):
            active = pres[mask] >= THRESHOLD
            duplicates[str(value)][name] = {
                "candidate": _dup(np.asarray(merged["event_candidate"])[mask], active),
                "time": _dup(np.asarray(merged["event_time"])[mask], active),
            }

    truth_mean = float(np.mean(k))
    old_excess = c172["soft_presence_mass"] - truth_mean
    new_excess = c173["soft_presence_mass"] - truth_mean
    excess_removed = old_excess - new_excess
    excess_removed_fraction = excess_removed / old_excess if old_excess else None

    gshares = c173["per_true_k"]
    common_grad_share = gshares["0"]["count_gradient_l1_share"] + gshares["1"]["count_gradient_l1_share"]
    tail_grad_share = gshares["5"]["count_gradient_l1_share"] + gshares["6"]["count_gradient_l1_share"]

    comparison = r173["comparison"]
    findings = {
        "global_f1_gain_v173_vs_v172_pp": 100.0 * comparison["delta_v173_minus_v172_f1"],
        "global_f1_gap_v173_vs_v104_pp": 100.0 * comparison["delta_v173_minus_v104_f1"],
        "fold_wins_v173_vs_v172": int(comparison["folds_v173_beats_v172"]),
        "poly_exact_delta_pp": 100.0 * comparison["delta_poly_v173_minus_v172"],
        "soft_mass_excess_v172": old_excess,
        "soft_mass_excess_v173": new_excess,
        "soft_mass_excess_removed_fraction": float(excess_removed_fraction),
        "count_nll_gradient_share_k0_k1": float(common_grad_share),
        "count_nll_gradient_share_k5_k6": float(tail_grad_share),
        "v173_mean_slot_activity_gini": spec173["mean_active_rate_gini"],
        "v173_mean_effective_active_slots_of_6": spec173["mean_effective_active_slots"],
        "v173_min_activity_occupancy_correlation": spec173["min_active_occupancy_correlation"],
        "v160_v17_query_specific_parameterization_persists": True,
        "dominant_measured_cause": (
            "Permutation-invariant supervision did not make the proposal parameterization exchangeable: "
            "query-specific proposal/objectness branches retain a strong ordinal-like specialization. "
            "The Poisson-binomial term improves global calibration slightly but is symmetric and cannot "
            "repair ownership/allocation across specialized proposal slots; its gradient mass is also "
            "dominated by common K=0/1 rows."
        ),
        "next_architecture_test": (
            "share the objectness classifier across all six proposal features (same hidden+sigmoid weights), "
            "while keeping V17.3 mass-preserving matching/count objective/runtime threshold unchanged"
        ),
    }

    assert comparison["delta_v173_minus_v172_f1"] > 0.0
    assert comparison["delta_poly_v173_minus_v172"] < 0.0
    assert c173["soft_presence_mass"] / truth_mean - 1.0 > 0.15
    assert excess_removed_fraction < 0.10
    assert spec173["mean_active_rate_gini"] > 0.50
    assert spec173["mean_effective_active_slots"] < 3.20
    assert spec173["min_active_occupancy_correlation"] > 0.99
    assert common_grad_share > 0.60
    assert tail_grad_share < 0.04
    assert comparison["delta_k2_exact_v173_minus_v172"] < 0.0
    assert comparison["delta_k5_exact_v173_minus_v172"] < 0.0
    assert comparison["delta_k6_exact_v173_minus_v172"] < 0.0

    result = {
        "schema_version": 1,
        "protocol": {
            "audit_only": True,
            "model_trained": False,
            "threshold_tuned": False,
            "presence_threshold": THRESHOLD,
            "locked12_indexed_or_evaluated": False,
            "same_76768_outer_clean_rows": True,
            "same_seed": 16061,
        },
        "canonical_summary_comparison": comparison,
        "strata": r173["strata"],
        "v172_count": c172,
        "v173_count": c173,
        "paired_count_transition_v172_rows_to_v173_columns": _transition(hard172, hard173),
        "paired_exact_transitions_by_true_k": _paired_exact_transitions(k, hard172, hard173),
        "slot_specialization": {"v172": spec172, "v173": spec173},
        "duplicates_by_true_k": duplicates,
        "findings": findings,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (args.output_dir / "summary.json").write_text(json.dumps(findings, indent=2, sort_keys=True) + "\n")
    print(json.dumps(findings, indent=2, sort_keys=True))
    return result


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--v172-folds", type=Path, required=True)
    p.add_argument("--v173-folds", type=Path, required=True)
    p.add_argument("--v172-summary", type=Path, required=True)
    p.add_argument("--v173-summary", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    return p


def main():
    audit(parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
