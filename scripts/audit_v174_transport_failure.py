"""Deep post-V17.4 audit of continuous candidate-capacity transport.

Analysis only. Consumes canonical V17.3 and V17.4 outer-clean fold artifacts.
It does not train, tune thresholds, touch Locked12, or alter model behavior.

Primary question: can V17.4 satisfy its continuous column-capacity constraint while
still allowing multiple active queries to share the same discrete argmax candidate?
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

SLOTS = 6
THRESHOLD = 0.5
EPS = 1e-12


def _one(root: Path, pattern: str) -> Path:
    hits = sorted(root.glob(pattern))
    if len(hits) != 1:
        raise RuntimeError(f"expected one {pattern!r} under {root}, got {len(hits)}")
    return hits[0]


def _load_json(path: Path):
    return json.loads(path.read_text())


def _load_version(root: Path, version: str):
    reports, parts = [], []
    for fold in range(5):
        rp = _one(root, f"**/report-fold-{fold}.json")
        npz = _one(root, f"**/predictions-fold-{fold}.npz")
        report = _load_json(rp)
        if int(report["outer_fold"]) != fold:
            raise RuntimeError(f"fold mismatch: {rp}")
        p = report.get("protocol", {})
        if version == "v173":
            assert p.get("v173_poibin_count_consistency") is True
            assert p.get("runtime_presence_threshold") == THRESHOLD
        elif version == "v174":
            assert p.get("v174_presence_mass_candidate_transport") is True
            assert p.get("runtime_presence_threshold") == THRESHOLD
            assert p.get("candidate_transport_presence_stop_gradient") is True
        else:
            raise RuntimeError(version)
        reports.append(report)
        with np.load(npz, allow_pickle=False) as z:
            n = len(z["global_index"])
            part = {
                key: np.asarray(z[key])
                for key in z.files
                if np.asarray(z[key]).ndim and len(np.asarray(z[key])) == n
            }
        parts.append(part)

    common = set(parts[0])
    for part in parts[1:]:
        common &= set(part)
    required = {"global_index", "k", "presence", "event_candidate", "event_time"}
    missing = required - common
    if missing:
        raise RuntimeError(f"{version}: missing required arrays {sorted(missing)}; common={sorted(common)}")
    merged = {key: np.concatenate([part[key] for part in parts], axis=0) for key in common}
    order = np.argsort(merged["global_index"], kind="stable")
    merged = {key: value[order] for key, value in merged.items()}
    if len(merged["global_index"]) != 76768 or len(np.unique(merged["global_index"])) != 76768:
        raise RuntimeError(f"{version}: invalid outer-clean coverage")
    return reports, merged


def _norm_rows(candidate: np.ndarray):
    x = np.maximum(np.asarray(candidate, dtype=np.float64), 0.0)
    sums = np.sum(x, axis=2, keepdims=True)
    norm = np.divide(x, np.maximum(sums, EPS), out=np.zeros_like(x), where=sums > EPS)
    return x, norm, sums[..., 0]


def _entropy(norm: np.ndarray):
    return -np.sum(np.where(norm > 0, norm * np.log(np.clip(norm, EPS, 1.0)), 0.0), axis=-1)


def _top_margin(norm: np.ndarray):
    if norm.shape[-1] < 2:
        return np.max(norm, axis=-1)
    part = np.partition(norm, -2, axis=-1)
    return part[..., -1] - part[..., -2]


def _pair_collision_matrix(argmax: np.ndarray, active: np.ndarray, rows: np.ndarray):
    mat_num = np.zeros((SLOTS, SLOTS), dtype=np.int64)
    mat_den = np.zeros((SLOTS, SLOTS), dtype=np.int64)
    for q in range(SLOTS):
        for r in range(q + 1, SLOTS):
            m = rows & active[:, q] & active[:, r]
            mat_den[q, r] = mat_den[r, q] = int(np.sum(m))
            if np.any(m):
                v = int(np.sum(argmax[m, q] == argmax[m, r]))
                mat_num[q, r] = mat_num[r, q] = v
    rate = np.divide(mat_num, mat_den, out=np.zeros_like(mat_num, dtype=np.float64), where=mat_den > 0)
    return {"numerator": mat_num.tolist(), "denominator": mat_den.tolist(), "rate": rate.tolist()}


def _row_duplicate(argmax: np.ndarray, active: np.ndarray):
    out = np.zeros(len(active), dtype=bool)
    pair_rate = np.full(len(active), np.nan, dtype=np.float64)
    for i in range(len(active)):
        ids = np.flatnonzero(active[i])
        if len(ids) < 2:
            continue
        a = argmax[i, ids]
        total = len(ids) * (len(ids) - 1) // 2
        dup = 0
        for x in range(len(a)):
            for y in range(x + 1, len(a)):
                dup += int(a[x] == a[y])
        out[i] = dup > 0
        pair_rate[i] = dup / total if total else np.nan
    return out, pair_rate


def _shared_argmax_loads(candidate: np.ndarray, presence: np.ndarray, active: np.ndarray):
    """For each row, measure load/slack on columns shared by >=2 active argmaxes.

    V17.4 transport mass is presence * transported conditional candidate output.
    The configured candidate capacity is normally 1 because valid candidates far
    outnumber the <=6 active proposal mass. If duplicate argmaxes remain while the
    shared column has large slack below 1, the continuous constraint is not binding
    at the discrete decision.
    """
    arg = np.argmax(candidate, axis=2)
    mass = candidate * presence[:, :, None]
    col_load = np.sum(mass, axis=1)
    max_shared_load = np.full(len(active), np.nan, dtype=np.float64)
    min_shared_slack = np.full(len(active), np.nan, dtype=np.float64)
    shared_columns = np.zeros(len(active), dtype=np.int32)
    for i in range(len(active)):
        ids = np.flatnonzero(active[i])
        if len(ids) < 2:
            continue
        vals, cnt = np.unique(arg[i, ids], return_counts=True)
        shared = vals[cnt >= 2]
        if len(shared) == 0:
            continue
        loads = col_load[i, shared]
        shared_columns[i] = len(shared)
        max_shared_load[i] = float(np.max(loads))
        min_shared_slack[i] = float(1.0 - np.max(loads))
    return {
        "max_shared_load": max_shared_load,
        "min_shared_slack_to_capacity1": min_shared_slack,
        "shared_columns": shared_columns,
        "column_load": col_load,
    }


def _mean(v):
    v = np.asarray(v, dtype=np.float64)
    v = v[np.isfinite(v)]
    return float(np.mean(v)) if len(v) else None


def _quantiles(v):
    v = np.asarray(v, dtype=np.float64)
    v = v[np.isfinite(v)]
    if not len(v):
        return None
    return {str(q): float(np.quantile(v, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)}


def _candidate_diagnostics(k: np.ndarray, presence: np.ndarray, candidate: np.ndarray):
    c, norm, row_sum = _norm_rows(candidate)
    active = presence >= THRESHOLD
    hard = np.sum(active, axis=1).astype(np.int32)
    arg = np.argmax(c, axis=2)
    dup, pair_rate = _row_duplicate(arg, active)
    entropy = _entropy(norm)
    margin = _top_margin(norm)
    effective = np.exp(entropy)
    loads = _shared_argmax_loads(c, presence, active)
    max_col_load = np.max(loads["column_load"], axis=1)

    out = {
        "global": {
            "hard_mean_k": float(np.mean(hard)),
            "candidate_row_sum_mean_all_slots": float(np.mean(row_sum)),
            "candidate_row_sum_mean_active_slots": _mean(row_sum[active]),
            "candidate_row_deficit_mean_active_slots": _mean(1.0 - row_sum[active]),
            "normalized_entropy_mean_active_slots": _mean(entropy[active]),
            "effective_candidates_mean_active_slots": _mean(effective[active]),
            "top1_top2_margin_mean_active_slots": _mean(margin[active]),
            "max_column_presence_mass_mean": float(np.mean(max_col_load)),
            "max_column_presence_mass_q90": float(np.quantile(max_col_load, 0.9)),
        },
        "per_true_k": {},
    }

    for value in range(2, SLOTS + 1):
        m = k == value
        exact = m & (hard == value)
        exact_dup = exact & dup
        exact_clean = exact & ~dup
        shared_load = loads["max_shared_load"]
        shared_slack = loads["min_shared_slack_to_capacity1"]
        row = {
            "clusters": int(np.sum(m)),
            "exact_count_clusters": int(np.sum(exact)),
            "exact_count_rate": float(np.mean(hard[m] == value)),
            "duplicate_argmax_rate_exact_count": float(np.mean(dup[exact])) if np.any(exact) else None,
            "duplicate_pair_rate_exact_count": _mean(pair_rate[exact]),
            "candidate_row_sum_mean_exact_active": _mean(row_sum[exact][active[exact]]) if np.any(exact) else None,
            "candidate_row_deficit_mean_exact_active": _mean(1.0 - row_sum[exact][active[exact]]) if np.any(exact) else None,
            "normalized_entropy_mean_exact_active": _mean(entropy[exact][active[exact]]) if np.any(exact) else None,
            "effective_candidates_mean_exact_active": _mean(effective[exact][active[exact]]) if np.any(exact) else None,
            "top1_top2_margin_mean_exact_active": _mean(margin[exact][active[exact]]) if np.any(exact) else None,
            "shared_argmax_max_column_load_mean": _mean(shared_load[exact_dup]),
            "shared_argmax_max_column_load_quantiles": _quantiles(shared_load[exact_dup]),
            "shared_argmax_capacity1_slack_mean": _mean(shared_slack[exact_dup]),
            "shared_argmax_capacity1_slack_quantiles": _quantiles(shared_slack[exact_dup]),
            "max_column_presence_mass_mean_exact": _mean(max_col_load[exact]),
            "duplicate_vs_clean": {
                "duplicate_rows": int(np.sum(exact_dup)),
                "clean_rows": int(np.sum(exact_clean)),
                "entropy_active_duplicate": _mean(entropy[exact_dup][active[exact_dup]]) if np.any(exact_dup) else None,
                "entropy_active_clean": _mean(entropy[exact_clean][active[exact_clean]]) if np.any(exact_clean) else None,
                "margin_active_duplicate": _mean(margin[exact_dup][active[exact_dup]]) if np.any(exact_dup) else None,
                "margin_active_clean": _mean(margin[exact_clean][active[exact_clean]]) if np.any(exact_clean) else None,
                "row_deficit_active_duplicate": _mean(1.0 - row_sum[exact_dup][active[exact_dup]]) if np.any(exact_dup) else None,
                "row_deficit_active_clean": _mean(1.0 - row_sum[exact_clean][active[exact_clean]]) if np.any(exact_clean) else None,
            },
            "pair_collision_matrix_exact_count": _pair_collision_matrix(arg, active, exact),
        }
        out["per_true_k"][str(value)] = row
    return out, {"hard": hard, "active": active, "arg": arg, "dup": dup, "row_sum": row_sum, "entropy": entropy, "margin": margin}


def _paired_k3(k: np.ndarray, s173: dict, s174: dict):
    m = k == 3
    common_exact = m & (s173["hard"] == 3) & (s174["hard"] == 3)
    a = s173["dup"]
    b = s174["dup"]
    transition = {
        "common_exact_count_rows": int(np.sum(common_exact)),
        "clean_to_clean": int(np.sum(common_exact & ~a & ~b)),
        "clean_to_duplicate": int(np.sum(common_exact & ~a & b)),
        "duplicate_to_clean": int(np.sum(common_exact & a & ~b)),
        "duplicate_to_duplicate": int(np.sum(common_exact & a & b)),
    }
    transition["net_new_duplicate_rows"] = transition["clean_to_duplicate"] - transition["duplicate_to_clean"]

    # Pairwise identity: which query pairs become newly colliding on common K3-exact rows?
    pair_changes = {}
    for q in range(SLOTS):
        for r in range(q + 1, SLOTS):
            both = common_exact & s173["active"][:, q] & s173["active"][:, r] & s174["active"][:, q] & s174["active"][:, r]
            old = s173["arg"][:, q] == s173["arg"][:, r]
            new = s174["arg"][:, q] == s174["arg"][:, r]
            pair_changes[f"{q}-{r}"] = {
                "rows": int(np.sum(both)),
                "new_collision": int(np.sum(both & ~old & new)),
                "resolved_collision": int(np.sum(both & old & ~new)),
                "net": int(np.sum(both & ~old & new) - np.sum(both & old & ~new)),
            }
    transition["pair_changes"] = pair_changes
    return transition


def _fold_metrics(reports173: list, reports174: list):
    out = {}
    for fold in range(5):
        a = next(r for r in reports173 if int(r["outer_fold"]) == fold)
        b = next(r for r in reports174 if int(r["outer_fold"]) == fold)
        g173 = a["strata"]["aggregate"]["v173_poibin"]["metrics"]["global"]
        g174 = b["strata"]["aggregate"]["v174_transport"]["metrics"]["global"]
        d173 = a["v173"].get("candidate_duplicate_argmax_by_true_k", {})
        d174 = b["v174"].get("candidate_duplicate_argmax_by_true_k", {})
        out[str(fold)] = {
            "v173_f1": float(g173["f1"]),
            "v174_f1": float(g174["f1"]),
            "delta_f1": float(g174["f1"] - g173["f1"]),
            "v173_pred_ref": float(g173["prediction_reference_ratio"]),
            "v174_pred_ref": float(g174["prediction_reference_ratio"]),
            "k3_duplicate_exact_v173_report": d173.get("3", {}).get("exact_count_rows"),
            "k3_duplicate_exact_v174_report": d174.get("3", {}).get("exact_count_rows"),
        }
    return out


def audit(args):
    r173, m173 = _load_version(args.v173_folds, "v173")
    r174, m174 = _load_version(args.v174_folds, "v174")
    for key in ("global_index", "k"):
        if not np.array_equal(np.asarray(m173[key]), np.asarray(m174[key])):
            raise RuntimeError(f"row mismatch: {key}")

    k = np.asarray(m174["k"], dtype=np.int32)
    p173 = np.asarray(m173["presence"], dtype=np.float64)
    p174 = np.asarray(m174["presence"], dtype=np.float64)
    c173 = np.asarray(m173["event_candidate"], dtype=np.float64)
    c174 = np.asarray(m174["event_candidate"], dtype=np.float64)

    d173, s173 = _candidate_diagnostics(k, p173, c173)
    d174, s174 = _candidate_diagnostics(k, p174, c174)
    paired_k3 = _paired_k3(k, s173, s174)
    folds = _fold_metrics(r173, r174)

    sum173 = _load_json(args.v173_summary / "report.json")
    sum174 = _load_json(args.v174_summary / "report.json")
    comp = sum174["comparison"]

    k3_173 = d173["per_true_k"]["3"]
    k3_174 = d174["per_true_k"]["3"]
    delta_dup_k3 = k3_174["duplicate_argmax_rate_exact_count"] - k3_173["duplicate_argmax_rate_exact_count"]
    slack = k3_174["shared_argmax_capacity1_slack_mean"]
    margin_dup = k3_174["duplicate_vs_clean"]["margin_active_duplicate"]
    margin_clean = k3_174["duplicate_vs_clean"]["margin_active_clean"]
    entropy_dup = k3_174["duplicate_vs_clean"]["entropy_active_duplicate"]
    entropy_clean = k3_174["duplicate_vs_clean"]["entropy_active_clean"]

    gates = {
        "transport_improved_global_f1": bool(comp["delta_v174_minus_v173_f1"] > 0),
        "transport_reduced_poly_exact_duplicates": bool(comp["candidate_duplicate_poly_exact_count_delta"] < 0),
        "k3_duplicate_rate_increased": bool(delta_dup_k3 > 0),
        "k3_duplicate_increase_gt_5pp": bool(delta_dup_k3 > 0.05),
        "k3_common_exact_net_new_duplicates_positive": bool(paired_k3["net_new_duplicate_rows"] > 0),
        "k3_shared_argmax_has_mean_capacity_slack_gt_0p10": bool(slack is not None and slack > 0.10),
        "k3_duplicate_rows_have_lower_margin_than_clean": bool(margin_dup is not None and margin_clean is not None and margin_dup < margin_clean),
        "k3_duplicate_rows_have_higher_entropy_than_clean": bool(entropy_dup is not None and entropy_clean is not None and entropy_dup > entropy_clean),
    }

    interpretation = []
    if gates["k3_duplicate_rate_increased"]:
        interpretation.append("V17.4 concentrates additional discrete candidate collisions in true-K3 exact-count rows.")
    if gates["k3_common_exact_net_new_duplicates_positive"]:
        interpretation.append("On row-aligned K3 cases where both models predict K=3 exactly, V17.4 creates more new duplicate-argmax rows than it resolves.")
    if gates["k3_shared_argmax_has_mean_capacity_slack_gt_0p10"]:
        interpretation.append("K3 duplicate argmaxes commonly remain while the shared candidate column is materially below capacity 1; the continuous capacity constraint is therefore not binding at the hard ownership decision.")
    if gates["k3_duplicate_rows_have_lower_margin_than_clean"] and gates["k3_duplicate_rows_have_higher_entropy_than_clean"]:
        interpretation.append("Duplicate K3 rows are more diffuse and less decisive than clean rows, consistent with mass spreading that satisfies transport constraints without producing distinct argmax ownership.")
    if not gates["transport_reduced_poly_exact_duplicates"]:
        interpretation.append("The tested continuous projection fails its primary mechanism gate: aggregate duplicate ownership on poly exact-count rows does not decrease.")

    result = {
        "schema_version": 1,
        "protocol": {
            "analysis_only": True,
            "canonical_v173_run": 33829092700,
            "canonical_v174_run": 33838069830,
            "outer_clean_rows": 76768,
            "same_rows": True,
            "threshold": THRESHOLD,
            "threshold_tuned": False,
            "locked12_indexed_or_evaluated": False,
        },
        "summary_comparison": comp,
        "v173_candidate_diagnostics": d173,
        "v174_candidate_diagnostics": d174,
        "paired_k3": paired_k3,
        "folds": folds,
        "gates": gates,
        "interpretation": interpretation,
        "reference_global_f1_v173": float(sum173["comparison"]["global_f1_v173"]),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    compact = {
        "global_f1_delta_v174_minus_v173": comp["delta_v174_minus_v173_f1"],
        "poly_duplicate_delta": comp["candidate_duplicate_poly_exact_count_delta"],
        "k3_duplicate_exact_v173": k3_173["duplicate_argmax_rate_exact_count"],
        "k3_duplicate_exact_v174": k3_174["duplicate_argmax_rate_exact_count"],
        "k3_duplicate_delta": delta_dup_k3,
        "k3_shared_argmax_capacity_slack_mean_v174": slack,
        "k3_common_exact_net_new_duplicate_rows": paired_k3["net_new_duplicate_rows"],
        "k3_margin_duplicate": margin_dup,
        "k3_margin_clean": margin_clean,
        "k3_entropy_duplicate": entropy_dup,
        "k3_entropy_clean": entropy_clean,
        "gates": gates,
        "interpretation": interpretation,
    }
    print(json.dumps(compact, indent=2, sort_keys=True))
    return result


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--v173-folds", type=Path, required=True)
    p.add_argument("--v174-folds", type=Path, required=True)
    p.add_argument("--v173-summary", type=Path, required=True)
    p.add_argument("--v174-summary", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    return p


def main(argv: Optional[Sequence[str]] = None):
    audit(parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
