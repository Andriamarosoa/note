"""Aggregate V23.1 categorical-cardinality dense transport over five outer-clean folds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from scripts import summarize_v171_controlled_assignment_ab as s171
from scripts import train_v190_dense_birth_centers as v190
from scripts import train_v231_cardinality_transport as v231

MODEL_KEY = v231.MODEL_KEY
PRED_KEY = v231.PRED_KEY
OUTER_ROWS = 76768
PREAUDIT_RUN = 34126377071
V19_GLOBAL_F1_PREAUDIT = 0.7945832758065259
TRUE_K_ORACLE_F1_PREAUDIT = 0.9729886591835076


def _safe_global(report, stratum, model_key):
    row = report.get("strata", {}).get(stratum)
    if not row or model_key not in row:
        return None
    return row[model_key].get("metrics", {}).get("global")


def _load_v231(root: Path):
    reports, parts = [], []
    for fold in range(5):
        hits = []
        for rp in root.glob(f"**/report-fold-{fold}.json"):
            report = json.loads(rp.read_text())
            if report.get("protocol", {}).get("v231_cardinality_conditioned_transport") is True:
                hits.append((rp, report))
        if len(hits) != 1:
            raise RuntimeError(f"v231 fold={fold}: expected one report, got {len(hits)}")
        rp, report = hits[0]
        reports.append(report)
        npzs = list(rp.parent.glob(f"predictions-fold-{fold}.npz"))
        if len(npzs) != 1:
            raise RuntimeError(f"v231 fold={fold}: predictions missing")
        with np.load(npzs[0], allow_pickle=False) as z:
            n = len(z["global_index"])
            parts.append(
                {
                    key: np.asarray(z[key])
                    for key in z.files
                    if np.asarray(z[key]).ndim and len(np.asarray(z[key])) == n
                }
            )
    common = set(parts[0])
    for part in parts[1:]:
        common &= set(part)
    merged = {key: np.concatenate([part[key] for part in parts], axis=0) for key in common}
    order = np.argsort(merged["global_index"], kind="stable")
    merged = {key: value[order] for key, value in merged.items()}
    idx = np.asarray(merged["global_index"], dtype=np.int64)
    if len(idx) != OUTER_ROWS or len(np.unique(idx)) != OUTER_ROWS:
        raise RuntimeError(f"v231 invalid outer-clean coverage: {len(idx)} / {len(np.unique(idx))}")
    return reports, merged


def _load_v190_summary(root: Path):
    report_path = root / "report.json"
    pred_path = root / "predictions.npz"
    if not report_path.exists() or not pred_path.exists():
        raise RuntimeError(f"canonical V19 summary files missing under {root}")
    report = json.loads(report_path.read_text())
    if not report.get("protocol", {}).get("v190_dense_birth_centers"):
        raise RuntimeError("wrong canonical V19 summary")
    with np.load(pred_path, allow_pickle=False) as z:
        pred = {key: np.asarray(z[key]) for key in z.files}
    return report, pred


def _weighted_dense_diag(reports, key: str):
    pos_n = poly_n = pos_exact = poly_exact = poly_hits = 0.0
    per_k = {str(k): {"rows": 0, "exact_sum": 0.0, "hit_sum": 0.0} for k in range(1, 7)}
    for report in reports:
        diag = report["v231"]["architecture"][key]
        pn = int(diag["eligible_positive_rows"])
        qn = int(diag["eligible_poly_rows"])
        pos_n += pn
        poly_n += qn
        if diag["top6_exact_center_coverage_positive"] is not None:
            pos_exact += pn * float(diag["top6_exact_center_coverage_positive"])
        if diag["top6_exact_center_coverage_poly"] is not None:
            poly_exact += qn * float(diag["top6_exact_center_coverage_poly"])
        if diag["top6_mean_center_hit_fraction_poly"] is not None:
            poly_hits += qn * float(diag["top6_mean_center_hit_fraction_poly"])
        for k in range(1, 7):
            row = diag["per_true_k"][str(k)]
            n = int(row["rows"])
            per_k[str(k)]["rows"] += n
            if row["top6_exact_center_coverage"] is not None:
                per_k[str(k)]["exact_sum"] += n * float(row["top6_exact_center_coverage"])
            if row["top6_mean_center_hit_fraction"] is not None:
                per_k[str(k)]["hit_sum"] += n * float(row["top6_mean_center_hit_fraction"])
    return {
        "eligible_positive_rows": int(pos_n),
        "eligible_poly_rows": int(poly_n),
        "top6_exact_center_coverage_positive": float(pos_exact / pos_n) if pos_n else None,
        "top6_exact_center_coverage_poly": float(poly_exact / poly_n) if poly_n else None,
        "top6_mean_center_hit_fraction_poly": float(poly_hits / poly_n) if poly_n else None,
        "per_true_k": {
            k: {
                "rows": int(row["rows"]),
                "top6_exact_center_coverage": float(row["exact_sum"] / row["rows"]) if row["rows"] else None,
                "top6_mean_center_hit_fraction": float(row["hit_sum"] / row["rows"]) if row["rows"] else None,
            }
            for k, row in per_k.items()
        },
    }


def summarize(args):
    reports, merged = _load_v231(args.input_dir)
    old, old_pred = _load_v190_summary(args.v190_summary_dir)

    idx = np.asarray(merged["global_index"], dtype=np.int64)
    k = np.asarray(merged["k"], dtype=np.int32)
    if not np.array_equal(idx, np.asarray(old_pred["global_index"], dtype=np.int64)):
        raise RuntimeError("V23.1/V19 global_index mismatch")
    if not np.array_equal(k, np.asarray(old_pred["k"], dtype=np.int32)):
        raise RuntimeError("V23.1/V19 K mismatch")

    pred104 = np.asarray(merged["pred104"], dtype=np.int32)
    pred231 = np.asarray(merged[PRED_KEY], dtype=np.int32)
    cardinality_prob = np.asarray(merged["cardinality_prob"], dtype=np.float64)
    categorical_k = np.asarray(merged["categorical_k"], dtype=np.int32)
    row_mass = np.asarray(merged["transport_row_mass"], dtype=np.float64)
    if not np.array_equal(pred231, categorical_k):
        raise RuntimeError("headline V23.1 prediction is not categorical argmax K")

    card231 = s171._card(k, pred231)
    per_k231 = s171._per_k(k, {MODEL_KEY: pred231})
    categorical_diag = v231._cardinality_diag(cardinality_prob, k)
    transport_mass = v231._transport_mass_diag(row_mass, cardinality_prob)
    center_prior = _weighted_dense_diag(reports, "birth_center_prior_diagnostics")
    transport_coverage = _weighted_dense_diag(reports, "transport_coverage_diagnostics")

    strata = {}
    stratum_names = ["aggregate", "comp", "solo", "player00", "player00_comp", "player00_rock_comp"]
    for name in stratum_names:
        new_metric = s171._metric_sum([_safe_global(r, name, MODEL_KEY) for r in reports])
        old_metric = old["strata"][name][v190.MODEL_KEY]
        baseline = s171._metric_sum([_safe_global(r, name, "v104") for r in reports])
        strata[name] = {
            "v104": baseline,
            v190.MODEL_KEY: old_metric,
            MODEL_KEY: new_metric,
            "delta_v231_minus_v190_f1": float(new_metric["f1"] - old_metric["f1"]),
            "delta_v231_minus_v190_precision": float(new_metric["precision"] - old_metric["precision"]),
            "delta_v231_minus_v190_recall": float(new_metric["recall"] - old_metric["recall"]),
            "delta_v231_minus_v190_pred_ref": float(
                new_metric["prediction_reference_ratio"] - old_metric["prediction_reference_ratio"]
            ),
        }

    folds = {}
    wins = 0
    for fold in range(5):
        report = next(r for r in reports if int(r["outer_fold"]) == fold)
        fnew = float(report["strata"]["aggregate"][MODEL_KEY]["metrics"]["global"]["f1"])
        fold_old = old["folds"][str(fold)]
        fold_v190 = float(fold_old["v190_f1"])
        wins += int(fnew > fold_v190)
        folds[str(fold)] = {
            "selected_epochs": int(report["data"]["selected_epochs"]),
            "v104_f1": float(report["strata"]["aggregate"]["v104"]["metrics"]["global"]["f1"]),
            "v190_f1": fold_v190,
            "v231_f1": fnew,
            "delta_v231_minus_v190_f1": float(fnew - fold_v190),
            "categorical_k_exact": float(report["v231"]["architecture"]["categorical_cardinality"]["exact"]),
            "categorical_k_poly_exact": float(report["v231"]["architecture"]["categorical_cardinality"]["poly_exact"]),
            "transport_exact_k_mass": float(report["v231"]["architecture"]["transport_mass"]["exact_k_mass_decode_rate"]),
            "coverage_top6_exact_poly": report["v231"]["architecture"]["transport_coverage_diagnostics"]["top6_exact_center_coverage_poly"],
        }

    agg = strata["aggregate"]
    old_global_f1 = float(agg[v190.MODEL_KEY]["f1"])
    new_global_f1 = float(agg[MODEL_KEY]["f1"])
    oracle_span = TRUE_K_ORACLE_F1_PREAUDIT - V19_GLOBAL_F1_PREAUDIT
    gap_closed = (new_global_f1 - V19_GLOBAL_F1_PREAUDIT) / oracle_span if oracle_span > 0 else None
    old_poly = float(old["cardinality"][v190.MODEL_KEY]["poly_accuracy"])
    new_poly = float(card231["poly_accuracy"])

    comparison = {
        "global_f1_v104": float(agg["v104"]["f1"]),
        "global_f1_v190": old_global_f1,
        "global_f1_v231": new_global_f1,
        "delta_v231_minus_v190_f1": float(new_global_f1 - old_global_f1),
        "true_k_oracle_f1_from_preaudit": TRUE_K_ORACLE_F1_PREAUDIT,
        "fraction_of_v19_to_true_k_oracle_gap_closed": float(gap_closed) if gap_closed is not None else None,
        "folds_v231_beats_v190": int(wins),
        "poly_exact_v190": old_poly,
        "poly_exact_v231": new_poly,
        "delta_poly_v231_minus_v190": float(new_poly - old_poly),
        "categorical_k_exact": categorical_diag["exact"],
        "categorical_k_poly_exact": categorical_diag["poly_exact"],
        "transport_exact_k_mass_decode_rate": transport_mass["exact_k_mass_decode_rate"],
        "transport_max_absolute_total_mass_error": transport_mass["max_absolute_total_mass_error"],
        "transport_prefix_violation_rate": transport_mass["prefix_violation_rate"],
        "transport_coverage_top6_exact_poly": transport_coverage["top6_exact_center_coverage_poly"],
        "transport_coverage_top6_hit_fraction_poly": transport_coverage["top6_mean_center_hit_fraction_poly"],
        "birth_center_prior_top6_exact_poly": center_prior["top6_exact_center_coverage_poly"],
        "player00_rock_comp_f1_v190": float(strata["player00_rock_comp"][v190.MODEL_KEY]["f1"]),
        "player00_rock_comp_f1_v231": float(strata["player00_rock_comp"][MODEL_KEY]["f1"]),
    }
    for value in range(2, 7):
        old_exact = float(old["per_true_k"][str(value)][v190.MODEL_KEY]["exact"])
        new_exact = float(per_k231[str(value)][MODEL_KEY]["exact"])
        comparison[f"delta_k{value}_exact_v231_minus_v190"] = new_exact - old_exact

    gates = {
        "headline_prediction_is_categorical_k": bool(np.array_equal(pred231, categorical_k)),
        "transport_mass_is_exact_k": transport_mass["exact_k_mass_decode_rate"] >= 0.999999,
        "transport_prefix_never_violated": transport_mass["prefix_violation_rate"] <= 1e-9,
        "global_f1_improved_vs_v190": comparison["delta_v231_minus_v190_f1"] > 0.0,
        "majority_folds_improved_vs_v190": wins >= 3,
        "poly_exact_improved_vs_v190": comparison["delta_poly_v231_minus_v190"] > 0.0,
        "transport_coverage_top6_exact_poly_above_90pct": (
            transport_coverage["top6_exact_center_coverage_poly"] is not None
            and transport_coverage["top6_exact_center_coverage_poly"] > 0.90
        ),
        "protected_player00_rock_comp_not_regressed": (
            comparison["player00_rock_comp_f1_v231"] >= comparison["player00_rock_comp_f1_v190"]
        ),
    }
    gates["v231_architecture_success"] = bool(all(gates.values()))

    result = {
        "schema_version": 1,
        "protocol": {
            "v231_cardinality_conditioned_transport": True,
            "mandatory_post_implementation_audit": True,
            "outer_clean_rows": OUTER_ROWS,
            "same_rows_as_canonical_v190": True,
            "same_seed": v231.DEFAULT_SEED,
            "preaudit_run": PREAUDIT_RUN,
            "categorical_cardinality_head_exists": True,
            "six_independent_presence_bernoullis_exist": False,
            "poisson_binomial_cardinality_objective_exists": False,
            "runtime_cardinality_decode": "argmax categorical P(K=0..6)",
            "transport_grid": [int(v190.v100.TIME_FRAMES), int(v190.v100.SPECTRAL_BANDS)],
            "locked12_indexed_or_evaluated": False,
        },
        "strata": strata,
        "cardinality": {v190.MODEL_KEY: old["cardinality"][v190.MODEL_KEY], MODEL_KEY: card231},
        "categorical_cardinality": categorical_diag,
        "per_true_k": per_k231,
        "folds": folds,
        "transport_mass": transport_mass,
        "birth_center_prior": center_prior,
        "transport_coverage": transport_coverage,
        "comparison": comparison,
        "gates": gates,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(
        args.output_dir / "predictions.npz",
        global_index=idx,
        k=k,
        pred104=pred104,
        pred190=np.asarray(old_pred["pred190"], dtype=np.int32),
        pred231=pred231,
        cardinality_prob=cardinality_prob.astype(np.float32),
        transport_row_mass=row_mass.astype(np.float32),
    )
    print(json.dumps(comparison, indent=2, sort_keys=True))
    print(json.dumps(gates, indent=2, sort_keys=True))
    return result


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--v190-summary-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    return p


def main(argv: Optional[Sequence[str]] = None):
    summarize(parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
