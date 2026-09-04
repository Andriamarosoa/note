"""Aggregate V17.3 and compare directly with V17.2-C and V10.4."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from scripts import summarize_v171_controlled_assignment_ab as s171

MODEL_KEY = "v173_poibin"
PRED_KEY = "pred173_poibin"


def _safe_global(report, stratum, model_key):
    row = report.get("strata", {}).get(stratum)
    if not row:
        return None
    model = row.get(model_key)
    if not model:
        return None
    return model.get("metrics", {}).get("global")


def _load_v173(input_dir: Path):
    reports, parts = [], []
    for fold in range(5):
        candidates = []
        for rp in input_dir.glob(f"**/report-fold-{fold}.json"):
            r = json.loads(rp.read_text())
            p = r.get("protocol", {})
            if p.get("v173_poibin_count_consistency") is True:
                candidates.append((rp, r))
        if len(candidates) != 1:
            raise RuntimeError(f"fold={fold}: expected one V17.3 report, got {len(candidates)}")
        rp, report = candidates[0]
        p = report["protocol"]
        assert p["v173_base_arm"] == "mass_permutation"
        assert p["runtime_graph_unchanged_from_v172_c"] is True
        assert p["runtime_decode_unchanged_from_v172_c"] is True
        assert p["runtime_presence_threshold"] == 0.5
        assert p["runtime_presence_threshold_tuned"] is False
        assert p["categorical_cardinality_head_exists"] is False
        assert p["poisson_binomial_cardinality_consistency"] is True
        assert p["event_count_norm_mse_loss_weight"] == 0.0
        assert p["historical_validation_or_locked12_indexed_or_evaluated"] is False
        reports.append(report)

        npzs = list(rp.parent.glob(f"predictions-fold-{fold}.npz"))
        if len(npzs) != 1:
            raise RuntimeError(f"fold={fold}: predictions shard missing")
        with np.load(npzs[0], allow_pickle=False) as z:
            n = len(z["global_index"])
            part = {key: np.asarray(z[key]) for key in z.files if np.asarray(z[key]).ndim and len(z[key]) == n}
        if PRED_KEY not in part:
            raise RuntimeError(f"fold={fold}: {PRED_KEY} missing")
        parts.append(part)

    common = set(parts[0])
    for part in parts[1:]:
        common &= set(part)
    merged = {key: np.concatenate([p[key] for p in parts], axis=0) for key in common}
    order = np.argsort(merged["global_index"], kind="stable")
    merged = {key: value[order] for key, value in merged.items()}
    if len(merged["global_index"]) != 76768 or len(np.unique(merged["global_index"])) != 76768:
        raise RuntimeError("invalid V17.3 outer row coverage")
    return reports, merged


def _aggregate_presence(reports):
    total = sum(int(r["data"]["outer_clusters"]) for r in reports)
    out = {}
    for q in range(6):
        rows = []
        for r in reports:
            n = int(r["data"]["outer_clusters"])
            rows.append((n, r["v173"]["outer_presence"][str(q)]))
        out[str(q)] = {
            "mean": float(sum(n * x["mean"] for n, x in rows) / total),
            "active_rate_at_0p5": float(sum(n * x["active_rate_at_0p5"] for n, x in rows) / total),
            "by_fold": {str(r["outer_fold"]): r["v173"]["outer_presence"][str(q)] for r in reports},
        }
    return out


def summarize(args):
    reports, merged = _load_v173(args.input_dir)
    v172_report = json.loads((args.v172_summary_dir / "report.json").read_text())
    with np.load(args.v172_summary_dir / "predictions.npz", allow_pickle=False) as z:
        v172_global_index = np.asarray(z["global_index"])
        v172_k = np.asarray(z["k"], dtype=np.int32)
        pred172 = np.asarray(z["pred172_mass_permutation"], dtype=np.int32)
        pred104_old = np.asarray(z["pred104"], dtype=np.int32)

    if not np.array_equal(np.asarray(merged["global_index"]), v172_global_index):
        raise RuntimeError("V17.3 rows do not align with V17.2 summary")
    k = np.asarray(merged["k"], dtype=np.int32)
    if not np.array_equal(k, v172_k):
        raise RuntimeError("truth K mismatch V17.3 vs V17.2")
    pred104 = np.asarray(merged["pred104"], dtype=np.int32)
    if not np.array_equal(pred104, pred104_old):
        raise RuntimeError("V10.4 baseline mismatch V17.3 vs V17.2")
    pred173 = np.asarray(merged[PRED_KEY], dtype=np.int32)

    strata_names = ["aggregate", "comp", "solo", "player00", "player00_comp", "player00_rock_comp"]
    strata = {}
    for name in strata_names:
        v104 = s171._metric_sum([_safe_global(r, name, "v104") for r in reports])
        v173 = s171._metric_sum([_safe_global(r, name, MODEL_KEY) for r in reports])
        v172 = v172_report["strata"][name]["mass_permutation"]
        strata[name] = {
            "v104": v104,
            "v172_mass_permutation": v172,
            "v173_poibin": v173,
            "delta_v173_minus_v172_f1": float(v173["f1"] - v172["f1"]),
            "delta_v173_minus_v104_f1": float(v173["f1"] - v104["f1"]),
            "delta_v173_minus_v172_precision": float(v173["precision"] - v172["precision"]),
            "delta_v173_minus_v172_recall": float(v173["recall"] - v172["recall"]),
            "delta_v173_minus_v172_pred_ref": float(v173["prediction_reference_ratio"] - v172["prediction_reference_ratio"]),
        }

    preds = {
        "v104": pred104,
        "v172_mass_permutation": pred172,
        "v173_poibin": pred173,
    }
    cards = {name: s171._card(k, pred) for name, pred in preds.items()}
    per_k = s171._per_k(k, preds)

    folds = {}
    wins_v172 = 0
    wins_v104 = 0
    for fold in range(5):
        r = next(x for x in reports if int(x["outer_fold"]) == fold)
        g173 = r["strata"]["aggregate"][MODEL_KEY]["metrics"]["global"]
        g104 = r["strata"]["aggregate"]["v104"]["metrics"]["global"]
        f172 = v172_report["folds"][str(fold)]
        f173 = float(g173["f1"])
        f104 = float(g104["f1"])
        old = float(f172["mass_permutation_f1"])
        dv172 = f173 - old
        dv104 = f173 - f104
        wins_v172 += int(dv172 > 0)
        wins_v104 += int(dv104 > 0)
        folds[str(fold)] = {
            "selected_epochs": int(r["data"]["selected_epochs"]),
            "v104_f1": f104,
            "v172_mass_permutation_f1": old,
            "v173_poibin_f1": f173,
            "delta_v173_minus_v172_f1": dv172,
            "delta_v173_minus_v104_f1": dv104,
            "v173_pred_ref": float(g173["prediction_reference_ratio"]),
        }

    presence = _aggregate_presence(reports)
    soft_count = float(sum(presence[str(q)]["mean"] for q in range(6)))
    truth_mean = float(cards["v173_poibin"]["mean_true_k"])
    hard_mean = float(cards["v173_poibin"]["mean_predicted_k"])
    old_soft_count = float(
        sum(v172_report["diagnostics"]["mass_permutation"]["presence"][str(q)]["mean"] for q in range(6))
    )

    comparison = {
        "global_f1_v104": float(strata["aggregate"]["v104"]["f1"]),
        "global_f1_v172_mass_permutation": float(strata["aggregate"]["v172_mass_permutation"]["f1"]),
        "global_f1_v173_poibin": float(strata["aggregate"]["v173_poibin"]["f1"]),
        "delta_v173_minus_v172_f1": float(strata["aggregate"]["delta_v173_minus_v172_f1"]),
        "delta_v173_minus_v104_f1": float(strata["aggregate"]["delta_v173_minus_v104_f1"]),
        "delta_v173_minus_v172_precision": float(strata["aggregate"]["delta_v173_minus_v172_precision"]),
        "delta_v173_minus_v172_recall": float(strata["aggregate"]["delta_v173_minus_v172_recall"]),
        "folds_v173_beats_v172": int(wins_v172),
        "folds_v173_beats_v104": int(wins_v104),
        "v172_soft_presence_mass": old_soft_count,
        "v173_soft_presence_mass": soft_count,
        "truth_mean_k": truth_mean,
        "v173_soft_mass_bias_fraction": float(soft_count / truth_mean - 1.0),
        "v173_hard_mean_k": hard_mean,
        "v173_hard_count_bias_fraction": float(hard_mean / truth_mean - 1.0),
        "soft_mass_bias_reduction": float(abs(old_soft_count - truth_mean) - abs(soft_count - truth_mean)),
        "poly_exact_v104": float(cards["v104"]["poly_accuracy"]),
        "poly_exact_v172": float(cards["v172_mass_permutation"]["poly_accuracy"]),
        "poly_exact_v173": float(cards["v173_poibin"]["poly_accuracy"]),
        "delta_poly_v173_minus_v172": float(cards["v173_poibin"]["poly_accuracy"] - cards["v172_mass_permutation"]["poly_accuracy"]),
        "delta_k2_exact_v173_minus_v172": float(per_k["2"]["v173_poibin"]["exact"] - per_k["2"]["v172_mass_permutation"]["exact"]),
        "delta_k3_exact_v173_minus_v172": float(per_k["3"]["v173_poibin"]["exact"] - per_k["3"]["v172_mass_permutation"]["exact"]),
        "delta_k4_exact_v173_minus_v172": float(per_k["4"]["v173_poibin"]["exact"] - per_k["4"]["v172_mass_permutation"]["exact"]),
        "delta_k5_exact_v173_minus_v172": float(per_k["5"]["v173_poibin"]["exact"] - per_k["5"]["v172_mass_permutation"]["exact"]),
        "delta_k6_exact_v173_minus_v172": float(per_k["6"]["v173_poibin"]["exact"] - per_k["6"]["v172_mass_permutation"]["exact"]),
    }

    result = {
        "schema_version": 1,
        "protocol": {
            "v173_exact_poisson_binomial_count_consistency": True,
            "outer_clean_rows": 76768,
            "same_rows_as_v172": True,
            "same_seed": 16061,
            "same_runtime_threshold": 0.5,
            "threshold_tuned": False,
            "runtime_graph_unchanged_from_v172_c": True,
            "runtime_decode_unchanged_from_v172_c": True,
            "categorical_cardinality_head_exists": False,
            "locked12_indexed_or_evaluated": False,
            "only_training_objective_change": "event_count_norm MSE contribution -> exact Poisson-binomial NLL",
        },
        "strata": strata,
        "cardinality": cards,
        "per_true_k": per_k,
        "folds": folds,
        "presence": presence,
        "comparison": comparison,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(
        args.output_dir / "predictions.npz",
        global_index=np.asarray(merged["global_index"]),
        k=k,
        pred104=pred104,
        pred172_mass_permutation=pred172,
        pred173_poibin=pred173,
    )
    print(json.dumps(comparison, indent=2, sort_keys=True))
    return result


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--v172-summary-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    return p


def main(argv: Optional[Sequence[str]] = None):
    summarize(parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
