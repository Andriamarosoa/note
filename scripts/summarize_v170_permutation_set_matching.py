"""Robustly aggregate five V17 outer-clean folds.

Unlike the legacy V13 summarizer, only row-wise NPZ arrays are concatenated;
metadata arrays such as schema_version=(1,) are ignored.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

MODELS = ("v104", "v170")
SLOT_COUNT = 6


def _metric_sum(rows):
    rows = [r for r in rows if r is not None]
    tp = int(sum(int(r["true_positive"]) for r in rows))
    fp = int(sum(int(r["false_positive"]) for r in rows))
    fn = int(sum(int(r["false_negative"]) for r in rows))
    pred = int(sum(int(r["prediction_count"]) for r in rows))
    ref = int(sum(int(r["reference_count"]) for r in rows))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "prediction_count": pred,
        "reference_count": ref,
        "prediction_reference_ratio": pred / ref if ref else None,
    }


def _card(k, pred):
    k = np.asarray(k, dtype=np.int32)
    pred = np.asarray(pred, dtype=np.int32)
    exact = pred == k
    poly = k >= 2
    return {
        "clusters": int(len(k)),
        "accuracy": float(np.mean(exact)),
        "poly_accuracy": float(np.mean(exact[poly])) if np.any(poly) else None,
        "mae": float(np.mean(np.abs(pred - k))),
        "under_rate": float(np.mean(pred < k)),
        "over_rate": float(np.mean(pred > k)),
        "mean_true_k": float(np.mean(k)),
        "mean_predicted_k": float(np.mean(pred)),
        "target_histogram": {str(v): int(np.sum(k == v)) for v in range(SLOT_COUNT + 1)},
        "predicted_histogram": {str(v): int(np.sum(pred == v)) for v in range(SLOT_COUNT + 1)},
        "confusion_true_rows_pred_columns": [
            [int(np.sum((k == y) & (pred == p))) for p in range(SLOT_COUNT + 1)]
            for y in range(SLOT_COUNT + 1)
        ],
    }


def _load(args):
    reports, parts = [], []
    for fold in range(5):
        rp = sorted(args.input_dir.glob(f"**/report-fold-{fold}.json"))
        npz = sorted(args.input_dir.glob(f"**/predictions-fold-{fold}.npz"))
        if len(rp) != 1 or len(npz) != 1:
            raise RuntimeError(f"fold {fold}: expected exactly one report and one prediction shard")
        report = json.loads(rp[0].read_text())
        p = report["protocol"]
        assert p["historical_validation_or_locked12_indexed_or_evaluated"] is False
        assert p["outer_fold_used_for_training"] is False
        assert p["outer_fold_used_for_epoch_selection"] is False
        assert p["categorical_cardinality_head_exists"] is False
        assert p["presence_threshold_tuned"] is False
        assert p["permutation_invariant_set_matching"] is True
        reports.append(report)
        with np.load(npz[0], allow_pickle=False) as z:
            n = len(z["global_index"])
            part = {
                key: np.asarray(z[key])
                for key in z.files
                if np.asarray(z[key]).ndim and len(z[key]) == n
            }
            if "pred170" not in part:
                raise RuntimeError(f"fold {fold}: pred170 missing")
            parts.append(part)

    common = set(parts[0])
    for part in parts[1:]:
        common &= set(part)
    merged = {key: np.concatenate([p[key] for p in parts], axis=0) for key in common}
    order = np.argsort(merged["global_index"], kind="stable")
    merged = {key: value[order] for key, value in merged.items()}
    if len(np.unique(merged["global_index"])) != len(merged["global_index"]):
        raise RuntimeError("outer-fold predictions overlap")
    if sorted(set(np.asarray(merged["outer_fold"], dtype=np.int32).tolist())) != list(range(5)):
        raise RuntimeError("missing outer fold")
    return reports, merged


def summarize(args):
    reports, merged = _load(args)
    k = np.asarray(merged["k"], dtype=np.int32)
    if len(k) != 76768:
        raise RuntimeError(f"expected 76768 outer rows, got {len(k)}")
    preds = {
        "v104": np.asarray(merged["pred104"], dtype=np.int32),
        "v170": np.asarray(merged["pred170"], dtype=np.int32),
    }

    strata_names = [
        "aggregate", "comp", "solo", "player00", "player00_comp",
        "player00_solo", "player00_rock_comp", "player01", "player02",
        "player03", "player04",
    ]
    strata = {}
    for stratum in strata_names:
        row = {}
        for model in MODELS:
            pieces = []
            for report in reports:
                sr = report["strata"].get(stratum)
                if sr is not None:
                    pieces.append(sr[model]["metrics"]["global"])
            row[model] = _metric_sum(pieces)
        strata[stratum] = row

    per_k = {}
    for value in range(SLOT_COUNT + 1):
        mask = k == value
        row = {"clusters": int(np.sum(mask))}
        for model in MODELS:
            p = preds[model][mask]
            row[model] = {
                "exact": float(np.mean(p == value)) if len(p) else None,
                "under_rate": float(np.mean(p < value)) if len(p) else None,
                "over_rate": float(np.mean(p > value)) if len(p) else None,
                "mae": float(np.mean(np.abs(p - value))) if len(p) else None,
                "mean_predicted_k": float(np.mean(p)) if len(p) else None,
                "prediction_distribution": {
                    str(v): int(np.sum(p == v)) for v in range(SLOT_COUNT + 1)
                },
            }
        per_k[str(value)] = row

    cards = {m: _card(k, preds[m]) for m in MODELS}
    folds = {}
    for report in reports:
        f = str(report["outer_fold"])
        folds[f] = {
            "selected_epochs": int(report["data"]["selected_epochs"]),
            "v104_f1": float(report["strata"]["aggregate"]["v104"]["metrics"]["global"]["f1"]),
            "v170_f1": float(report["strata"]["aggregate"]["v170"]["metrics"]["global"]["f1"]),
        }

    g104 = strata["aggregate"]["v104"]["f1"]
    g170 = strata["aggregate"]["v170"]["f1"]
    result = {
        "schema_version": 1,
        "protocol": {
            "five_outer_composition_folds": True,
            "every_row_evaluated_once_outer_clean": True,
            "historical_validation_or_locked12_indexed_or_evaluated": False,
            "permutation_invariant_set_matching": True,
            "ordered_query_presence_targets": False,
            "categorical_cardinality_head_exists": False,
            "presence_threshold_tuned": False,
            "presence_threshold": 0.5,
        },
        "data": {
            "clusters": int(len(k)),
            "unique_global_indices": int(len(np.unique(merged["global_index"]))),
            "outer_folds": 5,
        },
        "strata": strata,
        "cardinality": cards,
        "per_true_k": per_k,
        "event_presence": {
            "mean_by_query": np.mean(np.asarray(merged["presence"], dtype=np.float64), axis=0).tolist(),
            "active_rate_by_query": np.mean(np.asarray(merged["presence"], dtype=np.float64) >= 0.5, axis=0).tolist(),
            "prefix_violation_rate": float(np.mean(np.asarray(merged["prefix_violation"]) > 0)),
        },
        "folds": folds,
        "comparison": {
            "v170_minus_v104_global_f1": g170 - g104,
            "v170_minus_v104_poly_exact": cards["v170"]["poly_accuracy"] - cards["v104"]["poly_accuracy"],
            "v170_minus_v104_k5_exact": per_k["5"]["v170"]["exact"] - per_k["5"]["v104"]["exact"],
            "v170_minus_v104_k6_exact": per_k["6"]["v170"]["exact"] - per_k["6"]["v104"]["exact"],
            "folds_won_vs_v104": int(sum(row["v170_f1"] > row["v104_f1"] for row in folds.values())),
            "promotion_candidate": bool(
                g170 > g104
                and strata["player00_rock_comp"]["v170"]["f1"] >= strata["player00_rock_comp"]["v104"]["f1"]
                and cards["v170"]["poly_accuracy"] >= cards["v104"]["poly_accuracy"]
            ),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(args.output_dir / "predictions.npz", **merged)
    print(json.dumps({
        "v104_global_f1": g104,
        "v170_global_f1": g170,
        "v104_poly_exact": cards["v104"]["poly_accuracy"],
        "v170_poly_exact": cards["v170"]["poly_accuracy"],
        "v170_k2": per_k["2"]["v170"]["exact"],
        "v170_k3": per_k["3"]["v170"]["exact"],
        "v170_k4": per_k["4"]["v170"]["exact"],
        "v170_k5": per_k["5"]["v170"]["exact"],
        "v170_k6": per_k["6"]["v170"]["exact"],
        "folds_won_vs_v104": result["comparison"]["folds_won_vs_v104"],
        "promotion_candidate": result["comparison"]["promotion_candidate"],
    }, indent=2, sort_keys=True))
    return result


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    return p


def main(argv: Optional[Sequence[str]] = None):
    summarize(parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
