"""Aggregate the V17.1 controlled assignment A/B over the same outer-clean rows."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

ARMS = ("ordered", "permutation")
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
        "predicted_histogram": {str(v): int(np.sum(pred == v)) for v in range(SLOT_COUNT + 1)},
        "confusion_true_rows_pred_columns": [[int(np.sum((k == y) & (pred == p))) for p in range(SLOT_COUNT + 1)] for y in range(SLOT_COUNT + 1)],
    }


def _load_arm(input_dir: Path, arm: str):
    reports, parts = [], []
    pred_key = f"pred171_{arm}"
    for fold in range(5):
        candidates = []
        for rp in input_dir.glob(f"**/report-fold-{fold}.json"):
            r = json.loads(rp.read_text())
            if r.get("protocol", {}).get("assignment_arm") == arm:
                candidates.append((rp, r))
        if len(candidates) != 1:
            raise RuntimeError(f"arm={arm} fold={fold}: expected one report, got {len(candidates)}")
        rp, report = candidates[0]
        npzs = list(rp.parent.glob(f"predictions-fold-{fold}.npz"))
        if len(npzs) != 1:
            raise RuntimeError(f"arm={arm} fold={fold}: prediction shard missing")
        p = report["protocol"]
        assert p["controlled_assignment_ab"] is True
        assert p["only_assignment_differs_between_arms"] is True
        assert p["v16_proposal_graph_unchanged"] is True
        assert p["shared_seed"] == 16061
        assert p["fit_fold_only_presence_weighting"] is True
        assert p["symmetric_object_no_object_weighting"] is True
        assert p["presence_threshold_tuned"] is False and p["presence_threshold"] == 0.5
        reports.append(report)
        with np.load(npzs[0], allow_pickle=False) as z:
            n = len(z["global_index"])
            part = {key: np.asarray(z[key]) for key in z.files if np.asarray(z[key]).ndim and len(z[key]) == n}
        if pred_key not in part:
            raise RuntimeError(f"arm={arm} fold={fold}: {pred_key} missing")
        parts.append(part)
    common = set(parts[0])
    for p in parts[1:]:
        common &= set(p)
    merged = {key: np.concatenate([p[key] for p in parts], axis=0) for key in common}
    order = np.argsort(merged["global_index"], kind="stable")
    merged = {key: value[order] for key, value in merged.items()}
    if len(merged["global_index"]) != 76768:
        raise RuntimeError(f"arm={arm}: expected 76768 rows")
    if len(np.unique(merged["global_index"])) != 76768:
        raise RuntimeError(f"arm={arm}: duplicate outer rows")
    return reports, merged


def _per_k(k, preds):
    out = {}
    for value in range(SLOT_COUNT + 1):
        mask = k == value
        row = {"clusters": int(np.sum(mask))}
        for name, pred in preds.items():
            p = np.asarray(pred)[mask]
            row[name] = {
                "exact": float(np.mean(p == value)) if len(p) else None,
                "under_rate": float(np.mean(p < value)) if len(p) else None,
                "over_rate": float(np.mean(p > value)) if len(p) else None,
                "mae": float(np.mean(np.abs(p - value))) if len(p) else None,
                "mean_predicted_k": float(np.mean(p)) if len(p) else None,
                "prediction_distribution": {str(v): int(np.sum(p == v)) for v in range(SLOT_COUNT + 1)},
            }
        out[str(value)] = row
    return out


def _transition(a, b):
    counts = [[int(np.sum((a == i) & (b == j))) for j in range(7)] for i in range(7)]
    pct = []
    for row in counts:
        denom = sum(row)
        pct.append([float(v / denom) if denom else None for v in row])
    return {"rows_ordered_k_columns_permutation_k": counts, "row_percentages": pct}


def _aggregate_occupancy(reports):
    out = {}
    for q in range(6):
        rows = []
        for r in reports:
            occ = r["controlled_ab"]["outer_match_occupancy"][str(q)]
            n = int(r["data"]["outer_clusters"])
            rows.append((n, occ))
        total = sum(n for n, _ in rows)
        out[str(q)] = {
            "matched_object_rate": sum(n * x["matched_object_rate"] for n, x in rows) / total,
            "matched_no_object_rate": sum(n * x["matched_no_object_rate"] for n, x in rows) / total,
            "matched_valid_detail_rate": sum(n * x["matched_valid_detail_rate"] for n, x in rows) / total,
            "by_fold": {str(r["outer_fold"]): r["controlled_ab"]["outer_match_occupancy"][str(q)] for r in reports},
        }
    return out


def summarize(args):
    ro, mo = _load_arm(args.input_dir, "ordered")
    rp, mp = _load_arm(args.input_dir, "permutation")
    for key in ("global_index", "k", "member", "pred104"):
        if not np.array_equal(np.asarray(mo[key]).astype(str), np.asarray(mp[key]).astype(str)):
            raise RuntimeError(f"A/B row mismatch: {key}")
    k = np.asarray(mo["k"], dtype=np.int32)
    preds = {
        "v104": np.asarray(mo["pred104"], dtype=np.int32),
        "ordered": np.asarray(mo["pred171_ordered"], dtype=np.int32),
        "permutation": np.asarray(mp["pred171_permutation"], dtype=np.int32),
    }
    strata_names = ["aggregate", "comp", "solo", "player00", "player00_comp", "player00_rock_comp"]
    strata = {}
    for s in strata_names:
        row = {}
        row["v104"] = _metric_sum([r["strata"].get(s, {}).get("v104", {}).get("metrics", {}).get("global") for r in ro])
        for arm, reports in (("ordered", ro), ("permutation", rp)):
            model_key = f"v171_{arm}"
            row[arm] = _metric_sum([r["strata"].get(s, {}).get(model_key, {}).get("metrics", {}).get("global") for r in reports])
        strata[s] = row
    cards = {name: _card(k, pred) for name, pred in preds.items()}
    per_k = _per_k(k, preds)
    folds = {}
    for fold in range(5):
        a = next(r for r in ro if int(r["outer_fold"]) == fold)
        b = next(r for r in rp if int(r["outer_fold"]) == fold)
        folds[str(fold)] = {
            "ordered_selected_epochs": int(a["data"]["selected_epochs"]),
            "permutation_selected_epochs": int(b["data"]["selected_epochs"]),
            "ordered_f1": float(a["strata"]["aggregate"]["v171_ordered"]["metrics"]["global"]["f1"]),
            "permutation_f1": float(b["strata"]["aggregate"]["v171_permutation"]["metrics"]["global"]["f1"]),
            "delta_permutation_minus_ordered_f1": float(b["strata"]["aggregate"]["v171_permutation"]["metrics"]["global"]["f1"] - a["strata"]["aggregate"]["v171_ordered"]["metrics"]["global"]["f1"]),
            "ordered_final_weights": a["controlled_ab"]["final_fit_presence_weights"],
            "permutation_final_weights": b["controlled_ab"]["final_fit_presence_weights"],
            "ordered_gradient_mass": a["controlled_ab"].get("final_model_presence_gradient_mass"),
            "permutation_gradient_mass": b["controlled_ab"].get("final_model_presence_gradient_mass"),
        }
        if a["controlled_ab"]["final_fit_presence_weights"] != b["controlled_ab"]["final_fit_presence_weights"]:
            raise RuntimeError(f"fold {fold}: class weighting differs between arms")
    result = {
        "schema_version": 1,
        "protocol": {
            "causal_assignment_ab": True,
            "outer_clean_rows": 76768,
            "same_rows_both_arms": True,
            "v16_proposal_graph_unchanged": True,
            "same_seed": 16061,
            "same_threshold": 0.5,
            "same_fit_fold_only_symmetric_presence_weighting": True,
            "only_treatment": "fixed ordered identity assignment vs exact 6! minimum assignment",
            "locked12_indexed_or_evaluated": False,
        },
        "strata": strata,
        "cardinality": cards,
        "per_true_k": per_k,
        "kpred_transition": _transition(preds["ordered"], preds["permutation"]),
        "match_occupancy": {"ordered": _aggregate_occupancy(ro), "permutation": _aggregate_occupancy(rp)},
        "folds": folds,
        "comparison": {
            "global_f1_delta_permutation_minus_ordered": strata["aggregate"]["permutation"]["f1"] - strata["aggregate"]["ordered"]["f1"],
            "precision_delta_permutation_minus_ordered": strata["aggregate"]["permutation"]["precision"] - strata["aggregate"]["ordered"]["precision"],
            "recall_delta_permutation_minus_ordered": strata["aggregate"]["permutation"]["recall"] - strata["aggregate"]["ordered"]["recall"],
            "poly_exact_delta_permutation_minus_ordered": cards["permutation"]["poly_accuracy"] - cards["ordered"]["poly_accuracy"],
            "k4_delta": per_k["4"]["permutation"]["exact"] - per_k["4"]["ordered"]["exact"],
            "k5_delta": per_k["5"]["permutation"]["exact"] - per_k["5"]["ordered"]["exact"],
            "k6_delta": per_k["6"]["permutation"]["exact"] - per_k["6"]["ordered"]["exact"],
            "folds_permutation_wins": int(sum(v["delta_permutation_minus_ordered_f1"] > 0 for v in folds.values())),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(args.output_dir / "predictions.npz", global_index=np.asarray(mo["global_index"]), k=k, pred104=preds["v104"], pred171_ordered=preds["ordered"], pred171_permutation=preds["permutation"])
    print(json.dumps({
        "ordered_f1": strata["aggregate"]["ordered"]["f1"],
        "permutation_f1": strata["aggregate"]["permutation"]["f1"],
        "delta_f1": result["comparison"]["global_f1_delta_permutation_minus_ordered"],
        "ordered_poly": cards["ordered"]["poly_accuracy"],
        "permutation_poly": cards["permutation"]["poly_accuracy"],
        "ordered_k5": per_k["5"]["ordered"]["exact"],
        "permutation_k5": per_k["5"]["permutation"]["exact"],
        "ordered_k6": per_k["6"]["ordered"]["exact"],
        "permutation_k6": per_k["6"]["permutation"]["exact"],
        "folds_permutation_wins": result["comparison"]["folds_permutation_wins"],
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
