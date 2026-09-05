"""Aggregate V19 dense birth-center decoder across five outer-clean folds."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Optional, Sequence
import numpy as np
from scripts import summarize_v171_controlled_assignment_ab as s171
from scripts import summarize_v176_shared_set_decoder as s176
from scripts import train_v190_dense_birth_centers as v190

MODEL_KEY = v190.MODEL_KEY
PRED_KEY = v190.PRED_KEY
THRESHOLD = v190.PRESENCE_THRESHOLD


def _safe_global(report, stratum, model_key):
    row = report.get("strata", {}).get(stratum)
    if not row or model_key not in row:
        return None
    return row[model_key].get("metrics", {}).get("global")


def _load_v190(root):
    reports, parts = [], []
    for fold in range(5):
        hits = []
        for rp in Path(root).glob(f"**/report-fold-{fold}.json"):
            r = json.loads(rp.read_text())
            if r.get("protocol", {}).get("v190_dense_birth_centers") is True:
                hits.append((rp, r))
        if len(hits) != 1:
            raise RuntimeError(f"v190 fold={fold}: expected one report, got {len(hits)}")
        rp, r = hits[0]; reports.append(r)
        npzs = list(rp.parent.glob(f"predictions-fold-{fold}.npz"))
        if len(npzs) != 1:
            raise RuntimeError(f"v190 fold={fold}: predictions missing")
        with np.load(npzs[0], allow_pickle=False) as z:
            n = len(z["global_index"])
            parts.append({key: np.asarray(z[key]) for key in z.files if np.asarray(z[key]).ndim and len(np.asarray(z[key])) == n})
    common = set(parts[0])
    for p in parts[1:]: common &= set(p)
    merged = {key: np.concatenate([p[key] for p in parts], axis=0) for key in common}
    order = np.argsort(merged["global_index"], kind="stable"); merged = {key: value[order] for key, value in merged.items()}
    if len(merged["global_index"]) != 76768 or len(np.unique(merged["global_index"])) != 76768:
        raise RuntimeError("v190 invalid outer-clean coverage")
    return reports, merged


def _weighted_center(reports):
    pos_n = poly_n = pos_exact = poly_exact = poly_hits = 0.0
    per_k = {str(k): {"rows": 0, "exact_sum": 0.0, "hit_sum": 0.0} for k in range(1, 7)}
    for r in reports:
        d = r["v190"]["architecture"]["dense_center_diagnostics"]
        pn = int(d["eligible_positive_rows"]); qn = int(d["eligible_poly_rows"])
        pos_n += pn; poly_n += qn
        if d["top6_exact_center_coverage_positive"] is not None: pos_exact += pn * float(d["top6_exact_center_coverage_positive"])
        if d["top6_exact_center_coverage_poly"] is not None: poly_exact += qn * float(d["top6_exact_center_coverage_poly"])
        if d["top6_mean_center_hit_fraction_poly"] is not None: poly_hits += qn * float(d["top6_mean_center_hit_fraction_poly"])
        for k in range(1, 7):
            x = d["per_true_k"][str(k)]; n = int(x["rows"]); per_k[str(k)]["rows"] += n
            if x["top6_exact_center_coverage"] is not None: per_k[str(k)]["exact_sum"] += n * float(x["top6_exact_center_coverage"])
            if x["top6_mean_center_hit_fraction"] is not None: per_k[str(k)]["hit_sum"] += n * float(x["top6_mean_center_hit_fraction"])
    return {
        "eligible_positive_rows": int(pos_n), "eligible_poly_rows": int(poly_n),
        "top6_exact_center_coverage_positive": float(pos_exact / pos_n) if pos_n else None,
        "top6_exact_center_coverage_poly": float(poly_exact / poly_n) if poly_n else None,
        "top6_mean_center_hit_fraction_poly": float(poly_hits / poly_n) if poly_n else None,
        "per_true_k": {k: {"rows": int(v["rows"]), "top6_exact_center_coverage": float(v["exact_sum"] / v["rows"]) if v["rows"] else None, "top6_mean_center_hit_fraction": float(v["hit_sum"] / v["rows"]) if v["rows"] else None} for k, v in per_k.items()},
    }


def summarize(args):
    r190, m190 = _load_v190(args.input_dir)
    r173, m173 = s176._load_version(args.v173_fold_dir, "v173")
    v173_summary = json.loads((args.v173_summary_dir / "report.json").read_text())
    for key in ("global_index", "k", "member", "pred104"):
        if not np.array_equal(np.asarray(m190[key]).astype(str), np.asarray(m173[key]).astype(str)):
            raise RuntimeError(f"row mismatch: {key}")
    k = np.asarray(m190["k"], dtype=np.int32)
    pred104 = np.asarray(m190["pred104"], dtype=np.int32); pred173 = np.asarray(m173["pred173_poibin"], dtype=np.int32); pred190 = np.asarray(m190[PRED_KEY], dtype=np.int32)
    preds = {"v104": pred104, "v173_poibin": pred173, MODEL_KEY: pred190}
    strata = {}
    names = ["aggregate", "comp", "solo", "player00", "player00_comp", "player00_rock_comp"]
    for name in names:
        b = s171._metric_sum([_safe_global(r, name, "v104") for r in r190]); n = s171._metric_sum([_safe_global(r, name, MODEL_KEY) for r in r190]); o = v173_summary["strata"][name]["v173_poibin"]
        strata[name] = {"v104": b, "v173_poibin": o, MODEL_KEY: n, "delta_v190_minus_v173_f1": float(n["f1"] - o["f1"]), "delta_v190_minus_v104_f1": float(n["f1"] - b["f1"]), "delta_v190_minus_v173_precision": float(n["precision"] - o["precision"]), "delta_v190_minus_v173_recall": float(n["recall"] - o["recall"]), "delta_v190_minus_v173_pred_ref": float(n["prediction_reference_ratio"] - o["prediction_reference_ratio"])}
    cards = {name: s171._card(k, pred) for name, pred in preds.items()}; per_k = s171._per_k(k, preds)
    folds = {}; wins173 = wins104 = 0
    for fold in range(5):
        old = next(r for r in r173 if int(r["outer_fold"]) == fold); new = next(r for r in r190 if int(r["outer_fold"]) == fold)
        f173 = float(old["strata"]["aggregate"]["v173_poibin"]["metrics"]["global"]["f1"]); f190 = float(new["strata"]["aggregate"][MODEL_KEY]["metrics"]["global"]["f1"]); f104 = float(new["strata"]["aggregate"]["v104"]["metrics"]["global"]["f1"])
        wins173 += int(f190 > f173); wins104 += int(f190 > f104)
        a = new["v190"]["architecture"]; c = a["dense_center_diagnostics"]
        folds[str(fold)] = {"selected_epochs": int(new["data"]["selected_epochs"]), "v104_f1": f104, "v173_f1": f173, "v190_f1": f190, "delta_v190_minus_v173_f1": f190-f173, "delta_v190_minus_v104_f1": f190-f104, "activity_gini": float(a["outer_activity_gini"]), "effective_active_slots": float(a["outer_effective_active_slots"]), "active_occupancy_correlation": float(a["outer_active_occupancy_correlation"]), "center_top6_exact_poly": c["top6_exact_center_coverage_poly"], "center_top6_hit_fraction_poly": c["top6_mean_center_hit_fraction_poly"]}
    spec173 = s176._specialization(r173, "v173"); spec190 = s176._specialization(r190, "v190")
    dup173 = s176._duplicate_slices(k, np.asarray(m173["presence"], dtype=np.float64), np.asarray(m173["event_candidate"], dtype=np.float64)); dup190 = s176._duplicate_slices(k, np.asarray(m190["presence"], dtype=np.float64), np.asarray(m190["event_candidate"], dtype=np.float64))
    center = _weighted_center(r190)
    soft173 = float(np.mean(np.sum(np.asarray(m173["presence"], dtype=np.float64), axis=1))); soft190 = float(np.mean(np.sum(np.asarray(m190["presence"], dtype=np.float64), axis=1)))
    comparison = {
        "global_f1_v104": float(strata["aggregate"]["v104"]["f1"]), "global_f1_v173": float(strata["aggregate"]["v173_poibin"]["f1"]), "global_f1_v190": float(strata["aggregate"][MODEL_KEY]["f1"]),
        "delta_v190_minus_v173_f1": float(strata["aggregate"]["delta_v190_minus_v173_f1"]), "delta_v190_minus_v104_f1": float(strata["aggregate"]["delta_v190_minus_v104_f1"]), "delta_v190_minus_v173_precision": float(strata["aggregate"]["delta_v190_minus_v173_precision"]), "delta_v190_minus_v173_recall": float(strata["aggregate"]["delta_v190_minus_v173_recall"]),
        "folds_v190_beats_v173": int(wins173), "folds_v190_beats_v104": int(wins104),
        "poly_exact_v104": float(cards["v104"]["poly_accuracy"]), "poly_exact_v173": float(cards["v173_poibin"]["poly_accuracy"]), "poly_exact_v190": float(cards[MODEL_KEY]["poly_accuracy"]), "delta_poly_v190_minus_v173": float(cards[MODEL_KEY]["poly_accuracy"] - cards["v173_poibin"]["poly_accuracy"]),
        "soft_mass_v173": soft173, "soft_mass_v190": soft190,
        "activity_gini_v173": spec173["mean_active_rate_gini"], "activity_gini_v190": spec190["mean_active_rate_gini"], "effective_active_slots_v173": spec173["mean_effective_active_slots"], "effective_active_slots_v190": spec190["mean_effective_active_slots"], "min_active_occupancy_correlation_v173": spec173["min_active_occupancy_correlation"], "min_active_occupancy_correlation_v190": spec190["min_active_occupancy_correlation"],
        "raw_candidate_duplicate_poly_exact_v173": dup173["raw_duplicate_poly_exact_count"], "raw_candidate_duplicate_poly_exact_v190": dup190["raw_duplicate_poly_exact_count"],
        "center_top6_exact_poly": center["top6_exact_center_coverage_poly"], "center_top6_hit_fraction_poly": center["top6_mean_center_hit_fraction_poly"],
        "player00_rock_comp_f1_v104": float(strata["player00_rock_comp"]["v104"]["f1"]), "player00_rock_comp_f1_v173": float(strata["player00_rock_comp"]["v173_poibin"]["f1"]), "player00_rock_comp_f1_v190": float(strata["player00_rock_comp"][MODEL_KEY]["f1"]),
    }
    for value in range(2, 7): comparison[f"delta_k{value}_exact_v190_minus_v173"] = float(per_k[str(value)][MODEL_KEY]["exact"] - per_k[str(value)]["v173_poibin"]["exact"])
    gates = {"global_f1_improved_vs_v173": comparison["delta_v190_minus_v173_f1"] > 0, "majority_folds_improved_vs_v173": wins173 >= 3, "beats_v104_global": comparison["delta_v190_minus_v104_f1"] > 0, "poly_exact_improved_vs_v173": comparison["delta_poly_v190_minus_v173"] > 0, "dense_center_top6_exact_poly_above_90pct": center["top6_exact_center_coverage_poly"] is not None and center["top6_exact_center_coverage_poly"] > 0.90, "protected_player00_rock_comp_above_v104": comparison["player00_rock_comp_f1_v190"] > comparison["player00_rock_comp_f1_v104"]}
    result = {"schema_version": 1, "protocol": {"v190_dense_birth_centers": True, "mandatory_post_implementation_audit": True, "outer_clean_rows": 76768, "same_rows_as_v173": True, "same_seed": 16061, "presence_threshold": THRESHOLD, "threshold_tuned": False, "dense_center_grid": [23, 64], "center_map_loss_weight": v190.CENTER_MAP_WEIGHT, "center_map_weight_tuned": False, "v173_poisson_binomial_count_objective_unchanged": True, "mass_preserving_exchangeable_weights_unchanged": True, "exact_720_truth_matching_unchanged": True, "categorical_cardinality_head_exists": False, "locked12_indexed_or_evaluated": False}, "strata": strata, "cardinality": cards, "per_true_k": per_k, "folds": folds, "specialization": {"v173": spec173, "v190": spec190}, "duplicates": {"v173": dup173, "v190": dup190}, "dense_center": center, "comparison": comparison, "gates": gates}
    args.output_dir.mkdir(parents=True, exist_ok=True); (args.output_dir / "report.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(args.output_dir / "predictions.npz", global_index=np.asarray(m190["global_index"]), k=k, pred104=pred104, pred173=pred173, pred190=pred190, presence173=np.asarray(m173["presence"]), presence190=np.asarray(m190["presence"]))
    print(json.dumps(comparison, indent=2, sort_keys=True)); print(json.dumps(gates, indent=2, sort_keys=True)); return result


def parser():
    p = argparse.ArgumentParser(); p.add_argument("--input-dir", type=Path, required=True); p.add_argument("--v173-fold-dir", type=Path, required=True); p.add_argument("--v173-summary-dir", type=Path, required=True); p.add_argument("--output-dir", type=Path, required=True); return p


def main(argv: Optional[Sequence[str]] = None): summarize(parser().parse_args(argv)); return 0
if __name__ == "__main__": raise SystemExit(main())
