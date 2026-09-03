"""Deep post-hoc audit for the V17.1 controlled assignment A/B.

Consumes the corrected V17.1 summary artifact only. No training, threshold tuning,
or Locked12 access. The audit isolates the dominant loss/calibration failure and
quantifies assignment as a secondary treatment effect.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

SLOTS = 6


def _v16_balanced_ratio(p: float) -> dict:
    if not (0.0 < p < 1.0):
        return {"prevalence": p, "positive_weight": None, "negative_weight": None, "ratio": None}
    wp = float(np.clip(math.sqrt(1.0 / (2.0 * p)), 0.35, 8.0))
    wn = float(np.clip(math.sqrt(1.0 / (2.0 * (1.0 - p))), 0.35, 8.0))
    mean = p * wp + (1.0 - p) * wn
    wp /= mean
    wn /= mean
    return {"prevalence": p, "positive_weight": wp, "negative_weight": wn, "ratio": wp / wn}


def _entropy_norm(values):
    x = np.asarray(values, dtype=np.float64)
    s = float(np.sum(x))
    if s <= 0:
        return 0.0
    p = x / s
    nz = p > 0
    return float(-np.sum(p[nz] * np.log(p[nz])) / math.log(len(p)))


def audit(summary_dir: Path, output_dir: Path):
    report = json.loads((summary_dir / "report.json").read_text())
    with np.load(summary_dir / "predictions.npz", allow_pickle=False) as z:
        k = np.asarray(z["k"], dtype=np.int32)
        ordered = np.asarray(z["pred171_ordered"], dtype=np.int32)
        permutation = np.asarray(z["pred171_permutation"], dtype=np.int32)

    strata = report["strata"]["aggregate"]
    cards = report["cardinality"]
    comp = report["comparison"]

    transition = {
        "changed_rate": float(np.mean(ordered != permutation)),
        "up_rate": float(np.mean(permutation > ordered)),
        "down_rate": float(np.mean(permutation < ordered)),
        "mean_delta_k": float(np.mean(permutation - ordered)),
        "mean_abs_delta_k": float(np.mean(np.abs(permutation - ordered))),
        "by_true_k": {},
    }
    for value in range(SLOTS + 1):
        m = k == value
        transition["by_true_k"][str(value)] = {
            "rows": int(np.sum(m)),
            "up_rate": float(np.mean(permutation[m] > ordered[m])),
            "down_rate": float(np.mean(permutation[m] < ordered[m])),
            "same_rate": float(np.mean(permutation[m] == ordered[m])),
            "mean_delta_k": float(np.mean(permutation[m] - ordered[m])),
        }

    weighting = {"folds": {}, "aggregate_distortion": {}}
    distortions = {str(q): [] for q in range(SLOTS)}
    for fold in range(5):
        f = report["folds"][str(fold)]
        w = f["ordered_final_weights"]
        common_ratio = float(w["positive_weight"] / w["negative_weight"])
        qrows = {}
        for q in range(SLOTS):
            p = float(report["match_occupancy"]["ordered"][str(q)]["by_fold"][str(fold)]["matched_object_rate"])
            v16 = _v16_balanced_ratio(p)
            distortion = common_ratio / v16["ratio"] if v16["ratio"] else None
            if distortion is not None:
                distortions[str(q)].append(distortion)
            qrows[str(q)] = {**v16, "v171_common_ratio": common_ratio, "ratio_distortion_v171_over_v16": distortion}
        weighting["folds"][str(fold)] = {
            "v171_positive_weight": w["positive_weight"],
            "v171_negative_weight": w["negative_weight"],
            "v171_ratio": common_ratio,
            "queries": qrows,
        }
    for q, vals in distortions.items():
        weighting["aggregate_distortion"][q] = {
            "mean": float(np.mean(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }

    occupancy = {"ordered": {}, "permutation": {}, "permutation_fold_entropy": {}}
    for arm in ("ordered", "permutation"):
        for q in range(SLOTS):
            src = report["match_occupancy"][arm][str(q)]
            vals = np.asarray([src["by_fold"][str(f)]["matched_object_rate"] for f in range(5)], dtype=np.float64)
            occupancy[arm][str(q)] = {
                "aggregate_object_rate": float(src["matched_object_rate"]),
                "fold_min": float(vals.min()),
                "fold_max": float(vals.max()),
                "fold_cv": float(vals.std() / vals.mean()) if vals.mean() else None,
            }
    for fold in range(5):
        vals = [report["match_occupancy"]["permutation"][str(q)]["by_fold"][str(fold)]["matched_object_rate"] for q in range(SLOTS)]
        occupancy["permutation_fold_entropy"][str(fold)] = {
            "object_mass": float(sum(vals)),
            "normalized_slot_entropy": _entropy_norm(vals),
            "slot_object_rates": [float(x) for x in vals],
        }

    gradients = {"folds": {}, "mean": {}}
    agg = {"ordered": [], "permutation": []}
    for fold in range(5):
        gradients["folds"][str(fold)] = {}
        for arm in ("ordered", "permutation"):
            g = report["folds"][str(fold)][f"{arm}_gradient_mass"]
            ratio = float(g["positive_gradient_global_norm"] / g["negative_gradient_global_norm"])
            row = {**g, "positive_negative_gradient_ratio": ratio}
            gradients["folds"][str(fold)][arm] = row
            agg[arm].append([g["positive_presence_loss"], g["negative_presence_loss"], g["positive_gradient_global_norm"], g["negative_gradient_global_norm"], ratio])
    for arm, vals in agg.items():
        a = np.asarray(vals, dtype=np.float64)
        gradients["mean"][arm] = {
            "positive_presence_loss": float(a[:, 0].mean()),
            "negative_presence_loss": float(a[:, 1].mean()),
            "positive_gradient_global_norm": float(a[:, 2].mean()),
            "negative_gradient_global_norm": float(a[:, 3].mean()),
            "positive_negative_gradient_ratio": float(a[:, 4].mean()),
        }

    v104_f1 = float(strata["v104"]["f1"])
    ordered_f1 = float(strata["ordered"]["f1"])
    perm_f1 = float(strata["permutation"]["f1"])
    matching_effect = abs(perm_f1 - ordered_f1)
    controlled_gap = abs(v104_f1 - ordered_f1)

    result = {
        "schema_version": 1,
        "protocol": {
            "source_summary_rows": int(len(k)),
            "training_performed": False,
            "threshold_tuned": False,
            "locked12_indexed_or_evaluated": False,
        },
        "headline": {
            "v104_f1": v104_f1,
            "ordered_f1": ordered_f1,
            "permutation_f1": perm_f1,
            "matching_delta_f1": float(comp["global_f1_delta_permutation_minus_ordered"]),
            "matching_effect_fraction_of_controlled_gap": float(matching_effect / controlled_gap) if controlled_gap else None,
            "ordered_precision": float(strata["ordered"]["precision"]),
            "ordered_recall": float(strata["ordered"]["recall"]),
            "ordered_pred_ref": float(strata["ordered"]["prediction_reference_ratio"]),
            "permutation_precision": float(strata["permutation"]["precision"]),
            "permutation_recall": float(strata["permutation"]["recall"]),
            "permutation_pred_ref": float(strata["permutation"]["prediction_reference_ratio"]),
            "v104_over_rate": float(cards["v104"]["over_rate"]),
            "ordered_over_rate": float(cards["ordered"]["over_rate"]),
            "permutation_over_rate": float(cards["permutation"]["over_rate"]),
            "truth_mean_k": float(cards["ordered"]["mean_true_k"]),
            "ordered_mean_k": float(cards["ordered"]["mean_predicted_k"]),
            "permutation_mean_k": float(cards["permutation"]["mean_predicted_k"]),
        },
        "per_true_k": report["per_true_k"],
        "transition": transition,
        "weighting": weighting,
        "occupancy": occupancy,
        "gradients": gradients,
        "diagnosis": {
            "dominant_failure": "V17.1 equal-coefficient-mass linear class weighting overdrives presence and causes global overcount; assignment matching is secondary.",
            "evidence": [
                "ordered/permutation pred-ref ratios are far above 1 while V10.4 is below 1",
                "K0 false-positive cardinality rises from V10.4 5.6% to roughly 27-30%",
                "K2/K3 are overcounted in roughly 58% of rows under both controlled arms",
                "V17.1 applies one approximately 10x positive/negative ratio to every query whereas V16 sqrt-balanced query ratios are much smaller for common early queries",
                "permutation changes only about 18% of K decisions and its aggregate F1 effect is small relative to the controlled-loss gap",
                "permutation slot identity is fold-unstable, so hard matching redistributes ownership but does not explain the dominant calibration collapse",
            ],
            "next_experiment": "Mass-preserving exchangeable cardinality-conditioned presence weighting: derive V16 per-query sqrt weights on each fit fold, compress their total positive/no-object mass by true K so all object slots are exchangeable, then compare ordered vs exact permutation matching with all else fixed.",
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "dominant_failure": result["diagnosis"]["dominant_failure"],
        "ordered_pred_ref": result["headline"]["ordered_pred_ref"],
        "permutation_pred_ref": result["headline"]["permutation_pred_ref"],
        "matching_delta_f1": result["headline"]["matching_delta_f1"],
        "matching_effect_fraction_of_controlled_gap": result["headline"]["matching_effect_fraction_of_controlled_gap"],
        "q0_weight_ratio_distortion_mean": result["weighting"]["aggregate_distortion"]["0"]["mean"],
        "next_experiment": result["diagnosis"]["next_experiment"],
    }, indent=2, sort_keys=True))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--summary-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    audit(args.summary_dir, args.output_dir)


if __name__ == "__main__":
    main()
