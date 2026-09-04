"""Deep post-hoc audit for V17.2 mass-preserving exchangeable causal decomposition.

Consumes the V17.2 summary artifact plus the 15 completed fold reports. No
training, threshold tuning, or Locked12 access. The audit identifies what A->B
and B->C actually fixed, where the remaining gap to V10.4 lives, and which
architectural constraint should be tested next.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

ARMS = ("ranked_ordered", "mass_ordered", "mass_permutation")
SLOTS = 6


def _entropy_norm(values):
    x = np.asarray(values, dtype=np.float64)
    s = float(np.sum(x))
    if s <= 0.0:
        return 0.0
    p = x / s
    nz = p > 0
    return float(-np.sum(p[nz] * np.log(p[nz])) / math.log(len(p)))


def _load_fold_reports(folds_dir: Path):
    found = {arm: {} for arm in ARMS}
    for rp in folds_dir.glob("**/report-fold-*.json"):
        try:
            r = json.loads(rp.read_text())
        except Exception:
            continue
        p = r.get("protocol", {})
        if p.get("v172_mass_preserving_decomposition") is not True:
            continue
        arm = p.get("assignment_arm")
        fold = int(r.get("outer_fold", -1))
        if arm in found and 0 <= fold < 5:
            if fold in found[arm]:
                raise RuntimeError(f"duplicate fold report arm={arm} fold={fold}")
            found[arm][fold] = r
    for arm in ARMS:
        missing = sorted(set(range(5)) - set(found[arm]))
        if missing:
            raise RuntimeError(f"missing V17.2 fold reports arm={arm}: {missing}")
    return found


def _transition(k, src, dst):
    k = np.asarray(k, dtype=np.int32)
    src = np.asarray(src, dtype=np.int32)
    dst = np.asarray(dst, dtype=np.int32)
    es = np.abs(src - k)
    ed = np.abs(dst - k)
    out = {
        "changed_rate": float(np.mean(src != dst)),
        "up_rate": float(np.mean(dst > src)),
        "down_rate": float(np.mean(dst < src)),
        "cardinality_error_improved_rate": float(np.mean(ed < es)),
        "cardinality_error_worsened_rate": float(np.mean(ed > es)),
        "cardinality_error_same_rate": float(np.mean(ed == es)),
        "mean_delta_k": float(np.mean(dst - src)),
        "by_true_k": {},
    }
    for value in range(SLOTS + 1):
        m = k == value
        out["by_true_k"][str(value)] = {
            "rows": int(np.sum(m)),
            "changed_rate": float(np.mean(src[m] != dst[m])),
            "up_rate": float(np.mean(dst[m] > src[m])),
            "down_rate": float(np.mean(dst[m] < src[m])),
            "cardinality_error_improved_rate": float(np.mean(ed[m] < es[m])),
            "cardinality_error_worsened_rate": float(np.mean(ed[m] > es[m])),
            "mean_src_k": float(np.mean(src[m])),
            "mean_dst_k": float(np.mean(dst[m])),
        }
    return out


def _strata_audit(report):
    out = {}
    for name, row in report["strata"].items():
        v = row["v104"]
        a = row["ranked_ordered"]
        b = row["mass_ordered"]
        c = row["mass_permutation"]
        out[name] = {
            "v104": v,
            "ranked_ordered": a,
            "mass_ordered": b,
            "mass_permutation": c,
            "delta_B_minus_A_f1": float(b["f1"] - a["f1"]),
            "delta_C_minus_B_f1": float(c["f1"] - b["f1"]),
            "delta_C_minus_v104_f1": float(c["f1"] - v["f1"]),
            "delta_C_minus_v104_precision": float(c["precision"] - v["precision"]),
            "delta_C_minus_v104_recall": float(c["recall"] - v["recall"]),
            "delta_C_minus_v104_pred_ref": float(c["prediction_reference_ratio"] - v["prediction_reference_ratio"]),
        }
    return out


def _cardinality_audit(report):
    cards = report["cardinality"]
    per_k = report["per_true_k"]
    out = {
        "global": {},
        "per_true_k": {},
    }
    for arm in ("v104",) + ARMS:
        x = cards[arm]
        out["global"][arm] = {
            "accuracy": float(x["accuracy"]),
            "poly_accuracy": float(x["poly_accuracy"]),
            "mae": float(x["mae"]),
            "over_rate": float(x["over_rate"]),
            "under_rate": float(x["under_rate"]),
            "mean_predicted_k": float(x["mean_predicted_k"]),
            "mean_true_k": float(x["mean_true_k"]),
            "prediction_reference_ratio": float(x["mean_predicted_k"] / x["mean_true_k"]),
            "confusion_true_rows_pred_columns": x["confusion_true_rows_pred_columns"],
        }
    out["global"]["C_minus_v104"] = {
        "accuracy": float(cards["mass_permutation"]["accuracy"] - cards["v104"]["accuracy"]),
        "poly_accuracy": float(cards["mass_permutation"]["poly_accuracy"] - cards["v104"]["poly_accuracy"]),
        "mae": float(cards["mass_permutation"]["mae"] - cards["v104"]["mae"]),
        "over_rate": float(cards["mass_permutation"]["over_rate"] - cards["v104"]["over_rate"]),
        "under_rate": float(cards["mass_permutation"]["under_rate"] - cards["v104"]["under_rate"]),
        "mean_predicted_k": float(cards["mass_permutation"]["mean_predicted_k"] - cards["v104"]["mean_predicted_k"]),
    }
    for value in range(SLOTS + 1):
        s = str(value)
        row = {arm: per_k[s][arm] for arm in ("v104",) + ARMS}
        row["delta_B_minus_A_exact"] = float(row["mass_ordered"]["exact"] - row["ranked_ordered"]["exact"])
        row["delta_C_minus_B_exact"] = float(row["mass_permutation"]["exact"] - row["mass_ordered"]["exact"])
        row["delta_C_minus_v104_exact"] = float(row["mass_permutation"]["exact"] - row["v104"]["exact"])
        row["delta_C_minus_v104_over_rate"] = float(row["mass_permutation"]["over_rate"] - row["v104"]["over_rate"])
        row["delta_C_minus_v104_under_rate"] = float(row["mass_permutation"]["under_rate"] - row["v104"]["under_rate"])
        out["per_true_k"][s] = row
    return out


def _presence_budget(report):
    true_mean = float(report["cardinality"]["v104"]["mean_true_k"])
    out = {"truth_mean_k": true_mean, "arms": {}}
    for arm in ARMS:
        presence = report["diagnostics"][arm]["presence"]
        probability_mass = float(sum(float(presence[str(q)]["mean"]) for q in range(SLOTS)))
        hard_mean = float(report["cardinality"][arm]["mean_predicted_k"])
        out["arms"][arm] = {
            "mean_sum_presence_probability": probability_mass,
            "probability_mass_bias": probability_mass - true_mean,
            "probability_mass_ratio_to_truth": probability_mass / true_mean,
            "hard_mean_predicted_k": hard_mean,
            "hard_count_bias": hard_mean - true_mean,
            "hard_count_ratio_to_truth": hard_mean / true_mean,
            "soft_minus_hard_mean_count": probability_mass - hard_mean,
            "query_mean_presence": [float(presence[str(q)]["mean"]) for q in range(SLOTS)],
            "query_active_rate_at_0p5": [float(presence[str(q)]["active_rate_at_0p5"]) for q in range(SLOTS)],
        }
    return out


def _fold_and_weight_audit(summary, folds):
    out = {
        "paired_f1": {},
        "mass_weights": {"by_fold": {}, "aggregate_by_k": {}},
        "permutation_occupancy": {"by_fold": {}},
        "gradient_diagnostic": {"by_arm": {}},
    }
    delta_ab, delta_bc, delta_cv = [], [], []
    for fold in range(5):
        a = folds["ranked_ordered"][fold]
        b = folds["mass_ordered"][fold]
        c = folds["mass_permutation"][fold]
        ga = a["strata"]["aggregate"]["v172_ranked_ordered"]["metrics"]["global"]
        gb = b["strata"]["aggregate"]["v172_mass_ordered"]["metrics"]["global"]
        gc = c["strata"]["aggregate"]["v172_mass_permutation"]["metrics"]["global"]
        gv = a["strata"]["aggregate"]["v104"]["metrics"]["global"]
        dab = float(gb["f1"] - ga["f1"])
        dbc = float(gc["f1"] - gb["f1"])
        dcv = float(gc["f1"] - gv["f1"])
        delta_ab.append(dab); delta_bc.append(dbc); delta_cv.append(dcv)
        out["paired_f1"][str(fold)] = {
            "v104_f1": float(gv["f1"]),
            "ranked_ordered_f1": float(ga["f1"]),
            "mass_ordered_f1": float(gb["f1"]),
            "mass_permutation_f1": float(gc["f1"]),
            "delta_B_minus_A": dab,
            "delta_C_minus_B": dbc,
            "delta_C_minus_v104": dcv,
            "C_pred_ref": float(gc["prediction_reference_ratio"]),
        }
        spec = b["v172"]["final_fit_weight_spec"]["mass"]
        out["mass_weights"]["by_fold"][str(fold)] = spec

        occ = c["v172"]["outer_match_occupancy"]
        rates = [float(occ[str(q)]["matched_object_rate"]) for q in range(SLOTS)]
        out["permutation_occupancy"]["by_fold"][str(fold)] = {
            "slot_object_rates": rates,
            "normalized_slot_entropy": _entropy_norm(rates),
            "max_minus_min_object_rate": float(max(rates) - min(rates)),
        }

    out["paired_f1"]["aggregate"] = {
        "B_beats_A_folds": int(np.sum(np.asarray(delta_ab) > 0)),
        "C_beats_B_folds": int(np.sum(np.asarray(delta_bc) > 0)),
        "C_beats_v104_folds": int(np.sum(np.asarray(delta_cv) > 0)),
        "B_minus_A_mean": float(np.mean(delta_ab)),
        "B_minus_A_std": float(np.std(delta_ab)),
        "C_minus_B_mean": float(np.mean(delta_bc)),
        "C_minus_B_std": float(np.std(delta_bc)),
        "C_minus_v104_mean": float(np.mean(delta_cv)),
        "C_minus_v104_std": float(np.std(delta_cv)),
    }

    for kval in range(SLOTS + 1):
        objs, nulls, totals = [], [], []
        for fold in range(5):
            spec = out["mass_weights"]["by_fold"][str(fold)]
            objs.append(float(spec["object_by_k"][kval]))
            nulls.append(float(spec["no_object_by_k"][kval]))
            totals.append(float(spec["proof_by_k"][str(kval)]["compressed_total_presence_coefficient_mass"]))
        out["mass_weights"]["aggregate_by_k"][str(kval)] = {
            "object_weight_mean": float(np.mean(objs)),
            "object_weight_min": float(np.min(objs)),
            "object_weight_max": float(np.max(objs)),
            "no_object_weight_mean": float(np.mean(nulls)),
            "no_object_weight_min": float(np.min(nulls)),
            "no_object_weight_max": float(np.max(nulls)),
            "total_presence_coefficient_mass_mean": float(np.mean(totals)),
        }

    for arm in ARMS:
        errors, good = [], 0
        for fold in range(5):
            v = folds[arm][fold]["v172"]
            if v.get("final_model_presence_gradient_mass"):
                good += 1
            if v.get("final_model_presence_gradient_mass_error"):
                errors.append(v["final_model_presence_gradient_mass_error"])
        out["gradient_diagnostic"]["by_arm"][arm] = {
            "successful_folds": good,
            "error_folds": len(errors),
            "errors": errors,
            "diagnostic_only": True,
        }
    return out


def audit(summary_dir: Path, folds_dir: Path, output_dir: Path):
    summary = json.loads((summary_dir / "report.json").read_text())
    with np.load(summary_dir / "predictions.npz", allow_pickle=False) as z:
        k = np.asarray(z["k"], dtype=np.int32)
        a = np.asarray(z["pred172_ranked_ordered"], dtype=np.int32)
        b = np.asarray(z["pred172_mass_ordered"], dtype=np.int32)
        c = np.asarray(z["pred172_mass_permutation"], dtype=np.int32)
        v = np.asarray(z["pred104"], dtype=np.int32)
    if len(k) != 76768:
        raise RuntimeError(f"unexpected summary rows {len(k)}")
    folds = _load_fold_reports(folds_dir)

    strata = _strata_audit(summary)
    card = _cardinality_audit(summary)
    budget = _presence_budget(summary)
    fold_audit = _fold_and_weight_audit(summary, folds)

    transitions = {
        "A_to_B": _transition(k, a, b),
        "B_to_C": _transition(k, b, c),
        "v104_to_C": _transition(k, v, c),
    }

    agg = strata["aggregate"]
    c_global = card["global"]["mass_permutation"]
    v_global = card["global"]["v104"]
    k2 = card["per_true_k"]["2"]
    k3 = card["per_true_k"]["3"]
    k4 = card["per_true_k"]["4"]
    k5 = card["per_true_k"]["5"]
    k6 = card["per_true_k"]["6"]

    result = {
        "schema_version": 1,
        "protocol": {
            "source_run": 33815575062,
            "source_rows": int(len(k)),
            "training_performed": False,
            "threshold_tuned": False,
            "threshold": 0.5,
            "locked12_indexed_or_evaluated": False,
            "same_outer_rows_all_arms": True,
            "same_seed": 16061,
        },
        "headline": {
            "v104_f1": float(agg["v104"]["f1"]),
            "A_ranked_ordered_f1": float(agg["ranked_ordered"]["f1"]),
            "B_mass_ordered_f1": float(agg["mass_ordered"]["f1"]),
            "C_mass_permutation_f1": float(agg["mass_permutation"]["f1"]),
            "A_to_B_f1_delta": float(agg["delta_B_minus_A_f1"]),
            "B_to_C_f1_delta": float(agg["delta_C_minus_B_f1"]),
            "C_to_v104_f1_delta": float(agg["delta_C_minus_v104_f1"]),
            "C_to_v104_precision_delta": float(agg["delta_C_minus_v104_precision"]),
            "C_to_v104_recall_delta": float(agg["delta_C_minus_v104_recall"]),
            "C_pred_ref": float(agg["mass_permutation"]["prediction_reference_ratio"]),
            "v104_pred_ref": float(agg["v104"]["prediction_reference_ratio"]),
            "C_exact_k": float(c_global["accuracy"]),
            "v104_exact_k": float(v_global["accuracy"]),
            "C_poly_exact_k": float(c_global["poly_accuracy"]),
            "v104_poly_exact_k": float(v_global["poly_accuracy"]),
        },
        "strata": strata,
        "cardinality": card,
        "presence_budget": budget,
        "transitions": transitions,
        "folds_and_weights": fold_audit,
        "diagnosis": {
            "what_v172_fixed": [
                "Mass-preserving exchangeable weighting is a real improvement: B beats A on 5/5 folds and improves both F1 and poly exact-K while removing ranked query economics.",
                "Exact permutation matching is viable but secondary: C beats B on 4/5 folds with a much smaller aggregate F1 effect than A->B.",
                "The catastrophic V17/V17.1 calibration failure is gone; C pred/ref is close to 1 rather than grossly overcounting.",
            ],
            "remaining_gap": [
                "The global gap to V10.4 is now primarily precision, not recall: C has lower precision while recall is higher.",
                "The remaining cardinality deficit is concentrated in K2/K3 and K1; C already beats V10.4 exact-K at K4/K5/K6, so a generic high-K boost would attack the wrong problem.",
                "C overcounts medium-cardinality rows much more than V10.4, especially K3, while simultaneously undercounting less at K2-K6. The missing behavior is K-conditional allocation, not a single global threshold.",
                "The existing event_count_norm is only sum(presence)/6 with MSE supervision. Its outer probability mass remains biased high even when the hard thresholded mean K is close to truth, so expected-count supervision is not enough to align the discrete count decision.",
                "Hard permutation matching redistributes slot ownership and improves K4/K5 but does not provide a count-conserving mechanism; K6 remains fold-unstable.",
            ],
            "protected_gains": [
                "Preserve the A->B exchangeable mass-preserving weighting result.",
                "Preserve the player00 rock comp gain over V10.4 instead of optimizing only global F1.",
                "Preserve C's recall advantage and K4/K5 gains; do not recover precision by simply lowering all presence activations.",
            ],
            "next_architecture_hypothesis": (
                "Start from the exchangeable mass-preserving set decoder and add an explicit permutation-invariant count-conserving objectness mechanism, not another weight or threshold tweak. "
                "The strongest candidate is a count potential coupled to the six presence logits with an exact/Poisson-binomial cardinality consistency objective or a differentiable budget projection, so total active mass and its discrete cardinality are constrained jointly while object identity remains exchangeable. "
                "Do not add a categorical K classifier and do not tune the 0.5 threshold."
            ),
            "why_not_simple_tweaks": [
                "Increasing K5/K6 weights is contradicted by the audit because C already beats V10.4 exact-K for K4-K6.",
                "Lowering or raising one global threshold cannot simultaneously fix K2/K3 overcount and preserve the recovered K4/K5 behavior.",
                "Hard matching alone is too small an effect to close the remaining 1.37 pp global F1 gap.",
            ],
        },
        "derived_checks": {
            "C_precision_below_v104": bool(agg["mass_permutation"]["precision"] < agg["v104"]["precision"]),
            "C_recall_above_v104": bool(agg["mass_permutation"]["recall"] > agg["v104"]["recall"]),
            "C_K2_below_v104": bool(k2["delta_C_minus_v104_exact"] < 0),
            "C_K3_below_v104": bool(k3["delta_C_minus_v104_exact"] < 0),
            "C_K4_above_v104": bool(k4["delta_C_minus_v104_exact"] > 0),
            "C_K5_above_v104": bool(k5["delta_C_minus_v104_exact"] > 0),
            "C_K6_above_v104": bool(k6["delta_C_minus_v104_exact"] > 0),
            "C_medium_K_overcount_exceeds_v104": bool(k2["delta_C_minus_v104_over_rate"] > 0 and k3["delta_C_minus_v104_over_rate"] > 0),
            "C_soft_probability_mass_bias_fraction": float(budget["arms"]["mass_permutation"]["probability_mass_ratio_to_truth"] - 1.0),
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    brief = {
        "A_to_B_f1_delta_pp": 100.0 * result["headline"]["A_to_B_f1_delta"],
        "B_to_C_f1_delta_pp": 100.0 * result["headline"]["B_to_C_f1_delta"],
        "C_to_v104_f1_delta_pp": 100.0 * result["headline"]["C_to_v104_f1_delta"],
        "C_to_v104_precision_delta_pp": 100.0 * result["headline"]["C_to_v104_precision_delta"],
        "C_to_v104_recall_delta_pp": 100.0 * result["headline"]["C_to_v104_recall_delta"],
        "C_K2_delta_exact_pp": 100.0 * k2["delta_C_minus_v104_exact"],
        "C_K3_delta_exact_pp": 100.0 * k3["delta_C_minus_v104_exact"],
        "C_K4_delta_exact_pp": 100.0 * k4["delta_C_minus_v104_exact"],
        "C_K5_delta_exact_pp": 100.0 * k5["delta_C_minus_v104_exact"],
        "C_K6_delta_exact_pp": 100.0 * k6["delta_C_minus_v104_exact"],
        "C_soft_probability_mass_bias_pct": 100.0 * result["derived_checks"]["C_soft_probability_mass_bias_fraction"],
        "next_architecture_hypothesis": result["diagnosis"]["next_architecture_hypothesis"],
    }
    print(json.dumps(brief, indent=2, sort_keys=True))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--summary-dir", type=Path, required=True)
    p.add_argument("--folds-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    audit(args.summary_dir, args.folds_dir, args.output_dir)


if __name__ == "__main__":
    main()
