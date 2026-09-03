"""Aggregate the V17.2 three-arm mass-preserving causal decomposition."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from scripts import summarize_v171_controlled_assignment_ab as s171

ARMS = ("ranked_ordered", "mass_ordered", "mass_permutation")


def _load_arm(input_dir: Path, arm: str):
    reports, parts = [], []
    pred_key = f"pred172_{arm}"
    for fold in range(5):
        candidates = []
        for rp in input_dir.glob(f"**/report-fold-{fold}.json"):
            r = json.loads(rp.read_text())
            p = r.get("protocol", {})
            if p.get("v172_mass_preserving_decomposition") is True and p.get("assignment_arm") == arm:
                candidates.append((rp, r))
        if len(candidates) != 1:
            raise RuntimeError(f"arm={arm} fold={fold}: expected one report, got {len(candidates)}")
        rp, report = candidates[0]
        npzs = list(rp.parent.glob(f"predictions-fold-{fold}.npz"))
        if len(npzs) != 1:
            raise RuntimeError(f"arm={arm} fold={fold}: prediction shard missing")
        p = report["protocol"]
        assert p["v16_proposal_graph_unchanged"] is True
        assert p["shared_seed"] == 16061
        assert p["presence_threshold"] == 0.5 and p["presence_threshold_tuned"] is False
        assert p["fit_fold_only_sqrt_presence_weighting"] is True
        assert p["linear_equal_mass_weighting"] is False
        assert p["historical_validation_or_locked12_indexed_or_evaluated"] is False
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
    if len(merged["global_index"]) != 76768 or len(np.unique(merged["global_index"])) != 76768:
        raise RuntimeError(f"arm={arm}: invalid outer row coverage")
    return reports, merged


def _safe_global(report, stratum, model_key):
    row = report.get("strata", {}).get(stratum)
    if not row:
        return None
    model = row.get(model_key)
    if not model:
        return None
    return model.get("metrics", {}).get("global")


def _aggregate_presence(reports):
    out = {}
    total = sum(int(r["data"]["outer_clusters"]) for r in reports)
    for q in range(6):
        rows = []
        for r in reports:
            n = int(r["data"]["outer_clusters"])
            x = r["v172"]["outer_presence"][str(q)]
            rows.append((n, x))
        out[str(q)] = {
            "mean": sum(n * x["mean"] for n, x in rows) / total,
            "active_rate_at_0p5": sum(n * x["active_rate_at_0p5"] for n, x in rows) / total,
            "by_fold": {str(r["outer_fold"]): r["v172"]["outer_presence"][str(q)] for r in reports},
        }
    return out


def _aggregate_occupancy(reports):
    out = {}
    for q in range(6):
        rows = []
        for r in reports:
            n = int(r["data"]["outer_clusters"])
            x = r["v172"]["outer_match_occupancy"][str(q)]
            rows.append((n, x))
        total = sum(n for n, _ in rows)
        out[str(q)] = {
            "matched_object_rate": sum(n * x["matched_object_rate"] for n, x in rows) / total,
            "matched_no_object_rate": sum(n * x["matched_no_object_rate"] for n, x in rows) / total,
            "matched_valid_detail_rate": sum(n * x["matched_valid_detail_rate"] for n, x in rows) / total,
            "by_fold": {str(r["outer_fold"]): r["v172"]["outer_match_occupancy"][str(q)] for r in reports},
        }
    return out


def _aggregate_gradients(reports):
    rows = []
    by_fold = {}
    for r in reports:
        g = r["v172"].get("final_model_presence_gradient_mass")
        by_fold[str(r["outer_fold"])] = g
        if g:
            rows.append(g)
    if not rows:
        return {"mean": None, "by_fold": by_fold}
    keys = ["positive_presence_loss", "negative_presence_loss", "positive_gradient_global_norm", "negative_gradient_global_norm"]
    mean = {k: float(np.mean([x[k] for x in rows])) for k in keys}
    mean["positive_negative_gradient_ratio"] = float(mean["positive_gradient_global_norm"] / mean["negative_gradient_global_norm"]) if mean["negative_gradient_global_norm"] else None
    return {"mean": mean, "by_fold": by_fold}


def summarize(args):
    loaded = {arm: _load_arm(args.input_dir, arm) for arm in ARMS}
    base_reports, base_merged = loaded[ARMS[0]]
    for arm in ARMS[1:]:
        reports, merged = loaded[arm]
        for key in ("global_index", "k", "member", "pred104"):
            if not np.array_equal(np.asarray(base_merged[key]).astype(str), np.asarray(merged[key]).astype(str)):
                raise RuntimeError(f"row mismatch {ARMS[0]} vs {arm}: {key}")
        for fold in range(5):
            a = next(r for r in base_reports if int(r["outer_fold"]) == fold)
            b = next(r for r in reports if int(r["outer_fold"]) == fold)
            if a["v172"]["final_fit_weight_spec"] != b["v172"]["final_fit_weight_spec"]:
                raise RuntimeError(f"fold {fold}: weight spec mismatch for {arm}")

    k = np.asarray(base_merged["k"], dtype=np.int32)
    preds = {"v104": np.asarray(base_merged["pred104"], dtype=np.int32)}
    for arm in ARMS:
        preds[arm] = np.asarray(loaded[arm][1][f"pred172_{arm}"], dtype=np.int32)

    strata_names = ["aggregate", "comp", "solo", "player00", "player00_comp", "player00_rock_comp"]
    strata = {}
    for s in strata_names:
        row = {}
        row["v104"] = s171._metric_sum([_safe_global(r, s, "v104") for r in base_reports])
        for arm in ARMS:
            reports = loaded[arm][0]
            row[arm] = s171._metric_sum([_safe_global(r, s, f"v172_{arm}") for r in reports])
        strata[s] = row

    cards = {name: s171._card(k, pred) for name, pred in preds.items()}
    per_k = s171._per_k(k, preds)
    folds = {}
    for fold in range(5):
        row = {}
        for arm in ARMS:
            r = next(x for x in loaded[arm][0] if int(x["outer_fold"]) == fold)
            key = f"v172_{arm}"
            row[f"{arm}_epochs"] = int(r["data"]["selected_epochs"])
            row[f"{arm}_f1"] = float(r["strata"]["aggregate"][key]["metrics"]["global"]["f1"])
            row[f"{arm}_pred_ref"] = float(r["strata"]["aggregate"][key]["metrics"]["global"]["prediction_reference_ratio"])
        row["delta_mass_ordered_minus_ranked_f1"] = row["mass_ordered_f1"] - row["ranked_ordered_f1"]
        row["delta_mass_permutation_minus_mass_ordered_f1"] = row["mass_permutation_f1"] - row["mass_ordered_f1"]
        folds[str(fold)] = row

    diagnostics = {}
    for arm in ARMS:
        reports = loaded[arm][0]
        diagnostics[arm] = {
            "presence": _aggregate_presence(reports),
            "match_occupancy": _aggregate_occupancy(reports),
            "gradient_mass": _aggregate_gradients(reports),
        }

    comparison = {
        "exchangeable_weighting_delta_f1_mass_ordered_minus_ranked": strata["aggregate"]["mass_ordered"]["f1"] - strata["aggregate"]["ranked_ordered"]["f1"],
        "matching_delta_f1_mass_permutation_minus_mass_ordered": strata["aggregate"]["mass_permutation"]["f1"] - strata["aggregate"]["mass_ordered"]["f1"],
        "total_delta_f1_mass_permutation_minus_ranked": strata["aggregate"]["mass_permutation"]["f1"] - strata["aggregate"]["ranked_ordered"]["f1"],
        "matching_delta_precision": strata["aggregate"]["mass_permutation"]["precision"] - strata["aggregate"]["mass_ordered"]["precision"],
        "matching_delta_recall": strata["aggregate"]["mass_permutation"]["recall"] - strata["aggregate"]["mass_ordered"]["recall"],
        "exchangeable_weighting_delta_poly": cards["mass_ordered"]["poly_accuracy"] - cards["ranked_ordered"]["poly_accuracy"],
        "matching_delta_poly": cards["mass_permutation"]["poly_accuracy"] - cards["mass_ordered"]["poly_accuracy"],
        "matching_delta_k4": per_k["4"]["mass_permutation"]["exact"] - per_k["4"]["mass_ordered"]["exact"],
        "matching_delta_k5": per_k["5"]["mass_permutation"]["exact"] - per_k["5"]["mass_ordered"]["exact"],
        "matching_delta_k6": per_k["6"]["mass_permutation"]["exact"] - per_k["6"]["mass_ordered"]["exact"],
        "folds_mass_ordered_beats_ranked": int(sum(v["delta_mass_ordered_minus_ranked_f1"] > 0 for v in folds.values())),
        "folds_mass_permutation_beats_mass_ordered": int(sum(v["delta_mass_permutation_minus_mass_ordered_f1"] > 0 for v in folds.values())),
    }
    comparison["mass_permutation_beats_v104_global"] = bool(strata["aggregate"]["mass_permutation"]["f1"] > strata["aggregate"]["v104"]["f1"])

    result = {
        "schema_version": 1,
        "protocol": {
            "v172_three_arm_causal_decomposition": True,
            "outer_clean_rows": 76768,
            "same_rows_all_arms": True,
            "same_seed": 16061,
            "same_threshold": 0.5,
            "fit_fold_only_sqrt_weighting": True,
            "only_A_to_B_change": "query-ranked sqrt weights -> K-conditioned exchangeable mass-preserving weights",
            "only_B_to_C_change": "fixed ordered identity assignment -> exact 6! assignment",
            "locked12_indexed_or_evaluated": False,
        },
        "strata": strata,
        "cardinality": cards,
        "per_true_k": per_k,
        "folds": folds,
        "diagnostics": diagnostics,
        "comparison": comparison,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(
        args.output_dir / "predictions.npz",
        global_index=np.asarray(base_merged["global_index"]),
        k=k,
        pred104=preds["v104"],
        pred172_ranked_ordered=preds["ranked_ordered"],
        pred172_mass_ordered=preds["mass_ordered"],
        pred172_mass_permutation=preds["mass_permutation"],
    )
    print(json.dumps({
        "v104_f1": strata["aggregate"]["v104"]["f1"],
        "ranked_ordered_f1": strata["aggregate"]["ranked_ordered"]["f1"],
        "mass_ordered_f1": strata["aggregate"]["mass_ordered"]["f1"],
        "mass_permutation_f1": strata["aggregate"]["mass_permutation"]["f1"],
        "ranked_pred_ref": strata["aggregate"]["ranked_ordered"]["prediction_reference_ratio"],
        "mass_ordered_pred_ref": strata["aggregate"]["mass_ordered"]["prediction_reference_ratio"],
        "mass_permutation_pred_ref": strata["aggregate"]["mass_permutation"]["prediction_reference_ratio"],
        **comparison,
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
