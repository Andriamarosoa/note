"""Aggregate V18 and audit evidence-seeded proposal competition."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from scripts import summarize_v171_controlled_assignment_ab as s171
from scripts import summarize_v176_shared_set_decoder as s176
from scripts import train_v180_evidence_seeded_competition as v180

MODEL_KEY = v180.MODEL_KEY
PRED_KEY = "pred180_evidence_seeded"
THRESHOLD = v180.PRESENCE_THRESHOLD


def _safe_global(report, stratum, model_key):
    row = report.get("strata", {}).get(stratum)
    if not row:
        return None
    model = row.get(model_key)
    return model.get("metrics", {}).get("global") if model else None


def _load_v180(root: Path):
    reports, parts = [], []
    for fold in range(5):
        candidates = []
        for rp in root.glob(f"**/report-fold-{fold}.json"):
            r = json.loads(rp.read_text())
            if r.get("protocol", {}).get("v180_evidence_seeded_competition") is True:
                candidates.append((rp, r))
        if len(candidates) != 1:
            raise RuntimeError(f"v180 fold={fold}: expected one report, got {len(candidates)}")
        rp, report = candidates[0]
        reports.append(report)
        npzs = list(rp.parent.glob(f"predictions-fold-{fold}.npz"))
        if len(npzs) != 1:
            raise RuntimeError(f"v180 fold={fold}: predictions missing")
        with np.load(npzs[0], allow_pickle=False) as z:
            n = len(z["global_index"])
            parts.append({
                key: np.asarray(z[key])
                for key in z.files
                if np.asarray(z[key]).ndim and len(np.asarray(z[key])) == n
            })
    common = set(parts[0])
    for part in parts[1:]:
        common &= set(part)
    merged = {key: np.concatenate([part[key] for part in parts], axis=0) for key in common}
    order = np.argsort(merged["global_index"], kind="stable")
    merged = {key: value[order] for key, value in merged.items()}
    if len(merged["global_index"]) != 76768 or len(np.unique(merged["global_index"])) != 76768:
        raise RuntimeError("v180: invalid outer-clean coverage")
    return reports, merged


def summarize(args):
    r180, m180 = _load_v180(args.input_dir)
    r173, m173 = s176._load_version(args.v173_fold_dir, "v173")
    v173_summary = json.loads((args.v173_summary_dir / "report.json").read_text())

    for key in ("global_index", "k", "member", "pred104"):
        if not np.array_equal(np.asarray(m180[key]).astype(str), np.asarray(m173[key]).astype(str)):
            raise RuntimeError(f"row mismatch: {key}")

    k = np.asarray(m180["k"], dtype=np.int32)
    pred104 = np.asarray(m180["pred104"], dtype=np.int32)
    pred173 = np.asarray(m173["pred173_poibin"], dtype=np.int32)
    pred180 = np.asarray(m180[PRED_KEY], dtype=np.int32)
    preds = {"v104": pred104, "v173_poibin": pred173, MODEL_KEY: pred180}

    strata_names = ["aggregate", "comp", "solo", "player00", "player00_comp", "player00_rock_comp"]
    strata = {}
    for name in strata_names:
        base = s171._metric_sum([_safe_global(r, name, "v104") for r in r180])
        new = s171._metric_sum([_safe_global(r, name, MODEL_KEY) for r in r180])
        old = v173_summary["strata"][name]["v173_poibin"]
        strata[name] = {
            "v104": base,
            "v173_poibin": old,
            MODEL_KEY: new,
            "delta_v180_minus_v173_f1": float(new["f1"] - old["f1"]),
            "delta_v180_minus_v104_f1": float(new["f1"] - base["f1"]),
            "delta_v180_minus_v173_precision": float(new["precision"] - old["precision"]),
            "delta_v180_minus_v173_recall": float(new["recall"] - old["recall"]),
            "delta_v180_minus_v173_pred_ref": float(new["prediction_reference_ratio"] - old["prediction_reference_ratio"]),
        }

    cards = {name: s171._card(k, pred) for name, pred in preds.items()}
    per_k = s171._per_k(k, preds)

    folds = {}
    wins173 = wins104 = 0
    for fold in range(5):
        old = next(r for r in r173 if int(r["outer_fold"]) == fold)
        new = next(r for r in r180 if int(r["outer_fold"]) == fold)
        f173 = float(old["strata"]["aggregate"]["v173_poibin"]["metrics"]["global"]["f1"])
        f180 = float(new["strata"]["aggregate"][MODEL_KEY]["metrics"]["global"]["f1"])
        f104 = float(new["strata"]["aggregate"]["v104"]["metrics"]["global"]["f1"])
        wins173 += int(f180 > f173)
        wins104 += int(f180 > f104)
        a = new["v180"]["architecture"]
        folds[str(fold)] = {
            "selected_epochs": int(new["data"]["selected_epochs"]),
            "v104_f1": f104,
            "v173_f1": f173,
            "v180_f1": f180,
            "delta_v180_minus_v173_f1": f180 - f173,
            "delta_v180_minus_v104_f1": f180 - f104,
            "activity_gini": float(a["outer_activity_gini"]),
            "effective_active_slots": float(a["outer_effective_active_slots"]),
            "active_occupancy_correlation": float(a["outer_active_occupancy_correlation"]),
            "rows_candidate_infeasible_c_lt_k": int(a["rows_candidate_infeasible_c_lt_k"]),
        }

    spec173 = s176._specialization(r173, "v173")
    spec180 = s176._specialization(r180, "v180")
    dup173 = s176._duplicate_slices(
        k,
        np.asarray(m173["presence"], dtype=np.float64),
        np.asarray(m173["event_candidate"], dtype=np.float64),
    )
    dup180 = s176._duplicate_slices(
        k,
        np.asarray(m180["presence"], dtype=np.float64),
        np.asarray(m180["event_candidate"], dtype=np.float64),
    )

    soft173 = float(np.mean(np.sum(np.asarray(m173["presence"], dtype=np.float64), axis=1)))
    soft180 = float(np.mean(np.sum(np.asarray(m180["presence"], dtype=np.float64), axis=1)))

    comparison = {
        "global_f1_v104": float(strata["aggregate"]["v104"]["f1"]),
        "global_f1_v173": float(strata["aggregate"]["v173_poibin"]["f1"]),
        "global_f1_v180": float(strata["aggregate"][MODEL_KEY]["f1"]),
        "delta_v180_minus_v173_f1": float(strata["aggregate"]["delta_v180_minus_v173_f1"]),
        "delta_v180_minus_v104_f1": float(strata["aggregate"]["delta_v180_minus_v104_f1"]),
        "delta_v180_minus_v173_precision": float(strata["aggregate"]["delta_v180_minus_v173_precision"]),
        "delta_v180_minus_v173_recall": float(strata["aggregate"]["delta_v180_minus_v173_recall"]),
        "folds_v180_beats_v173": int(wins173),
        "folds_v180_beats_v104": int(wins104),
        "poly_exact_v104": float(cards["v104"]["poly_accuracy"]),
        "poly_exact_v173": float(cards["v173_poibin"]["poly_accuracy"]),
        "poly_exact_v180": float(cards[MODEL_KEY]["poly_accuracy"]),
        "delta_poly_v180_minus_v173": float(cards[MODEL_KEY]["poly_accuracy"] - cards["v173_poibin"]["poly_accuracy"]),
        "soft_mass_v173": soft173,
        "soft_mass_v180": soft180,
        "activity_gini_v173": spec173["mean_active_rate_gini"],
        "activity_gini_v180": spec180["mean_active_rate_gini"],
        "delta_activity_gini": spec180["mean_active_rate_gini"] - spec173["mean_active_rate_gini"],
        "effective_active_slots_v173": spec173["mean_effective_active_slots"],
        "effective_active_slots_v180": spec180["mean_effective_active_slots"],
        "delta_effective_active_slots": spec180["mean_effective_active_slots"] - spec173["mean_effective_active_slots"],
        "min_active_occupancy_correlation_v173": spec173["min_active_occupancy_correlation"],
        "min_active_occupancy_correlation_v180": spec180["min_active_occupancy_correlation"],
        "raw_candidate_duplicate_poly_exact_v173": dup173["raw_duplicate_poly_exact_count"],
        "raw_candidate_duplicate_poly_exact_v180": dup180["raw_duplicate_poly_exact_count"],
        "player00_rock_comp_f1_v104": float(strata["player00_rock_comp"]["v104"]["f1"]),
        "player00_rock_comp_f1_v173": float(strata["player00_rock_comp"]["v173_poibin"]["f1"]),
        "player00_rock_comp_f1_v180": float(strata["player00_rock_comp"][MODEL_KEY]["f1"]),
    }
    for value in range(2, 7):
        comparison[f"delta_k{value}_exact_v180_minus_v173"] = float(
            per_k[str(value)][MODEL_KEY]["exact"] - per_k[str(value)]["v173_poibin"]["exact"]
        )
        comparison[f"raw_duplicate_k{value}_exact_v173"] = dup173["per_true_k"][str(value)]["raw_duplicate_exact_count"]
        comparison[f"raw_duplicate_k{value}_exact_v180"] = dup180["per_true_k"][str(value)]["raw_duplicate_exact_count"]

    gates = {
        "global_f1_improved_vs_v173": bool(comparison["delta_v180_minus_v173_f1"] > 0),
        "majority_folds_improved_vs_v173": bool(wins173 >= 3),
        "beats_v104_global": bool(comparison["delta_v180_minus_v104_f1"] > 0),
        "activity_gini_reduced_vs_v173": bool(comparison["delta_activity_gini"] < 0),
        "effective_active_slots_increased_vs_v173": bool(comparison["delta_effective_active_slots"] > 0),
        "protected_player00_rock_comp_above_v104": bool(
            comparison["player00_rock_comp_f1_v180"] > comparison["player00_rock_comp_f1_v104"]
        ),
    }

    result = {
        "schema_version": 1,
        "protocol": {
            "v180_evidence_seeded_competition": True,
            "mandatory_post_implementation_audit": True,
            "outer_clean_rows": 76768,
            "same_rows_as_v173": True,
            "same_seed": 16061,
            "presence_threshold": THRESHOLD,
            "threshold_tuned": False,
            "proposal_anchor_source": "top-6 valid candidates by frozen V8.8 fused score",
            "fixed_or_learned_anonymous_seed_count": 0,
            "v173_poisson_binomial_count_objective_unchanged": True,
            "mass_preserving_exchangeable_weights_unchanged": True,
            "exact_720_truth_matching_unchanged": True,
            "v175_injective_ownership_used": False,
            "categorical_cardinality_head_exists": False,
            "locked12_indexed_or_evaluated": False,
        },
        "strata": strata,
        "cardinality": cards,
        "per_true_k": per_k,
        "folds": folds,
        "specialization": {"v173": spec173, "v180": spec180},
        "duplicates": {"v173": dup173, "v180": dup180},
        "comparison": comparison,
        "gates": gates,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(
        args.output_dir / "predictions.npz",
        global_index=np.asarray(m180["global_index"]),
        k=k,
        pred104=pred104,
        pred173=pred173,
        pred180=pred180,
        presence173=np.asarray(m173["presence"]),
        presence180=np.asarray(m180["presence"]),
    )
    print(json.dumps(comparison, indent=2, sort_keys=True))
    print(json.dumps(gates, indent=2, sort_keys=True))
    return result


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--v173-fold-dir", type=Path, required=True)
    p.add_argument("--v173-summary-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    return p


def main(argv: Optional[Sequence[str]] = None):
    summarize(parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
