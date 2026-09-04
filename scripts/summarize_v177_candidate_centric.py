"""Aggregate V17.7 and audit candidate-centric object generation and realization."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from scripts import summarize_v171_controlled_assignment_ab as s171
from scripts import train_v177_candidate_centric as v177

MODEL_KEY = v177.MODEL_KEY
PRED_KEY = "pred177_candidate_centric"
THRESHOLD = v177.PRESENCE_THRESHOLD


def _safe_global(report, stratum, model_key):
    row = report.get("strata", {}).get(stratum)
    if not row:
        return None
    model = row.get(model_key)
    return model.get("metrics", {}).get("global") if model else None


def _load_version(root: Path, version: str):
    reports, parts = [], []
    for fold in range(5):
        candidates = []
        for rp in root.glob(f"**/report-fold-{fold}.json"):
            r = json.loads(rp.read_text())
            p = r.get("protocol", {})
            if version == "v177":
                ok = p.get("v177_candidate_centric") is True
            elif version == "v173":
                ok = (
                    p.get("v173_poibin_count_consistency") is True
                    and p.get("v177_candidate_centric") is not True
                    and p.get("v176_shared_set_decoder") is not True
                    and p.get("v175_exact_injective_candidate_ownership") is not True
                    and p.get("v174_presence_mass_candidate_transport") is not True
                )
            else:
                raise RuntimeError(version)
            if ok:
                candidates.append((rp, r))
        if len(candidates) != 1:
            raise RuntimeError(f"{version} fold={fold}: expected one report, got {len(candidates)}")
        rp, report = candidates[0]
        reports.append(report)
        npzs = list(rp.parent.glob(f"predictions-fold-{fold}.npz"))
        if len(npzs) != 1:
            raise RuntimeError(f"{version} fold={fold}: predictions missing")
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
        raise RuntimeError(f"{version}: invalid outer-clean coverage")
    return reports, merged


def _metric_sum(items):
    return s171._metric_sum(items)


def _count_only_global(reports, stratum):
    items = []
    for r in reports:
        block = r["v177"]["count_only_strata"].get(stratum)
        items.append(block.get("global") if block else None)
    return _metric_sum(items)


def _selected_unique_rate(selected):
    ok = []
    for row in np.asarray(selected, dtype=np.int32):
        keep = row[row >= 0]
        ok.append(len(keep) == len(np.unique(keep)))
    return float(np.mean(ok)) if ok else 1.0


def summarize(args):
    r177, m177 = _load_version(args.input_dir, "v177")
    r173, m173 = _load_version(args.v173_fold_dir, "v173")
    v173_summary = json.loads((args.v173_summary_dir / "report.json").read_text())
    v176_summary = None
    if args.v176_summary_dir is not None:
        path = args.v176_summary_dir / "report.json"
        if path.exists():
            v176_summary = json.loads(path.read_text())

    for key in ("global_index", "k", "member", "pred104"):
        if not np.array_equal(np.asarray(m177[key]).astype(str), np.asarray(m173[key]).astype(str)):
            raise RuntimeError(f"row mismatch: {key}")

    k = np.asarray(m177["k"], dtype=np.int32)
    pred104 = np.asarray(m177["pred104"], dtype=np.int32)
    pred173 = np.asarray(m173["pred173_poibin"], dtype=np.int32)
    pred177 = np.asarray(m177[PRED_KEY], dtype=np.int32)
    preds = {"v104": pred104, "v173_poibin": pred173, MODEL_KEY: pred177}

    strata_names = ["aggregate", "comp", "solo", "player00", "player00_comp", "player00_rock_comp"]
    strata = {}
    for name in strata_names:
        base = _metric_sum([_safe_global(r, name, "v104") for r in r177])
        direct = _metric_sum([_safe_global(r, name, MODEL_KEY) for r in r177])
        old = v173_summary["strata"][name]["v173_poibin"]
        count_only = _count_only_global(r177, name)
        strata[name] = {
            "v104": base,
            "v173_poibin": old,
            "v177_count_only_frozen_ranking": count_only,
            MODEL_KEY: direct,
            "delta_direct_minus_v173_f1": float(direct["f1"] - old["f1"]),
            "delta_direct_minus_v104_f1": float(direct["f1"] - base["f1"]),
            "delta_direct_minus_count_only_f1": float(direct["f1"] - count_only["f1"]),
            "delta_count_only_minus_v173_f1": float(count_only["f1"] - old["f1"]),
            "delta_direct_minus_v173_precision": float(direct["precision"] - old["precision"]),
            "delta_direct_minus_v173_recall": float(direct["recall"] - old["recall"]),
        }

    cards = {name: s171._card(k, pred) for name, pred in preds.items()}
    per_k = s171._per_k(k, preds)

    folds = {}
    wins173 = wins104 = wins_count = 0
    for fold in range(5):
        old = next(r for r in r173 if int(r["outer_fold"]) == fold)
        new = next(r for r in r177 if int(r["outer_fold"]) == fold)
        f173 = float(old["strata"]["aggregate"]["v173_poibin"]["metrics"]["global"]["f1"])
        f177 = float(new["strata"]["aggregate"][MODEL_KEY]["metrics"]["global"]["f1"])
        f104 = float(new["strata"]["aggregate"]["v104"]["metrics"]["global"]["f1"])
        fcount = float(new["v177"]["count_only_strata"]["aggregate"]["global"]["f1"])
        wins173 += int(f177 > f173)
        wins104 += int(f177 > f104)
        wins_count += int(f177 > fcount)
        arch = new["v177"]["architecture"]
        folds[str(fold)] = {
            "selected_epochs": int(new["data"]["selected_epochs"]),
            "v104_f1": f104,
            "v173_f1": f173,
            "v177_count_only_f1": fcount,
            "v177_direct_f1": f177,
            "delta_direct_minus_v173_f1": f177 - f173,
            "delta_direct_minus_v104_f1": f177 - f104,
            "delta_direct_minus_count_only_f1": f177 - fcount,
            "soft_object_count": float(arch["outer_soft_object_count"]),
            "hard_object_count": float(arch["outer_hard_object_count"]),
            "true_count": float(arch["outer_true_count"]),
            "selected_unique_rate": float(arch["selected_candidate_ids_unique_rate"]),
        }

    objectness = np.asarray(m177["candidate_objectness"], dtype=np.float64)
    selected = np.asarray(m177["candidate_selected_ids"], dtype=np.int32)
    soft_count177 = float(np.mean(np.sum(objectness, axis=1)))
    hard_count177 = float(np.mean(pred177))
    true_mean = float(np.mean(k))
    soft_count173 = float(np.mean(np.sum(np.asarray(m173["presence"], dtype=np.float64), axis=1)))
    unique_rate = _selected_unique_rate(selected)

    mass_error = 0.0
    valid_candidate_min = 999
    for r in r177:
        for row in r["v177"]["architecture"]["mass_preservation_by_true_k"].values():
            mass_error = max(mass_error, float(row["max_abs_mass_error"]))
            valid_candidate_min = min(valid_candidate_min, int(row["valid_candidate_count_min"]))

    aggregate = strata["aggregate"]
    p00rock = strata["player00_rock_comp"]
    comparison = {
        "global_f1_v104": float(aggregate["v104"]["f1"]),
        "global_f1_v173": float(aggregate["v173_poibin"]["f1"]),
        "global_f1_v176": (
            float(v176_summary["comparison"]["global_f1_v176"]) if v176_summary else None
        ),
        "global_f1_v177_count_only": float(aggregate["v177_count_only_frozen_ranking"]["f1"]),
        "global_f1_v177_direct": float(aggregate[MODEL_KEY]["f1"]),
        "delta_direct_minus_v173_f1": float(aggregate["delta_direct_minus_v173_f1"]),
        "delta_direct_minus_v104_f1": float(aggregate["delta_direct_minus_v104_f1"]),
        "delta_direct_minus_count_only_f1": float(aggregate["delta_direct_minus_count_only_f1"]),
        "delta_count_only_minus_v173_f1": float(aggregate["delta_count_only_minus_v173_f1"]),
        "delta_direct_minus_v173_precision": float(aggregate["delta_direct_minus_v173_precision"]),
        "delta_direct_minus_v173_recall": float(aggregate["delta_direct_minus_v173_recall"]),
        "folds_direct_beats_v173": int(wins173),
        "folds_direct_beats_v104": int(wins104),
        "folds_direct_beats_own_count_only": int(wins_count),
        "poly_exact_v104": float(cards["v104"]["poly_accuracy"]),
        "poly_exact_v173": float(cards["v173_poibin"]["poly_accuracy"]),
        "poly_exact_v177": float(cards[MODEL_KEY]["poly_accuracy"]),
        "delta_poly_v177_minus_v173": float(cards[MODEL_KEY]["poly_accuracy"] - cards["v173_poibin"]["poly_accuracy"]),
        "soft_object_count_v173_six_queries": soft_count173,
        "soft_object_count_v177_candidates": soft_count177,
        "hard_count_v177": hard_count177,
        "true_mean_count": true_mean,
        "soft_count_abs_error_v173": abs(soft_count173 - true_mean),
        "soft_count_abs_error_v177": abs(soft_count177 - true_mean),
        "selected_candidate_ids_unique_rate": unique_rate,
        "max_mass_preservation_error": mass_error,
        "minimum_valid_candidates_seen": int(valid_candidate_min),
        "player00_rock_comp_f1_v104": float(p00rock["v104"]["f1"]),
        "player00_rock_comp_f1_v173": float(p00rock["v173_poibin"]["f1"]),
        "player00_rock_comp_f1_v177_count_only": float(p00rock["v177_count_only_frozen_ranking"]["f1"]),
        "player00_rock_comp_f1_v177_direct": float(p00rock[MODEL_KEY]["f1"]),
    }
    if comparison["global_f1_v176"] is not None:
        comparison["delta_direct_minus_v176_f1"] = (
            comparison["global_f1_v177_direct"] - comparison["global_f1_v176"]
        )
    for value in range(2, 7):
        comparison[f"delta_k{value}_exact_v177_minus_v173"] = float(
            per_k[str(value)][MODEL_KEY]["exact"] - per_k[str(value)]["v173_poibin"]["exact"]
        )

    gates = {
        "direct_global_f1_improved_vs_v173": bool(comparison["delta_direct_minus_v173_f1"] > 0),
        "count_only_global_f1_improved_vs_v173": bool(comparison["delta_count_only_minus_v173_f1"] > 0),
        "candidate_realization_improves_own_count_only": bool(comparison["delta_direct_minus_count_only_f1"] > 0),
        "majority_folds_direct_improve_vs_v173": bool(wins173 >= 3),
        "majority_folds_candidate_realization_improves_count_only": bool(wins_count >= 3),
        "beats_v104_global": bool(comparison["delta_direct_minus_v104_f1"] > 0),
        "protected_player00_rock_comp_above_v104": bool(
            comparison["player00_rock_comp_f1_v177_direct"] > comparison["player00_rock_comp_f1_v104"]
        ),
        "selected_candidate_ids_unique": bool(unique_rate == 1.0),
        "mass_preservation_exact": bool(mass_error < 1e-6),
        "soft_count_calibration_improved_vs_v173": bool(
            comparison["soft_count_abs_error_v177"] < comparison["soft_count_abs_error_v173"]
        ),
    }

    result = {
        "schema_version": 1,
        "protocol": {
            "v177_candidate_centric": True,
            "mandatory_post_implementation_audit": True,
            "outer_clean_rows": 76768,
            "same_rows_as_v173": True,
            "same_seed": 16061,
            "presence_threshold": THRESHOLD,
            "threshold_tuned": False,
            "fixed_or_trainable_anonymous_seed_count": 0,
            "candidate_object_token_count": 48,
            "candidate_set_dp_states": 64,
            "candidate_set_dp_permutation_invariant": True,
            "v173_count_nll_weight": 0.35,
            "v173_mass_coefficient_preservation": True,
            "headline_realization": "direct selected candidate objects",
            "count_only_control_realization": "same predicted K through frozen V9+ ranking",
            "categorical_cardinality_head_exists": False,
            "v175_injective_ownership_used": False,
            "locked12_indexed_or_evaluated": False,
        },
        "strata": strata,
        "cardinality": cards,
        "per_true_k": per_k,
        "folds": folds,
        "comparison": comparison,
        "gates": gates,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(comparison, indent=2, sort_keys=True))
    print(json.dumps(gates, indent=2, sort_keys=True))
    return result


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--v173-fold-dir", type=Path, required=True)
    p.add_argument("--v173-summary-dir", type=Path, required=True)
    p.add_argument("--v176-summary-dir", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    return p


def main(argv: Optional[Sequence[str]] = None):
    summarize(parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
