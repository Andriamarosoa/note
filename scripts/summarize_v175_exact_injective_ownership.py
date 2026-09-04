"""Aggregate V17.5 and perform the mandatory ownership mechanism audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from scripts import summarize_v171_controlled_assignment_ab as s171
from scripts import train_v175_exact_injective_ownership as v175

MODEL_KEY = v175.MODEL_KEY
PRED_KEY = "pred175_injective"
THRESHOLD = v175.PRESENCE_THRESHOLD


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
            if version == "v175":
                ok = p.get("v175_exact_injective_candidate_ownership") is True
            else:
                ok = (
                    p.get("v173_poibin_count_consistency") is True
                    and p.get("v175_exact_injective_candidate_ownership") is not True
                    and p.get("v174_presence_mass_candidate_transport") is not True
                )
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


def _raw_duplicate_rate(candidate, active, rows):
    values = []
    arg = np.argmax(candidate, axis=2)
    for row in np.flatnonzero(rows):
        qs = np.flatnonzero(active[row])
        if len(qs) < 2:
            continue
        values.append(len(np.unique(arg[row, qs])) < len(qs))
    return float(np.mean(values)) if values else None


def _structured_duplicate_rate(ids, active, rows):
    values = []
    for row in np.flatnonzero(rows):
        qs = np.flatnonzero(active[row])
        if len(qs) < 2:
            continue
        real = np.asarray(ids[row, qs], dtype=np.int32)
        real = real[real >= 0]
        values.append(len(np.unique(real)) < len(real))
    return float(np.mean(values)) if values else None


def _duplicate_slices(k, presence, candidate, ids=None, feasible=None):
    active = presence >= THRESHOLD
    hard = np.sum(active, axis=1).astype(np.int32)
    out = {}
    for value in range(2, 7):
        rows = k == value
        exact = rows & (hard == value)
        row = {
            "clusters": int(np.sum(rows)),
            "exact_count_clusters": int(np.sum(exact)),
            "raw_duplicate_exact_count": _raw_duplicate_rate(candidate, active, exact),
        }
        if ids is not None:
            row["structured_duplicate_real_id_exact_count"] = _structured_duplicate_rate(ids, active, exact)
        if feasible is not None:
            row["real_assignment_feasible_exact_count"] = float(np.mean(feasible[exact])) if np.any(exact) else None
        out[str(value)] = row
    poly_exact = (k >= 2) & (hard == k)
    result = {
        "per_true_k": out,
        "raw_duplicate_poly_exact_count": _raw_duplicate_rate(candidate, active, poly_exact),
    }
    if ids is not None:
        result["structured_duplicate_real_id_poly_exact_count"] = _structured_duplicate_rate(ids, active, poly_exact)
    if feasible is not None:
        result["real_assignment_feasible_poly_exact_count"] = float(np.mean(feasible[poly_exact])) if np.any(poly_exact) else None
    return result


def _presence_summary(reports, block):
    total = sum(int(r["data"]["outer_clusters"]) for r in reports)
    rows = []
    for q in range(6):
        vals = [(int(r["data"]["outer_clusters"]), r[block]["outer_presence"][str(q)]) for r in reports]
        rows.append({
            "mean": float(sum(n * v["mean"] for n, v in vals) / total),
            "active_rate": float(sum(n * v["active_rate_at_0p5"] for n, v in vals) / total),
        })
    return {
        "queries": {str(q): row for q, row in enumerate(rows)},
        "soft_mass": float(sum(row["mean"] for row in rows)),
    }


def summarize(args):
    r175, m175 = _load_version(args.input_dir, "v175")
    r173, m173 = _load_version(args.v173_fold_dir, "v173")
    v173_summary = json.loads((args.v173_summary_dir / "report.json").read_text())
    for key in ("global_index", "k", "member", "pred104"):
        if not np.array_equal(np.asarray(m175[key]).astype(str), np.asarray(m173[key]).astype(str)):
            raise RuntimeError(f"row mismatch: {key}")

    k = np.asarray(m175["k"], dtype=np.int32)
    pred104 = np.asarray(m175["pred104"], dtype=np.int32)
    pred173 = np.asarray(m173["pred173_poibin"], dtype=np.int32)
    pred175 = np.asarray(m175[PRED_KEY], dtype=np.int32)
    preds = {"v104": pred104, "v173_poibin": pred173, "v175_injective": pred175}

    strata_names = ["aggregate", "comp", "solo", "player00", "player00_comp", "player00_rock_comp"]
    strata = {}
    for name in strata_names:
        base = s171._metric_sum([_safe_global(r, name, "v104") for r in r175])
        new = s171._metric_sum([_safe_global(r, name, MODEL_KEY) for r in r175])
        old = v173_summary["strata"][name]["v173_poibin"]
        strata[name] = {
            "v104": base,
            "v173_poibin": old,
            "v175_injective": new,
            "delta_v175_minus_v173_f1": float(new["f1"] - old["f1"]),
            "delta_v175_minus_v104_f1": float(new["f1"] - base["f1"]),
            "delta_v175_minus_v173_precision": float(new["precision"] - old["precision"]),
            "delta_v175_minus_v173_recall": float(new["recall"] - old["recall"]),
            "delta_v175_minus_v173_pred_ref": float(new["prediction_reference_ratio"] - old["prediction_reference_ratio"]),
        }

    cards = {name: s171._card(k, pred) for name, pred in preds.items()}
    per_k = s171._per_k(k, preds)
    folds = {}
    wins173 = wins104 = 0
    for fold in range(5):
        old = next(r for r in r173 if int(r["outer_fold"]) == fold)
        new = next(r for r in r175 if int(r["outer_fold"]) == fold)
        f173 = float(old["strata"]["aggregate"]["v173_poibin"]["metrics"]["global"]["f1"])
        f175 = float(new["strata"]["aggregate"][MODEL_KEY]["metrics"]["global"]["f1"])
        f104 = float(new["strata"]["aggregate"]["v104"]["metrics"]["global"]["f1"])
        wins173 += int(f175 > f173)
        wins104 += int(f175 > f104)
        folds[str(fold)] = {
            "selected_epochs": int(new["data"]["selected_epochs"]),
            "v104_f1": f104,
            "v173_f1": f173,
            "v175_f1": f175,
            "delta_v175_minus_v173_f1": f175 - f173,
            "delta_v175_minus_v104_f1": f175 - f104,
        }

    dup173 = _duplicate_slices(
        k,
        np.asarray(m173["presence"], dtype=np.float64),
        np.asarray(m173["event_candidate"], dtype=np.float64),
    )
    if "ownership_candidate_id" not in m175 or "ownership_real_feasible" not in m175:
        raise RuntimeError("V17.5 ownership runtime arrays missing")
    dup175 = _duplicate_slices(
        k,
        np.asarray(m175["presence"], dtype=np.float64),
        np.asarray(m175["event_candidate"], dtype=np.float64),
        np.asarray(m175["ownership_candidate_id"], dtype=np.int32),
        np.asarray(m175["ownership_real_feasible"], dtype=np.int8) > 0,
    )
    presence173 = _presence_summary(r173, "v173")
    presence175 = _presence_summary(r175, "v175")

    comparison = {
        "global_f1_v104": float(strata["aggregate"]["v104"]["f1"]),
        "global_f1_v173": float(strata["aggregate"]["v173_poibin"]["f1"]),
        "global_f1_v175": float(strata["aggregate"]["v175_injective"]["f1"]),
        "delta_v175_minus_v173_f1": float(strata["aggregate"]["delta_v175_minus_v173_f1"]),
        "delta_v175_minus_v104_f1": float(strata["aggregate"]["delta_v175_minus_v104_f1"]),
        "delta_v175_minus_v173_precision": float(strata["aggregate"]["delta_v175_minus_v173_precision"]),
        "delta_v175_minus_v173_recall": float(strata["aggregate"]["delta_v175_minus_v173_recall"]),
        "folds_v175_beats_v173": int(wins173),
        "folds_v175_beats_v104": int(wins104),
        "poly_exact_v104": float(cards["v104"]["poly_accuracy"]),
        "poly_exact_v173": float(cards["v173_poibin"]["poly_accuracy"]),
        "poly_exact_v175": float(cards["v175_injective"]["poly_accuracy"]),
        "delta_poly_v175_minus_v173": float(cards["v175_injective"]["poly_accuracy"] - cards["v173_poibin"]["poly_accuracy"]),
        "soft_mass_v173": presence173["soft_mass"],
        "soft_mass_v175": presence175["soft_mass"],
        "raw_candidate_duplicate_poly_exact_v173": dup173["raw_duplicate_poly_exact_count"],
        "raw_candidate_duplicate_poly_exact_v175": dup175["raw_duplicate_poly_exact_count"],
        "raw_candidate_duplicate_poly_exact_delta": float(dup175["raw_duplicate_poly_exact_count"] - dup173["raw_duplicate_poly_exact_count"]),
        "structured_candidate_duplicate_poly_exact_v175": dup175["structured_duplicate_real_id_poly_exact_count"],
        "real_assignment_feasible_poly_exact_v175": dup175["real_assignment_feasible_poly_exact_count"],
        "player00_rock_comp_f1_v104": float(strata["player00_rock_comp"]["v104"]["f1"]),
        "player00_rock_comp_f1_v173": float(strata["player00_rock_comp"]["v173_poibin"]["f1"]),
        "player00_rock_comp_f1_v175": float(strata["player00_rock_comp"]["v175_injective"]["f1"]),
    }
    for value in range(2, 7):
        comparison[f"delta_k{value}_exact_v175_minus_v173"] = float(
            per_k[str(value)]["v175_injective"]["exact"] - per_k[str(value)]["v173_poibin"]["exact"]
        )
        comparison[f"raw_duplicate_k{value}_exact_v173"] = dup173["per_true_k"][str(value)]["raw_duplicate_exact_count"]
        comparison[f"raw_duplicate_k{value}_exact_v175"] = dup175["per_true_k"][str(value)]["raw_duplicate_exact_count"]
        comparison[f"structured_duplicate_k{value}_exact_v175"] = dup175["per_true_k"][str(value)]["structured_duplicate_real_id_exact_count"]

    gates = {
        "global_f1_improved_vs_v173": bool(comparison["delta_v175_minus_v173_f1"] > 0),
        "majority_folds_improved_vs_v173": bool(wins173 >= 3),
        "raw_poly_duplicate_rate_reduced": bool(comparison["raw_candidate_duplicate_poly_exact_delta"] < 0),
        "structured_real_candidate_duplicates_zero": bool(abs(comparison["structured_candidate_duplicate_poly_exact_v175"] or 0.0) < 1e-12),
        "k3_raw_duplicate_rate_reduced": bool(
            comparison["raw_duplicate_k3_exact_v175"] is not None
            and comparison["raw_duplicate_k3_exact_v173"] is not None
            and comparison["raw_duplicate_k3_exact_v175"] < comparison["raw_duplicate_k3_exact_v173"]
        ),
        "protected_player00_rock_comp_above_v104": bool(
            comparison["player00_rock_comp_f1_v175"] > comparison["player00_rock_comp_f1_v104"]
        ),
    }

    result = {
        "schema_version": 1,
        "protocol": {
            "v175_exact_injective_candidate_ownership": True,
            "mandatory_post_implementation_audit": True,
            "outer_clean_rows": 76768,
            "same_rows_as_v173": True,
            "same_seed": 16061,
            "presence_threshold": THRESHOLD,
            "threshold_tuned": False,
            "v173_poisson_binomial_count_objective_unchanged": True,
            "mass_preserving_exchangeable_weights_unchanged": True,
            "exact_720_truth_matching_unchanged": True,
            "categorical_cardinality_head_exists": False,
            "locked12_indexed_or_evaluated": False,
        },
        "strata": strata,
        "cardinality": cards,
        "per_true_k": per_k,
        "folds": folds,
        "presence": {"v173": presence173, "v175": presence175},
        "ownership": {"v173_raw": dup173, "v175": dup175},
        "comparison": comparison,
        "gates": gates,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(
        args.output_dir / "predictions.npz",
        global_index=np.asarray(m175["global_index"]),
        k=k,
        pred104=pred104,
        pred173_poibin=pred173,
        pred175_injective=pred175,
        ownership_candidate_id=np.asarray(m175["ownership_candidate_id"], dtype=np.int16),
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
