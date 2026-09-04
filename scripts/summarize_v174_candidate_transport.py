"""Aggregate V17.4 and perform the mandatory post-implementation mechanism audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from scripts import summarize_v171_controlled_assignment_ab as s171
from scripts import train_v174_candidate_transport as v174

MODEL_KEY = v174.MODEL_KEY
PRED_KEY = "pred174_transport"


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
            ok = (
                p.get("v174_presence_mass_candidate_transport") is True
                if version == "v174"
                else p.get("v173_poibin_count_consistency") is True
                and p.get("v174_presence_mass_candidate_transport") is not True
            )
            if ok:
                candidates.append((rp, r))
        if len(candidates) != 1:
            raise RuntimeError(f"{version} fold={fold}: expected one report, got {len(candidates)}")
        rp, r = candidates[0]
        reports.append(r)
        npzs = list(rp.parent.glob(f"predictions-fold-{fold}.npz"))
        if len(npzs) != 1:
            raise RuntimeError(f"{version} fold={fold}: predictions missing")
        with np.load(npzs[0], allow_pickle=False) as z:
            n = len(z["global_index"])
            parts.append({
                k: np.asarray(z[k])
                for k in z.files
                if np.asarray(z[k]).ndim and len(np.asarray(z[k])) == n
            })

    common = set(parts[0])
    for part in parts[1:]:
        common &= set(part)
    merged = {k: np.concatenate([p[k] for p in parts], axis=0) for k in common}
    order = np.argsort(merged["global_index"], kind="stable")
    merged = {k: v[order] for k, v in merged.items()}
    if len(merged["global_index"]) != 76768 or len(np.unique(merged["global_index"])) != 76768:
        raise RuntimeError(f"{version}: invalid outer coverage")
    return reports, merged


def _duplicate_rate(dist, active):
    values = []
    for row in range(len(active)):
        ids = np.flatnonzero(active[row])
        if len(ids) < 2:
            continue
        arg = np.argmax(dist[row, ids], axis=1)
        values.append(len(np.unique(arg)) < len(arg))
    return float(np.mean(values)) if values else None


def _duplicate_slices(k, presence, candidate, time):
    hard = np.sum(presence >= 0.5, axis=1).astype(np.int32)
    out = {}
    for value in range(2, 7):
        m = k == value
        exact = m & (hard == value)
        out[str(value)] = {
            "clusters": int(np.sum(m)),
            "exact_count_clusters": int(np.sum(exact)),
            "candidate_all": _duplicate_rate(candidate[m], presence[m] >= 0.5),
            "candidate_exact_count": _duplicate_rate(candidate[exact], presence[exact] >= 0.5),
            "time_all": _duplicate_rate(time[m], presence[m] >= 0.5),
            "time_exact_count": _duplicate_rate(time[exact], presence[exact] >= 0.5),
        }
    poly_exact = (k >= 2) & (hard == k)
    return {
        "per_true_k": out,
        "candidate_poly_exact_count": _duplicate_rate(candidate[poly_exact], presence[poly_exact] >= 0.5),
        "time_poly_exact_count": _duplicate_rate(time[poly_exact], presence[poly_exact] >= 0.5),
    }


def _presence_summary(reports, block):
    total = sum(int(r["data"]["outer_clusters"]) for r in reports)
    per_q = []
    for q in range(6):
        vals = [(int(r["data"]["outer_clusters"]), r[block]["outer_presence"][str(q)]) for r in reports]
        per_q.append({
            "mean": float(sum(n * v["mean"] for n, v in vals) / total),
            "active_rate": float(sum(n * v["active_rate_at_0p5"] for n, v in vals) / total),
        })
    active = np.asarray([x["active_rate"] for x in per_q], dtype=np.float64)
    sorted_active = np.sort(active)
    if np.sum(active):
        n = len(active)
        gini = float(2 * np.sum(np.arange(1, n + 1) * sorted_active) / (n * np.sum(active)) - (n + 1) / n)
        effective = float(np.sum(active) ** 2 / np.sum(active * active))
    else:
        gini = effective = 0.0
    return {
        "queries": {str(q): row for q, row in enumerate(per_q)},
        "soft_mass": float(sum(x["mean"] for x in per_q)),
        "activity_gini": gini,
        "effective_active_slots": effective,
    }


def summarize(args):
    r174, m174 = _load_version(args.input_dir, "v174")
    r173, m173 = _load_version(args.v173_fold_dir, "v173")
    v173_summary = json.loads((args.v173_summary_dir / "report.json").read_text())

    for key in ("global_index", "k", "member", "pred104"):
        if not np.array_equal(np.asarray(m174[key]).astype(str), np.asarray(m173[key]).astype(str)):
            raise RuntimeError(f"row mismatch: {key}")

    k = np.asarray(m174["k"], dtype=np.int32)
    pred104 = np.asarray(m174["pred104"], dtype=np.int32)
    pred173 = np.asarray(m173["pred173_poibin"], dtype=np.int32)
    pred174 = np.asarray(m174[PRED_KEY], dtype=np.int32)
    preds = {"v104": pred104, "v173_poibin": pred173, "v174_transport": pred174}

    strata_names = ["aggregate", "comp", "solo", "player00", "player00_comp", "player00_rock_comp"]
    strata = {}
    for name in strata_names:
        v104 = s171._metric_sum([_safe_global(r, name, "v104") for r in r174])
        new = s171._metric_sum([_safe_global(r, name, MODEL_KEY) for r in r174])
        old = v173_summary["strata"][name]["v173_poibin"]
        strata[name] = {
            "v104": v104,
            "v173_poibin": old,
            "v174_transport": new,
            "delta_v174_minus_v173_f1": float(new["f1"] - old["f1"]),
            "delta_v174_minus_v104_f1": float(new["f1"] - v104["f1"]),
            "delta_v174_minus_v173_precision": float(new["precision"] - old["precision"]),
            "delta_v174_minus_v173_recall": float(new["recall"] - old["recall"]),
            "delta_v174_minus_v173_pred_ref": float(new["prediction_reference_ratio"] - old["prediction_reference_ratio"]),
        }

    cards = {name: s171._card(k, pred) for name, pred in preds.items()}
    per_k = s171._per_k(k, preds)
    folds = {}
    wins173 = wins104 = 0
    for fold in range(5):
        a = next(r for r in r173 if int(r["outer_fold"]) == fold)
        b = next(r for r in r174 if int(r["outer_fold"]) == fold)
        f173 = float(a["strata"]["aggregate"]["v173_poibin"]["metrics"]["global"]["f1"])
        f174 = float(b["strata"]["aggregate"][MODEL_KEY]["metrics"]["global"]["f1"])
        f104 = float(b["strata"]["aggregate"]["v104"]["metrics"]["global"]["f1"])
        wins173 += int(f174 > f173)
        wins104 += int(f174 > f104)
        folds[str(fold)] = {
            "selected_epochs": int(b["data"]["selected_epochs"]),
            "v104_f1": f104,
            "v173_f1": f173,
            "v174_f1": f174,
            "delta_v174_minus_v173_f1": f174 - f173,
            "delta_v174_minus_v104_f1": f174 - f104,
        }

    dup173 = _duplicate_slices(
        k,
        np.asarray(m173["presence"], dtype=np.float64),
        np.asarray(m173["event_candidate"], dtype=np.float64),
        np.asarray(m173["event_time"], dtype=np.float64),
    )
    dup174 = _duplicate_slices(
        k,
        np.asarray(m174["presence"], dtype=np.float64),
        np.asarray(m174["event_candidate"], dtype=np.float64),
        np.asarray(m174["event_time"], dtype=np.float64),
    )
    presence173 = _presence_summary(r173, "v173")
    presence174 = _presence_summary(r174, "v174")

    duplicate_deltas = {}
    for value in range(2, 7):
        o = dup173["per_true_k"][str(value)]["candidate_exact_count"]
        n = dup174["per_true_k"][str(value)]["candidate_exact_count"]
        duplicate_deltas[str(value)] = None if o is None or n is None else float(n - o)

    comparison = {
        "global_f1_v104": float(strata["aggregate"]["v104"]["f1"]),
        "global_f1_v173": float(strata["aggregate"]["v173_poibin"]["f1"]),
        "global_f1_v174": float(strata["aggregate"]["v174_transport"]["f1"]),
        "delta_v174_minus_v173_f1": float(strata["aggregate"]["delta_v174_minus_v173_f1"]),
        "delta_v174_minus_v104_f1": float(strata["aggregate"]["delta_v174_minus_v104_f1"]),
        "delta_v174_minus_v173_precision": float(strata["aggregate"]["delta_v174_minus_v173_precision"]),
        "delta_v174_minus_v173_recall": float(strata["aggregate"]["delta_v174_minus_v173_recall"]),
        "folds_v174_beats_v173": int(wins173),
        "folds_v174_beats_v104": int(wins104),
        "poly_exact_v104": float(cards["v104"]["poly_accuracy"]),
        "poly_exact_v173": float(cards["v173_poibin"]["poly_accuracy"]),
        "poly_exact_v174": float(cards["v174_transport"]["poly_accuracy"]),
        "delta_poly_v174_minus_v173": float(cards["v174_transport"]["poly_accuracy"] - cards["v173_poibin"]["poly_accuracy"]),
        "soft_mass_v173": presence173["soft_mass"],
        "soft_mass_v174": presence174["soft_mass"],
        "candidate_duplicate_poly_exact_count_v173": dup173["candidate_poly_exact_count"],
        "candidate_duplicate_poly_exact_count_v174": dup174["candidate_poly_exact_count"],
        "candidate_duplicate_poly_exact_count_delta": float(dup174["candidate_poly_exact_count"] - dup173["candidate_poly_exact_count"]),
        "candidate_duplicate_exact_count_delta_by_true_k": duplicate_deltas,
        "player00_rock_comp_f1_v104": float(strata["player00_rock_comp"]["v104"]["f1"]),
        "player00_rock_comp_f1_v173": float(strata["player00_rock_comp"]["v173_poibin"]["f1"]),
        "player00_rock_comp_f1_v174": float(strata["player00_rock_comp"]["v174_transport"]["f1"]),
    }
    for value in range(2, 7):
        comparison[f"delta_k{value}_exact_v174_minus_v173"] = float(
            per_k[str(value)]["v174_transport"]["exact"] - per_k[str(value)]["v173_poibin"]["exact"]
        )

    result = {
        "schema_version": 1,
        "protocol": {
            "v174_presence_mass_candidate_transport": True,
            "mandatory_post_implementation_audit": True,
            "outer_clean_rows": 76768,
            "same_rows_as_v173": True,
            "same_seed": 16061,
            "presence_threshold": 0.5,
            "threshold_tuned": False,
            "poisson_binomial_count_objective_unchanged": True,
            "mass_preserving_exchangeable_weights_unchanged": True,
            "exact_720_matching_unchanged": True,
            "categorical_cardinality_head_exists": False,
            "locked12_indexed_or_evaluated": False,
        },
        "strata": strata,
        "cardinality": cards,
        "per_true_k": per_k,
        "folds": folds,
        "presence": {"v173": presence173, "v174": presence174},
        "ownership_duplicates": {"v173": dup173, "v174": dup174},
        "comparison": comparison,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(
        args.output_dir / "predictions.npz",
        global_index=np.asarray(m174["global_index"]),
        k=k,
        pred104=pred104,
        pred173_poibin=pred173,
        pred174_transport=pred174,
    )
    print(json.dumps(comparison, indent=2, sort_keys=True))
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
