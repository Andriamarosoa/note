"""Aggregate five outer-clean V15 folds. locked12 is never accessed."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Optional, Sequence

import numpy as np

MODELS = ("v101", "v102", "v104", "v130", "v150")
STEPS = 6


def _metric_sum(rows):
    rows = [r for r in rows if r]
    tp = sum(int(r.get("true_positive", 0)) for r in rows)
    fp = sum(int(r.get("false_positive", 0)) for r in rows)
    fn = sum(int(r.get("false_negative", 0)) for r in rows)
    pred = sum(int(r.get("prediction_count", 0)) for r in rows)
    ref = sum(int(r.get("reference_count", 0)) for r in rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "f1": f1, "precision": precision, "recall": recall,
        "true_positive": tp, "false_positive": fp, "false_negative": fn,
        "prediction_count": pred, "reference_count": ref,
        "prediction_reference_ratio": pred / ref if ref else None,
    }


def _card(k, pred):
    k = np.asarray(k, dtype=np.int32)
    pred = np.asarray(pred, dtype=np.int32)
    exact = pred == k
    birth = k == 1
    poly = k >= 2
    return {
        "accuracy": float(np.mean(exact)),
        "birth_accuracy": float(np.mean(exact[birth])) if np.any(birth) else None,
        "poly_accuracy": float(np.mean(exact[poly])) if np.any(poly) else None,
        "mae": float(np.mean(np.abs(pred - k))),
        "mean_predicted_count": float(np.mean(pred)),
        "mean_true_count": float(np.mean(k)),
        "predicted_histogram": {str(v): int(np.sum(pred == v)) for v in range(7)},
        "target_histogram": {str(v): int(np.sum(k == v)) for v in range(7)},
    }


def _player(member):
    return str(member).split("_", 1)[0]


def _mode(member):
    s = str(member)
    return "comp" if s.endswith("_comp.jams") else "solo" if s.endswith("_solo.jams") else "other"


def _genre(member):
    s = str(member).split("_", 1)[1] if "_" in str(member) else str(member)
    m = re.match(r"^([A-Za-z]+)", s)
    return m.group(1) if m else "unknown"


def summarize(args):
    reports = []
    parts = []
    for fold in range(5):
        rp = sorted(args.input_dir.glob(f"**/report-fold-{fold}.json"))
        npz = sorted(args.input_dir.glob(f"**/predictions-fold-{fold}.npz"))
        if len(rp) != 1 or len(npz) != 1:
            raise RuntimeError(f"fold {fold}: expected one report and one prediction shard")
        report = json.loads(rp[0].read_text())
        protocol = report["protocol"]
        assert protocol["historical_validation_or_locked12_indexed_or_evaluated"] is False
        assert protocol["outer_fold_used_for_training"] is False
        assert protocol["outer_fold_used_for_epoch_selection"] is False
        assert protocol["categorical_cardinality_head_exists"] is False
        assert protocol["stop_threshold_tuned"] is False
        assert protocol["destructive_explaining_away"] is False
        assert protocol["full_evidence_visible_every_step"] is True
        assert protocol["novelty_memory"] is True
        reports.append(report)
        with np.load(npz[0], allow_pickle=False) as z:
            parts.append({key: np.asarray(z[key]) for key in z.files})

    keys = set(parts[0])
    if any(set(part) != keys for part in parts):
        raise RuntimeError("prediction shard schema mismatch")
    merged = {key: np.concatenate([part[key] for part in parts], axis=0) for key in keys}
    order = np.argsort(merged["global_index"], kind="stable")
    merged = {key: value[order] for key, value in merged.items()}
    gi = np.asarray(merged["global_index"], dtype=np.int64)
    if len(np.unique(gi)) != len(gi):
        raise RuntimeError("outer prediction overlap")
    if set(np.asarray(merged["outer_fold"], dtype=np.int32).tolist()) != set(range(5)):
        raise RuntimeError("missing outer fold")

    members = np.asarray(merged["member"]).astype(str)
    players = np.asarray([_player(x) for x in members])
    modes = np.asarray([_mode(x) for x in members])
    genres = np.asarray([_genre(x) for x in members])
    k = np.asarray(merged["k"], dtype=np.int32)
    pred = {m: np.asarray(merged[f"pred{m[1:]}"] , dtype=np.int32) for m in MODELS}

    masks = {
        "aggregate": np.ones(len(k), dtype=bool),
        "comp": modes == "comp",
        "solo": modes == "solo",
        "player00": players == "00",
        "player00_comp": (players == "00") & (modes == "comp"),
        "player00_solo": (players == "00") & (modes == "solo"),
        "player00_rock_comp": (players == "00") & (modes == "comp") & (genres == "Rock"),
    }
    for player in ("00", "01", "02", "03", "04"):
        masks[f"player{player}"] = players == player

    strata = {}
    for stratum, mask in masks.items():
        row = {"clusters": int(np.sum(mask))}
        for model in MODELS:
            pieces = []
            for report in reports:
                sr = report["strata"].get(stratum)
                if sr and model in sr:
                    pieces.append(sr[model]["metrics"]["global"])
            row[model] = {
                "metrics": {"global": _metric_sum(pieces)},
                "cardinality": _card(k[mask], pred[model][mask]),
            }
        strata[stratum] = row

    per_k = {}
    for value in range(7):
        mask = k == value
        row = {"clusters": int(np.sum(mask))}
        for model in MODELS:
            pp = pred[model][mask]
            row[model] = {
                "exact": float(np.mean(pp == value)) if len(pp) else None,
                "under_rate": float(np.mean(pp < value)) if len(pp) else None,
                "over_rate": float(np.mean(pp > value)) if len(pp) else None,
                "mae": float(np.mean(np.abs(pp - value))) if len(pp) else None,
            }
        per_k[str(value)] = row

    cont = np.asarray(merged["continue_probability"], dtype=np.float64)
    novelty_tf = np.asarray(merged["tf_mass"], dtype=np.float64)
    novelty_cand = np.asarray(merged["candidate_mass"], dtype=np.float64)
    coverage_tf = np.asarray(merged["coverage_tf"], dtype=np.float64)
    coverage_cand = np.asarray(merged["coverage_candidate"], dtype=np.float64)
    conditional = {}
    for q in range(STEPS):
        reachable = k >= q
        positive = reachable & (k > q)
        negative = reachable & (k == q)
        conditional[str(q)] = {
            "reachable": int(np.sum(reachable)),
            "positive": int(np.sum(positive)),
            "negative": int(np.sum(negative)),
            "mean_probability_positive": float(np.mean(cont[positive, q])) if np.any(positive) else None,
            "mean_probability_negative": float(np.mean(cont[negative, q])) if np.any(negative) else None,
            "novelty_tf_positive": float(np.mean(novelty_tf[positive, q])) if np.any(positive) else None,
            "novelty_tf_negative": float(np.mean(novelty_tf[negative, q])) if np.any(negative) else None,
            "novelty_candidate_positive": float(np.mean(novelty_cand[positive, q])) if np.any(positive) else None,
            "novelty_candidate_negative": float(np.mean(novelty_cand[negative, q])) if np.any(negative) else None,
            "coverage_tf_mean": float(np.mean(coverage_tf[reachable, q])) if np.any(reachable) else None,
            "coverage_candidate_mean": float(np.mean(coverage_cand[reachable, q])) if np.any(reachable) else None,
        }

    f104 = strata["aggregate"]["v104"]["metrics"]["global"]["f1"]
    f130 = strata["aggregate"]["v130"]["metrics"]["global"]["f1"]
    f150 = strata["aggregate"]["v150"]["metrics"]["global"]["f1"]
    rock104 = strata["player00_rock_comp"]["v104"]["metrics"]["global"]["f1"]
    rock150 = strata["player00_rock_comp"]["v150"]["metrics"]["global"]["f1"]
    poly104 = strata["aggregate"]["v104"]["cardinality"]["poly_accuracy"]
    poly150 = strata["aggregate"]["v150"]["cardinality"]["poly_accuracy"]

    result = {
        "schema_version": 1,
        "protocol": {
            "five_outer_composition_folds": True,
            "every_row_evaluated_once_outer_clean": True,
            "historical_validation_or_locked12_indexed_or_evaluated": False,
            "categorical_cardinality_head_exists": False,
            "conditional_stop_decoder": True,
            "destructive_explaining_away": False,
            "full_evidence_visible_every_step": True,
            "coverage_novelty_memory": True,
            "stop_threshold_tuned": False,
            "stop_threshold": 0.5,
        },
        "data": {"clusters": int(len(k)), "unique_global_indices": int(len(np.unique(gi))), "outer_folds": 5},
        "strata": strata,
        "per_true_k": per_k,
        "conditional_steps": conditional,
        "folds": {
            str(r["outer_fold"]): {
                "selected_epochs": r["data"]["selected_epochs"],
                "v104_f1": r["strata"]["aggregate"]["v104"]["metrics"]["global"]["f1"],
                "v130_f1": r["strata"]["aggregate"]["v130"]["metrics"]["global"]["f1"],
                "v150_f1": r["strata"]["aggregate"]["v150"]["metrics"]["global"]["f1"],
            }
            for r in reports
        },
        "comparison": {
            "v150_minus_v104_global_f1": f150 - f104,
            "v150_minus_v130_global_f1": f150 - f130,
            "v150_minus_v104_player00_rock_comp_f1": rock150 - rock104,
            "v150_minus_v104_poly_exact": poly150 - poly104,
            "v150_minus_v104_k4_exact": per_k["4"]["v150"]["exact"] - per_k["4"]["v104"]["exact"],
            "v150_minus_v104_k5_exact": per_k["5"]["v150"]["exact"] - per_k["5"]["v104"]["exact"],
            "v150_minus_v104_k6_exact": per_k["6"]["v150"]["exact"] - per_k["6"]["v104"]["exact"],
            "folds_won_vs_v104": int(sum(
                r["strata"]["aggregate"]["v150"]["metrics"]["global"]["f1"]
                > r["strata"]["aggregate"]["v104"]["metrics"]["global"]["f1"]
                for r in reports
            )),
            "folds_won_vs_v130": int(sum(
                r["strata"]["aggregate"]["v150"]["metrics"]["global"]["f1"]
                > r["strata"]["aggregate"]["v130"]["metrics"]["global"]["f1"]
                for r in reports
            )),
            "promotion_candidate": bool(f150 > f104 and rock150 >= rock104 and poly150 >= poly104),
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(args.output_dir / "predictions.npz", **merged)
    print(json.dumps({
        "global": {m: strata["aggregate"][m]["metrics"]["global"]["f1"] for m in MODELS},
        "precision": {m: strata["aggregate"][m]["metrics"]["global"]["precision"] for m in ("v104", "v130", "v150")},
        "recall": {m: strata["aggregate"][m]["metrics"]["global"]["recall"] for m in ("v104", "v130", "v150")},
        "player00_rock_comp": {m: strata["player00_rock_comp"][m]["metrics"]["global"]["f1"] for m in ("v104", "v130", "v150")},
        "poly": {m: strata["aggregate"][m]["cardinality"]["poly_accuracy"] for m in ("v104", "v130", "v150")},
        "k4_k5_k6_v150": {kval: per_k[kval]["v150"]["exact"] for kval in ("4", "5", "6")},
        "counts": {m: strata["aggregate"][m]["cardinality"]["mean_predicted_count"] for m in ("v104", "v130", "v150")},
        "q4": conditional["4"], "q5": conditional["5"],
        "comparison": result["comparison"],
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
