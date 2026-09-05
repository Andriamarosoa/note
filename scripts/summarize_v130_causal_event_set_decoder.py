"""Aggregate the five outer-clean V13.0 folds without touching locked12."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Optional, Sequence

import numpy as np

SLOT_COUNT = 6
MODELS = ("v101", "v102", "v104", "v130")


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
        "clusters": int(len(k)),
        "accuracy": float(np.mean(exact)) if len(k) else None,
        "birth_accuracy": float(np.mean(exact[birth])) if np.any(birth) else None,
        "poly_accuracy": float(np.mean(exact[poly])) if np.any(poly) else None,
        "mae": float(np.mean(np.abs(pred - k))) if len(k) else None,
        "predicted_histogram": {str(v): int(np.sum(pred == v)) for v in range(SLOT_COUNT + 1)},
        "target_histogram": {str(v): int(np.sum(k == v)) for v in range(SLOT_COUNT + 1)},
    }


def _player(member):
    return str(member).split("_", 1)[0]


def _mode(member):
    s = str(member)
    if s.endswith("_comp.jams"):
        return "comp"
    if s.endswith("_solo.jams"):
        return "solo"
    return "other"


def _genre(member):
    s = str(member)
    core = s.split("_", 1)[1] if "_" in s else s
    m = re.match(r"^([A-Za-z]+)", core)
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
        assert protocol["presence_threshold_tuned"] is False
        reports.append(report)
        with np.load(npz[0], allow_pickle=False) as z:
            parts.append({k: np.asarray(z[k]) for k in z.files})

    merged = {key: np.concatenate([p[key] for p in parts], axis=0) for key in parts[0].keys()}
    order = np.argsort(merged["global_index"], kind="stable")
    merged = {k: v[order] for k, v in merged.items()}
    if len(np.unique(merged["global_index"])) != len(merged["global_index"]):
        raise RuntimeError("outer-fold predictions overlap")
    if sorted(set(np.asarray(merged["outer_fold"], dtype=np.int32).tolist())) != list(range(5)):
        raise RuntimeError("missing outer fold")

    members = np.asarray(merged["member"]).astype(str)
    players = np.asarray([_player(x) for x in members])
    modes = np.asarray([_mode(x) for x in members])
    genres = np.asarray([_genre(x) for x in members])
    k = np.asarray(merged["k"], dtype=np.int32)
    preds = {
        "v101": np.asarray(merged["pred101"], dtype=np.int32),
        "v102": np.asarray(merged["pred102"], dtype=np.int32),
        "v104": np.asarray(merged["pred104"], dtype=np.int32),
        "v130": np.asarray(merged["pred130"], dtype=np.int32),
    }

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
            onset = {}
            for channel in ("global", "solo", "comp"):
                pieces = []
                for report in reports:
                    sr = report["strata"].get(stratum)
                    if sr is not None:
                        pieces.append(sr[model]["metrics"][channel])
                onset[channel] = _metric_sum(pieces)
            row[model] = {"metrics": onset, "cardinality": _card(k[mask], preds[model][mask])}
        strata[stratum] = row

    per_k = {}
    for value in range(SLOT_COUNT + 1):
        mask = k == value
        row = {"clusters": int(np.sum(mask))}
        for model in MODELS:
            p = preds[model][mask]
            row[model] = {
                "exact": float(np.mean(p == value)) if len(p) else None,
                "under_rate": float(np.mean(p < value)) if len(p) else None,
                "over_rate": float(np.mean(p > value)) if len(p) else None,
                "mae": float(np.mean(np.abs(p - value))) if len(p) else None,
            }
        per_k[str(value)] = row

    presence = np.asarray(merged["presence"], dtype=np.float64)
    prefix = np.asarray(merged["prefix_violation"], dtype=np.int8) > 0
    global104 = strata["aggregate"]["v104"]["metrics"]["global"]["f1"]
    global130 = strata["aggregate"]["v130"]["metrics"]["global"]["f1"]
    rock104 = strata["player00_rock_comp"]["v104"]["metrics"]["global"]["f1"]
    rock130 = strata["player00_rock_comp"]["v130"]["metrics"]["global"]["f1"]
    poly104 = strata["aggregate"]["v104"]["cardinality"]["poly_accuracy"]
    poly130 = strata["aggregate"]["v130"]["cardinality"]["poly_accuracy"]

    result = {
        "schema_version": 1,
        "protocol": {
            "five_outer_composition_folds": True,
            "every_row_evaluated_once_outer_clean": True,
            "historical_validation_or_locked12_indexed_or_evaluated": False,
            "categorical_cardinality_head_exists": False,
            "cardinality_is_event_presence_count": True,
            "presence_threshold_tuned": False,
            "presence_threshold": 0.5,
        },
        "data": {
            "clusters": int(len(k)), "outer_folds": 5,
            "unique_global_indices": int(len(np.unique(merged["global_index"]))),
        },
        "strata": strata,
        "per_true_k": per_k,
        "event_presence": {
            "mean_by_query": np.mean(presence, axis=0).tolist(),
            "active_rate_by_query": np.mean(presence >= 0.5, axis=0).tolist(),
            "prefix_violation_rate": float(np.mean(prefix)),
            "mean_predicted_count": float(np.mean(preds["v130"])),
            "mean_true_count": float(np.mean(k)),
        },
        "folds": {
            str(r["outer_fold"]): {
                "selected_epochs": r["data"]["selected_epochs"],
                "v104_f1": r["strata"]["aggregate"]["v104"]["metrics"]["global"]["f1"],
                "v130_f1": r["strata"]["aggregate"]["v130"]["metrics"]["global"]["f1"],
                "event_diagnostics": r["event_diagnostics"],
            }
            for r in reports
        },
        "comparison": {
            "v130_minus_v104_global_f1": global130 - global104,
            "v130_minus_v104_player00_f1": (
                strata["player00"]["v130"]["metrics"]["global"]["f1"]
                - strata["player00"]["v104"]["metrics"]["global"]["f1"]
            ),
            "v130_minus_v104_player00_comp_f1": (
                strata["player00_comp"]["v130"]["metrics"]["global"]["f1"]
                - strata["player00_comp"]["v104"]["metrics"]["global"]["f1"]
            ),
            "v130_minus_v104_player00_rock_comp_f1": rock130 - rock104,
            "v130_minus_v104_poly_exact": poly130 - poly104,
            "v130_minus_v104_k6_exact": per_k["6"]["v130"]["exact"] - per_k["6"]["v104"]["exact"],
            "folds_won_vs_v104": int(sum(
                r["strata"]["aggregate"]["v130"]["metrics"]["global"]["f1"]
                > r["strata"]["aggregate"]["v104"]["metrics"]["global"]["f1"]
                for r in reports
            )),
            "promotion_candidate": bool(global130 > global104 and rock130 >= rock104 and poly130 >= poly104),
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(args.output_dir / "predictions.npz", **merged)
    print(json.dumps({
        "global": {m: strata["aggregate"][m]["metrics"]["global"]["f1"] for m in MODELS},
        "player00_comp": {m: strata["player00_comp"][m]["metrics"]["global"]["f1"] for m in MODELS},
        "player00_rock_comp": {m: strata["player00_rock_comp"][m]["metrics"]["global"]["f1"] for m in MODELS},
        "poly_exact": {m: strata["aggregate"][m]["cardinality"]["poly_accuracy"] for m in MODELS},
        "k6_exact": {m: per_k["6"][m]["exact"] for m in MODELS},
        "event_presence": result["event_presence"],
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
