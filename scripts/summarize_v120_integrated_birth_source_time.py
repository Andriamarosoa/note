"""Aggregate the five outer-clean V12.0 folds without touching locked12."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Optional, Sequence

import numpy as np

SLOT_COUNT = 6
MODELS = ("v101", "v102", "v104", "v120")
STRATA = (
    "aggregate", "comp", "solo", "player00", "player00_comp",
    "player00_solo", "player00_rock_comp", "player00", "player01",
    "player02", "player03", "player04",
)


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
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "prediction_count": pred,
        "reference_count": ref,
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
        reports.append(json.loads(rp[0].read_text()))
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
    preds = {name: np.asarray(merged[f"pred{name[1:] if name.startswith('v') else name}"], dtype=np.int32) for name in ()}
    preds = {
        "v101": np.asarray(merged["pred101"], dtype=np.int32),
        "v102": np.asarray(merged["pred102"], dtype=np.int32),
        "v104": np.asarray(merged["pred104"], dtype=np.int32),
        "v120": np.asarray(merged["pred120"], dtype=np.int32),
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
            row[model] = {
                "metrics": onset,
                "cardinality": _card(k[mask], preds[model][mask]),
            }
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

    birth = np.asarray(merged["birth"], dtype=np.float64)
    global_v104 = strata["aggregate"]["v104"]["metrics"]["global"]["f1"]
    global_v120 = strata["aggregate"]["v120"]["metrics"]["global"]["f1"]
    rock_v104 = strata["player00_rock_comp"]["v104"]["metrics"]["global"]["f1"]
    rock_v120 = strata["player00_rock_comp"]["v120"]["metrics"]["global"]["f1"]
    poly104 = strata["aggregate"]["v104"]["cardinality"]["poly_accuracy"]
    poly120 = strata["aggregate"]["v120"]["cardinality"]["poly_accuracy"]

    result = {
        "schema_version": 1,
        "protocol": {
            "five_outer_composition_folds": True,
            "every_row_evaluated_once_outer_clean": True,
            "historical_validation_or_locked12_indexed_or_evaluated": False,
            "v120_expert_predictions_as_inputs": False,
            "birth_threshold_tuned": False,
            "birth_threshold": 0.5,
        },
        "data": {
            "clusters": int(len(k)),
            "outer_folds": 5,
            "unique_global_indices": int(len(np.unique(merged["global_index"]))),
        },
        "strata": strata,
        "per_true_k": per_k,
        "birth": {
            "mean_probability": float(np.mean(birth)),
            "positive_rate_at_050": float(np.mean(birth >= 0.5)),
            "true_birth_rate": float(np.mean(k > 0)),
        },
        "folds": {
            str(r["outer_fold"]): {
                "selected_epochs": r["data"]["selected_epochs"],
                "v104_f1": r["strata"]["aggregate"]["v104"]["metrics"]["global"]["f1"],
                "v120_f1": r["strata"]["aggregate"]["v120"]["metrics"]["global"]["f1"],
            }
            for r in reports
        },
        "comparison": {
            "v120_minus_v104_global_f1": global_v120 - global_v104,
            "v120_minus_v104_player00_f1": (
                strata["player00"]["v120"]["metrics"]["global"]["f1"]
                - strata["player00"]["v104"]["metrics"]["global"]["f1"]
            ),
            "v120_minus_v104_player00_rock_comp_f1": rock_v120 - rock_v104,
            "v120_minus_v104_poly_exact": poly120 - poly104,
            "promotion_candidate": bool(
                global_v120 > global_v104
                and rock_v120 >= rock_v104
                and poly120 >= poly104
            ),
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(args.output_dir / "predictions.npz", **merged)
    print(json.dumps({
        "global": {m: strata["aggregate"][m]["metrics"]["global"]["f1"] for m in MODELS},
        "player00": {m: strata["player00"][m]["metrics"]["global"]["f1"] for m in MODELS},
        "player00_rock_comp": {m: strata["player00_rock_comp"][m]["metrics"]["global"]["f1"] for m in MODELS},
        "poly_exact": {m: strata["aggregate"][m]["cardinality"]["poly_accuracy"] for m in MODELS},
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
