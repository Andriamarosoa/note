"""Audit player00 V10.4 failure modes from the completed nested outer predictions.

This is a train-only descriptive/diagnostic audit.  It reuses the five nested
outer-fold prediction artifacts, which together cover the GuitarSet training
split exactly once with outer-clean predictions.  Historical validation and
locked12 are never indexed or evaluated.

The goal is to separate two possibilities on player00:
  1. V10.4 loses because cardinality K is worse; or
  2. V10.4 can improve exact K while the resulting candidate-count change
     worsens onset precision/recall on particular compositions/tracks.

No model is trained and no threshold is tuned here.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import sys
from typing import Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for _p in (ROOT, SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from causal_note.guitarset import SLOT_COUNT
from scripts.train_boundaries import group_stem
from scripts.train_v91_ordinal_cardinality import _dataset_split
from scripts.train_v100_spectral_string_slots import _load_spectral_caches
from scripts import train_v102_source_time_assignment as v102
from scripts import train_v104_class_conditional_fusion as v104

MODELS = {
    "v101_deployment": "pred101_deploy",
    "v102_deployment": "pred102_deploy",
    "v104_probe_ensemble": "pred104_probe_ensemble",
    "v104_deployment": "pred104_deploy",
}


def _load_nested(eval_dir: Path, expected_count: int):
    paths = sorted(eval_dir.glob("**/v104-nested-eval-*.npz"))
    if len(paths) != 5:
        raise RuntimeError(f"expected 5 nested eval shards, found {len(paths)}")
    parts = []
    for path in paths:
        with np.load(path, allow_pickle=False) as z:
            required = {"global_index", "k", "member", *MODELS.values()}
            if not required.issubset(z.files):
                raise RuntimeError(f"missing keys in {path}: {sorted(required - set(z.files))}")
            parts.append({k: np.asarray(z[k]) for k in required})
    merged = {k: np.concatenate([p[k] for p in parts], axis=0) for k in parts[0]}
    order = np.argsort(merged["global_index"], kind="stable")
    merged = {k: v[order] for k, v in merged.items()}
    idx = np.asarray(merged["global_index"], dtype=np.int64)
    if len(idx) != expected_count:
        raise RuntimeError(f"nested coverage {len(idx)} != cache {expected_count}")
    if not np.array_equal(idx, np.arange(expected_count, dtype=np.int64)):
        raise RuntimeError("nested shards are not exact one-time train coverage")
    return merged, [str(p) for p in paths]


def _card(k, pred):
    return v102._cardinality_report(np.asarray(k, dtype=np.int32), np.asarray(pred, dtype=np.int32))


def _delta(a, b):
    if a is None or b is None:
        return None
    return float(a) - float(b)


def _mode(member: str):
    if member.endswith("_comp.jams"):
        return "comp"
    if member.endswith("_solo.jams"):
        return "solo"
    return "other"


def _genre(group: str):
    m = re.match(r"^([A-Za-z]+)", group)
    return m.group(1) if m else "unknown"


def _metric_row(cache, train_split, idx, pred):
    metrics = v104._metrics_for_indices(cache, train_split, idx, pred)
    g = metrics["global"]
    return {
        "f1": float(g["f1"]),
        "precision": float(g["precision"]),
        "recall": float(g["recall"]),
        "true_positive": int(g["true_positive"]),
        "false_positive": int(g["false_positive"]),
        "false_negative": int(g["false_negative"]),
        "prediction_count": int(g["prediction_count"]),
        "reference_count": int(g["reference_count"]),
        "prediction_reference_ratio": float(g["prediction_reference_ratio"]),
    }


def _slice_report(cache, train_split, k_all, predictions, idx):
    idx = np.asarray(idx, dtype=np.int64)
    out = {"clusters": int(len(idx))}
    ko = k_all[idx]
    for name, pred_all in predictions.items():
        pred = np.asarray(pred_all[idx], dtype=np.int32)
        out[name] = {
            "onset": _metric_row(cache, train_split, idx, pred),
            "cardinality": _card(ko, pred),
        }
    a = out["v101_deployment"]
    b = out["v104_deployment"]
    p = out["v104_probe_ensemble"]
    out["delta_v104_deploy_minus_v101"] = {
        "f1": b["onset"]["f1"] - a["onset"]["f1"],
        "precision": b["onset"]["precision"] - a["onset"]["precision"],
        "recall": b["onset"]["recall"] - a["onset"]["recall"],
        "true_positive": b["onset"]["true_positive"] - a["onset"]["true_positive"],
        "false_positive": b["onset"]["false_positive"] - a["onset"]["false_positive"],
        "false_negative": b["onset"]["false_negative"] - a["onset"]["false_negative"],
        "cardinality_accuracy": _delta(b["cardinality"]["accuracy"], a["cardinality"]["accuracy"]),
        "birth_accuracy": _delta(b["cardinality"]["birth_cluster_accuracy"], a["cardinality"]["birth_cluster_accuracy"]),
        "poly_accuracy": _delta(b["cardinality"]["poly_cluster_accuracy"], a["cardinality"]["poly_cluster_accuracy"]),
        "mae": _delta(b["cardinality"]["mean_absolute_class_error"], a["cardinality"]["mean_absolute_class_error"]),
    }
    out["delta_v104_probe_minus_deploy"] = {
        "f1": p["onset"]["f1"] - b["onset"]["f1"],
        "precision": p["onset"]["precision"] - b["onset"]["precision"],
        "recall": p["onset"]["recall"] - b["onset"]["recall"],
        "cardinality_accuracy": _delta(p["cardinality"]["accuracy"], b["cardinality"]["accuracy"]),
        "poly_accuracy": _delta(p["cardinality"]["poly_cluster_accuracy"], b["cardinality"]["poly_cluster_accuracy"]),
    }
    return out


def _compact(rows, n=8):
    def one(name, r):
        d = r["delta_v104_deploy_minus_v101"]
        pd = r["delta_v104_probe_minus_deploy"]
        return {
            "name": name,
            "clusters": r["clusters"],
            "v101_f1": r["v101_deployment"]["onset"]["f1"],
            "v104_deploy_f1": r["v104_deployment"]["onset"]["f1"],
            "v104_probe_f1": r["v104_probe_ensemble"]["onset"]["f1"],
            "v104_minus_v101_f1": d["f1"],
            "probe_minus_deploy_f1": pd["f1"],
            "v104_minus_v101_cardinality_accuracy": d["cardinality_accuracy"],
            "v104_minus_v101_poly_accuracy": d["poly_accuracy"],
            "delta_tp": d["true_positive"],
            "delta_fp": d["false_positive"],
            "delta_fn": d["false_negative"],
        }
    ordered = sorted(rows.items(), key=lambda kv: kv[1]["delta_v104_deploy_minus_v101"]["f1"])
    return {
        "worst": [one(name, r) for name, r in ordered[:n]],
        "best": [one(name, r) for name, r in ordered[-n:][::-1]],
    }


def audit(args):
    cache = _load_spectral_caches(args.cache_dir)
    _, train_split, validation = _dataset_split(args.dataset_dir)
    train_members = {t.annotation_member for t in train_split}
    validation_members = {t.annotation_member for t in validation}
    if set(cache["track_members"]) != train_members:
        raise RuntimeError("cache/train split mismatch")
    if train_members & validation_members:
        raise RuntimeError("train/validation leakage")

    nested, shard_paths = _load_nested(args.eval_dir, len(cache["members"]))
    cache_members = np.asarray([str(x) for x in cache["members"]])
    if not np.array_equal(np.asarray(nested["member"]).astype(str), cache_members):
        raise RuntimeError("nested member order does not match frozen cache")
    k_all = np.minimum(np.asarray(cache["exact"], dtype=np.int32), SLOT_COUNT)
    if not np.array_equal(np.asarray(nested["k"], dtype=np.int32), k_all):
        raise RuntimeError("nested K labels do not match frozen cache")

    predictions = {
        name: np.asarray(nested[key], dtype=np.int32) for name, key in MODELS.items()
    }
    player00_idx = np.flatnonzero(np.char.startswith(cache_members.astype("U"), "00_")).astype(np.int64)
    if len(player00_idx) == 0:
        raise RuntimeError("no player00 rows")

    by_member_track = {t.annotation_member: t for t in train_split}
    groups = np.asarray([group_stem(by_member_track[m]) for m in cache_members], dtype="U64")
    modes = np.asarray([_mode(m) for m in cache_members], dtype="U8")
    genres = np.asarray([_genre(g) for g in groups], dtype="U16")

    overall = _slice_report(cache, train_split, k_all, predictions, player00_idx)

    by_group = {}
    for group in sorted(set(groups[player00_idx].tolist())):
        idx = player00_idx[groups[player00_idx] == group]
        by_group[group] = _slice_report(cache, train_split, k_all, predictions, idx)

    by_track = {}
    player_members = cache_members[player00_idx]
    for member in sorted(set(player_members.tolist())):
        idx = player00_idx[player_members == member]
        by_track[member] = _slice_report(cache, train_split, k_all, predictions, idx)

    by_mode = {}
    for mode in sorted(set(modes[player00_idx].tolist())):
        idx = player00_idx[modes[player00_idx] == mode]
        by_mode[mode] = _slice_report(cache, train_split, k_all, predictions, idx)

    by_genre = {}
    for genre in sorted(set(genres[player00_idx].tolist())):
        idx = player00_idx[genres[player00_idx] == genre]
        by_genre[genre] = _slice_report(cache, train_split, k_all, predictions, idx)

    # Pure cardinality diagnostics by true K do not invoke onset evaluation on
    # partial tracks.  This keeps the onset metric semantically valid.
    by_true_k = {}
    for kval in range(SLOT_COUNT + 1):
        idx = player00_idx[k_all[player00_idx] == kval]
        if len(idx) == 0:
            continue
        row = {"clusters": int(len(idx))}
        for name, pred_all in predictions.items():
            pred = pred_all[idx]
            row[name] = {
                "accuracy": float(np.mean(pred == kval)),
                "mean_error": float(np.mean(pred - kval)),
                "mae": float(np.mean(np.abs(pred - kval))),
                "under_rate": float(np.mean(pred < kval)),
                "over_rate": float(np.mean(pred > kval)),
            }
        by_true_k[str(kval)] = row

    group_regress = [r for r in by_group.values() if r["delta_v104_deploy_minus_v101"]["f1"] < 0]
    group_count_better_f1_worse = [
        r for r in by_group.values()
        if r["delta_v104_deploy_minus_v101"]["cardinality_accuracy"] > 0
        and r["delta_v104_deploy_minus_v101"]["f1"] < 0
    ]
    track_regress = [r for r in by_track.values() if r["delta_v104_deploy_minus_v101"]["f1"] < 0]
    track_count_better_f1_worse = [
        r for r in by_track.values()
        if r["delta_v104_deploy_minus_v101"]["cardinality_accuracy"] > 0
        and r["delta_v104_deploy_minus_v101"]["f1"] < 0
    ]

    result = {
        "schema_version": 1,
        "protocol": {
            "train_only_nested_predictions": True,
            "historical_validation_or_locked12_indexed_or_evaluated": False,
            "models_retrained_or_tuned": False,
            "thresholds_tuned": False,
            "player": "00",
        },
        "data": {
            "nested_shards": shard_paths,
            "train_clusters": int(len(cache_members)),
            "player00_clusters": int(len(player00_idx)),
            "player00_tracks": int(len(by_track)),
            "player00_composition_groups": int(len(by_group)),
        },
        "overall_player00": overall,
        "failure_mode_counts": {
            "groups_f1_regress_v104_vs_v101": int(len(group_regress)),
            "groups_cardinality_improves_but_f1_regresses": int(len(group_count_better_f1_worse)),
            "tracks_f1_regress_v104_vs_v101": int(len(track_regress)),
            "tracks_cardinality_improves_but_f1_regresses": int(len(track_count_better_f1_worse)),
        },
        "group_ranking": _compact(by_group, n=10),
        "track_ranking": _compact(by_track, n=12),
        "by_mode": by_mode,
        "by_genre": by_genre,
        "by_true_k_cardinality_only": by_true_k,
        "by_group": by_group,
        "by_track": by_track,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "player00_v101_f1": overall["v101_deployment"]["onset"]["f1"],
        "player00_v104_f1": overall["v104_deployment"]["onset"]["f1"],
        "groups_f1_regress": len(group_regress),
        "groups_count_better_f1_worse": len(group_count_better_f1_worse),
        "tracks_f1_regress": len(track_regress),
        "tracks_count_better_f1_worse": len(track_count_better_f1_worse),
        "worst_groups": result["group_ranking"]["worst"][:5],
    }, indent=2))
    return result


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("dataset_dir", nargs="?", type=Path, default=ROOT / "data" / "GuitarSet")
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--eval-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    return p


def main(argv: Optional[Sequence[str]] = None):
    audit(parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
