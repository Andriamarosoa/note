"""Train-only V10.4 audit split by GuitarSet player 00..04.

No validation/locked track is indexed or evaluated.  The leakage-safe OOF
predictions are evaluated per player, then the same train clusters are passed
through the historical full-training experts only as a descriptive input-shift
comparison.  This audit answers whether player 00 is unusually difficult in the
same OOF protocol that produced V10.4's strong aggregate train result.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
from typing import Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for _p in (ROOT, SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from causal_note.guitarset import ALLOWED_PLAYERS, SLOT_COUNT
from scripts.train_v91_ordinal_cardinality import _dataset_split
from scripts.train_v100_spectral_string_slots import _load_spectral_caches
from scripts import train_v101_string_query_attention as v101
from scripts import train_v102_source_time_assignment as v102
from scripts import run_v102_competitive_mass as v102_mass
from scripts import train_v103_residual_soft_fusion as v103
from scripts import train_v104_class_conditional_fusion as v104
from scripts import audit_v104_transfer as transfer


def _player(member: str) -> str:
    p = str(member).split("_", 1)[0]
    if p not in ALLOWED_PLAYERS:
        raise RuntimeError(f"unexpected player prefix {p!r} for {member!r}")
    return p


def _metrics(cache, train_split, idx, pred):
    return v104._metrics_for_indices(
        cache, train_split, np.asarray(idx, dtype=np.int64), np.asarray(pred, dtype=np.int32)
    )


def _card(k, pred):
    return v102._cardinality_report(np.asarray(k, dtype=np.int32), np.asarray(pred, dtype=np.int32))


def _tv(a, b):
    return 0.5 * np.sum(
        np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)), axis=1
    )


def audit(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    import tensorflow as tf
    tf.random.set_seed(args.seed)

    cache = _load_spectral_caches(args.cache_dir)
    _, train_split, validation = _dataset_split(args.dataset_dir)
    train_members = {t.annotation_member for t in train_split}
    validation_members = {t.annotation_member for t in validation}
    if set(cache["track_members"]) != train_members:
        raise RuntimeError("cache/train split mismatch")
    if train_members & validation_members:
        raise RuntimeError("train/validation leakage")

    k = np.minimum(np.asarray(cache["exact"], dtype=np.int32), SLOT_COUNT)
    oof, _, _ = v104._load_oof(args.oof_dir, len(k))
    if not np.array_equal(k, np.asarray(oof["k"], dtype=np.int32)):
        raise RuntimeError("OOF labels differ from cache")

    frozen = json.loads(args.v104_report.read_text())
    selected_epochs = int(frozen["configuration"]["selected_epochs"])
    crossfit_pred, crossfit = transfer._crossfit_fusion(
        cache, train_split, oof, selected_epochs, args.seed + 1000
    )

    report101 = json.loads(args.v101_report.read_text())
    report102 = json.loads(args.v102_report.read_text())
    mode101 = str(report101["configuration"]["decode_mode"])
    mode102 = str(report102["configuration"]["decode_mode"])
    model101, _, _ = v101._build_model()
    model101.load_weights(args.v101_weights)
    model102, _, _ = v102_mass._build_model_mass_aware()
    model102.load_weights(args.v102_weights)
    full = v103._prepare_fusion_inputs(model101, model102, cache, None, mode101, mode102)

    with np.load(args.v104_scaler) as z:
        mean = np.asarray(z["mean"], dtype=np.float32)
        std = np.asarray(z["std"], dtype=np.float32)
    fusion, _, _ = v104._build_fusion(np.asarray(oof["features"]).shape[1])
    fusion.load_weights(args.v104_weights)
    x_full = v104._standardize_apply(np.asarray(full["features"], dtype=np.float32), mean, std)
    p_full = np.asarray(
        fusion.predict(v104._inputs(x_full, full["anchor"], full["p102"]), batch_size=256, verbose=0)
    )
    pred_full104 = np.argmax(p_full, axis=1).astype(np.int32)

    members = np.asarray([str(x) for x in cache["members"]])
    players = np.asarray([_player(x) for x in members])
    out = {}
    for player in ALLOWED_PLAYERS:
        idx = np.flatnonzero(players == player).astype(np.int64)
        if len(idx) == 0:
            raise RuntimeError(f"player {player} has no train clusters")
        kp = k[idx]
        poly = kp >= 2
        tv101 = _tv(np.asarray(oof["p101"])[idx], np.asarray(full["p101"])[idx])
        tv102 = _tv(np.asarray(oof["p102"])[idx], np.asarray(full["p102"])[idx])
        out[player] = {
            "clusters": int(len(idx)),
            "tracks": int(len({str(members[i]) for i in idx})),
            "true_cardinality_histogram": {
                str(kk): int(np.sum(kp == kk)) for kk in range(SLOT_COUNT + 1)
            },
            "strict_oof": {
                "v101_metrics": _metrics(cache, train_split, idx, np.asarray(oof["pred101"])[idx]),
                "v102_metrics": _metrics(cache, train_split, idx, np.asarray(oof["pred102"])[idx]),
                "v104_crossfit_metrics": _metrics(cache, train_split, idx, crossfit_pred[idx]),
                "v101_cardinality": _card(kp, np.asarray(oof["pred101"])[idx]),
                "v102_cardinality": _card(kp, np.asarray(oof["pred102"])[idx]),
                "v104_crossfit_cardinality": _card(kp, crossfit_pred[idx]),
            },
            "same_train_rows_full_experts_descriptive_only": {
                "v101_metrics": _metrics(cache, train_split, idx, np.asarray(full["pred101"])[idx]),
                "v102_metrics": _metrics(cache, train_split, idx, np.asarray(full["pred102"])[idx]),
                "v104_frozen_metrics": _metrics(cache, train_split, idx, pred_full104[idx]),
                "v101_cardinality": _card(kp, np.asarray(full["pred101"])[idx]),
                "v102_cardinality": _card(kp, np.asarray(full["pred102"])[idx]),
                "v104_frozen_cardinality": _card(kp, pred_full104[idx]),
                "v101_oof_full_decoded_agreement": float(
                    np.mean(np.asarray(oof["pred101"])[idx] == np.asarray(full["pred101"])[idx])
                ),
                "v102_oof_full_decoded_agreement": float(
                    np.mean(np.asarray(oof["pred102"])[idx] == np.asarray(full["pred102"])[idx])
                ),
                "v101_mean_total_variation": float(np.mean(tv101)),
                "v102_mean_total_variation": float(np.mean(tv102)),
                "v101_poly_mean_total_variation": float(np.mean(tv101[poly])) if np.any(poly) else None,
                "v102_poly_mean_total_variation": float(np.mean(tv102[poly])) if np.any(poly) else None,
            },
        }

    result = {
        "schema_version": 1,
        "protocol": {
            "train_only_audit": True,
            "locked12_indexed_or_evaluated": False,
            "players": list(ALLOWED_PLAYERS),
            "strict_oof_player_comparison_is_leakage_safe": True,
            "same_train_rows_full_expert_comparison_is_descriptive_not_causal": True,
        },
        "data": {
            "clusters": int(len(k)),
            "train_tracks": int(len(train_split)),
            "validation_tracks_not_evaluated": int(len(validation)),
        },
        "crossfit_aggregate": crossfit["aggregate"],
        "by_player": out,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        p: {
            "clusters": out[p]["clusters"],
            "oof_v101_f1": out[p]["strict_oof"]["v101_metrics"]["global"]["f1"],
            "oof_v102_f1": out[p]["strict_oof"]["v102_metrics"]["global"]["f1"],
            "oof_v104_f1": out[p]["strict_oof"]["v104_crossfit_metrics"]["global"]["f1"],
            "oof_v104_poly": out[p]["strict_oof"]["v104_crossfit_cardinality"]["poly_cluster_accuracy"],
        }
        for p in ALLOWED_PLAYERS
    }, indent=2, sort_keys=True))
    return result


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("dataset_dir", nargs="?", type=Path, default=ROOT / "data" / "GuitarSet")
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--oof-dir", type=Path, required=True)
    p.add_argument("--v101-weights", type=Path, required=True)
    p.add_argument("--v101-report", type=Path, required=True)
    p.add_argument("--v102-weights", type=Path, required=True)
    p.add_argument("--v102-report", type=Path, required=True)
    p.add_argument("--v104-weights", type=Path, required=True)
    p.add_argument("--v104-scaler", type=Path, required=True)
    p.add_argument("--v104-report", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=10471)
    return p


def main(argv: Optional[Sequence[str]] = None):
    audit(parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
