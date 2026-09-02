"""V10.4 strict out-of-fold expert prediction shard.

Train fresh V10.1 and V10.2 experts on four of five composition-group folds and
predict the fifth fold.  The held fold is never used for training, validation,
early stopping, decoder selection, or epoch selection.  Epoch counts and decode
modes are copied from the historical train-only calibrated V10.1/V10.2 reports.

The output is a compact OOF feature shard for the V10.4 class-conditional
residual fusion model.  No locked validation track is indexed or evaluated here.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import sys
from typing import Dict, Optional, Sequence

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
from scripts import train_v101_string_query_attention as v101
from scripts import train_v102_source_time_assignment as v102
from scripts import run_v102_competitive_mass as v102_mass
from scripts import train_v103_residual_soft_fusion as v103

DEFAULT_SEED_V104 = 10431
FOLD_COUNT = 5


class V104OOFError(RuntimeError):
    pass


def _balanced_group_folds(cache, train_split, fold_count: int = FOLD_COUNT):
    """Deterministically balance composition groups by cached cluster count."""
    by_member = {t.annotation_member: t for t in train_split}
    counts: Counter[str] = Counter()
    for member in cache["members"]:
        key = str(member)
        if key not in by_member:
            raise V104OOFError(f"cached member absent from train split: {key}")
        counts[group_stem(by_member[key])] += 1
    if len(counts) < fold_count:
        raise V104OOFError(f"only {len(counts)} groups for {fold_count} folds")

    loads = [0] * fold_count
    groups_per_fold = [[] for _ in range(fold_count)]
    assignment: Dict[str, int] = {}
    for group, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        fold = min(range(fold_count), key=lambda f: (loads[f], len(groups_per_fold[f]), f))
        assignment[group] = fold
        groups_per_fold[fold].append(group)
        loads[fold] += int(n)
    return assignment, groups_per_fold, loads, counts


def _train_v101(cache, fit_idx, pitch_targets, pitch_mask, k, epochs: int, seed: int):
    import tensorflow as tf
    tf.random.set_seed(seed)
    model, _, _ = v101._build_model()
    model.fit(
        v101._inputs(cache, fit_idx),
        v101._targets(cache["slot_targets"][fit_idx], pitch_targets[fit_idx], k[fit_idx]),
        sample_weight=v101._sample_weights(cache["slot_targets"][fit_idx], pitch_mask[fit_idx], k[fit_idx]),
        epochs=epochs,
        batch_size=64,
        shuffle=True,
        verbose=2,
    )
    return model


def _train_v102(cache, fit_idx, pitch_targets, time_mask, time_targets, k, epochs: int, seed: int):
    import tensorflow as tf
    tf.random.set_seed(seed)
    model, _, _ = v102_mass._build_model_mass_aware()
    model.fit(
        v102._inputs(cache, fit_idx),
        v102._targets(
            cache["slot_targets"][fit_idx],
            pitch_targets[fit_idx],
            time_targets[fit_idx],
            k[fit_idx],
        ),
        sample_weight=v102._sample_weights(
            cache["slot_targets"][fit_idx], time_mask[fit_idx], k[fit_idx]
        ),
        epochs=epochs,
        batch_size=64,
        shuffle=True,
        verbose=2,
    )
    return model


def run_fold(args):
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    random.seed(args.seed + args.fold)
    np.random.seed(args.seed + args.fold)

    if not 0 <= args.fold < FOLD_COUNT:
        raise V104OOFError(f"fold must be in [0,{FOLD_COUNT})")

    cache = _load_spectral_caches(args.cache_dir)
    indexed, train_split, validation = _dataset_split(args.dataset_dir)
    train_members = {t.annotation_member for t in train_split}
    validation_members = {t.annotation_member for t in validation}
    if train_members & validation_members:
        raise V104OOFError("train/validation leakage")
    if set(cache["track_members"]) != train_members:
        raise V104OOFError("spectral cache does not exactly cover train split")
    if any(str(m) in validation_members for m in cache["members"]):
        raise V104OOFError("locked validation member present in train cache")

    assignment, groups_per_fold, loads, group_counts = _balanced_group_folds(cache, train_split)
    by_member = {t.annotation_member: t for t in train_split}
    row_fold = np.asarray(
        [assignment[group_stem(by_member[str(m)])] for m in cache["members"]], dtype=np.int16
    )
    hold_idx = np.flatnonzero(row_fold == args.fold).astype(np.int64)
    fit_idx = np.flatnonzero(row_fold != args.fold).astype(np.int64)
    if len(hold_idx) < 1000 or len(fit_idx) < 10000:
        raise V104OOFError(f"unexpected fold sizes fit={len(fit_idx)} hold={len(hold_idx)}")

    report101 = json.loads(args.v101_report.read_text())
    report102 = json.loads(args.v102_report.read_text())
    epochs101 = int(report101["configuration"]["epochs_ran"])
    epochs102 = int(report102["configuration"]["epochs_ran"])
    mode101 = str(report101["configuration"]["decode_mode"])
    mode102 = str(report102["configuration"]["decode_mode"])
    if report101["configuration"].get("decode_mode_selected_on_locked_validation") is not False:
        raise V104OOFError("V10.1 decode mode leakage")
    if report102["configuration"].get("decode_mode_selected_on_locked_validation") is not False:
        raise V104OOFError("V10.2 decode mode leakage")

    k = np.minimum(np.asarray(cache["exact"], dtype=np.int32), SLOT_COUNT)
    print(
        f"V10.4 OOF fold={args.fold}/{FOLD_COUNT} fit={len(fit_idx)} hold={len(hold_idx)} "
        f"groups={groups_per_fold[args.fold]} epochs101={epochs101} epochs102={epochs102}"
    )

    # Use the V10.2 global event-to-cluster assignment for both experts' pitch
    # auxiliaries.  It is the corrected exact assignment used by the successful
    # V10.2 run and is verified against the cached six-string occupancy labels.
    # This avoids V10.1's older local-row pitch helper, whose row indexing is not
    # valid when reused as a full-cache cross-fitting primitive.
    candidate_samples, reconstruction = v102._reconstruct_candidates(cache)
    pitch_targets, time_mask, time_targets, time_sample, time_diag = v102._derive_supervision(
        [str(x) for x in cache["members"]],
        candidate_samples,
        args.dataset_dir,
        expected_slot_targets=cache["slot_targets"],
    )
    time_diag["cluster_reconstruction"] = reconstruction
    pitch_mask = np.asarray(time_mask, dtype=np.float32)
    pitch_diag = {
        "source": "V10.2 exact global event-to-cluster assignment",
        "shared_with_v102": True,
        "slot_mask_agreement": time_diag.get("slot_mask_agreement"),
        "active_slot_pitch_coverage": time_diag.get("active_slot_time_coverage"),
        "assigned_events": time_diag.get("assigned_events"),
        "unassigned_events": time_diag.get("unassigned_events"),
    }

    model101 = _train_v101(
        cache, fit_idx, pitch_targets, pitch_mask, k, epochs101, args.seed + 100 + args.fold
    )
    model102 = _train_v102(
        cache, fit_idx, pitch_targets, time_mask, time_targets, k, epochs102, args.seed + 200 + args.fold
    )

    inputs102 = v102._inputs(cache, hold_idx)
    slots101, _, cond101 = v101._predict(model101, v101._inputs(cache, hold_idx))
    slots102, _, times102, p102, _ = v102._predict(model102, inputs102)
    p101 = v103._exact_from_ordinal(cond101)
    pred101 = v101._decode(slots101, cond101, mode101)
    pred102 = v102._decode(slots102, p102, mode102)
    anchor = v103._anchor_distribution(p101, pred101)
    assignment_features = v103._source_assignment_features(model102, inputs102)
    features = v103._feature_matrix(
        p101,
        p102,
        slots101,
        slots102,
        times102,
        assignment_features,
        cache["stats"][hold_idx],
    )

    groups = np.asarray(
        [group_stem(by_member[str(cache["members"][i])]) for i in hold_idx], dtype="U64"
    )
    members = np.asarray([str(cache["members"][i]) for i in hold_idx], dtype="U96")
    shard_path = args.output_dir / f"v104-oof-fold-{args.fold}.npz"
    np.savez_compressed(
        shard_path,
        schema_version=np.asarray([1], dtype=np.int16),
        fold=np.full(len(hold_idx), args.fold, dtype=np.int16),
        global_index=hold_idx,
        member=members,
        group=groups,
        k=k[hold_idx].astype(np.int16),
        features=features.astype(np.float32),
        p101=p101.astype(np.float32),
        anchor=anchor.astype(np.float32),
        p102=p102.astype(np.float32),
        pred101=pred101.astype(np.int16),
        pred102=pred102.astype(np.int16),
    )

    audit = v103._disagreement_audit(k[hold_idx], pred101, pred102)
    report = {
        "schema_version": 1,
        "fold": args.fold,
        "fold_count": FOLD_COUNT,
        "seed": args.seed,
        "protocol": {
            "held_fold_used_for_training": False,
            "held_fold_used_for_validation": False,
            "early_stopping_used": False,
            "epochs_selected_from_historical_train_only_reports": True,
            "locked12_indexed_or_evaluated": False,
            "runtime_inputs_use_annotations": False,
        },
        "epochs": {"v101": epochs101, "v102": epochs102},
        "decode_modes": {"v101": mode101, "v102": mode102},
        "data": {
            "indexed_track_count": len(indexed),
            "train_track_count": len(train_split),
            "validation_track_count": len(validation),
            "cached_cluster_count": len(k),
            "fit_cluster_count": len(fit_idx),
            "hold_cluster_count": len(hold_idx),
            "held_groups": groups_per_fold[args.fold],
            "all_fold_groups": groups_per_fold,
            "fold_cluster_loads": loads,
            "group_cluster_counts": dict(sorted(group_counts.items())),
        },
        "supervision": {"v101_pitch": pitch_diag, "v102_source_time": time_diag},
        "oof_disagreement": audit,
        "oof_cardinality": {
            "v101": v102._cardinality_report(k[hold_idx], pred101),
            "v102": v102._cardinality_report(k[hold_idx], pred102),
        },
        "feature_dim": int(features.shape[1]),
        "shard": shard_path.name,
    }
    (args.output_dir / f"v104-oof-fold-{args.fold}.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({
        "fold": args.fold,
        "fit_clusters": len(fit_idx),
        "hold_clusters": len(hold_idx),
        "held_groups": groups_per_fold[args.fold],
        "feature_dim": int(features.shape[1]),
        "disagreement": audit,
    }, indent=2, sort_keys=True))
    return report


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("dataset_dir", nargs="?", type=Path, default=ROOT / "data" / "GuitarSet")
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--v101-report", type=Path, required=True)
    p.add_argument("--v102-report", type=Path, required=True)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED_V104)
    return p


def main(argv: Optional[Sequence[str]] = None):
    args = parser().parse_args(argv)
    run_fold(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
