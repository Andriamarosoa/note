"""Nested train-only V10.4 expert prediction job.

For an outer composition fold, this script either:
- trains V10.1/V10.2 on three of the four outer-train folds and predicts one
  inner fold (``mode=inner``), while also predicting the untouched outer fold
  as an in-domain lower-data outer probe, or
- trains V10.1/V10.2 on all four outer-train folds and predicts the untouched
  outer fold (``mode=outer``), matching deployment retraining.

The historical validation/locked split is never evaluated. Epoch counts and
decode modes are frozen from historical train-only V10.1/V10.2 reports.
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

from causal_note.guitarset import SLOT_COUNT
from scripts.train_boundaries import group_stem
from scripts.train_v91_ordinal_cardinality import _dataset_split
from scripts.train_v100_spectral_string_slots import _load_spectral_caches
from scripts import train_v101_string_query_attention as v101
from scripts import train_v102_source_time_assignment as v102
from scripts import run_v102_competitive_mass as v102_mass
from scripts import train_v103_residual_soft_fusion as v103
from scripts import train_v104_oof_fold as oofmod


def _predict_pack(model101, model102, cache, idx, mode101, mode102):
    inputs102 = v102._inputs(cache, idx)
    slots101, _, cond101 = v101._predict(model101, v101._inputs(cache, idx))
    slots102, _, times102, p102, _ = v102._predict(model102, inputs102)
    p101 = v103._exact_from_ordinal(cond101)
    pred101 = v101._decode(slots101, cond101, mode101)
    pred102 = v102._decode(slots102, p102, mode102)
    anchor = v103._anchor_distribution(p101, pred101)
    assignment_features = v103._source_assignment_features(model102, inputs102)
    features = v103._feature_matrix(
        p101, p102, slots101, slots102, times102, assignment_features, cache["stats"][idx]
    )
    return {
        "features": features.astype(np.float32),
        "p101": p101.astype(np.float32),
        "anchor": anchor.astype(np.float32),
        "p102": p102.astype(np.float32),
        "pred101": pred101.astype(np.int16),
        "pred102": pred102.astype(np.int16),
    }


def _save_pack(path, pack, cache, idx, k, outer_fold, held_fold, role):
    np.savez_compressed(
        path,
        schema_version=np.asarray([2], dtype=np.int16),
        role=np.asarray([role]),
        outer_fold=np.full(len(idx), outer_fold, dtype=np.int16),
        held_fold=np.full(len(idx), held_fold, dtype=np.int16),
        global_index=idx,
        k=k[idx].astype(np.int16),
        member=np.asarray([str(cache["members"][i]) for i in idx], dtype="U96"),
        **pack,
    )


def run(args):
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    if not 0 <= args.outer_fold < oofmod.FOLD_COUNT:
        raise ValueError("outer-fold outside range")
    if args.mode == "inner":
        if args.inner_fold is None or not 0 <= args.inner_fold < oofmod.FOLD_COUNT:
            raise ValueError("inner mode requires valid --inner-fold")
        if args.inner_fold == args.outer_fold:
            raise ValueError("inner fold must differ from outer fold")
    elif args.inner_fold is not None:
        raise ValueError("outer mode must not specify --inner-fold")

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

    assignment, groups_per_fold, loads, _ = oofmod._balanced_group_folds(cache, train_split)
    by_member = {t.annotation_member: t for t in train_split}
    row_fold = np.asarray(
        [assignment[group_stem(by_member[str(m)])] for m in cache["members"]], dtype=np.int16
    )
    outer_idx = np.flatnonzero(row_fold == args.outer_fold).astype(np.int64)
    if args.mode == "inner":
        hold_fold = int(args.inner_fold)
        hold_idx = np.flatnonzero(row_fold == hold_fold).astype(np.int64)
        fit_idx = np.flatnonzero((row_fold != args.outer_fold) & (row_fold != hold_fold)).astype(np.int64)
    else:
        hold_fold = int(args.outer_fold)
        hold_idx = outer_idx
        fit_idx = np.flatnonzero(row_fold != args.outer_fold).astype(np.int64)

    if len(fit_idx) < 10000 or len(hold_idx) < 1000:
        raise RuntimeError(f"unexpected fit/hold sizes {len(fit_idx)}/{len(hold_idx)}")
    if np.intersect1d(fit_idx, outer_idx).size:
        raise RuntimeError("outer fold leaked into expert training")
    if args.mode == "inner" and np.intersect1d(fit_idx, hold_idx).size:
        raise RuntimeError("inner held fold leaked into expert training")

    r101 = json.loads(args.v101_report.read_text())
    r102 = json.loads(args.v102_report.read_text())
    epochs101 = int(r101["configuration"]["epochs_ran"])
    epochs102 = int(r102["configuration"]["epochs_ran"])
    mode101 = str(r101["configuration"]["decode_mode"])
    mode102 = str(r102["configuration"]["decode_mode"])
    if r101["configuration"].get("decode_mode_selected_on_locked_validation") is not False:
        raise RuntimeError("V10.1 decode leakage")
    if r102["configuration"].get("decode_mode_selected_on_locked_validation") is not False:
        raise RuntimeError("V10.2 decode leakage")

    k = np.minimum(np.asarray(cache["exact"], dtype=np.int32), SLOT_COUNT)
    candidate_samples, reconstruction = v102._reconstruct_candidates(cache)
    pitch_targets, time_mask, time_targets, _, time_diag = v102._derive_supervision(
        [str(x) for x in cache["members"]], candidate_samples, args.dataset_dir,
        expected_slot_targets=cache["slot_targets"],
    )
    pitch_mask = np.asarray(time_mask, dtype=np.float32)

    print(
        f"nested expert mode={args.mode} outer={args.outer_fold} hold={hold_fold} "
        f"fit={len(fit_idx)} hold_rows={len(hold_idx)} epochs101={epochs101} epochs102={epochs102}"
    )
    model101 = oofmod._train_v101(
        cache, fit_idx, pitch_targets, pitch_mask, k, epochs101, args.seed + 101
    )
    model102 = oofmod._train_v102(
        cache, fit_idx, pitch_targets, time_mask, time_targets, k, epochs102, args.seed + 202
    )

    held_pack = _predict_pack(model101, model102, cache, hold_idx, mode101, mode102)
    tag = (
        f"outer-{args.outer_fold}-inner-{hold_fold}"
        if args.mode == "inner" else f"outer-{args.outer_fold}-deploy"
    )
    held_npz = args.output_dir / f"v104-nested-{tag}.npz"
    _save_pack(held_npz, held_pack, cache, hold_idx, k, args.outer_fold, hold_fold, args.mode)

    outer_probe_npz = None
    if args.mode == "inner":
        outer_pack = _predict_pack(model101, model102, cache, outer_idx, mode101, mode102)
        outer_probe_npz = args.output_dir / f"v104-nested-outer-{args.outer_fold}-probe-from-inner-{hold_fold}.npz"
        _save_pack(
            outer_probe_npz, outer_pack, cache, outer_idx, k,
            args.outer_fold, hold_fold, "outer_probe_from_inner_expert",
        )

    report = {
        "schema_version": 2,
        "protocol": {
            "train_only": True,
            "historical_validation_or_locked_evaluated": False,
            "outer_fold_used_for_expert_training": False,
            "inner_held_fold_used_for_expert_training": False if args.mode == "inner" else None,
            "fixed_historical_train_only_epochs": True,
            "runtime_inputs_use_annotations": False,
        },
        "mode": args.mode,
        "outer_fold": args.outer_fold,
        "held_fold": hold_fold,
        "data": {
            "fit_clusters": int(len(fit_idx)),
            "hold_clusters": int(len(hold_idx)),
            "outer_clusters": int(len(outer_idx)),
            "outer_groups": groups_per_fold[args.outer_fold],
            "held_groups": groups_per_fold[hold_fold],
            "fold_cluster_loads": loads,
        },
        "epochs": {"v101": epochs101, "v102": epochs102},
        "decode_modes": {"v101": mode101, "v102": mode102},
        "supervision": {
            "slot_mask_agreement": time_diag.get("slot_mask_agreement"),
            "cluster_reconstruction": reconstruction,
        },
        "cardinality": {
            "v101": v102._cardinality_report(k[hold_idx], held_pack["pred101"]),
            "v102": v102._cardinality_report(k[hold_idx], held_pack["pred102"]),
        },
        "feature_dim": int(held_pack["features"].shape[1]),
        "held_npz": held_npz.name,
        "outer_probe_npz": outer_probe_npz.name if outer_probe_npz else None,
    }
    (args.output_dir / f"v104-nested-{tag}.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({
        "mode": args.mode, "outer": args.outer_fold, "held": hold_fold,
        "fit": len(fit_idx), "hold": len(hold_idx),
        "v101_exact": report["cardinality"]["v101"]["accuracy"],
        "v102_exact": report["cardinality"]["v102"]["accuracy"],
        "outer_probe_saved": outer_probe_npz is not None,
    }, indent=2))
    return report


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("dataset_dir", nargs="?", type=Path, default=ROOT / "data" / "GuitarSet")
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--v101-report", type=Path, required=True)
    p.add_argument("--v102-report", type=Path, required=True)
    p.add_argument("--mode", choices=("inner", "outer"), required=True)
    p.add_argument("--outer-fold", type=int, required=True)
    p.add_argument("--inner-fold", type=int)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=10491)
    return p


def main(argv: Optional[Sequence[str]] = None):
    run(parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
