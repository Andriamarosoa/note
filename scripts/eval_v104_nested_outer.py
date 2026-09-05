"""Evaluate one outer fold with a deployment-matched nested V10.4 protocol.

Four inner-OOF shards are produced without the outer fold. They train and
calibrate a fresh V10.4 fusion. Four lower-data expert pairs also predict the
same outer fold (OOF-like probes). A fifth expert pair is retrained on all four
outer-train folds and predicts the outer fold (deployment experts).

Comparing the probe ensemble with deployment experts on the exact same unseen
outer rows measures the effect of retraining the experts before deployment,
without using historical validation/locked tracks.
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
from scripts.train_boundaries import group_stem
from scripts.train_v91_ordinal_cardinality import _dataset_split
from scripts.train_v100_spectral_string_slots import _load_spectral_caches
from scripts import train_v102_source_time_assignment as v102
from scripts import train_v104_class_conditional_fusion as v104
from scripts import train_v104_oof_fold as oofmod


def _load_npz(path: Path):
    with np.load(path, allow_pickle=False) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def _tv(a, b):
    return 0.5 * np.sum(
        np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)), axis=1
    )


def _metrics(cache, train_split, idx, pred):
    return v104._metrics_for_indices(cache, train_split, idx, pred)


def _card(k, pred):
    return v102._cardinality_report(np.asarray(k, dtype=np.int32), np.asarray(pred, dtype=np.int32))


def _player(member):
    p = str(member).split("_", 1)[0]
    if p not in ALLOWED_PLAYERS:
        raise RuntimeError(f"unexpected player {p}")
    return p


def _concat_inner(paths):
    parts = [_load_npz(p) for p in paths]
    keys = {"global_index", "k", "member", "features", "p101", "anchor", "p102", "pred101", "pred102", "held_fold"}
    for p in parts:
        if not keys.issubset(p):
            raise RuntimeError("inner shard missing keys")
    merged = {k: np.concatenate([p[k] for p in parts], axis=0) for k in keys}
    order = np.argsort(merged["global_index"], kind="stable")
    return {k: v[order] for k, v in merged.items()}


def evaluate(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    import tensorflow as tf
    from tensorflow import keras
    tf.random.set_seed(args.seed)

    cache = _load_spectral_caches(args.cache_dir)
    _, train_split, validation = _dataset_split(args.dataset_dir)
    train_members = {t.annotation_member for t in train_split}
    validation_members = {t.annotation_member for t in validation}
    if set(cache["track_members"]) != train_members:
        raise RuntimeError("cache/train split mismatch")
    if train_members & validation_members:
        raise RuntimeError("train/validation leakage")

    assignment, groups_per_fold, _, _ = oofmod._balanced_group_folds(cache, train_split)
    by_member = {t.annotation_member: t for t in train_split}
    row_fold = np.asarray(
        [assignment[group_stem(by_member[str(m)])] for m in cache["members"]], dtype=np.int16
    )
    outer_idx = np.flatnonzero(row_fold == args.outer_fold).astype(np.int64)
    outer_train_idx = np.flatnonzero(row_fold != args.outer_fold).astype(np.int64)
    k_all = np.minimum(np.asarray(cache["exact"], dtype=np.int32), SLOT_COUNT)

    inner_paths = sorted(args.inner_dir.glob(f"**/v104-nested-outer-{args.outer_fold}-inner-*.npz"))
    if len(inner_paths) != 4:
        raise RuntimeError(f"outer {args.outer_fold}: expected 4 inner shards, found {len(inner_paths)}")
    inner = _concat_inner(inner_paths)
    if not np.array_equal(np.asarray(inner["global_index"], dtype=np.int64), outer_train_idx):
        raise RuntimeError("inner OOF does not exactly cover outer-train rows")
    if not np.array_equal(np.asarray(inner["k"], dtype=np.int32), k_all[outer_train_idx]):
        raise RuntimeError("inner OOF labels mismatch")
    held = np.asarray(inner["held_fold"], dtype=np.int16)
    inner_ids = sorted(set(held.tolist()))
    expected_inner = sorted(set(range(oofmod.FOLD_COUNT)) - {args.outer_fold})
    if inner_ids != expected_inner:
        raise RuntimeError(f"inner fold IDs {inner_ids} != {expected_inner}")

    features = np.asarray(inner["features"], dtype=np.float32)
    anchor = np.asarray(inner["anchor"], dtype=np.float32)
    p102 = np.asarray(inner["p102"], dtype=np.float32)
    k = np.asarray(inner["k"], dtype=np.int32)

    # Outer-clean epoch selection: the lowest remaining inner fold is meta-val.
    meta_fold = inner_ids[0]
    meta_val = np.flatnonzero(held == meta_fold).astype(np.int64)
    meta_fit = np.flatnonzero(held != meta_fold).astype(np.int64)
    x_fit, probe_mean, probe_std = v104._standardize_fit(features[meta_fit])
    x_val = v104._standardize_apply(features[meta_val], probe_mean, probe_std)
    probe, _, _ = v104._build_fusion(features.shape[1])
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=8, min_delta=2e-4, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=5e-5
        ),
    ]
    hist = probe.fit(
        v104._inputs(x_fit, anchor[meta_fit], p102[meta_fit]),
        np.eye(SLOT_COUNT + 1, dtype=np.float32)[k[meta_fit]],
        sample_weight=v104._mild_count_weights(k[meta_fit]),
        validation_data=(
            v104._inputs(x_val, anchor[meta_val], p102[meta_val]),
            np.eye(SLOT_COUNT + 1, dtype=np.float32)[k[meta_val]],
            v104._mild_count_weights(k[meta_val]),
        ),
        epochs=v104.MAX_META_EPOCHS,
        batch_size=128,
        shuffle=True,
        callbacks=callbacks,
        verbose=0,
    )
    selected_epochs = max(2, int(np.argmin(np.asarray(hist.history["val_loss"])) + 1))

    x_all, mean, std = v104._standardize_fit(features)
    fusion, _, _ = v104._build_fusion(features.shape[1])
    fusion.fit(
        v104._inputs(x_all, anchor, p102),
        np.eye(SLOT_COUNT + 1, dtype=np.float32)[k],
        sample_weight=v104._mild_count_weights(k),
        epochs=selected_epochs,
        batch_size=128,
        shuffle=True,
        verbose=0,
    )

    deploy_paths = sorted(args.outer_dir.glob(f"**/v104-nested-outer-{args.outer_fold}-deploy.npz"))
    if len(deploy_paths) != 1:
        raise RuntimeError(f"expected one deployment shard, found {len(deploy_paths)}")
    deploy = _load_npz(deploy_paths[0])
    if not np.array_equal(np.asarray(deploy["global_index"], dtype=np.int64), outer_idx):
        raise RuntimeError("deployment shard does not exactly cover outer fold")

    x_deploy = v104._standardize_apply(np.asarray(deploy["features"], dtype=np.float32), mean, std)
    p_deploy104 = np.asarray(
        fusion.predict(v104._inputs(x_deploy, deploy["anchor"], deploy["p102"]), batch_size=256, verbose=0)
    )
    pred_deploy104 = np.argmax(p_deploy104, axis=1).astype(np.int32)

    probe_paths = sorted(args.inner_dir.glob(f"**/v104-nested-outer-{args.outer_fold}-probe-from-inner-*.npz"))
    if len(probe_paths) != 4:
        raise RuntimeError(f"expected 4 outer probe shards, found {len(probe_paths)}")
    probe_probs = []
    probe_details = {}
    tv101_all, tv102_all = [], []
    for path in probe_paths:
        pp = _load_npz(path)
        if not np.array_equal(np.asarray(pp["global_index"], dtype=np.int64), outer_idx):
            raise RuntimeError(f"probe coverage mismatch: {path}")
        x = v104._standardize_apply(np.asarray(pp["features"], dtype=np.float32), mean, std)
        p104 = np.asarray(
            fusion.predict(v104._inputs(x, pp["anchor"], pp["p102"]), batch_size=256, verbose=0)
        )
        pred104 = np.argmax(p104, axis=1).astype(np.int32)
        inner_source = int(np.asarray(pp["held_fold"]).reshape(-1)[0])
        tv101 = _tv(pp["p101"], deploy["p101"])
        tv102 = _tv(pp["p102"], deploy["p102"])
        tv101_all.append(tv101)
        tv102_all.append(tv102)
        probe_probs.append(p104)
        probe_details[str(inner_source)] = {
            "metrics": _metrics(cache, train_split, outer_idx, pred104),
            "cardinality": _card(k_all[outer_idx], pred104),
            "v101_to_deploy_mean_tv": float(np.mean(tv101)),
            "v102_to_deploy_mean_tv": float(np.mean(tv102)),
            "v104_to_deploy_decoded_agreement": float(np.mean(pred104 == pred_deploy104)),
        }

    p_probe_ensemble = np.mean(np.stack(probe_probs, axis=0), axis=0)
    pred_probe_ensemble = np.argmax(p_probe_ensemble, axis=1).astype(np.int32)
    ko = k_all[outer_idx]

    players = np.asarray([_player(cache["members"][i]) for i in outer_idx])
    by_player = {}
    for player in ALLOWED_PLAYERS:
        local = np.flatnonzero(players == player).astype(np.int64)
        if len(local) == 0:
            continue
        global_idx = outer_idx[local]
        by_player[player] = {
            "clusters": int(len(local)),
            "probe_ensemble_metrics": _metrics(cache, train_split, global_idx, pred_probe_ensemble[local]),
            "deployment_metrics": _metrics(cache, train_split, global_idx, pred_deploy104[local]),
            "probe_ensemble_cardinality": _card(ko[local], pred_probe_ensemble[local]),
            "deployment_cardinality": _card(ko[local], pred_deploy104[local]),
        }

    result = {
        "schema_version": 1,
        "protocol": {
            "train_only_nested_outer_holdout": True,
            "historical_validation_or_locked_evaluated": False,
            "outer_fold_seen_by_any_expert_training": False,
            "outer_fold_seen_by_fusion_training_or_epoch_selection": False,
            "fusion_epoch_selection_outer_clean": True,
            "outer_probe_experts_train_on_three_inner_folds": True,
            "deployment_experts_train_on_all_four_outer_train_folds": True,
        },
        "outer_fold": args.outer_fold,
        "outer_groups": groups_per_fold[args.outer_fold],
        "data": {
            "outer_clusters": int(len(outer_idx)),
            "outer_train_clusters": int(len(outer_train_idx)),
            "inner_oof_clusters": int(len(k)),
            "meta_validation_inner_fold": int(meta_fold),
            "selected_fusion_epochs": int(selected_epochs),
        },
        "outer_metrics": {
            "v101_deployment": _metrics(cache, train_split, outer_idx, deploy["pred101"]),
            "v102_deployment": _metrics(cache, train_split, outer_idx, deploy["pred102"]),
            "v104_probe_ensemble": _metrics(cache, train_split, outer_idx, pred_probe_ensemble),
            "v104_deployment": _metrics(cache, train_split, outer_idx, pred_deploy104),
        },
        "outer_cardinality": {
            "v101_deployment": _card(ko, deploy["pred101"]),
            "v102_deployment": _card(ko, deploy["pred102"]),
            "v104_probe_ensemble": _card(ko, pred_probe_ensemble),
            "v104_deployment": _card(ko, pred_deploy104),
        },
        "retraining_effect_same_outer_rows": {
            "probe_variants": probe_details,
            "mean_v101_probe_to_deploy_tv": float(np.mean(np.concatenate(tv101_all))),
            "mean_v102_probe_to_deploy_tv": float(np.mean(np.concatenate(tv102_all))),
            "probe_ensemble_vs_deploy_decoded_agreement": float(
                np.mean(pred_probe_ensemble == pred_deploy104)
            ),
            "probe_ensemble_minus_deploy_global_f1": float(
                _metrics(cache, train_split, outer_idx, pred_probe_ensemble)["global"]["f1"]
                - _metrics(cache, train_split, outer_idx, pred_deploy104)["global"]["f1"]
            ),
        },
        "by_player": by_player,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / f"v104-nested-eval-{args.outer_fold}.json"
    report_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(
        args.output_dir / f"v104-nested-eval-{args.outer_fold}.npz",
        schema_version=np.asarray([1], dtype=np.int16),
        outer_fold=np.full(len(outer_idx), args.outer_fold, dtype=np.int16),
        global_index=outer_idx,
        k=ko.astype(np.int16),
        member=np.asarray([str(cache["members"][i]) for i in outer_idx], dtype="U96"),
        pred101_deploy=np.asarray(deploy["pred101"], dtype=np.int16),
        pred102_deploy=np.asarray(deploy["pred102"], dtype=np.int16),
        pred104_probe_ensemble=pred_probe_ensemble.astype(np.int16),
        pred104_deploy=pred_deploy104.astype(np.int16),
    )
    print(json.dumps({
        "outer": args.outer_fold,
        "epochs": selected_epochs,
        "probe_f1": result["outer_metrics"]["v104_probe_ensemble"]["global"]["f1"],
        "deploy_f1": result["outer_metrics"]["v104_deployment"]["global"]["f1"],
        "delta_probe_minus_deploy": result["retraining_effect_same_outer_rows"]["probe_ensemble_minus_deploy_global_f1"],
    }, indent=2))
    return result


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("dataset_dir", nargs="?", type=Path, default=ROOT / "data" / "GuitarSet")
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--inner-dir", type=Path, required=True)
    p.add_argument("--outer-dir", type=Path, required=True)
    p.add_argument("--outer-fold", type=int, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=10521)
    return p


def main(argv: Optional[Sequence[str]] = None):
    evaluate(parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
