"""V10.4 class-conditional residual fusion trained from strict OOF experts.

Five OOF shards cover the complete GuitarSet train split.  Every shard was
predicted by fresh V10.1/V10.2 experts that were trained without that shard's
composition groups.  V10.4 learns seven independent class gates in log-count
space plus a small bounded residual:

    logits_k = log(P101_anchor_k)
             + alpha_k(x) * (log(P102_k) - log(P101_anchor_k))
             + residual_scale * tanh(delta_k(x))

The scalar V10.3 gate could not keep V10.2's K4 gain without importing its K0/K1
false positives.  Per-class gates explicitly remove that coupling.  Fold 0 is a
meta-validation fold used only to select the number of fusion epochs; the final
fusion is then retrained on all strict OOF predictions.  Locked12 is evaluated
once after architecture, scaler, epochs and weights are frozen.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for _p in (ROOT, SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from causal_note.guitarset import SAMPLE_RATE, SLOT_COUNT
from scripts.train_boundaries import group_stem
from scripts.train_v90_structured_cluster_cardinality import _cluster_data, _prediction_map, _represent_full
from scripts.train_v91_ordinal_cardinality import _dataset_split, _load_frozen_stack
from scripts.train_v100_spectral_string_slots import _cached_prediction_map, _load_spectral_caches, _spectral_maps_for_runtime
from scripts import train_v101_string_query_attention as v101
from scripts import train_v102_source_time_assignment as v102
from scripts import run_v102_competitive_mass as v102_mass
from scripts import train_v103_residual_soft_fusion as v103
from scripts.train_v104_oof_fold import FOLD_COUNT

DEFAULT_SEED_V104 = 10431
META_VALIDATION_FOLD = 0
FUSION_L2 = 4e-3
RESIDUAL_SCALE = 0.30
MAX_META_EPOCHS = 80
EPS = 1e-7


class V104FusionError(RuntimeError):
    pass


def _load_oof(oof_dir: Path, expected_count: int):
    paths = sorted(oof_dir.glob("v104-oof-fold-*.npz"))
    if len(paths) != FOLD_COUNT:
        raise V104FusionError(f"expected {FOLD_COUNT} OOF shards, found {len(paths)}")
    parts = []
    for path in paths:
        with np.load(path, allow_pickle=False) as z:
            schema = int(np.asarray(z["schema_version"]).reshape(-1)[0])
            if schema != 1:
                raise V104FusionError(f"unsupported OOF schema {schema}: {path}")
            parts.append({key: np.asarray(z[key]) for key in z.files if key != "schema_version"})
    keys = set(parts[0])
    if any(set(p) != keys for p in parts[1:]):
        raise V104FusionError("OOF shard key mismatch")
    merged = {key: np.concatenate([p[key] for p in parts], axis=0) for key in keys}
    order = np.argsort(merged["global_index"], kind="stable")
    merged = {key: value[order] for key, value in merged.items()}
    idx = np.asarray(merged["global_index"], dtype=np.int64)
    if len(idx) != expected_count:
        raise V104FusionError(f"OOF coverage {len(idx)} != cache {expected_count}")
    if not np.array_equal(idx, np.arange(expected_count, dtype=np.int64)):
        raise V104FusionError("OOF global indices are not exact one-time full coverage")
    fold = np.asarray(merged["fold"], dtype=np.int16)
    if set(np.unique(fold).tolist()) != set(range(FOLD_COUNT)):
        raise V104FusionError(f"OOF fold IDs invalid: {np.unique(fold).tolist()}")

    # A composition group must belong to exactly one held fold.
    group_folds: Dict[str, set] = {}
    for g, f in zip(merged["group"], fold):
        group_folds.setdefault(str(g), set()).add(int(f))
    bad = {g: sorted(v) for g, v in group_folds.items() if len(v) != 1}
    if bad:
        raise V104FusionError(f"composition group spans OOF folds: {bad}")

    for name in ("p101", "anchor", "p102"):
        p = np.asarray(merged[name], dtype=np.float64)
        if p.shape != (expected_count, SLOT_COUNT + 1):
            raise V104FusionError(f"{name} shape {p.shape}")
        if np.max(np.abs(np.sum(p, axis=1) - 1.0)) > 2e-4:
            raise V104FusionError(f"{name} is not normalized")
    return merged, paths, group_folds


def _standardize_fit(x: np.ndarray):
    mean = np.mean(x, axis=0).astype(np.float32)
    std = np.std(x, axis=0).astype(np.float32)
    std[std < 1e-5] = 1.0
    return ((x - mean) / std).astype(np.float32), mean, std


def _standardize_apply(x, mean, std):
    return ((np.asarray(x, dtype=np.float32) - mean) / std).astype(np.float32)


def _build_fusion(feature_dim: int):
    import tensorflow as tf
    from tensorflow import keras

    feat = keras.Input((feature_dim,), name="fusion_features")
    anchor = keras.Input((SLOT_COUNT + 1,), name="v101_anchor")
    p2 = keras.Input((SLOT_COUNT + 1,), name="v102_count")

    x = keras.layers.Dense(
        48,
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(FUSION_L2),
        name="fusion_hidden1",
    )(feat)
    x = keras.layers.Dropout(0.08, name="fusion_dropout")(x)
    x = keras.layers.Dense(
        24,
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(FUSION_L2),
        name="fusion_hidden2",
    )(x)
    alpha = keras.layers.Dense(SLOT_COUNT + 1, activation="sigmoid", name="class_alpha")(x)
    residual = keras.layers.Dense(SLOT_COUNT + 1, activation="tanh", name="class_residual")(x)

    final = keras.layers.Lambda(
        lambda z: tf.nn.softmax(
            tf.math.log(tf.clip_by_value(z[1], EPS, 1.0))
            + z[0] * (
                tf.math.log(tf.clip_by_value(z[2], EPS, 1.0))
                - tf.math.log(tf.clip_by_value(z[1], EPS, 1.0))
            )
            + RESIDUAL_SCALE * z[3],
            axis=-1,
        ),
        name="final_count",
    )([alpha, anchor, p2, residual])

    model = keras.Model(
        {"fusion_features": feat, "v101_anchor": anchor, "v102_count": p2},
        final,
        name="v104_class_conditional_residual_fusion",
    )
    model.compile(optimizer=keras.optimizers.Adam(7e-4), loss="categorical_crossentropy")
    alpha_model = keras.Model(model.inputs, alpha)
    residual_model = keras.Model(model.inputs, residual)
    return model, alpha_model, residual_model


def _inputs(features, anchor, p102):
    return {
        "fusion_features": np.asarray(features, dtype=np.float32),
        "v101_anchor": np.asarray(anchor, dtype=np.float32),
        "v102_count": np.asarray(p102, dtype=np.float32),
    }


def _mild_count_weights(k: np.ndarray):
    base = np.asarray(v102._count_weights(np.asarray(k, dtype=np.int32)), dtype=np.float64)
    out = 0.65 + 0.35 * base
    out /= np.mean(out)
    return np.clip(out, 0.65, 2.25).astype(np.float32)


def _alpha_report(alpha: np.ndarray):
    a = np.asarray(alpha, dtype=np.float64)
    return {
        str(k): {
            "mean": float(np.mean(a[:, k])),
            "median": float(np.median(a[:, k])),
            "p90": float(np.percentile(a[:, k], 90)),
        }
        for k in range(SLOT_COUNT + 1)
    }


def _metrics_for_indices(cache, train_split, indices, pred):
    member_set = {str(cache["members"][i]) for i in indices}
    tracks = tuple(t for t in train_split if t.annotation_member in member_set)
    return v101._metrics(tracks, _cached_prediction_map(cache, indices, pred))


def train_eval(args):
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    import tensorflow as tf
    from tensorflow import keras
    tf.random.set_seed(args.seed)

    cache = _load_spectral_caches(args.cache_dir)
    indexed, train_split, validation = _dataset_split(args.dataset_dir)
    if set(cache["track_members"]) != {t.annotation_member for t in train_split}:
        raise V104FusionError("cache/train split mismatch")
    if {t.annotation_member for t in train_split} & {t.annotation_member for t in validation}:
        raise V104FusionError("train/validation leakage")
    k_cache = np.minimum(np.asarray(cache["exact"], dtype=np.int32), SLOT_COUNT)

    oof, shard_paths, group_folds = _load_oof(args.oof_dir, len(k_cache))
    k = np.asarray(oof["k"], dtype=np.int32)
    if not np.array_equal(k, k_cache):
        raise V104FusionError("OOF cardinality labels do not match frozen cache")
    fold = np.asarray(oof["fold"], dtype=np.int16)
    features = np.asarray(oof["features"], dtype=np.float32)
    anchor = np.asarray(oof["anchor"], dtype=np.float32)
    p102 = np.asarray(oof["p102"], dtype=np.float32)
    pred101 = np.asarray(oof["pred101"], dtype=np.int32)
    pred102 = np.asarray(oof["pred102"], dtype=np.int32)

    meta_val_idx = np.flatnonzero(fold == META_VALIDATION_FOLD).astype(np.int64)
    meta_train_idx = np.flatnonzero(fold != META_VALIDATION_FOLD).astype(np.int64)
    if len(meta_val_idx) < 1000 or len(meta_train_idx) < 10000:
        raise V104FusionError("meta train/validation split unexpectedly small")

    x_train, probe_mean, probe_std = _standardize_fit(features[meta_train_idx])
    x_val = _standardize_apply(features[meta_val_idx], probe_mean, probe_std)
    probe, probe_alpha, _ = _build_fusion(features.shape[1])
    y_train = np.eye(SLOT_COUNT + 1, dtype=np.float32)[k[meta_train_idx]]
    y_val = np.eye(SLOT_COUNT + 1, dtype=np.float32)[k[meta_val_idx]]
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=8, min_delta=2e-4, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=5e-5
        ),
    ]
    print(
        f"V10.4 meta calibration train={len(meta_train_idx)} val={len(meta_val_idx)} "
        f"feature_dim={features.shape[1]} groups={len(group_folds)}"
    )
    history = probe.fit(
        _inputs(x_train, anchor[meta_train_idx], p102[meta_train_idx]),
        y_train,
        sample_weight=_mild_count_weights(k[meta_train_idx]),
        validation_data=(
            _inputs(x_val, anchor[meta_val_idx], p102[meta_val_idx]),
            y_val,
            _mild_count_weights(k[meta_val_idx]),
        ),
        epochs=MAX_META_EPOCHS,
        batch_size=128,
        shuffle=True,
        callbacks=callbacks,
        verbose=2,
    )
    val_loss = np.asarray(history.history["val_loss"], dtype=np.float64)
    selected_epochs = int(np.argmin(val_loss) + 1)
    if selected_epochs < 2:
        selected_epochs = 2

    probe_val = np.asarray(
        probe.predict(_inputs(x_val, anchor[meta_val_idx], p102[meta_val_idx]), batch_size=256, verbose=0)
    )
    probe_alpha_val = np.asarray(
        probe_alpha.predict(_inputs(x_val, anchor[meta_val_idx], p102[meta_val_idx]), batch_size=256, verbose=0)
    )
    pred_probe = np.argmax(probe_val, axis=1).astype(np.int32)
    meta_metrics101 = _metrics_for_indices(cache, train_split, meta_val_idx, pred101[meta_val_idx])
    meta_metrics102 = _metrics_for_indices(cache, train_split, meta_val_idx, pred102[meta_val_idx])
    meta_metrics104 = _metrics_for_indices(cache, train_split, meta_val_idx, pred_probe)

    # Refit scaler and fusion on the complete strict OOF dataset for the selected
    # train-only epoch count.  Locked12 still has not been touched.
    x_all, mean, std = _standardize_fit(features)
    final_model, final_alpha_model, final_residual_model = _build_fusion(features.shape[1])
    final_history = final_model.fit(
        _inputs(x_all, anchor, p102),
        np.eye(SLOT_COUNT + 1, dtype=np.float32)[k],
        sample_weight=_mild_count_weights(k),
        epochs=selected_epochs,
        batch_size=128,
        shuffle=True,
        verbose=2,
    )
    final_model.save_weights(args.output_dir / "v104-class-conditional-fusion.weights.h5")
    np.savez_compressed(args.output_dir / "v104-feature-scaler.npz", mean=mean, std=std)

    # Load historical full-train experts only after OOF fusion training is frozen.
    report101 = json.loads(args.v101_report.read_text())
    report102 = json.loads(args.v102_report.read_text())
    report103 = json.loads(args.v103_report.read_text())
    mode101 = str(report101["configuration"]["decode_mode"])
    mode102 = str(report102["configuration"]["decode_mode"])
    model101, _, _ = v101._build_model()
    model101.load_weights(args.v101_weights)
    model102, _, _ = v102_mass._build_model_mass_aware()
    model102.load_weights(args.v102_weights)

    # Historical locked12: exactly one evaluation after all V10.4 choices freeze.
    locked12 = tuple(validation[:12])
    floor, _, enc86, enc87, model88 = _load_frozen_stack(args)
    print("evaluating V10.4 once on historical locked12")
    score_streams, records, x88, out88 = _represent_full(
        locked12, args.base_model, floor, enc86, enc87, model88
    )
    clusters, fused, assignment, sequence, mask, stats, target, exact, truncated = _cluster_data(
        locked12, records, x88, out88
    )
    spectral = _spectral_maps_for_runtime(locked12, clusters, records)
    locked_cache = {"sequence": sequence, "mask": mask, "stats": stats, "spectral": spectral}
    locked = v103._prepare_fusion_inputs(model101, model102, locked_cache, None, mode101, mode102)
    locked_x = _standardize_apply(locked["features"], mean, std)
    locked_in = _inputs(locked_x, locked["anchor"], locked["p102"])
    p104 = np.asarray(final_model.predict(locked_in, batch_size=256, verbose=0))
    alpha_locked = np.asarray(final_alpha_model.predict(locked_in, batch_size=256, verbose=0))
    residual_locked = np.asarray(final_residual_model.predict(locked_in, batch_size=256, verbose=0))
    pred104 = np.argmax(p104, axis=1).astype(np.int32)
    oracle_k = np.minimum(np.asarray(exact, dtype=np.int32), SLOT_COUNT)

    metrics101 = v101._metrics(locked12, _prediction_map(clusters, records, fused, locked["pred101"]))
    metrics102 = v101._metrics(locked12, _prediction_map(clusters, records, fused, locked["pred102"]))
    metrics104 = v101._metrics(locked12, _prediction_map(clusters, records, fused, pred104))
    oracle = v101._metrics(locked12, _prediction_map(clusters, records, fused, oracle_k))
    historical103 = report103["locked12"]["v103_metrics"]

    report = {
        "schema_version": 1,
        "architecture": {
            "name": "V10.4 strict-OOF class-conditional residual fusion",
            "oof_fold_count": FOLD_COUNT,
            "class_gate_count": SLOT_COUNT + 1,
            "fusion_trainable_parameters": int(final_model.count_params()),
            "fusion": "log(P101_anchor)+alpha_k*(log(P102)-log(P101_anchor))+bounded_residual_k",
            "residual_scale": RESIDUAL_SCALE,
            "experts_trainable_during_fusion": False,
            "hard_routing": False,
            "runtime_inputs_use_annotations": False,
            "offset_stream_executed": False,
            "offset_weights_modified": False,
            "cluster_window_ms": v102.CLUSTER_WINDOW_MS,
            "maximum_verification_delay_ms": (
                (v102.MAX_HORIZON + v102.CLUSTER_WINDOW_SAMPLES) * 1000.0 / SAMPLE_RATE
            ),
        },
        "configuration": {
            "seed": args.seed,
            "meta_validation_fold": META_VALIDATION_FOLD,
            "selected_epochs": selected_epochs,
            "selected_epochs_from_locked12": False,
            "decode_mode": "argmax_class_conditional_distribution",
            "decode_mode_selected_on_locked_validation": False,
            "fusion_l2": FUSION_L2,
            "mild_count_weighting": True,
            "locked12_used_for_training_or_calibration": False,
            "v101_decode_mode": mode101,
            "v102_decode_mode": mode102,
        },
        "data": {
            "cached_train_cluster_count": len(k_cache),
            "oof_cluster_count": len(k),
            "oof_full_exact_coverage": True,
            "oof_shards": [p.name for p in shard_paths],
            "oof_group_count": len(group_folds),
            "meta_train_cluster_count": len(meta_train_idx),
            "meta_validation_cluster_count": len(meta_val_idx),
            "locked_cluster_count": len(oracle_k),
            "locked_truncated_cluster_count": int(np.sum(truncated > 0)),
        },
        "oof_meta_validation": {
            "v101_metrics": meta_metrics101,
            "v102_metrics": meta_metrics102,
            "v104_metrics": meta_metrics104,
            "v101_cardinality": v102._cardinality_report(k[meta_val_idx], pred101[meta_val_idx]),
            "v102_cardinality": v102._cardinality_report(k[meta_val_idx], pred102[meta_val_idx]),
            "v104_cardinality": v102._cardinality_report(k[meta_val_idx], pred_probe),
            "v104_poly_by_k": v102._poly_by_k(k[meta_val_idx], pred_probe),
            "class_alpha": _alpha_report(probe_alpha_val),
            "epochs_tested": len(history.history["loss"]),
            "selected_epochs": selected_epochs,
            "history": {key: [float(v) for v in values] for key, values in history.history.items()},
        },
        "full_oof_audit": {
            "disagreement": v103._disagreement_audit(k, pred101, pred102),
            "v101_cardinality": v102._cardinality_report(k, pred101),
            "v102_cardinality": v102._cardinality_report(k, pred102),
        },
        "locked12": {
            "v101_metrics": metrics101,
            "v102_metrics": metrics102,
            "v103_metrics": historical103,
            "v104_metrics": metrics104,
            "oracle_exact_cardinality_metrics": oracle,
            "v101_cardinality": v102._cardinality_report(oracle_k, locked["pred101"]),
            "v102_cardinality": v102._cardinality_report(oracle_k, locked["pred102"]),
            "v104_cardinality": v102._cardinality_report(oracle_k, pred104),
            "v104_poly_by_k": v102._poly_by_k(oracle_k, pred104),
            "class_alpha": _alpha_report(alpha_locked),
            "residual_abs_mean_per_class": {
                str(i): float(np.mean(np.abs(residual_locked[:, i]))) for i in range(SLOT_COUNT + 1)
            },
        },
        "final_training_history": {
            key: [float(v) for v in values] for key, values in final_history.history.items()
        },
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "oof_clusters": len(k),
        "oof_groups": len(group_folds),
        "meta_selected_epochs": selected_epochs,
        "meta_v101": meta_metrics101,
        "meta_v102": meta_metrics102,
        "meta_v104": meta_metrics104,
        "locked_v101": metrics101,
        "locked_v102": metrics102,
        "locked_v103": historical103,
        "locked_v104": metrics104,
        "locked_cardinality": report["locked12"]["v104_cardinality"],
        "locked_poly": report["locked12"]["v104_poly_by_k"],
        "locked_alpha": report["locked12"]["class_alpha"],
    }, indent=2, sort_keys=True))
    return report


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("dataset_dir", nargs="?", type=Path, default=ROOT / "data" / "GuitarSet")
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--oof-dir", type=Path, required=True)
    p.add_argument("--v101-weights", type=Path, required=True)
    p.add_argument("--v101-report", type=Path, required=True)
    p.add_argument("--v102-weights", type=Path, required=True)
    p.add_argument("--v102-report", type=Path, required=True)
    p.add_argument("--v103-report", type=Path, required=True)
    p.add_argument("--base-model", type=Path, required=True)
    p.add_argument("--v86-weights", type=Path, required=True)
    p.add_argument("--v86-report", type=Path, required=True)
    p.add_argument("--v87-weights", type=Path, required=True)
    p.add_argument("--v87-report", type=Path, required=True)
    p.add_argument("--v88-weights", type=Path, required=True)
    p.add_argument("--v88-report", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED_V104)
    return p


def main(argv: Optional[Sequence[str]] = None):
    args = parser().parse_args(argv)
    train_eval(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
