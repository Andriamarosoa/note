"""V10.3 confidence-gated residual soft fusion of frozen V10.1 and V10.2.

V10.1 is the conservative expert (high precision, strong K0/K1). V10.2 is the
source-time expert (higher recall and much better K3/K4). V10.3 does not retrain
or modify either expert. It learns one scalar soft gate alpha and mixes their
count distributions:

    P_final = (1-alpha) * P_V10.1_anchor + alpha * P_V10.2

The gate is trained only on composition groups that belong to BOTH historical
train-only holdouts, so neither frozen expert was trained on gate-fit examples.
Locked12 is evaluated exactly once after the gate and scaler are frozen.
"""
from __future__ import annotations

import argparse
import json
import math
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

from causal_note.guitarset import SLOT_COUNT, SAMPLE_RATE
from scripts.train_boundaries import group_stem
from scripts.train_v90_structured_cluster_cardinality import _cluster_data, _prediction_map, _represent_full
from scripts.train_v91_ordinal_cardinality import _dataset_split, _load_frozen_stack
from scripts.train_v100_spectral_string_slots import _cached_prediction_map, _load_spectral_caches, _spectral_maps_for_runtime
from scripts import train_v101_string_query_attention as v101
from scripts import train_v102_source_time_assignment as v102
from scripts import run_v102_competitive_mass as v102_mass

DEFAULT_SEED_V103 = 10331
ANCHOR_ONEHOT_WEIGHT = 0.75
GATE_L2 = 2e-3
GATE_EPOCHS = 80
MIN_COMMON_GROUPS = 3
EPS = 1e-8


class V103Error(RuntimeError):
    pass


def _exact_from_ordinal(conditionals: np.ndarray) -> np.ndarray:
    cumulative = np.asarray(v101._cumulative(conditionals), dtype=np.float64)
    out = np.zeros((len(cumulative), SLOT_COUNT + 1), dtype=np.float64)
    out[:, 0] = 1.0 - cumulative[:, 0]
    for k in range(1, SLOT_COUNT):
        out[:, k] = cumulative[:, k - 1] - cumulative[:, k]
    out[:, SLOT_COUNT] = cumulative[:, SLOT_COUNT - 1]
    out = np.clip(out, EPS, None)
    return out / np.sum(out, axis=1, keepdims=True)


def _anchor_distribution(p101: np.ndarray, pred101: np.ndarray) -> np.ndarray:
    onehot = np.eye(SLOT_COUNT + 1, dtype=np.float64)[np.asarray(pred101, dtype=np.int32)]
    out = ANCHOR_ONEHOT_WEIGHT * onehot + (1.0 - ANCHOR_ONEHOT_WEIGHT) * np.asarray(p101, dtype=np.float64)
    return out / np.sum(out, axis=1, keepdims=True)


def _entropy(p: np.ndarray, axis: int = -1) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1.0)
    return -np.sum(p * np.log(p), axis=axis)


def _margin(p: np.ndarray) -> np.ndarray:
    s = np.sort(np.asarray(p, dtype=np.float64), axis=1)
    return s[:, -1] - s[:, -2]


def _source_assignment_features(model, inputs) -> np.ndarray:
    from tensorflow import keras

    probe = keras.Model(inputs=model.inputs, outputs=model.get_layer("source_assignment_grid").output)
    a = np.asarray(probe.predict(inputs, batch_size=128, verbose=0), dtype=np.float64)
    mass = np.mean(a, axis=(1, 2))
    source_entropy = np.mean(_entropy(a, axis=-1), axis=(1, 2))[:, None]
    background = mass[:, SLOT_COUNT:SLOT_COUNT + 1]
    foreground = np.sum(mass[:, :SLOT_COUNT], axis=1, keepdims=True)
    return np.concatenate([mass, source_entropy, background, foreground], axis=1)


def _feature_matrix(
    p101: np.ndarray,
    p102: np.ndarray,
    slots101: np.ndarray,
    slots102: np.ndarray,
    times102: np.ndarray,
    assignment_features: np.ndarray,
    stats: np.ndarray,
) -> np.ndarray:
    classes = np.arange(SLOT_COUNT + 1, dtype=np.float64)
    e101 = p101 @ classes
    e102 = p102 @ classes
    time_prob = np.clip(np.asarray(times102, dtype=np.float64), EPS, 1.0)
    time_ms = np.sum(time_prob * v102.FRAME_CENTER_MS[None, None, :], axis=2)
    time_ent = _entropy(time_prob, axis=2) / math.log(v102.TIME_FRAMES)
    sw = np.clip(np.asarray(slots102, dtype=np.float64), 0.0, 1.0)
    sw_sum = np.sum(sw, axis=1) + EPS
    mean_t = np.sum(sw * time_ms, axis=1) / sw_sum
    spread_t = np.sqrt(np.sum(sw * (time_ms - mean_t[:, None]) ** 2, axis=1) / sw_sum)
    strong35 = np.sum(sw >= 0.35, axis=1).astype(np.float64)
    strong50 = np.sum(sw >= 0.50, axis=1).astype(np.float64)
    max101 = np.max(p101, axis=1)
    max102 = np.max(p102, axis=1)

    scalar = np.stack(
        [
            _entropy(p101), _entropy(p102), _margin(p101), _margin(p102),
            max101, max102, e101, e102, e102 - e101,
            np.sum(slots101, axis=1), np.sum(slots102, axis=1),
            spread_t, strong35, strong50,
        ],
        axis=1,
    )
    return np.concatenate(
        [
            p101, p102, p102 - p101,
            slots101, slots102, slots102 - slots101,
            time_ms / 40.0, time_ent,
            assignment_features,
            scalar,
            np.asarray(stats, dtype=np.float64),
        ],
        axis=1,
    ).astype(np.float32)


def _standardize_fit(x: np.ndarray):
    mean = np.mean(x, axis=0).astype(np.float32)
    std = np.std(x, axis=0).astype(np.float32)
    std[std < 1e-5] = 1.0
    return ((x - mean) / std).astype(np.float32), mean, std


def _standardize_apply(x: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return ((x - mean) / std).astype(np.float32)


def _build_gate(feature_dim: int):
    import tensorflow as tf
    from tensorflow import keras

    feat = keras.Input((feature_dim,), name="fusion_features")
    p1 = keras.Input((SLOT_COUNT + 1,), name="v101_anchor")
    p2 = keras.Input((SLOT_COUNT + 1,), name="v102_count")
    x = keras.layers.Dense(
        24, activation="relu", kernel_regularizer=keras.regularizers.l2(GATE_L2), name="gate_hidden1"
    )(feat)
    x = keras.layers.Dense(
        8, activation="relu", kernel_regularizer=keras.regularizers.l2(GATE_L2), name="gate_hidden2"
    )(x)
    alpha = keras.layers.Dense(1, activation="sigmoid", name="gate_alpha")(x)
    final = keras.layers.Lambda(
        lambda z: (1.0 - z[0]) * z[1] + z[0] * z[2], name="final_count"
    )([alpha, p1, p2])
    model = keras.Model(
        {"fusion_features": feat, "v101_anchor": p1, "v102_count": p2}, final,
        name="v103_residual_soft_fusion",
    )
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="categorical_crossentropy")
    alpha_model = keras.Model(model.inputs, alpha)
    return model, alpha_model


def _gate_inputs(features, anchor, p102):
    return {
        "fusion_features": np.asarray(features, dtype=np.float32),
        "v101_anchor": np.asarray(anchor, dtype=np.float32),
        "v102_count": np.asarray(p102, dtype=np.float32),
    }


def _disagreement_audit(k, p1, p2, alpha=None):
    k = np.asarray(k, dtype=np.int32)
    p1 = np.asarray(p1, dtype=np.int32)
    p2 = np.asarray(p2, dtype=np.int32)
    c1, c2 = p1 == k, p2 == k
    groups = {
        "A_both_correct": c1 & c2,
        "B_v101_only": c1 & ~c2,
        "C_v102_only": ~c1 & c2,
        "D_both_wrong": ~c1 & ~c2,
    }
    out: Dict[str, dict] = {}
    for name, m in groups.items():
        pm = m & (k >= 2)
        item = {
            "clusters": int(np.sum(m)),
            "fraction": float(np.mean(m)),
            "poly_clusters": int(np.sum(pm)),
        }
        if alpha is not None and np.any(m):
            item["mean_gate_alpha"] = float(np.mean(np.asarray(alpha)[m]))
            item["median_gate_alpha"] = float(np.median(np.asarray(alpha)[m]))
        out[name] = item
    return out


def _predict_experts(model101, model102, cache, indices=None):
    inputs = v102._inputs(cache, indices)
    s101, _, c101 = v101._predict(model101, v101._inputs(cache, indices))
    s102, _, t102, p102, _ = v102._predict(model102, inputs)
    return inputs, s101, c101, s102, t102, p102


def _prepare_fusion_inputs(model101, model102, cache, indices, mode101, mode102):
    inputs, s101, c101, s102, t102, p102 = _predict_experts(model101, model102, cache, indices)
    p101 = _exact_from_ordinal(c101)
    pred101 = v101._decode(s101, c101, mode101)
    pred102 = v102._decode(s102, p102, mode102)
    anchor = _anchor_distribution(p101, pred101)
    af = _source_assignment_features(model102, inputs)
    stats = cache["stats"] if indices is None else cache["stats"][indices]
    feat = _feature_matrix(p101, p102, s101, s102, t102, af, stats)
    return {
        "features": feat,
        "p101": p101,
        "anchor": anchor,
        "p102": p102,
        "pred101": pred101,
        "pred102": pred102,
        "slots101": s101,
        "slots102": s102,
        "times102": t102,
    }


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
    by_member = {t.annotation_member: t for t in indexed}
    k = np.minimum(np.asarray(cache["exact"], dtype=np.int32), SLOT_COUNT)

    report101 = json.loads(args.v101_report.read_text())
    report102 = json.loads(args.v102_report.read_text())
    mode101 = str(report101["configuration"]["decode_mode"])
    mode102 = str(report102["configuration"]["decode_mode"])
    h101 = set(report101["data"]["holdout_composition_groups"])
    h102 = set(report102["data"]["holdout_composition_groups"])
    common_groups = sorted(h101 & h102)
    if len(common_groups) < MIN_COMMON_GROUPS:
        raise V103Error(
            f"clean common holdout too small: {len(common_groups)} groups; refusing leaked meta-fit"
        )
    common_set = set(common_groups)
    common_idx = np.asarray([
        i for i, m in enumerate(cache["members"])
        if group_stem(by_member[str(m)]) in common_set
    ], dtype=np.int64)
    common_tracks = tuple(t for t in train_split if group_stem(t) in common_set)
    if len(common_idx) < 500:
        raise V103Error(f"clean gate-fit cluster count too small: {len(common_idx)}")

    model101, _, _ = v101._build_model()
    model101.load_weights(args.v101_weights)
    model102, _, _ = v102_mass._build_model_mass_aware()
    model102.load_weights(args.v102_weights)

    print(f"V10.3 clean meta-fit groups={len(common_groups)} tracks={len(common_tracks)} clusters={len(common_idx)}")
    cal = _prepare_fusion_inputs(model101, model102, cache, common_idx, mode101, mode102)
    audit_before = _disagreement_audit(k[common_idx], cal["pred101"], cal["pred102"])
    print("disagreement_before", json.dumps(audit_before, sort_keys=True))

    xz, mean, std = _standardize_fit(cal["features"])
    gate, alpha_model = _build_gate(xz.shape[1])
    target = np.eye(SLOT_COUNT + 1, dtype=np.float32)[k[common_idx]]
    weight = v102._count_weights(k[common_idx])
    callbacks = [keras.callbacks.EarlyStopping(monitor="loss", patience=10, min_delta=1e-4, restore_best_weights=True)]
    history = gate.fit(
        _gate_inputs(xz, cal["anchor"], cal["p102"]), target,
        sample_weight=weight,
        epochs=GATE_EPOCHS,
        batch_size=64,
        shuffle=True,
        callbacks=callbacks,
        verbose=2,
    )
    final_cal = np.asarray(gate.predict(_gate_inputs(xz, cal["anchor"], cal["p102"]), batch_size=128, verbose=0))
    alpha_cal = np.asarray(alpha_model.predict(_gate_inputs(xz, cal["anchor"], cal["p102"]), batch_size=128, verbose=0)).reshape(-1)
    pred_cal = np.argmax(final_cal, axis=1).astype(np.int32)
    audit_after = _disagreement_audit(k[common_idx], cal["pred101"], cal["pred102"], alpha_cal)

    cal_metrics101 = v101._metrics(common_tracks, _cached_prediction_map(cache, common_idx, cal["pred101"]))
    cal_metrics102 = v101._metrics(common_tracks, _cached_prediction_map(cache, common_idx, cal["pred102"]))
    cal_metrics103 = v101._metrics(common_tracks, _cached_prediction_map(cache, common_idx, pred_cal))

    gate.save_weights(args.output_dir / "v103-residual-soft-fusion.weights.h5")
    np.savez_compressed(args.output_dir / "v103-feature-scaler.npz", mean=mean, std=std)

    # Locked12 is touched only now, after all meta-model parameters are frozen.
    locked12 = tuple(validation[:12])
    floor, _, enc86, enc87, model88 = _load_frozen_stack(args)
    print("evaluating V10.3 once on historical locked12")
    score_streams, records, x88, out88 = _represent_full(locked12, args.base_model, floor, enc86, enc87, model88)
    clusters, fused, assignment, sequence, mask, stats, target_locked, exact, truncated = _cluster_data(locked12, records, x88, out88)
    spectral = _spectral_maps_for_runtime(locked12, clusters, records)
    locked_cache = {"sequence": sequence, "mask": mask, "stats": stats, "spectral": spectral}
    locked = _prepare_fusion_inputs(model101, model102, locked_cache, None, mode101, mode102)
    locked_x = _standardize_apply(locked["features"], mean, std)
    locked_gate_inputs = _gate_inputs(locked_x, locked["anchor"], locked["p102"])
    final_locked = np.asarray(gate.predict(locked_gate_inputs, batch_size=128, verbose=0))
    alpha_locked = np.asarray(alpha_model.predict(locked_gate_inputs, batch_size=128, verbose=0)).reshape(-1)
    pred103 = np.argmax(final_locked, axis=1).astype(np.int32)
    oracle_k = np.minimum(np.asarray(exact, dtype=np.int32), SLOT_COUNT)

    metrics101 = v101._metrics(locked12, _prediction_map(clusters, records, fused, locked["pred101"]))
    metrics102 = v101._metrics(locked12, _prediction_map(clusters, records, fused, locked["pred102"]))
    metrics103 = v101._metrics(locked12, _prediction_map(clusters, records, fused, pred103))
    oracle = v101._metrics(locked12, _prediction_map(clusters, records, fused, oracle_k))

    report = {
        "schema_version": 1,
        "architecture": {
            "name": "V10.3 confidence-gated residual soft fusion",
            "frozen_experts": ["V10.1 string-query ordinal", "V10.2 competitive source-time"],
            "experts_trainable": False,
            "gate_trainable_parameters": int(gate.count_params()),
            "fusion": "(1-alpha)*V10.1_anchor + alpha*V10.2_structured_count",
            "v101_anchor_onehot_weight": ANCHOR_ONEHOT_WEIGHT,
            "hard_routing": False,
            "runtime_inputs_use_annotations": False,
            "offset_stream_executed": False,
            "offset_weights_modified": False,
            "cluster_window_ms": v102.CLUSTER_WINDOW_MS,
            "maximum_verification_delay_ms": (v102.MAX_HORIZON + v102.CLUSTER_WINDOW_SAMPLES) * 1000.0 / SAMPLE_RATE,
        },
        "configuration": {
            "seed": args.seed,
            "v101_decode_mode": mode101,
            "v102_decode_mode": mode102,
            "v103_decode_mode": "argmax_soft_fused_distribution",
            "decode_mode_selected_on_locked_validation": False,
            "gate_epochs_ran": len(history.history["loss"]),
            "gate_l2": GATE_L2,
            "gate_fit_uses_locked12": False,
        },
        "data": {
            "v101_holdout_group_count": len(h101),
            "v102_holdout_group_count": len(h102),
            "clean_common_holdout_groups": common_groups,
            "clean_common_holdout_group_count": len(common_groups),
            "gate_fit_track_count": len(common_tracks),
            "gate_fit_cluster_count": len(common_idx),
            "gate_fit_poly_cluster_count": int(np.sum(k[common_idx] >= 2)),
            "locked_cluster_count": len(oracle_k),
            "locked_truncated_cluster_count": int(np.sum(truncated > 0)),
        },
        "gate_fit": {
            "disagreement_before": audit_before,
            "disagreement_with_gate_alpha": audit_after,
            "v101_metrics": cal_metrics101,
            "v102_metrics": cal_metrics102,
            "v103_metrics": cal_metrics103,
            "v101_cardinality": v102._cardinality_report(k[common_idx], cal["pred101"]),
            "v102_cardinality": v102._cardinality_report(k[common_idx], cal["pred102"]),
            "v103_cardinality": v102._cardinality_report(k[common_idx], pred_cal),
            "v103_poly_by_k": v102._poly_by_k(k[common_idx], pred_cal),
            "alpha_mean": float(np.mean(alpha_cal)),
            "alpha_median": float(np.median(alpha_cal)),
            "alpha_p90": float(np.percentile(alpha_cal, 90)),
            "training_history": {key: [float(x) for x in vals] for key, vals in history.history.items()},
        },
        "locked12": {
            "v101_metrics": metrics101,
            "v102_metrics": metrics102,
            "v103_metrics": metrics103,
            "oracle_exact_cardinality_metrics": oracle,
            "v101_cardinality": v102._cardinality_report(oracle_k, locked["pred101"]),
            "v102_cardinality": v102._cardinality_report(oracle_k, locked["pred102"]),
            "v103_cardinality": v102._cardinality_report(oracle_k, pred103),
            "v103_poly_by_k": v102._poly_by_k(oracle_k, pred103),
            "disagreement": _disagreement_audit(oracle_k, locked["pred101"], locked["pred102"], alpha_locked),
            "alpha_mean": float(np.mean(alpha_locked)),
            "alpha_median": float(np.median(alpha_locked)),
            "alpha_p90": float(np.percentile(alpha_locked, 90)),
        },
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "common_groups": len(common_groups),
        "gate_fit_clusters": len(common_idx),
        "gate_params": int(gate.count_params()),
        "gate_epochs": len(history.history["loss"]),
        "fit_audit": audit_before,
        "locked_v101": metrics101,
        "locked_v102": metrics102,
        "locked_v103": metrics103,
        "locked_poly": report["locked12"]["v103_poly_by_k"],
        "locked_cardinality": report["locked12"]["v103_cardinality"],
        "locked_alpha": {
            "mean": report["locked12"]["alpha_mean"],
            "median": report["locked12"]["alpha_median"],
            "p90": report["locked12"]["alpha_p90"],
        },
    }, indent=2, sort_keys=True))
    return report


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("dataset_dir", nargs="?", type=Path, default=ROOT / "data" / "GuitarSet")
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--v101-weights", type=Path, required=True)
    p.add_argument("--v101-report", type=Path, required=True)
    p.add_argument("--v102-weights", type=Path, required=True)
    p.add_argument("--v102-report", type=Path, required=True)
    p.add_argument("--base-model", type=Path, required=True)
    p.add_argument("--v86-weights", type=Path, required=True)
    p.add_argument("--v86-report", type=Path, required=True)
    p.add_argument("--v87-weights", type=Path, required=True)
    p.add_argument("--v87-report", type=Path, required=True)
    p.add_argument("--v88-weights", type=Path, required=True)
    p.add_argument("--v88-report", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED_V103)
    return p


def main(argv: Optional[Sequence[str]] = None):
    args = parser().parse_args(argv)
    train_eval(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
