"""V9.1 full-train ordinal cardinality decoder.

V9.0 proved that moving the decision unit from individual candidates to a
40 ms candidate group is a large architectural win (57.57 -> 71.97 locked12
F1), but its 7-way softmax still under-counted real polyphonic groups and never
predicted 5 or 6 on locked12.

V9.1 keeps V8.4/V8.6/V8.7/V8.8 frozen and makes two deliberate changes:

1. Mine the entire leakage-safe GuitarSet train split, not a 30-track pilot.
2. Replace one imbalanced 0..6 softmax with a conditional ordinal chain:

       q1 = P(K >= 1)
       q2 = P(K >= 2 | K >= 1)
       ...
       q6 = P(K >= 6 | K >= 5)

   Cumulative P(K >= j) is prod(q1..qj).  A K=6 example therefore supervises
   all six stages instead of one extremely rare class.

The script has two subcommands. ``mine`` is designed for parallel GitHub Action
shards and writes compact float16 frozen cluster caches. ``train-eval`` merges
those caches, performs a composition-group-safe internal fit/holdout split,
calibrates a fixed decoder family on train-only holdout, then evaluates once on
frozen locked12.  Runtime still uses audio-derived frozen features only; labels
are supervision/metrics only. Offset is never executed or modified.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from causal_note.guitarset import ALLOWED_PLAYERS, SAMPLE_RATE, index_guitarset
from scripts.evaluate_v8_boundaries import DEFAULT_SEED, DEFAULT_VALIDATION_FRACTION, _arrangement
from scripts.train_boundaries import group_stem, split_tracks_by_group
from scripts.train_v86_state_transition_proposals import MAX_HORIZON, TOLERANCE_MS, _candidate_ceiling
from scripts.train_v87_causal_candidate_memory import _load_v86_encoder
from scripts.train_v88_regime_moe import _load_v87_encoder, _retained_predictions
from scripts.evaluate_v89_hard_regime_routing import _load_v88
from scripts.evaluate_v90_cluster_oracles import _metrics, _slots, _topk
from scripts.train_v90_structured_cluster_cardinality import (
    CLUSTER_WINDOW_MS,
    CLUSTER_WINDOW_SAMPLES,
    COUNT_CLASSES,
    MAX_CANDIDATES,
    MAX_EXACT_COUNT,
    SET_CANDIDATE_DIM,
    SET_STATS_DIM,
    _build_cluster_model,
    _cluster_data,
    _decode_cardinality as _decode_v90_cardinality,
    _extended_candidate_features,
    _model_inputs,
    _prediction_map,
    _represent_full,
)

DEFAULT_SEED_V91 = 9131
INTERNAL_HOLDOUT_FRACTION = 0.20
ORDINAL_STAGES = 6
DEFAULT_SHARD_COUNT = 8
CACHE_SCHEMA_VERSION = 1
DECODE_MODES = (
    "conditional_040",
    "conditional_045",
    "conditional_050",
    "conditional_055",
    "conditional_060",
    "expected_minus_025",
    "expected_round",
    "expected_plus_025",
    "cumulative_050",
)


class V91Error(RuntimeError):
    pass


def _load_frozen_stack(args):
    r86 = json.loads(args.v86_report.read_text())
    r87 = json.loads(args.v87_report.read_text())
    r88 = json.loads(args.v88_report.read_text())
    floor = float(r86["configuration"]["candidate_floor"])
    threshold = float(r88["configuration"]["retain_threshold"])
    if r86["configuration"].get("candidate_floor_selected_on_locked_validation") is not False:
        raise V91Error("V8.6 floor leakage")
    if r87["configuration"].get("retain_threshold_selected_on_locked_validation") is not False:
        raise V91Error("V8.7 threshold leakage")
    if r88["configuration"].get("retain_threshold_selected_on_locked_validation") is not False:
        raise V91Error("V8.8 threshold leakage")
    _, enc86 = _load_v86_encoder(args.v86_weights)
    _, enc87 = _load_v87_encoder(args.v87_weights)
    model88 = _load_v88(args.v88_weights)
    return floor, threshold, enc86, enc87, model88


def _dataset_split(dataset_dir: Path):
    indexed = tuple(t for t in index_guitarset(dataset_dir) if t.player_id in ALLOWED_PLAYERS)
    train, validation = split_tracks_by_group(
        indexed, validation_fraction=DEFAULT_VALIDATION_FRACTION, seed=DEFAULT_SEED
    )
    return indexed, train, validation


def _top_samples(clusters, records, fused_scores) -> np.ndarray:
    result = np.full((len(clusters), MAX_EXACT_COUNT), -1, dtype=np.int32)
    for cid, cluster in enumerate(clusters):
        ranked = _topk(_slots(cluster, records, fused_scores), MAX_EXACT_COUNT)
        for j, slot in enumerate(ranked):
            result[cid, j] = int(slot["sample"])
    return result


def _save_cache(path: Path, *, sequence, mask, stats, target, exact, truncated, members, top_samples, track_members):
    np.savez_compressed(
        path,
        schema_version=np.asarray([CACHE_SCHEMA_VERSION], dtype=np.int16),
        sequence=np.asarray(sequence, dtype=np.float16),
        mask=np.asarray(mask, dtype=np.uint8),
        stats=np.asarray(stats, dtype=np.float16),
        target=np.asarray(target, dtype=np.uint8),
        exact=np.asarray(exact, dtype=np.uint8),
        truncated=np.asarray(truncated, dtype=np.int16),
        members=np.asarray(members, dtype="U96"),
        top_samples=np.asarray(top_samples, dtype=np.int32),
        track_members=np.asarray(track_members, dtype="U96"),
    )


def mine(args) -> dict:
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    _, train_split, locked_validation = _dataset_split(args.dataset_dir)
    locked_members = {t.annotation_member for t in locked_validation}
    if not 0 <= args.shard_index < args.shard_count:
        raise V91Error("shard-index must be in [0, shard-count)")
    tracks = tuple(
        t for i, t in enumerate(sorted(train_split, key=lambda x: x.annotation_member))
        if i % args.shard_count == args.shard_index
    )
    if not tracks:
        raise V91Error("empty cache shard")
    if {t.annotation_member for t in tracks} & locked_members:
        raise V91Error("cache shard overlaps validation")

    floor, _, enc86, enc87, model88 = _load_frozen_stack(args)
    print(f"mining V9.1 shard {args.shard_index}/{args.shard_count} tracks={len(tracks)}")
    score_streams, records, x88, out88 = _represent_full(
        tracks, args.base_model, floor, enc86, enc87, model88
    )
    (
        clusters,
        fused,
        assignment,
        sequence,
        mask,
        stats,
        target,
        exact,
        truncated,
    ) = _cluster_data(tracks, records, x88, out88)
    members = [cluster["member"] for cluster in clusters]
    top_samples = _top_samples(clusters, records, fused)
    cache_path = args.output_dir / f"v91-cache-shard-{args.shard_index:02d}.npz"
    _save_cache(
        cache_path,
        sequence=sequence,
        mask=mask,
        stats=stats,
        target=target,
        exact=exact,
        truncated=truncated,
        members=members,
        top_samples=top_samples,
        track_members=[t.annotation_member for t in tracks],
    )
    report = {
        "schema_version": 1,
        "architecture": {
            "name": "V9.1 frozen full-train cluster cache shard",
            "runtime_inputs_use_annotations": False,
            "annotations_used_only_for_train_targets": True,
            "offset_stream_executed": False,
            "offset_weights_modified": False,
            "all_frozen_models_trainable": False,
        },
        "configuration": {
            "shard_index": args.shard_index,
            "shard_count": args.shard_count,
            "cluster_window_ms": CLUSTER_WINDOW_MS,
            "candidate_floor": floor,
        },
        "data": {
            "train_split_track_count": len(train_split),
            "validation_track_count": len(locked_validation),
            "shard_track_count": len(tracks),
            "shard_cluster_count": len(clusters),
            "track_members": [t.annotation_member for t in tracks],
            "reference_assignment": assignment,
            "truncated_cluster_count": int(np.sum(truncated > 0)),
            "truncated_candidate_count": int(np.sum(truncated)),
            "target_histogram": {str(k): int(np.sum(target == k)) for k in range(COUNT_CLASSES)},
            "exact_max": int(np.max(exact)) if len(exact) else 0,
        },
        "candidate_ceiling": _candidate_ceiling(tracks, score_streams, floor),
        "cache": {
            "path": cache_path.name,
            "sequence_shape": list(sequence.shape),
            "sequence_storage_dtype": "float16",
        },
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"shard": args.shard_index, "tracks": len(tracks), "clusters": len(clusters), "histogram": report["data"]["target_histogram"]}, indent=2))
    return report


def _load_caches(cache_dir: Path):
    paths = sorted(cache_dir.rglob("v91-cache-shard-*.npz"))
    if not paths:
        raise V91Error(f"no V9.1 cache shards under {cache_dir}")
    arrays = defaultdict(list)
    seen_tracks: List[str] = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            version = int(data["schema_version"][0])
            if version != CACHE_SCHEMA_VERSION:
                raise V91Error(f"unsupported cache schema {version} in {path}")
            for key in ("sequence", "mask", "stats", "target", "exact", "truncated", "members", "top_samples"):
                arrays[key].append(np.asarray(data[key]))
            seen_tracks.extend(str(x) for x in data["track_members"])
    merged = {
        "sequence": np.concatenate(arrays["sequence"], axis=0),
        "mask": np.concatenate(arrays["mask"], axis=0),
        "stats": np.concatenate(arrays["stats"], axis=0),
        "target": np.concatenate(arrays["target"], axis=0).astype(np.int32),
        "exact": np.concatenate(arrays["exact"], axis=0).astype(np.int32),
        "truncated": np.concatenate(arrays["truncated"], axis=0),
        "members": np.concatenate(arrays["members"], axis=0).astype(str),
        "top_samples": np.concatenate(arrays["top_samples"], axis=0).astype(np.int32),
        "track_members": tuple(sorted(seen_tracks)),
        "shard_paths": [str(p) for p in paths],
    }
    n = len(merged["target"])
    for key in ("sequence", "mask", "stats", "exact", "truncated", "members", "top_samples"):
        if len(merged[key]) != n:
            raise V91Error(f"cache row mismatch for {key}")
    return merged


def _split_train_groups(train_tracks, seed: int):
    grouped: Dict[str, List] = defaultdict(list)
    for track in train_tracks:
        grouped[group_stem(track)].append(track)
    names = sorted(grouped)
    if len(names) < 4:
        raise V91Error("not enough composition groups for internal holdout")
    rng = random.Random(seed)
    rng.shuffle(names)
    n_holdout = max(1, round(len(names) * INTERNAL_HOLDOUT_FRACTION))
    holdout_groups = frozenset(names[:n_holdout])
    fit, holdout = [], []
    for name, tracks in grouped.items():
        (holdout if name in holdout_groups else fit).extend(tracks)
    return tuple(sorted(fit, key=lambda t: t.annotation_member)), tuple(sorted(holdout, key=lambda t: t.annotation_member)), tuple(sorted(holdout_groups))


def _build_ordinal_model():
    try:
        from tensorflow import keras
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required") from exc
    scaffold = _build_cluster_model()
    hidden = scaffold.get_layer("cluster_hidden2").output
    outputs = {
        f"ge{stage}": keras.layers.Dense(1, activation="sigmoid", name=f"ge{stage}")(hidden)
        for stage in range(1, ORDINAL_STAGES + 1)
    }
    model = keras.Model(scaffold.inputs, outputs, name="v91_conditional_ordinal_cardinality")
    # Later conditional stages have fewer eligible examples.  A moderate loss
    # boost keeps them visible without letting very rare K=6 samples dominate.
    loss_weights = {f"ge{stage}": float(math.sqrt(stage)) for stage in range(1, ORDINAL_STAGES + 1)}
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=3e-4),
        loss={name: "binary_crossentropy" for name in outputs},
        loss_weights=loss_weights,
    )
    return model, loss_weights


def _ordinal_targets(k: np.ndarray):
    return {f"ge{stage}": (k >= stage).astype(np.float32).reshape(-1, 1) for stage in range(1, ORDINAL_STAGES + 1)}


def _conditional_sample_weights(k: np.ndarray):
    n = len(k)
    result = {}
    diagnostics = {}
    for stage in range(1, ORDINAL_STAGES + 1):
        eligible = k >= (stage - 1)
        positive = k >= stage
        negative = eligible & ~positive
        n_pos = int(np.sum(positive))
        n_neg = int(np.sum(negative))
        n_eligible = n_pos + n_neg
        if n_pos == 0 or n_neg == 0:
            raise V91Error(f"ordinal stage {stage} lacks positive/negative fit examples: pos={n_pos} neg={n_neg}")
        w = np.zeros(n, dtype=np.float32)
        # Balanced within the eligible conditional population.  Normalize the
        # eligible mean to one; stage-level loss_weights provide the moderate
        # rarity compensation rather than huge per-example weights.
        w[positive] = n_eligible / (2.0 * n_pos)
        w[negative] = n_eligible / (2.0 * n_neg)
        result[f"ge{stage}"] = w
        diagnostics[str(stage)] = {
            "eligible": n_eligible,
            "positive": n_pos,
            "negative": n_neg,
            "positive_fraction_given_eligible": n_pos / n_eligible,
            "positive_weight": float(n_eligible / (2.0 * n_pos)),
            "negative_weight": float(n_eligible / (2.0 * n_neg)),
        }
    return result, diagnostics


def _predict_conditionals(model, inputs) -> np.ndarray:
    raw = model.predict(inputs, batch_size=256, verbose=0)
    return np.stack([np.asarray(raw[f"ge{stage}"]).reshape(-1) for stage in range(1, ORDINAL_STAGES + 1)], axis=1).astype(np.float64)


def _cumulative(conditionals: np.ndarray) -> np.ndarray:
    return np.cumprod(np.clip(conditionals, 0.0, 1.0), axis=1)


def _decode_ordinal(conditionals: np.ndarray, mode: str) -> np.ndarray:
    q = np.asarray(conditionals, dtype=np.float64)
    cumulative = _cumulative(q)
    if mode.startswith("conditional_"):
        threshold = int(mode.rsplit("_", 1)[1]) / 100.0
        result = np.zeros(len(q), dtype=np.int32)
        alive = np.ones(len(q), dtype=bool)
        for stage in range(ORDINAL_STAGES):
            accept = alive & (q[:, stage] >= threshold)
            result[accept] = stage + 1
            alive &= q[:, stage] >= threshold
        return result
    if mode == "cumulative_050":
        return np.sum(cumulative >= 0.5, axis=1).astype(np.int32)
    expected = np.sum(cumulative, axis=1)
    bias = {
        "expected_minus_025": -0.25,
        "expected_round": 0.0,
        "expected_plus_025": 0.25,
    }.get(mode)
    if bias is None:
        raise ValueError(f"unknown ordinal decode mode {mode}")
    return np.clip(np.floor(expected + 0.5 + bias), 0, MAX_EXACT_COUNT).astype(np.int32)


def _cached_prediction_map(members, top_samples, k_values, allowed_members: Optional[set] = None):
    retained: Dict[str, List[int]] = defaultdict(list)
    for member, samples, k in zip(members, top_samples, k_values):
        member = str(member)
        if allowed_members is not None and member not in allowed_members:
            continue
        for sample in samples[: int(k)]:
            if int(sample) >= 0:
                retained[member].append(int(sample))
    return {member: tuple(sorted(values)) for member, values in retained.items()}


def _ordinal_report(target: np.ndarray, pred: np.ndarray, conditionals: np.ndarray) -> dict:
    confusion = np.zeros((COUNT_CLASSES, COUNT_CLASSES), dtype=np.int64)
    for y, p in zip(target, pred):
        confusion[int(y), int(p)] += 1
    mask_birth = target > 0
    mask_poly = target >= 2
    per_stage = {}
    for stage in range(1, ORDINAL_STAGES + 1):
        eligible = target >= stage - 1
        truth = target[eligible] >= stage
        p = conditionals[eligible, stage - 1]
        binary = p >= 0.5
        per_stage[str(stage)] = {
            "eligible": int(np.sum(eligible)),
            "positive": int(np.sum(truth)),
            "conditional_accuracy_at_0_5": float(np.mean(binary == truth)) if len(truth) else None,
            "mean_probability_positive": float(np.mean(p[truth])) if np.any(truth) else None,
            "mean_probability_negative": float(np.mean(p[~truth])) if np.any(~truth) else None,
        }
    return {
        "accuracy": float(np.mean(pred == target)) if len(target) else 0.0,
        "birth_cluster_accuracy": float(np.mean(pred[mask_birth] == target[mask_birth])) if np.any(mask_birth) else None,
        "poly_cluster_accuracy": float(np.mean(pred[mask_poly] == target[mask_poly])) if np.any(mask_poly) else None,
        "mean_absolute_class_error": float(np.mean(np.abs(pred.astype(np.int64) - target.astype(np.int64)))) if len(target) else 0.0,
        "predicted_histogram": {str(k): int(np.sum(pred == k)) for k in range(COUNT_CLASSES)},
        "target_histogram": {str(k): int(np.sum(target == k)) for k in range(COUNT_CLASSES)},
        "confusion_true_rows_pred_columns": confusion.tolist(),
        "conditional_stages": per_stage,
    }


def _calibrate_modes(conditionals, target, members, top_samples, holdout_tracks, holdout_set):
    rows = []
    for mode in DECODE_MODES:
        pred = _decode_ordinal(conditionals, mode)
        predictions = _cached_prediction_map(members, top_samples, pred, holdout_set)
        metrics = _metrics(holdout_tracks, predictions)
        macro = 0.5 * (metrics["solo"]["f1"] + metrics["comp"]["f1"])
        rows.append({
            "mode": mode,
            "macro_f1": float(macro),
            "metrics": metrics,
            "ordinal": _ordinal_report(target, pred, conditionals),
        })
    best = max(rows, key=lambda r: (r["macro_f1"], r["metrics"]["global"]["f1"], r["ordinal"]["accuracy"]))
    return {
        "selection_rule": "fixed ordinal decode mode maximizing composition-safe train-holdout macro(solo,comp) F1",
        "best": best,
        "sweep": rows,
    }


def train_eval(args) -> dict:
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    try:
        import tensorflow as tf
        from tensorflow import keras
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required") from exc
    tf.random.set_seed(args.seed)

    indexed, train_split, locked_validation = _dataset_split(args.dataset_dir)
    locked12 = tuple(locked_validation[:12])
    train_members = {t.annotation_member for t in train_split}
    locked_members = {t.annotation_member for t in locked_validation}
    if train_members & locked_members:
        raise V91Error("train/validation leakage")

    cache = _load_caches(args.cache_dir)
    cached_track_members = set(cache["track_members"])
    if cached_track_members != train_members:
        missing = sorted(train_members - cached_track_members)
        extra = sorted(cached_track_members - train_members)
        raise V91Error(f"cache does not exactly cover train split: missing={len(missing)} extra={len(extra)}")
    if len(cache["track_members"]) != len(cached_track_members):
        raise V91Error("duplicate tracks across cache shards")

    fit_tracks, holdout_tracks, holdout_groups = _split_train_groups(train_split, args.seed)
    fit_set = {t.annotation_member for t in fit_tracks}
    holdout_set = {t.annotation_member for t in holdout_tracks}
    if fit_set & holdout_set or (fit_set | holdout_set) != train_members:
        raise V91Error("internal composition split is invalid")
    fit_idx = np.asarray([i for i, m in enumerate(cache["members"]) if m in fit_set], dtype=np.int64)
    holdout_idx = np.asarray([i for i, m in enumerate(cache["members"]) if m in holdout_set], dtype=np.int64)
    if not len(fit_idx) or not len(holdout_idx):
        raise V91Error("empty fit/holdout cache split")

    sequence = cache["sequence"]
    mask = cache["mask"]
    stats = cache["stats"]
    target = cache["target"]
    y_fit = target[fit_idx]
    y_holdout = target[holdout_idx]

    model, loss_weights = _build_ordinal_model()
    fit_weights, weight_diagnostics = _conditional_sample_weights(y_fit)
    holdout_weights, holdout_weight_diagnostics = _conditional_sample_weights(y_holdout)
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=2e-5),
    ]
    print(f"training V9.1 full-train ordinal clusters fit={len(fit_idx)} holdout={len(holdout_idx)} tracks={len(train_split)}")
    history = model.fit(
        _model_inputs(sequence, mask, stats, fit_idx),
        _ordinal_targets(y_fit),
        sample_weight=fit_weights,
        validation_data=(
            _model_inputs(sequence, mask, stats, holdout_idx),
            _ordinal_targets(y_holdout),
            holdout_weights,
        ),
        epochs=args.epochs,
        batch_size=192,
        shuffle=True,
        callbacks=callbacks,
        verbose=2,
    )

    holdout_conditional = _predict_conditionals(model, _model_inputs(sequence, mask, stats, holdout_idx))
    calibration = _calibrate_modes(
        holdout_conditional,
        y_holdout,
        cache["members"][holdout_idx],
        cache["top_samples"][holdout_idx],
        holdout_tracks,
        holdout_set,
    )
    decode_mode = calibration["best"]["mode"]
    model.save_weights(args.output_dir / "v91-ordinal-cardinality.weights.h5")

    floor, v88_threshold, enc86, enc87, model88 = _load_frozen_stack(args)
    print("evaluating V9.1 once on frozen locked12")
    locked_score_streams, locked_records, locked_x88, locked_out88 = _represent_full(
        locked12, args.base_model, floor, enc86, enc87, model88
    )
    (
        locked_clusters,
        locked_fused,
        locked_assignment,
        locked_sequence,
        locked_mask,
        locked_stats,
        locked_target,
        locked_exact,
        locked_truncated,
    ) = _cluster_data(locked12, locked_records, locked_x88, locked_out88)
    locked_conditional = _predict_conditionals(
        model, _model_inputs(locked_sequence, locked_mask, locked_stats)
    )
    locked_k = _decode_ordinal(locked_conditional, decode_mode)
    locked_predictions = _prediction_map(locked_clusters, locked_records, locked_fused, locked_k)
    locked_metrics = _metrics(locked12, locked_predictions)
    v88_baseline = _metrics(
        locked12, _retained_predictions(locked_records, locked_fused, v88_threshold)
    )
    oracle_k = np.minimum(MAX_EXACT_COUNT, locked_exact).astype(np.int32)
    locked_oracle = _metrics(
        locked12, _prediction_map(locked_clusters, locked_records, locked_fused, oracle_k)
    )
    locked_ceiling = _candidate_ceiling(locked12, locked_score_streams, floor)

    total_max_delay_samples = MAX_HORIZON + CLUSTER_WINDOW_SAMPLES
    report = {
        "schema_version": 1,
        "architecture": {
            "name": "V9.1 full-train conditional ordinal exact-cardinality decoder",
            "frozen_base": "V8.4 + V8.6 + V8.7 + V8.8",
            "base_trainable": False,
            "runtime_inputs_use_annotations": False,
            "ordinal_outputs": [f"P(K>={stage}|K>={stage-1})" for stage in range(1, ORDINAL_STAGES + 1)],
            "ordinal_monotonic_cumulative_probability": True,
            "cluster_window_ms": CLUSTER_WINDOW_MS,
            "candidate_set_cap": MAX_CANDIDATES,
            "candidate_feature_dim": SET_CANDIDATE_DIM,
            "cluster_stats_dim": SET_STATS_DIM,
            "cardinality_classes": ["0", "1", "2", "3", "4", "5", "6"],
            "candidate_selection_after_k": "top-K frozen V8.8 fused-score slots",
            "offset_stream_executed": False,
            "offset_weights_modified": False,
            "future_candidate_context": True,
            "future_candidate_context_ms": CLUSTER_WINDOW_MS,
            "maximum_verification_delay_samples": total_max_delay_samples,
            "maximum_verification_delay_ms": total_max_delay_samples * 1000.0 / SAMPLE_RATE,
            "v91_trainable_parameters": int(model.count_params()),
        },
        "configuration": {
            "seed": args.seed,
            "epochs_requested": args.epochs,
            "epochs_ran": len(history.history["loss"]),
            "decode_mode": decode_mode,
            "decode_mode_selected_on_locked_validation": False,
            "decode_candidates_fixed_before_locked_eval": list(DECODE_MODES),
            "internal_holdout_fraction_by_composition": INTERNAL_HOLDOUT_FRACTION,
            "matching_tolerance_ms": TOLERANCE_MS,
            "candidate_floor": floor,
            "frozen_v88_threshold": v88_threshold,
            "loss_weights": loss_weights,
        },
        "data": {
            "indexed_track_count": len(indexed),
            "full_train_track_count": len(train_split),
            "full_validation_track_count": len(locked_validation),
            "cache_shard_count": len(cache["shard_paths"]),
            "cached_cluster_count": len(target),
            "fit_track_count": len(fit_tracks),
            "holdout_track_count": len(holdout_tracks),
            "fit_cluster_count": len(fit_idx),
            "holdout_cluster_count": len(holdout_idx),
            "internal_holdout_composition_groups": list(holdout_groups),
            "composition_group_leakage": False,
            "locked_validation_members": [t.annotation_member for t in locked12],
            "locked_reference_assignment": locked_assignment,
            "cache_target_histogram": {str(k): int(np.sum(target == k)) for k in range(COUNT_CLASSES)},
            "fit_target_histogram": {str(k): int(np.sum(y_fit == k)) for k in range(COUNT_CLASSES)},
            "holdout_target_histogram": {str(k): int(np.sum(y_holdout == k)) for k in range(COUNT_CLASSES)},
            "fit_ordinal_weight_diagnostics": weight_diagnostics,
            "holdout_ordinal_weight_diagnostics": holdout_weight_diagnostics,
            "cached_truncated_cluster_count": int(np.sum(cache["truncated"] > 0)),
            "locked_truncated_cluster_count": int(np.sum(locked_truncated > 0)),
        },
        "training_history": {key: [float(v) for v in values] for key, values in history.history.items()},
        "holdout_decode_calibration": calibration,
        "locked12": {
            "candidate_ceiling": locked_ceiling,
            "frozen_v88_baseline_metrics": v88_baseline,
            "v91_metrics": locked_metrics,
            "v91_ordinal": _ordinal_report(locked_target, locked_k, locked_conditional),
            "oracle_exact_cardinality_metrics": locked_oracle,
        },
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "decode_mode": decode_mode,
        "full_train_tracks": len(train_split),
        "cached_clusters": len(target),
        "parameters": int(model.count_params()),
        "locked_ordinal": report["locked12"]["v91_ordinal"],
        "v88": v88_baseline,
        "v91": locked_metrics,
        "oracle": locked_oracle,
    }, indent=2, sort_keys=True))
    return report


def _add_frozen_args(p):
    p.add_argument("dataset_dir", nargs="?", type=Path, default=ROOT / "data" / "GuitarSet")
    p.add_argument("--base-model", type=Path, required=True)
    p.add_argument("--v86-weights", type=Path, required=True)
    p.add_argument("--v86-report", type=Path, required=True)
    p.add_argument("--v87-weights", type=Path, required=True)
    p.add_argument("--v87-report", type=Path, required=True)
    p.add_argument("--v88-weights", type=Path, required=True)
    p.add_argument("--v88-report", type=Path, required=True)


def create_argument_parser():
    p = argparse.ArgumentParser(description="V9.1 full-train ordinal cluster cardinality")
    sub = p.add_subparsers(dest="command", required=True)
    mine_p = sub.add_parser("mine", help="mine one frozen full-train cache shard")
    _add_frozen_args(mine_p)
    mine_p.add_argument("--shard-index", type=int, required=True)
    mine_p.add_argument("--shard-count", type=int, default=DEFAULT_SHARD_COUNT)
    mine_p.add_argument("--output-dir", type=Path, required=True)

    train_p = sub.add_parser("train-eval", help="train ordinal decoder from merged caches and evaluate locked12")
    _add_frozen_args(train_p)
    train_p.add_argument("--cache-dir", type=Path, required=True)
    train_p.add_argument("--output-dir", type=Path, required=True)
    train_p.add_argument("--epochs", type=int, default=30)
    train_p.add_argument("--seed", type=int, default=DEFAULT_SEED_V91)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = create_argument_parser().parse_args(argv)
    if args.command == "mine":
        mine(args)
    elif args.command == "train-eval":
        train_eval(args)
    else:
        raise AssertionError(args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
