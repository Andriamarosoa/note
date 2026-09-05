"""V9.0 structured exact-cardinality decoder over frozen V8.8 candidates.

V8.9 rejected hard expert routing. A train-only oracle study then showed that,
inside a <=40 ms candidate group and under the locked 50 ms onset tolerance,
correct local cardinality matters far more than combinatorial candidate subset
selection. It also showed a large loss from collapsing all counts >=3 into 3+.

V9.0 therefore changes the decision unit from one candidate to one causal
candidate group and predicts an exact guitar-scale cardinality 0..6 (class 6
also absorbs extremely rare >6 target groups). Candidate ranking inside a group
remains the frozen V8.8 fused score. The V8.4/V8.6/V8.7/V8.8 stack is frozen.

Runtime path:
  frozen candidates over 40 ms -> learned set encoder -> K in 0..6
  -> retain top-K frozen V8.8 candidate slots

The 40 ms grouping is runtime-only and adds at most 40 ms to the existing V8.6
1024-sample acoustic horizon. Annotations are used only for fit/holdout targets
and final locked metrics. Offset is never executed or modified.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from causal_note.guitarset import ALLOWED_PLAYERS, SAMPLE_RATE, index_guitarset
from scripts.evaluate_v8_boundaries import DEFAULT_SEED, DEFAULT_VALIDATION_FRACTION
from scripts.train_boundaries import split_tracks_by_group
from scripts.train_v85_transition_verifier import _split_members
from scripts.train_v86_state_transition_proposals import (
    MAX_HORIZON,
    TOLERANCE_MS,
    _candidate_ceiling,
    _predict_score_tracks,
    _records,
)
from scripts.train_v87_causal_candidate_memory import (
    _encode_records,
    _load_v86_encoder,
    _sequence_arrays,
)
from scripts.train_v88_regime_moe import (
    FEATURE_DIM as V88_FEATURE_DIM,
    _aggregate,
    _encode_v87,
    _feature_matrix,
    _fused_scores,
    _load_v87_encoder,
    _predictions,
    _retained_predictions,
)
from scripts.evaluate_v89_hard_regime_routing import _load_v88
from scripts.evaluate_v90_cluster_oracles import _assign_refs, _metrics, _references, _slots, _topk
from scripts.evaluate_v90_cardinality_realization import _clusters_window

DEFAULT_SEED_V90 = 9031
DEFAULT_TRAIN_MEMBERS = 30
CLUSTER_WINDOW_MS = 40.0
CLUSTER_WINDOW_SAMPLES = round(CLUSTER_WINDOW_MS * SAMPLE_RATE / 1000.0)
LOCAL_ASSIGNMENT_RADIUS_MS = 20.0
COUNT_CLASSES = 7  # 0,1,2,3,4,5,6 where 6 absorbs 6+
MAX_EXACT_COUNT = COUNT_CLASSES - 1
MAX_CANDIDATES = 48
# V8.8 feature vector + router + 4 local-cardinality probs + two experts + fused
FROZEN_CANDIDATE_DIM = V88_FEATURE_DIM + 8
# plus start-relative and center-relative candidate time
SET_CANDIDATE_DIM = FROZEN_CANDIDATE_DIM + 2
SET_STATS_DIM = 8
CALIBRATION_MODES = (
    "argmax",
    "expected_round",
    "expected_minus_025",
    "expected_plus_025",
)


class V90Error(RuntimeError):
    pass


def _represent_full(tracks, base_model: Path, floor: float, v86_encoder, v87_encoder, v88_model):
    score_by_member = _predict_score_tracks(base_model, tracks)
    records = _records(tracks, score_by_member, floor)
    emb86, prob86 = _encode_records(v86_encoder, records)
    sequences, _ = _sequence_arrays(records, emb86, prob86)
    hidden87, prob87 = _encode_v87(v87_encoder, sequences)
    x88 = _feature_matrix(records, emb86, prob86, hidden87, prob87)
    out88 = _predictions(v88_model, x88)
    return score_by_member, records, x88, out88


def _extended_candidate_features(x88: np.ndarray, out88: dict) -> Tuple[np.ndarray, np.ndarray]:
    router = np.asarray(out88["cluster_router"], dtype=np.float32).reshape(-1, 1)
    card = np.asarray(out88["local_cardinality"], dtype=np.float32)
    isolated = np.asarray(out88["isolated_birth"], dtype=np.float32).reshape(-1, 1)
    cluster = np.asarray(out88["cluster_birth"], dtype=np.float32).reshape(-1, 1)
    fused = np.asarray(out88["fused_birth"], dtype=np.float32).reshape(-1, 1)
    features = np.concatenate((x88, router, card, isolated, cluster, fused), axis=1).astype(np.float32)
    if features.shape[1] != FROZEN_CANDIDATE_DIM:
        raise V90Error(f"unexpected frozen candidate feature dim {features.shape}")
    return features, fused.reshape(-1)


def _cluster_arrays(clusters, assigned, records, candidate_features, out88, fused_scores):
    n = len(clusters)
    sequence = np.zeros((n, MAX_CANDIDATES, SET_CANDIDATE_DIM), dtype=np.float32)
    mask = np.zeros((n, MAX_CANDIDATES), dtype=np.float32)
    stats = np.zeros((n, SET_STATS_DIM), dtype=np.float32)
    target = np.zeros((n,), dtype=np.int32)
    exact = np.zeros((n,), dtype=np.int32)
    truncated = np.zeros((n,), dtype=np.int32)
    card = np.asarray(out88["local_cardinality"], dtype=np.float32)
    router = np.asarray(out88["cluster_router"], dtype=np.float32).reshape(-1)

    for cid, cluster in enumerate(clusters):
        indices = list(cluster["indices"])
        total_candidates = len(indices)
        if total_candidates > MAX_CANDIDATES:
            ranked = sorted(indices, key=lambda i: (-float(fused_scores[i]), int(records[i]["sample"]), i))[:MAX_CANDIDATES]
            indices = sorted(ranked, key=lambda i: (int(records[i]["sample"]), i))
            truncated[cid] = total_candidates - MAX_CANDIDATES
        samples = [int(records[i]["sample"]) for i in cluster["indices"]]
        start = min(samples)
        end = max(samples)
        center = 0.5 * (start + end)
        width = max(1, end - start)
        for slot, i in enumerate(indices):
            sample = int(records[i]["sample"])
            rel_start = np.clip((sample - start) / max(1.0, float(CLUSTER_WINDOW_SAMPLES)), 0.0, 1.0)
            rel_center = np.clip((sample - center) / max(1.0, float(CLUSTER_WINDOW_SAMPLES)), -0.5, 0.5)
            sequence[cid, slot] = np.concatenate(
                (candidate_features[i], np.asarray([rel_start, rel_center], dtype=np.float32))
            )
            mask[cid, slot] = 1.0

        cluster_indices = np.asarray(cluster["indices"], dtype=np.int64)
        f = np.asarray(fused_scores[cluster_indices], dtype=np.float64)
        r = np.asarray(router[cluster_indices], dtype=np.float64)
        cp = np.asarray(card[cluster_indices], dtype=np.float64)
        expected_local = cp @ np.arange(4, dtype=np.float64)
        weights = np.maximum(f, 1e-6)
        weighted_local = float(np.sum(expected_local * weights) / np.sum(weights))
        stats[cid] = np.asarray(
            [
                np.clip(math.log1p(total_candidates) / math.log1p(64.0), 0.0, 1.5),
                np.clip(width / max(1.0, float(CLUSTER_WINDOW_SAMPLES)), 0.0, 1.0),
                float(np.mean(f)),
                float(np.max(f)),
                float(np.mean(r)),
                float(np.mean(expected_local) / 3.0),
                float(weighted_local / 3.0),
                float(np.max(1.0 - cp[:, 0])),
            ],
            dtype=np.float32,
        )
        exact_count = len(assigned.get(cid, ()))
        exact[cid] = int(exact_count)
        target[cid] = int(min(MAX_EXACT_COUNT, exact_count))
    return sequence, mask, stats, target, exact, truncated


def _build_cluster_model():
    try:
        import tensorflow as tf
        from tensorflow import keras
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required for V9.0") from exc

    sequence = keras.Input((MAX_CANDIDATES, SET_CANDIDATE_DIM), name="candidate_set")
    mask = keras.Input((MAX_CANDIDATES,), name="candidate_mask")
    stats = keras.Input((SET_STATS_DIM,), name="cluster_stats")

    x = keras.layers.TimeDistributed(keras.layers.LayerNormalization(), name="candidate_norm")(sequence)
    x = keras.layers.TimeDistributed(keras.layers.Dense(96, activation="relu"), name="candidate_hidden1")(x)
    x = keras.layers.TimeDistributed(keras.layers.Dense(64, activation="relu"), name="candidate_hidden2")(x)

    mask_e = keras.layers.Lambda(lambda m: tf.expand_dims(m, axis=-1), name="mask_expand")(mask)
    masked = keras.layers.Multiply(name="masked_candidate_hidden")([x, mask_e])
    summed = keras.layers.Lambda(lambda t: tf.reduce_sum(t, axis=1), name="sum_pool")(masked)
    count = keras.layers.Lambda(
        lambda m: tf.maximum(tf.reduce_sum(m, axis=1, keepdims=True), 1.0), name="candidate_count"
    )(mask)
    mean = keras.layers.Lambda(lambda values: values[0] / values[1], name="mean_pool")([summed, count])
    scaled_sum = keras.layers.Lambda(
        lambda values: values[0] / tf.sqrt(values[1]), name="scaled_sum_pool"
    )([summed, count])

    masked_for_max = keras.layers.Lambda(
        lambda values: tf.where(values[1] > 0.0, values[0], tf.cast(-1e9, values[0].dtype)),
        name="max_mask",
    )([x, mask_e])
    maximum = keras.layers.Lambda(lambda t: tf.reduce_max(t, axis=1), name="max_pool")(masked_for_max)

    attention_logits = keras.layers.TimeDistributed(keras.layers.Dense(1), name="attention_logits")(x)
    masked_logits = keras.layers.Lambda(
        lambda values: tf.where(values[1] > 0.0, values[0], tf.cast(-1e9, values[0].dtype)),
        name="attention_mask",
    )([attention_logits, mask_e])
    attention = keras.layers.Softmax(axis=1, name="candidate_attention")(masked_logits)
    attended = keras.layers.Multiply(name="attention_weighted")([x, attention])
    attended = keras.layers.Lambda(lambda t: tf.reduce_sum(t, axis=1), name="attention_pool")(attended)

    pooled = keras.layers.Concatenate(name="collective_cluster_representation")(
        [mean, maximum, attended, scaled_sum, stats]
    )
    pooled = keras.layers.LayerNormalization(name="cluster_norm")(pooled)
    hidden = keras.layers.Dense(192, activation="relu", name="cluster_hidden1")(pooled)
    hidden = keras.layers.Dropout(0.15, name="cluster_dropout")(hidden)
    hidden = keras.layers.Dense(96, activation="relu", name="cluster_hidden2")(hidden)

    cardinality = keras.layers.Dense(COUNT_CLASSES, activation="softmax", name="exact_cardinality")(hidden)
    birth_present = keras.layers.Dense(1, activation="sigmoid", name="birth_present")(hidden)
    poly_present = keras.layers.Dense(1, activation="sigmoid", name="poly_present")(hidden)
    model = keras.Model(
        {"candidate_set": sequence, "candidate_mask": mask, "cluster_stats": stats},
        {"exact_cardinality": cardinality, "birth_present": birth_present, "poly_present": poly_present},
        name="v90_structured_cluster_cardinality",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=3e-4),
        loss={
            "exact_cardinality": "sparse_categorical_crossentropy",
            "birth_present": "binary_crossentropy",
            "poly_present": "binary_crossentropy",
        },
        loss_weights={"exact_cardinality": 1.0, "birth_present": 0.25, "poly_present": 0.25},
        metrics={"exact_cardinality": ["accuracy"]},
    )
    return model


def _balanced_weights(target: np.ndarray) -> np.ndarray:
    counts = np.bincount(target, minlength=COUNT_CLASSES).astype(np.float64)
    weights = np.ones(COUNT_CLASSES, dtype=np.float64)
    nonzero = counts > 0
    weights[nonzero] = np.sqrt(len(target) / (COUNT_CLASSES * counts[nonzero]))
    weights = np.clip(weights, 0.45, 4.0)
    sample = weights[target]
    sample /= max(1e-8, float(np.mean(sample)))
    return sample.astype(np.float32)


def _targets(target: np.ndarray):
    return {
        "exact_cardinality": target,
        "birth_present": (target > 0).astype(np.float32).reshape(-1, 1),
        "poly_present": (target >= 2).astype(np.float32).reshape(-1, 1),
    }


def _sample_weights(target: np.ndarray):
    base = _balanced_weights(target)
    return {"exact_cardinality": base, "birth_present": base, "poly_present": base}


def _model_inputs(sequence, mask, stats, indices=None):
    if indices is None:
        return {"candidate_set": sequence, "candidate_mask": mask, "cluster_stats": stats}
    return {
        "candidate_set": sequence[indices],
        "candidate_mask": mask[indices],
        "cluster_stats": stats[indices],
    }


def _decode_cardinality(probabilities: np.ndarray, mode: str) -> np.ndarray:
    p = np.asarray(probabilities, dtype=np.float64)
    if mode == "argmax":
        return np.argmax(p, axis=1).astype(np.int32)
    expected = p @ np.arange(COUNT_CLASSES, dtype=np.float64)
    bias = {
        "expected_round": 0.0,
        "expected_minus_025": -0.25,
        "expected_plus_025": 0.25,
    }.get(mode)
    if bias is None:
        raise ValueError(f"unknown decode mode {mode}")
    return np.clip(np.floor(expected + 0.5 + bias), 0, MAX_EXACT_COUNT).astype(np.int32)


def _prediction_map(clusters, records, fused_scores, k_values: np.ndarray, allowed_members: Optional[set] = None):
    retained: Dict[str, List[int]] = defaultdict(list)
    for cluster, k in zip(clusters, k_values):
        member = cluster["member"]
        if allowed_members is not None and member not in allowed_members:
            continue
        selected = _topk(_slots(cluster, records, fused_scores), int(k))
        retained[member].extend(slot["sample"] for slot in selected)
    return {member: tuple(sorted(values)) for member, values in retained.items()}


def _cardinality_report(target: np.ndarray, exact: np.ndarray, pred: np.ndarray) -> dict:
    confusion = np.zeros((COUNT_CLASSES, COUNT_CLASSES), dtype=np.int64)
    for y, p in zip(target, pred):
        confusion[int(y), int(p)] += 1
    mask_birth = target > 0
    mask_poly = target >= 2
    exact_counts = Counter(int(v) for v in exact)
    return {
        "accuracy": float(np.mean(pred == target)) if len(target) else 0.0,
        "birth_cluster_accuracy": float(np.mean(pred[mask_birth] == target[mask_birth])) if np.any(mask_birth) else None,
        "poly_cluster_accuracy": float(np.mean(pred[mask_poly] == target[mask_poly])) if np.any(mask_poly) else None,
        "mean_absolute_class_error": float(np.mean(np.abs(pred.astype(np.int64) - target.astype(np.int64)))) if len(target) else 0.0,
        "confusion_true_rows_pred_columns": confusion.tolist(),
        "target_class_histogram": {str(k): int(np.sum(target == k)) for k in range(COUNT_CLASSES)},
        "exact_count_histogram": {str(k): int(v) for k, v in sorted(exact_counts.items())},
        "exact_count_max": int(np.max(exact)) if len(exact) else 0,
        "overflow_above_6_count": int(np.sum(exact > MAX_EXACT_COUNT)),
    }


def _select_calibration_mode(probabilities, target, exact, clusters, records, fused_scores, holdout_tracks, holdout_set):
    sweep = []
    for mode in CALIBRATION_MODES:
        pred = _decode_cardinality(probabilities, mode)
        predictions = _prediction_map(clusters, records, fused_scores, pred, holdout_set)
        metrics = _metrics(holdout_tracks, predictions)
        macro = 0.5 * (metrics["solo"]["f1"] + metrics["comp"]["f1"])
        sweep.append(
            {
                "mode": mode,
                "macro_f1": float(macro),
                "metrics": metrics,
                "cardinality": _cardinality_report(target, exact, pred),
            }
        )
    best = max(sweep, key=lambda row: (row["macro_f1"], row["metrics"]["global"]["f1"], row["cardinality"]["accuracy"]))
    return {"selection_rule": "fixed decode rule maximizing train-only holdout macro(solo,comp) F1", "best": best, "sweep": sweep}


def _cluster_data(tracks, records, x88, out88):
    candidate_features, fused = _extended_candidate_features(x88, out88)
    clusters = _clusters_window(records, CLUSTER_WINDOW_MS)
    refs = _references(tracks)
    assigned, assignment = _assign_refs(clusters, records, refs)
    sequence, mask, stats, target, exact, truncated = _cluster_arrays(
        clusters, assigned, records, candidate_features, out88, fused
    )
    return clusters, fused, assignment, sequence, mask, stats, target, exact, truncated


def create_argument_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train V9.0 structured exact-cardinality cluster decoder.")
    p.add_argument("dataset_dir", nargs="?", type=Path, default=ROOT / "data" / "GuitarSet")
    p.add_argument("--base-model", type=Path, required=True)
    p.add_argument("--v86-weights", type=Path, required=True)
    p.add_argument("--v86-report", type=Path, required=True)
    p.add_argument("--v87-weights", type=Path, required=True)
    p.add_argument("--v87-report", type=Path, required=True)
    p.add_argument("--v88-weights", type=Path, required=True)
    p.add_argument("--v88-report", type=Path, required=True)
    p.add_argument("--train-audit", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--train-members", type=int, default=DEFAULT_TRAIN_MEMBERS)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED_V90)
    return p


def run(args) -> dict:
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

    indexed = tuple(t for t in index_guitarset(args.dataset_dir) if t.player_id in ALLOWED_PLAYERS)
    by_member = {t.annotation_member: t for t in indexed}
    _, locked_validation = split_tracks_by_group(indexed, validation_fraction=DEFAULT_VALIDATION_FRACTION, seed=DEFAULT_SEED)
    locked12 = tuple(locked_validation[:12])
    locked_members = {t.annotation_member for t in locked_validation}

    audit = json.loads(args.train_audit.read_text())
    source_members = list(audit["scope"]["members"])[: args.train_members]
    if len(source_members) != args.train_members:
        raise V90Error("not enough source train members")
    if set(source_members) & locked_members:
        raise V90Error("training members overlap locked validation")
    fit_members, holdout_members = _split_members(source_members, args.seed)
    fit_set, holdout_set = set(fit_members), set(holdout_members)
    train_tracks = tuple(by_member[m] for m in source_members)
    fit_tracks = tuple(by_member[m] for m in fit_members)
    holdout_tracks = tuple(by_member[m] for m in holdout_members)

    r86 = json.loads(args.v86_report.read_text())
    r87 = json.loads(args.v87_report.read_text())
    r88 = json.loads(args.v88_report.read_text())
    floor = float(r86["configuration"]["candidate_floor"])
    v88_threshold = float(r88["configuration"]["retain_threshold"])
    if r86["configuration"].get("candidate_floor_selected_on_locked_validation") is not False:
        raise V90Error("V8.6 floor leakage")
    if r87["configuration"].get("retain_threshold_selected_on_locked_validation") is not False:
        raise V90Error("V8.7 threshold leakage")
    if r88["configuration"].get("retain_threshold_selected_on_locked_validation") is not False:
        raise V90Error("V8.8 threshold leakage")

    _, v86_encoder = _load_v86_encoder(args.v86_weights)
    _, v87_encoder = _load_v87_encoder(args.v87_weights)
    v88_model = _load_v88(args.v88_weights)

    print("mining frozen V8.8 candidate sets on source train members")
    train_score_streams, train_records, train_x88, train_out88 = _represent_full(
        train_tracks, args.base_model, floor, v86_encoder, v87_encoder, v88_model
    )
    (
        train_clusters,
        train_fused,
        train_assignment,
        train_sequence,
        train_mask,
        train_stats,
        train_target,
        train_exact,
        train_truncated,
    ) = _cluster_data(train_tracks, train_records, train_x88, train_out88)

    fit_indices = np.asarray([i for i, c in enumerate(train_clusters) if c["member"] in fit_set], dtype=np.int64)
    holdout_indices = np.asarray([i for i, c in enumerate(train_clusters) if c["member"] in holdout_set], dtype=np.int64)
    if not len(fit_indices) or not len(holdout_indices):
        raise V90Error("empty fit or holdout cluster split")

    y_fit = train_target[fit_indices]
    y_holdout = train_target[holdout_indices]
    model = _build_cluster_model()
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=3e-5),
    ]
    history = model.fit(
        _model_inputs(train_sequence, train_mask, train_stats, fit_indices),
        _targets(y_fit),
        sample_weight=_sample_weights(y_fit),
        validation_data=(
            _model_inputs(train_sequence, train_mask, train_stats, holdout_indices),
            _targets(y_holdout),
            _sample_weights(y_holdout),
        ),
        epochs=args.epochs,
        batch_size=128,
        shuffle=True,
        callbacks=callbacks,
        verbose=2,
    )

    holdout_raw = model.predict(_model_inputs(train_sequence, train_mask, train_stats, holdout_indices), batch_size=256, verbose=0)
    holdout_prob = np.asarray(holdout_raw["exact_cardinality"], dtype=np.float32)
    holdout_clusters = [train_clusters[i] for i in holdout_indices]
    # Calibration helper expects k-values aligned to the supplied cluster list.
    calibration_sweep = []
    for mode in CALIBRATION_MODES:
        pred = _decode_cardinality(holdout_prob, mode)
        retained: Dict[str, List[int]] = defaultdict(list)
        for cluster, k in zip(holdout_clusters, pred):
            selected = _topk(_slots(cluster, train_records, train_fused), int(k))
            retained[cluster["member"]].extend(slot["sample"] for slot in selected)
        predictions = {m: tuple(sorted(v)) for m, v in retained.items()}
        metrics = _metrics(holdout_tracks, predictions)
        macro = 0.5 * (metrics["solo"]["f1"] + metrics["comp"]["f1"])
        calibration_sweep.append(
            {
                "mode": mode,
                "macro_f1": float(macro),
                "metrics": metrics,
                "cardinality": _cardinality_report(
                    train_target[holdout_indices], train_exact[holdout_indices], pred
                ),
            }
        )
    calibration_best = max(
        calibration_sweep,
        key=lambda row: (row["macro_f1"], row["metrics"]["global"]["f1"], row["cardinality"]["accuracy"]),
    )
    decode_mode = calibration_best["mode"]
    calibration = {
        "selection_rule": "fixed cardinality decode rule maximizing train-only holdout macro(solo,comp) F1",
        "best": calibration_best,
        "sweep": calibration_sweep,
    }

    model.save_weights(args.output_dir / "v90-structured-cardinality.weights.h5")

    print("evaluating V9.0 once on frozen locked12")
    locked_score_streams, locked_records, locked_x88, locked_out88 = _represent_full(
        locked12, args.base_model, floor, v86_encoder, v87_encoder, v88_model
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
    locked_raw = model.predict(_model_inputs(locked_sequence, locked_mask, locked_stats), batch_size=256, verbose=0)
    locked_prob = np.asarray(locked_raw["exact_cardinality"], dtype=np.float32)
    locked_k = _decode_cardinality(locked_prob, decode_mode)
    locked_predictions = _prediction_map(locked_clusters, locked_records, locked_fused, locked_k)
    locked_metrics = _metrics(locked12, locked_predictions)

    v88_baseline = _metrics(
        locked12, _retained_predictions(locked_records, locked_fused, v88_threshold)
    )
    oracle_exact_k = np.asarray(
        [min(MAX_EXACT_COUNT, len(_refs)) for _refs in (
            # preserve cluster ordering without exposing labels to runtime
            [None] * 0
        )], dtype=np.int32
    )
    oracle_exact_k = np.asarray(
        [min(MAX_EXACT_COUNT, int(value)) for value in locked_exact], dtype=np.int32
    )
    oracle_predictions = _prediction_map(
        locked_clusters, locked_records, locked_fused, oracle_exact_k
    )
    locked_oracle = _metrics(locked12, oracle_predictions)
    locked_ceiling = _candidate_ceiling(locked12, locked_score_streams, floor)

    total_max_delay_samples = MAX_HORIZON + CLUSTER_WINDOW_SAMPLES
    report = {
        "schema_version": 1,
        "architecture": {
            "name": "V9.0 structured exact-cardinality candidate-set decoder",
            "frozen_base": "V8.4 + V8.6 + V8.7 + V8.8",
            "base_trainable": False,
            "v90_trainable_parameters": int(model.count_params()),
            "cluster_window_ms": CLUSTER_WINDOW_MS,
            "candidate_set_cap": MAX_CANDIDATES,
            "candidate_feature_dim": SET_CANDIDATE_DIM,
            "cluster_stats_dim": SET_STATS_DIM,
            "cardinality_classes": ["0", "1", "2", "3", "4", "5", "6+"],
            "candidate_selection_after_k": "top-K frozen V8.8 fused score slots",
            "runtime_inputs_use_annotations": False,
            "future_candidate_context": True,
            "future_candidate_context_ms": CLUSTER_WINDOW_MS,
            "offset_stream_executed": False,
            "offset_weights_modified": False,
            "base_acoustic_horizon_samples": MAX_HORIZON,
            "base_acoustic_horizon_ms": MAX_HORIZON * 1000.0 / SAMPLE_RATE,
            "maximum_verification_delay_samples": total_max_delay_samples,
            "maximum_verification_delay_ms": total_max_delay_samples * 1000.0 / SAMPLE_RATE,
        },
        "configuration": {
            "seed": args.seed,
            "train_members": args.train_members,
            "candidate_floor": floor,
            "frozen_v88_threshold": v88_threshold,
            "requested_epochs": args.epochs,
            "epochs_ran": len(history.history["loss"]),
            "decode_mode": decode_mode,
            "decode_mode_selected_on_locked_validation": False,
            "matching_tolerance_ms": TOLERANCE_MS,
            "reference_assignment_radius_ms": LOCAL_ASSIGNMENT_RADIUS_MS,
        },
        "data": {
            "source_train_members": source_members,
            "fit_members": list(fit_members),
            "holdout_members": list(holdout_members),
            "locked_validation_members": [t.annotation_member for t in locked12],
            "train_reference_assignment": train_assignment,
            "locked_reference_assignment": locked_assignment,
            "fit_cluster_count": int(len(fit_indices)),
            "holdout_cluster_count": int(len(holdout_indices)),
            "locked_cluster_count": int(len(locked_clusters)),
            "train_truncated_cluster_count": int(np.sum(train_truncated > 0)),
            "locked_truncated_cluster_count": int(np.sum(locked_truncated > 0)),
            "train_truncated_candidate_count": int(np.sum(train_truncated)),
            "locked_truncated_candidate_count": int(np.sum(locked_truncated)),
            "fit_cardinality": _cardinality_report(
                train_target[fit_indices], train_exact[fit_indices],
                _decode_cardinality(
                    np.asarray(model.predict(_model_inputs(train_sequence, train_mask, train_stats, fit_indices), batch_size=256, verbose=0)["exact_cardinality"]),
                    decode_mode,
                ),
            ),
            "holdout_cardinality": calibration_best["cardinality"],
        },
        "training_history": {key: [float(v) for v in values] for key, values in history.history.items()},
        "holdout_decode_calibration": calibration,
        "locked12": {
            "candidate_ceiling": locked_ceiling,
            "frozen_v88_baseline_metrics": v88_baseline,
            "v90_metrics": locked_metrics,
            "v90_cardinality": _cardinality_report(locked_target, locked_exact, locked_k),
            "oracle_capped_6_exact_cardinality_metrics": locked_oracle,
        },
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "decode_mode": decode_mode,
                "parameters": int(model.count_params()),
                "max_delay_ms": report["architecture"]["maximum_verification_delay_ms"],
                "locked_cardinality": report["locked12"]["v90_cardinality"],
                "v88": v88_baseline,
                "v90": locked_metrics,
                "oracle": locked_oracle,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = create_argument_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
