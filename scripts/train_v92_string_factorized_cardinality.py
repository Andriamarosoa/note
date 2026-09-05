"""V9.2 string-factorized structured cardinality decoder.

V9.1 reached 76.01 locked12 F1 but still under-counted polyphonic births. The
oracle with exact K and the unchanged V8.8 top-K ranking already reaches 95.94
F1, proving that cardinality remains the dominant bottleneck and that a joint
candidate-subset model is premature.

GuitarSet provides note boundaries in six physical string slots. V9.2 uses that
structure only as training supervision and factorizes the count into six binary
string-birth decisions, while retaining the V9.1 conditional ordinal chain as a
second view of K. Runtime inputs remain the frozen audio-derived V8.8 cluster
representation. No string annotation is used at runtime.

The full-train V9.1 caches are reused; absolute cluster positions are recovered
from the cached relative candidate positions plus the top V8.8 sample. This is
validated against the cached exact cardinality before training.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
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

from causal_note.guitarset import (
    ALLOWED_PLAYERS,
    SAMPLE_RATE,
    SLOT_COUNT,
    index_guitarset,
    load_boundary_slots,
)
from scripts.evaluate_boundaries import milliseconds_to_samples
from scripts.evaluate_v8_boundaries import DEFAULT_SEED, DEFAULT_VALIDATION_FRACTION
from scripts.train_boundaries import group_stem, split_tracks_by_group
from scripts.train_v86_state_transition_proposals import MAX_HORIZON, TOLERANCE_MS, _candidate_ceiling
from scripts.train_v88_regime_moe import _retained_predictions
from scripts.evaluate_v90_cluster_oracles import _metrics, _slots, _topk
from scripts.train_v90_structured_cluster_cardinality import (
    CLUSTER_WINDOW_MS,
    CLUSTER_WINDOW_SAMPLES,
    MAX_CANDIDATES,
    MAX_EXACT_COUNT,
    SET_CANDIDATE_DIM,
    SET_STATS_DIM,
    _build_cluster_model,
    _cluster_data,
    _model_inputs,
    _prediction_map,
    _represent_full,
)
from scripts.train_v91_ordinal_cardinality import (
    ORDINAL_STAGES,
    _conditional_sample_weights,
    _cumulative,
    _dataset_split,
    _load_caches,
    _load_frozen_stack,
    _split_train_groups,
)

DEFAULT_SEED_V92 = 9231
LOCAL_RADIUS_MS = 20.0
LOCAL_RADIUS_SAMPLES = milliseconds_to_samples(LOCAL_RADIUS_MS)
RECONSTRUCT_TOLERANCE_SAMPLES = 3
SLOT_COUNT_LOSS_WEIGHT = 0.35
DECODE_MODES = (
    "slot_threshold_040",
    "slot_threshold_045",
    "slot_threshold_050",
    "slot_threshold_055",
    "slot_threshold_060",
    "slot_expected_round",
    "ordinal_cumulative_050",
    "hybrid_expected_25slot",
    "hybrid_expected_50slot",
    "hybrid_expected_75slot",
)


class V92Error(RuntimeError):
    pass


def _valid_rows(mask_row: np.ndarray) -> np.ndarray:
    return np.flatnonzero(np.asarray(mask_row) > 0)


def _candidate_relative_samples(sequence_row: np.ndarray, mask_row: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    rows = _valid_rows(mask_row)
    if not len(rows):
        return rows, np.zeros((0,), dtype=np.int32)
    rel = np.asarray(sequence_row[rows, -2], dtype=np.float64)
    relative_samples = np.rint(rel * float(CLUSTER_WINDOW_SAMPLES)).astype(np.int32)
    return rows, relative_samples


def _recover_cluster_start(sequence_row, mask_row, top_samples_row) -> Tuple[int, int]:
    rows, relative = _candidate_relative_samples(sequence_row, mask_row)
    top = [int(x) for x in np.asarray(top_samples_row).tolist() if int(x) >= 0]
    if not len(rows) or not top:
        raise V92Error("cannot reconstruct cluster without candidates/top sample")

    # Any top sample must equal start + one valid relative candidate position.
    # Float16 relative storage can move reconstruction by a sample, so score all
    # candidate starts and choose the one explaining the most cached top samples.
    best = None
    relative_unique = np.unique(relative)
    for sample in top:
        for rel in relative_unique:
            start = int(sample - int(rel))
            reconstructed = start + relative_unique
            matches = 0
            total_error = 0
            for observed in top:
                dist = int(np.min(np.abs(reconstructed - observed)))
                if dist <= RECONSTRUCT_TOLERANCE_SAMPLES:
                    matches += 1
                total_error += min(dist, 1000)
            key = (matches, -total_error, -abs(start))
            if best is None or key > best[0]:
                best = (key, start)
    assert best is not None
    start = int(best[1])
    return start, int(best[0][0])


def _reconstruct_candidates(cache) -> Tuple[List[np.ndarray], dict]:
    reconstructed: List[np.ndarray] = []
    full_top_matches = 0
    total_top = 0
    max_top_error = 0
    for seq, mask, top in zip(cache["sequence"], cache["mask"], cache["top_samples"]):
        start, _ = _recover_cluster_start(seq, mask, top)
        _, relative = _candidate_relative_samples(seq, mask)
        samples = np.sort(start + relative)
        reconstructed.append(samples.astype(np.int32))
        for observed in [int(x) for x in top if int(x) >= 0]:
            dist = int(np.min(np.abs(samples - observed))) if len(samples) else 10**9
            total_top += 1
            max_top_error = max(max_top_error, dist)
            if dist <= RECONSTRUCT_TOLERANCE_SAMPLES:
                full_top_matches += 1
    return reconstructed, {
        "top_sample_match_fraction": full_top_matches / total_top if total_top else None,
        "top_sample_count": total_top,
        "max_top_sample_error": max_top_error,
    }


def _slot_targets_for_member_clusters(track, cluster_ids, candidate_samples, *, exact_reference_count=None):
    targets = np.zeros((len(cluster_ids), SLOT_COUNT), dtype=np.float32)
    by_local = {cid: local for local, cid in enumerate(cluster_ids)}
    slots = load_boundary_slots(track.annotation_zip, track.annotation_member)
    assigned_events = 0
    total_events = 0
    same_slot_collisions = 0
    for slot_index, boundaries in enumerate(slots):
        for boundary in boundaries:
            ref = int(boundary.onset_sample)
            total_events += 1
            choices = []
            for cid in cluster_ids:
                samples = candidate_samples[cid]
                if not len(samples):
                    continue
                dist = int(np.min(np.abs(samples - ref)))
                if dist <= LOCAL_RADIUS_SAMPLES:
                    choices.append((dist, cid))
            if not choices:
                continue
            _, cid = min(choices)
            local = by_local[cid]
            if targets[local, slot_index] > 0.5:
                same_slot_collisions += 1
            targets[local, slot_index] = 1.0
            assigned_events += 1
    return targets, {
        "total_events": total_events,
        "assigned_events": assigned_events,
        "unassigned_events": total_events - assigned_events,
        "assigned_fraction": assigned_events / total_events if total_events else None,
        "same_slot_collisions": same_slot_collisions,
    }


def _derive_cache_slot_targets(cache, dataset_dir: Path):
    indexed = tuple(t for t in index_guitarset(dataset_dir) if t.player_id in ALLOWED_PLAYERS)
    by_member = {t.annotation_member: t for t in indexed}
    candidate_samples, reconstruction = _reconstruct_candidates(cache)
    targets = np.zeros((len(cache["target"]), SLOT_COUNT), dtype=np.float32)
    diagnostics = defaultdict(int)
    by_member_rows: Dict[str, List[int]] = defaultdict(list)
    for i, member in enumerate(cache["members"]):
        by_member_rows[str(member)].append(i)
    for member, ids in by_member_rows.items():
        track = by_member.get(member)
        if track is None:
            raise V92Error(f"cache member missing from GuitarSet index: {member}")
        local_targets, diag = _slot_targets_for_member_clusters(track, ids, candidate_samples)
        targets[np.asarray(ids, dtype=np.int64)] = local_targets
        for key, value in diag.items():
            if isinstance(value, (int, np.integer)):
                diagnostics[key] += int(value)
    occupancy_count = np.sum(targets, axis=1).astype(np.int32)
    cached_target = np.asarray(cache["target"], dtype=np.int32)
    cached_exact = np.asarray(cache["exact"], dtype=np.int32)
    capped_exact = np.minimum(MAX_EXACT_COUNT, cached_exact)
    count_match = occupancy_count == capped_exact
    # Some true groups can contain two births on the same physical string within
    # 40 ms; report rather than hide that irreducible occupancy mismatch.
    result_diag = {
        **reconstruction,
        "cluster_count": len(targets),
        "occupancy_exact_count_match_fraction": float(np.mean(count_match)),
        "occupancy_count_mae_vs_exact": float(np.mean(np.abs(occupancy_count - capped_exact))),
        "occupancy_count_histogram": {str(k): int(np.sum(occupancy_count == k)) for k in range(SLOT_COUNT + 1)},
        "cached_count_histogram": {str(k): int(np.sum(cached_target == k)) for k in range(SLOT_COUNT + 1)},
        "same_slot_collisions": int(diagnostics["same_slot_collisions"]),
        "assigned_events": int(diagnostics["assigned_events"]),
        "unassigned_events": int(diagnostics["unassigned_events"]),
    }
    return targets, candidate_samples, result_diag


def _slot_targets_for_runtime_clusters(tracks, clusters, records):
    targets = np.zeros((len(clusters), SLOT_COUNT), dtype=np.float32)
    by_member_ids: Dict[str, List[int]] = defaultdict(list)
    candidate_samples = []
    for cid, cluster in enumerate(clusters):
        by_member_ids[cluster["member"]].append(cid)
        candidate_samples.append(np.asarray([int(records[i]["sample"]) for i in cluster["indices"]], dtype=np.int32))
    by_member_track = {t.annotation_member: t for t in tracks}
    diagnostics = defaultdict(int)
    for member, ids in by_member_ids.items():
        local_targets, diag = _slot_targets_for_member_clusters(by_member_track[member], ids, candidate_samples)
        targets[np.asarray(ids, dtype=np.int64)] = local_targets
        for key, value in diag.items():
            if isinstance(value, (int, np.integer)):
                diagnostics[key] += int(value)
    return targets, {key: int(value) for key, value in diagnostics.items()}


def _build_model():
    try:
        import tensorflow as tf
        from tensorflow import keras
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required") from exc
    scaffold = _build_cluster_model()
    hidden = scaffold.get_layer("cluster_hidden2").output

    string_outputs = [
        keras.layers.Dense(1, activation="sigmoid", name=f"string_{slot}")(hidden)
        for slot in range(SLOT_COUNT)
    ]
    string_stack = keras.layers.Concatenate(name="string_birth_vector")(string_outputs)
    slot_count = keras.layers.Lambda(
        lambda x: tf.reduce_sum(x, axis=1, keepdims=True) / float(SLOT_COUNT),
        name="slot_count",
    )(string_stack)
    ordinal_outputs = [
        keras.layers.Dense(1, activation="sigmoid", name=f"ge{stage}")(hidden)
        for stage in range(1, ORDINAL_STAGES + 1)
    ]
    outputs = {f"string_{slot}": out for slot, out in enumerate(string_outputs)}
    outputs["slot_count"] = slot_count
    outputs.update({f"ge{stage}": out for stage, out in enumerate(ordinal_outputs, start=1)})

    loss = {f"string_{slot}": "binary_crossentropy" for slot in range(SLOT_COUNT)}
    loss["slot_count"] = "mse"
    loss.update({f"ge{stage}": "binary_crossentropy" for stage in range(1, ORDINAL_STAGES + 1)})
    loss_weights = {f"string_{slot}": 0.75 for slot in range(SLOT_COUNT)}
    loss_weights["slot_count"] = SLOT_COUNT_LOSS_WEIGHT
    loss_weights.update({f"ge{stage}": 0.35 * math.sqrt(stage) for stage in range(1, ORDINAL_STAGES + 1)})

    model = keras.Model(scaffold.inputs, outputs, name="v92_string_factorized_cardinality")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=3e-4), loss=loss, loss_weights=loss_weights)
    return model, loss_weights


def _balanced_binary_weights(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    pos = y > 0.5
    n_pos = int(np.sum(pos))
    n_neg = len(y) - n_pos
    if not n_pos or not n_neg:
        raise V92Error(f"binary target lacks both classes pos={n_pos} neg={n_neg}")
    w = np.empty(len(y), dtype=np.float32)
    w[pos] = len(y) / (2.0 * n_pos)
    w[~pos] = len(y) / (2.0 * n_neg)
    return np.clip(w, 0.25, 8.0)


def _training_targets(slot_targets: np.ndarray, k: np.ndarray):
    result = {f"string_{slot}": slot_targets[:, slot].reshape(-1, 1) for slot in range(SLOT_COUNT)}
    result["slot_count"] = (np.minimum(k, SLOT_COUNT).astype(np.float32) / float(SLOT_COUNT)).reshape(-1, 1)
    result.update({f"ge{stage}": (k >= stage).astype(np.float32).reshape(-1, 1) for stage in range(1, ORDINAL_STAGES + 1)})
    return result


def _sample_weights(slot_targets: np.ndarray, k: np.ndarray):
    result = {f"string_{slot}": _balanced_binary_weights(slot_targets[:, slot]) for slot in range(SLOT_COUNT)}
    class_counts = np.bincount(np.minimum(k, SLOT_COUNT), minlength=SLOT_COUNT + 1).astype(np.float64)
    class_weight = np.ones(SLOT_COUNT + 1, dtype=np.float64)
    nonzero = class_counts > 0
    class_weight[nonzero] = np.sqrt(len(k) / ((SLOT_COUNT + 1) * class_counts[nonzero]))
    class_weight = np.clip(class_weight, 0.35, 4.0)
    result["slot_count"] = class_weight[np.minimum(k, SLOT_COUNT)].astype(np.float32)
    ordinal, _ = _conditional_sample_weights(k)
    result.update(ordinal)
    return result


def _predict(model, inputs):
    raw = model.predict(inputs, batch_size=256, verbose=0)
    slots = np.stack([np.asarray(raw[f"string_{slot}"]).reshape(-1) for slot in range(SLOT_COUNT)], axis=1)
    conditionals = np.stack([np.asarray(raw[f"ge{stage}"]).reshape(-1) for stage in range(1, ORDINAL_STAGES + 1)], axis=1)
    return slots.astype(np.float64), conditionals.astype(np.float64)


def _decode(slots: np.ndarray, conditionals: np.ndarray, mode: str) -> np.ndarray:
    slot_expected = np.sum(slots, axis=1)
    cumulative = _cumulative(conditionals)
    ordinal_expected = np.sum(cumulative, axis=1)
    if mode.startswith("slot_threshold_"):
        threshold = int(mode.rsplit("_", 1)[1]) / 100.0
        return np.sum(slots >= threshold, axis=1).astype(np.int32)
    if mode == "slot_expected_round":
        value = slot_expected
    elif mode == "ordinal_cumulative_050":
        return np.sum(cumulative >= 0.5, axis=1).astype(np.int32)
    elif mode.startswith("hybrid_expected_"):
        slot_weight = int(mode.rsplit("_", 1)[1].replace("slot", "")) / 100.0
        value = slot_weight * slot_expected + (1.0 - slot_weight) * ordinal_expected
    else:
        raise ValueError(f"unknown decode mode {mode}")
    return np.clip(np.floor(value + 0.5), 0, SLOT_COUNT).astype(np.int32)


def _cached_prediction_map(cache, indices: np.ndarray, k_values: np.ndarray):
    retained: Dict[str, List[int]] = defaultdict(list)
    for idx, k in zip(indices, k_values):
        member = str(cache["members"][idx])
        samples = [int(v) for v in cache["top_samples"][idx] if int(v) >= 0]
        retained[member].extend(samples[: int(k)])
    return {member: tuple(sorted(values)) for member, values in retained.items()}


def _slot_report(truth: np.ndarray, probabilities: np.ndarray) -> dict:
    pred = probabilities >= 0.5
    truth_b = truth >= 0.5
    per_slot = {}
    for slot in range(SLOT_COUNT):
        tp = int(np.sum(pred[:, slot] & truth_b[:, slot]))
        fp = int(np.sum(pred[:, slot] & ~truth_b[:, slot]))
        fn = int(np.sum(~pred[:, slot] & truth_b[:, slot]))
        per_slot[str(slot)] = {
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "positive_count": int(np.sum(truth_b[:, slot])),
        }
    exact_vector = float(np.mean(np.all(pred == truth_b, axis=1))) if len(truth) else 0.0
    return {
        "exact_string_vector_accuracy_at_0_5": exact_vector,
        "per_string": per_slot,
        "mean_predicted_string_count_at_0_5": float(np.mean(np.sum(pred, axis=1))) if len(pred) else 0.0,
        "mean_true_string_count": float(np.mean(np.sum(truth_b, axis=1))) if len(truth_b) else 0.0,
    }


def _cardinality_report(target: np.ndarray, pred: np.ndarray) -> dict:
    target = np.minimum(np.asarray(target, dtype=np.int32), SLOT_COUNT)
    pred = np.asarray(pred, dtype=np.int32)
    confusion = np.zeros((SLOT_COUNT + 1, SLOT_COUNT + 1), dtype=np.int64)
    for y, p in zip(target, pred):
        confusion[int(y), int(p)] += 1
    birth = target > 0
    poly = target >= 2
    return {
        "accuracy": float(np.mean(target == pred)) if len(target) else 0.0,
        "birth_cluster_accuracy": float(np.mean(target[birth] == pred[birth])) if np.any(birth) else None,
        "poly_cluster_accuracy": float(np.mean(target[poly] == pred[poly])) if np.any(poly) else None,
        "mean_absolute_class_error": float(np.mean(np.abs(target - pred))) if len(target) else 0.0,
        "target_histogram": {str(k): int(np.sum(target == k)) for k in range(SLOT_COUNT + 1)},
        "predicted_histogram": {str(k): int(np.sum(pred == k)) for k in range(SLOT_COUNT + 1)},
        "confusion_true_rows_pred_columns": confusion.tolist(),
    }


def create_parser():
    p = argparse.ArgumentParser()
    p.add_argument("dataset_dir", nargs="?", type=Path, default=ROOT / "data" / "GuitarSet")
    p.add_argument("--base-model", type=Path, required=True)
    p.add_argument("--v86-weights", type=Path, required=True)
    p.add_argument("--v86-report", type=Path, required=True)
    p.add_argument("--v87-weights", type=Path, required=True)
    p.add_argument("--v87-report", type=Path, required=True)
    p.add_argument("--v88-weights", type=Path, required=True)
    p.add_argument("--v88-report", type=Path, required=True)
    p.add_argument("--v91-report", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED_V92)
    return p


def run(args):
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

    _, train_split, validation = _dataset_split(args.dataset_dir)
    locked12 = tuple(validation[:12])
    locked_members = {t.annotation_member for t in locked12}
    cache = _load_caches(args.cache_dir)
    if set(cache["track_members"]) & {t.annotation_member for t in validation}:
        raise V92Error("full-train cache overlaps validation")
    if len(cache["track_members"]) != len(train_split):
        raise V92Error(f"expected full train cache {len(train_split)} tracks, got {len(cache['track_members'])}")

    print("deriving six-string supervision from cached full-train clusters")
    slot_targets, reconstructed_samples, reconstruction = _derive_cache_slot_targets(cache, args.dataset_dir)
    if reconstruction["top_sample_match_fraction"] is None or reconstruction["top_sample_match_fraction"] < 0.999:
        raise V92Error(f"cache position reconstruction too weak: {reconstruction}")
    if reconstruction["occupancy_exact_count_match_fraction"] < 0.95:
        raise V92Error(f"string occupancy supervision does not explain cardinality well enough: {reconstruction}")

    fit_tracks, holdout_tracks, holdout_groups = _split_train_groups(train_split, args.seed)
    fit_set = {t.annotation_member for t in fit_tracks}
    holdout_set = {t.annotation_member for t in holdout_tracks}
    fit_idx = np.asarray([i for i, m in enumerate(cache["members"]) if str(m) in fit_set], dtype=np.int64)
    holdout_idx = np.asarray([i for i, m in enumerate(cache["members"]) if str(m) in holdout_set], dtype=np.int64)
    if not len(fit_idx) or not len(holdout_idx):
        raise V92Error("empty fit/holdout")

    model, loss_weights = _build_model()
    k = np.minimum(np.asarray(cache["exact"], dtype=np.int32), SLOT_COUNT)
    y_fit = _training_targets(slot_targets[fit_idx], k[fit_idx])
    y_holdout = _training_targets(slot_targets[holdout_idx], k[holdout_idx])
    sw_fit = _sample_weights(slot_targets[fit_idx], k[fit_idx])
    sw_holdout = _sample_weights(slot_targets[holdout_idx], k[holdout_idx])
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=3e-5),
    ]
    print(f"training V9.2 fit_clusters={len(fit_idx)} holdout_clusters={len(holdout_idx)}")
    history = model.fit(
        _model_inputs(cache["sequence"], cache["mask"], cache["stats"], fit_idx),
        y_fit,
        sample_weight=sw_fit,
        validation_data=(
            _model_inputs(cache["sequence"], cache["mask"], cache["stats"], holdout_idx),
            y_holdout,
            sw_holdout,
        ),
        epochs=args.epochs,
        batch_size=192,
        shuffle=True,
        callbacks=callbacks,
        verbose=2,
    )

    hold_slots, hold_cond = _predict(model, _model_inputs(cache["sequence"], cache["mask"], cache["stats"], holdout_idx))
    sweep = []
    for mode in DECODE_MODES:
        pred_k = _decode(hold_slots, hold_cond, mode)
        metrics = _metrics(holdout_tracks, _cached_prediction_map(cache, holdout_idx, pred_k))
        macro = 0.5 * (metrics["solo"]["f1"] + metrics["comp"]["f1"])
        sweep.append({
            "mode": mode,
            "macro_f1": float(macro),
            "metrics": metrics,
            "cardinality": _cardinality_report(k[holdout_idx], pred_k),
        })
    best = max(sweep, key=lambda row: (row["macro_f1"], row["metrics"]["global"]["f1"], row["cardinality"]["accuracy"]))
    decode_mode = best["mode"]
    model.save_weights(args.output_dir / "v92-string-factorized-cardinality.weights.h5")

    floor, v88_threshold, enc86, enc87, model88 = _load_frozen_stack(args)
    print("evaluating V9.2 once on frozen locked12")
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
    locked_slots, locked_cond = _predict(model, _model_inputs(locked_sequence, locked_mask, locked_stats))
    locked_k = _decode(locked_slots, locked_cond, decode_mode)
    locked_metrics = _metrics(locked12, _prediction_map(locked_clusters, locked_records, locked_fused, locked_k))
    v88_metrics = _metrics(locked12, _retained_predictions(locked_records, locked_fused, v88_threshold))
    oracle_k = np.minimum(np.asarray(locked_exact, dtype=np.int32), SLOT_COUNT)
    oracle_metrics = _metrics(locked12, _prediction_map(locked_clusters, locked_records, locked_fused, oracle_k))
    locked_slot_truth, locked_slot_assignment = _slot_targets_for_runtime_clusters(locked12, locked_clusters, locked_records)

    v91 = json.loads(args.v91_report.read_text())
    v91_metrics = v91["locked12"]["v91_metrics"]
    total_delay = MAX_HORIZON + CLUSTER_WINDOW_SAMPLES
    report = {
        "schema_version": 1,
        "architecture": {
            "name": "V9.2 dual-factorized six-string + ordinal cardinality decoder",
            "frozen_base": "V8.4 + V8.6 + V8.7 + V8.8",
            "base_trainable": False,
            "v92_trainable_parameters": int(model.count_params()),
            "runtime_inputs_use_annotations": False,
            "string_labels_training_only": True,
            "string_slot_count": SLOT_COUNT,
            "ordinal_stage_count": ORDINAL_STAGES,
            "offset_stream_executed": False,
            "offset_weights_modified": False,
            "cluster_window_ms": CLUSTER_WINDOW_MS,
            "maximum_verification_delay_ms": total_delay * 1000.0 / SAMPLE_RATE,
        },
        "configuration": {
            "seed": args.seed,
            "decode_mode": decode_mode,
            "decode_mode_selected_on_locked_validation": False,
            "decode_modes_fixed_before_locked_eval": list(DECODE_MODES),
            "loss_weights": loss_weights,
            "epochs_ran": len(history.history["loss"]),
            "matching_tolerance_ms": TOLERANCE_MS,
        },
        "data": {
            "full_train_track_count": len(train_split),
            "full_validation_track_count": len(validation),
            "cached_cluster_count": len(cache["target"]),
            "fit_track_count": len(fit_tracks),
            "holdout_track_count": len(holdout_tracks),
            "holdout_composition_groups": list(holdout_groups),
            "fit_cluster_count": len(fit_idx),
            "holdout_cluster_count": len(holdout_idx),
            "composition_group_leakage": bool({group_stem(t) for t in fit_tracks} & {group_stem(t) for t in holdout_tracks}),
            "cache_reconstruction": reconstruction,
            "locked_string_assignment": locked_slot_assignment,
        },
        "training_history": {key: [float(v) for v in values] for key, values in history.history.items()},
        "holdout_decode_calibration": {"best": best, "sweep": sweep},
        "locked12": {
            "frozen_v88_baseline_metrics": v88_metrics,
            "v91_metrics": v91_metrics,
            "v92_metrics": locked_metrics,
            "oracle_exact_cardinality_metrics": oracle_metrics,
            "v92_cardinality": _cardinality_report(np.minimum(locked_exact, SLOT_COUNT), locked_k),
            "v92_string_slots": _slot_report(locked_slot_truth, locked_slots),
            "candidate_ceiling": _candidate_ceiling(locked12, locked_score_streams, floor),
        },
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "decode_mode": decode_mode,
        "parameters": int(model.count_params()),
        "cache_reconstruction": reconstruction,
        "v88": v88_metrics,
        "v91": v91_metrics,
        "v92": locked_metrics,
        "oracle": oracle_metrics,
        "cardinality": report["locked12"]["v92_cardinality"],
        "string_slots": report["locked12"]["v92_string_slots"],
    }, indent=2, sort_keys=True))
    return report


def main(argv: Optional[Sequence[str]] = None):
    args = create_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
