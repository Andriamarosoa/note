"""V10.1 string-query harmonic attention cardinality decoder.

V10.0 proved that preserving a 23x64x3 time-frequency map improves anonymous
birth cardinality, especially on accompaniment, but its six string heads still
consume one globally flattened acoustic embedding.  V10.1 makes the physical
string factorization architectural rather than only supervisory:

* keep the frozen V8.4/V8.6/V8.7/V8.8 proposal/context stack;
* reuse the already-mined V10 spectral caches (no acoustic re-mining);
* preserve absolute time/frequency coordinates in the spectral encoder;
* form a token sequence instead of flattening the time-frequency map;
* let six independent string queries cross-attend to those tokens;
* supervise each query with string occupancy and training-only MIDI pitch;
* build the ordinal K decoder from the six disentangled string evidence vectors.

Pitch and string annotations are training-only. Runtime inputs remain audio plus
frozen model outputs. Locked12 is evaluated once after train-only calibration.
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
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from causal_note.guitarset import ALLOWED_PLAYERS, SAMPLE_RATE, SLOT_COUNT, index_guitarset
from scripts.train_boundaries import group_stem
from scripts.train_v86_state_transition_proposals import MAX_HORIZON, TOLERANCE_MS, _candidate_ceiling
from scripts.train_v88_regime_moe import _retained_predictions
from scripts.evaluate_v90_cluster_oracles import _metrics
from scripts.train_v90_structured_cluster_cardinality import (
    CLUSTER_WINDOW_MS,
    CLUSTER_WINDOW_SAMPLES,
    _build_cluster_model,
    _cluster_data,
    _prediction_map,
    _represent_full,
)
from scripts.train_v91_ordinal_cardinality import (
    ORDINAL_STAGES,
    _conditional_sample_weights,
    _cumulative,
    _dataset_split,
    _load_frozen_stack,
    _split_train_groups,
)
from scripts.train_v92_string_factorized_cardinality import (
    LOCAL_RADIUS_SAMPLES,
    _cardinality_report,
    _reconstruct_candidates,
    _slot_targets_for_runtime_clusters,
)
from scripts.train_v100_spectral_string_slots import (
    DECODE_MODES as V100_DECODE_MODES,
    SPECTRAL_BANDS,
    SPECTRAL_CHANNELS,
    TIME_FRAMES,
    _cached_prediction_map,
    _load_spectral_caches,
    _slot_report_local,
    _spectral_maps_for_runtime,
)

DEFAULT_SEED_V101 = 10131
PITCH_SCALE = 127.0
DECODE_MODES = tuple(V100_DECODE_MODES) + ("slot_poisson_binomial_argmax",)


class V101Error(RuntimeError):
    pass


def _slot_index(annotation: dict) -> int:
    metadata = annotation.get("annotation_metadata")
    if not isinstance(metadata, dict):
        raise V101Error("note_midi annotation_metadata must be an object")
    source = metadata.get("data_source")
    if isinstance(source, str) and source in {"0", "1", "2", "3", "4", "5"}:
        return int(source)
    if not isinstance(source, bool) and isinstance(source, int) and 0 <= source < SLOT_COUNT:
        return int(source)
    raise V101Error(f"invalid string data_source {source!r}")


def _midi_value(value) -> float:
    if not isinstance(value, bool) and isinstance(value, (int, float)):
        midi = float(value)
    elif isinstance(value, dict):
        candidate = None
        for key in ("midi", "note", "pitch", "value"):
            if key in value and not isinstance(value[key], bool) and isinstance(value[key], (int, float)):
                candidate = float(value[key])
                break
        if candidate is None:
            raise V101Error(f"unsupported note_midi value {value!r}")
        midi = candidate
    else:
        raise V101Error(f"unsupported note_midi value {value!r}")
    if not math.isfinite(midi) or midi < 0.0 or midi > PITCH_SCALE:
        raise V101Error(f"invalid MIDI pitch {midi}")
    return midi


def _pitch_events(track) -> Tuple[Tuple[int, int, float], ...]:
    try:
        with zipfile.ZipFile(track.annotation_zip, "r") as archive:
            raw = archive.read(track.annotation_member)
        document = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise V101Error(f"cannot parse pitch annotations for {track.annotation_member}") from exc
    annotations = document.get("annotations") if isinstance(document, dict) else None
    if not isinstance(annotations, list):
        raise V101Error("JAMS annotations must be a list")
    events: List[Tuple[int, int, float]] = []
    for annotation in annotations:
        if not isinstance(annotation, dict) or annotation.get("namespace") != "note_midi":
            continue
        slot = _slot_index(annotation)
        observations = annotation.get("data")
        if not isinstance(observations, list):
            raise V101Error("note_midi data must be a list")
        for observation in observations:
            if not isinstance(observation, dict):
                raise V101Error("note_midi observation must be an object")
            time = observation.get("time")
            if isinstance(time, bool) or not isinstance(time, (int, float)) or not math.isfinite(float(time)) or float(time) < 0:
                raise V101Error(f"invalid onset time {time!r}")
            onset = int(round(float(time) * SAMPLE_RATE))
            events.append((slot, onset, _midi_value(observation.get("value"))))
    return tuple(events)


def _derive_pitch_targets(cache, dataset_dir: Path):
    indexed = tuple(t for t in index_guitarset(dataset_dir) if t.player_id in ALLOWED_PLAYERS)
    by_member = {t.annotation_member: t for t in indexed}
    candidate_samples, reconstruction = _reconstruct_candidates(cache)
    pitch = np.zeros((len(cache["target"]), SLOT_COUNT), dtype=np.float32)
    mask = np.zeros_like(pitch)
    by_member_rows: Dict[str, List[int]] = defaultdict(list)
    for i, member in enumerate(cache["members"]):
        by_member_rows[str(member)].append(i)

    assigned = 0
    unassigned = 0
    collisions = 0
    values: List[float] = []
    for member, ids in by_member_rows.items():
        track = by_member.get(member)
        if track is None:
            raise V101Error(f"cache member missing from GuitarSet index: {member}")
        by_local = {cid: local for local, cid in enumerate(ids)}
        for slot, onset, midi in _pitch_events(track):
            choices = []
            for cid in ids:
                samples = candidate_samples[cid]
                if not len(samples):
                    continue
                dist = int(np.min(np.abs(samples - onset)))
                if dist <= LOCAL_RADIUS_SAMPLES:
                    choices.append((dist, cid))
            if not choices:
                unassigned += 1
                continue
            _, cid = min(choices)
            local = by_local[cid]
            if mask[local, slot] > 0.5:
                collisions += 1
            pitch[local, slot] = float(midi / PITCH_SCALE)
            mask[local, slot] = 1.0
            values.append(float(midi))
            assigned += 1

    slot_truth = np.asarray(cache["slot_targets"], dtype=np.float32) > 0.5
    pitch_truth = mask > 0.5
    agreement = float(np.mean(slot_truth == pitch_truth))
    active_agreement = float(np.mean(pitch_truth[slot_truth])) if np.any(slot_truth) else 1.0
    if agreement < 0.999 or active_agreement < 0.995:
        raise V101Error(f"pitch assignment does not reproduce slot supervision: agreement={agreement} active={active_agreement}")
    return pitch, mask, {
        **reconstruction,
        "assigned_pitch_events": assigned,
        "unassigned_pitch_events": unassigned,
        "same_slot_collisions": collisions,
        "slot_pitch_mask_agreement": agreement,
        "active_slot_pitch_coverage": active_agreement,
        "midi_min": min(values) if values else None,
        "midi_max": max(values) if values else None,
        "midi_mean": float(np.mean(values)) if values else None,
    }


def _build_model():
    try:
        import tensorflow as tf
        from tensorflow import keras
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required") from exc

    scaffold = _build_cluster_model()
    candidate_hidden = scaffold.get_layer("cluster_hidden2").output
    candidate_context = keras.layers.Dense(96, activation="relu", name="candidate_context")(candidate_hidden)

    spectral = keras.Input((TIME_FRAMES, SPECTRAL_BANDS, SPECTRAL_CHANNELS), name="spectral_map")
    x = keras.layers.LayerNormalization(axis=-1, name="spectral_channel_norm")(spectral)

    time_coord = np.linspace(-1.0, 1.0, TIME_FRAMES, dtype=np.float32)[:, None]
    freq_coord = np.linspace(-1.0, 1.0, SPECTRAL_BANDS, dtype=np.float32)[None, :]
    coord_grid = np.stack(
        (
            np.broadcast_to(time_coord, (TIME_FRAMES, SPECTRAL_BANDS)),
            np.broadcast_to(freq_coord, (TIME_FRAMES, SPECTRAL_BANDS)),
        ),
        axis=-1,
    ).astype(np.float32)
    coord_const = tf.constant(coord_grid, dtype=tf.float32)
    coords = keras.layers.Lambda(
        lambda t: tf.tile(coord_const[None, :, :, :], [tf.shape(t)[0], 1, 1, 1]),
        output_shape=(TIME_FRAMES, SPECTRAL_BANDS, 2),
        name="absolute_tf_coordinates",
    )(spectral)
    x = keras.layers.Concatenate(axis=-1, name="spectral_plus_coordinates")([x, coords])
    x = keras.layers.Conv2D(32, (3, 5), strides=(1, 2), padding="same", activation="relu", name="tfq_conv1")(x)
    x = keras.layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same", activation="relu", name="tfq_conv2")(x)
    x = keras.layers.Conv2D(96, (3, 3), strides=(1, 2), padding="same", activation="relu", name="tfq_conv3")(x)
    token_time = int(math.ceil(TIME_FRAMES / 2.0))
    token_freq = int(math.ceil(math.ceil(math.ceil(SPECTRAL_BANDS / 2.0) / 2.0) / 2.0))
    token_count = token_time * token_freq
    tokens = keras.layers.Reshape((token_count, 96), name="tf_tokens")(x)
    tokens = keras.layers.LayerNormalization(name="tf_token_norm")(tokens)
    tokens = keras.layers.Dense(96, activation="relu", name="tf_token_projection")(tokens)

    cross_attention = keras.layers.MultiHeadAttention(
        num_heads=4,
        key_dim=24,
        dropout=0.10,
        name="shared_string_cross_attention",
    )
    slot_features = []
    string_outputs = []
    pitch_outputs = []
    for slot in range(SLOT_COUNT):
        q = keras.layers.Dense(96, activation="relu", name=f"string_{slot}_query")(candidate_context)
        q = keras.layers.Reshape((1, 96), name=f"string_{slot}_query_token")(q)
        attended = cross_attention(query=q, value=tokens, key=tokens)
        attended = keras.layers.Add(name=f"string_{slot}_attention_residual")([q, attended])
        attended = keras.layers.LayerNormalization(name=f"string_{slot}_attention_norm")(attended)
        attended = keras.layers.Dense(96, activation="relu", name=f"string_{slot}_attention_ff")(attended)
        attended = keras.layers.Lambda(lambda t: tf.squeeze(t, axis=1), name=f"string_{slot}_evidence")(attended)
        feature = keras.layers.Concatenate(name=f"string_{slot}_fusion")([candidate_context, attended])
        feature = keras.layers.Dense(96, activation="relu", name=f"string_{slot}_hidden")(feature)
        feature = keras.layers.Dropout(0.10, name=f"string_{slot}_dropout")(feature)
        slot_features.append(feature)
        string_outputs.append(keras.layers.Dense(1, activation="sigmoid", name=f"string_{slot}")(feature))
        pitch_outputs.append(keras.layers.Dense(1, activation="sigmoid", name=f"pitch_{slot}")(feature))

    string_stack = keras.layers.Concatenate(name="string_birth_vector")(string_outputs)
    slot_count = keras.layers.Lambda(
        lambda t: tf.reduce_sum(t, axis=1, keepdims=True) / float(SLOT_COUNT),
        name="slot_count",
    )(string_stack)

    global_tokens = keras.layers.Concatenate(name="tf_global_pool")([
        keras.layers.GlobalAveragePooling1D(name="tf_global_average")(tokens),
        keras.layers.GlobalMaxPooling1D(name="tf_global_max")(tokens),
    ])
    all_slot_evidence = keras.layers.Concatenate(name="all_string_evidence")(slot_features)
    count_hidden = keras.layers.Concatenate(name="count_source_fusion")([
        candidate_context,
        global_tokens,
        all_slot_evidence,
    ])
    count_hidden = keras.layers.LayerNormalization(name="count_fusion_norm")(count_hidden)
    count_hidden = keras.layers.Dense(256, activation="relu", name="count_hidden1")(count_hidden)
    count_hidden = keras.layers.Dropout(0.15, name="count_dropout")(count_hidden)
    count_hidden = keras.layers.Dense(128, activation="relu", name="count_hidden2")(count_hidden)
    ordinal_outputs = [
        keras.layers.Dense(1, activation="sigmoid", name=f"ge{stage}")(count_hidden)
        for stage in range(1, ORDINAL_STAGES + 1)
    ]

    outputs = {f"string_{slot}": out for slot, out in enumerate(string_outputs)}
    outputs.update({f"pitch_{slot}": out for slot, out in enumerate(pitch_outputs)})
    outputs["slot_count"] = slot_count
    outputs.update({f"ge{stage}": out for stage, out in enumerate(ordinal_outputs, start=1)})

    loss = {f"string_{slot}": "binary_crossentropy" for slot in range(SLOT_COUNT)}
    loss.update({f"pitch_{slot}": "mse" for slot in range(SLOT_COUNT)})
    loss["slot_count"] = "mse"
    loss.update({f"ge{stage}": "binary_crossentropy" for stage in range(1, ORDINAL_STAGES + 1)})
    loss_weights = {f"string_{slot}": 0.70 for slot in range(SLOT_COUNT)}
    loss_weights.update({f"pitch_{slot}": 0.20 for slot in range(SLOT_COUNT)})
    loss_weights["slot_count"] = 0.35
    loss_weights.update({f"ge{stage}": 0.50 * math.sqrt(stage) for stage in range(1, ORDINAL_STAGES + 1)})

    model = keras.Model(
        {
            "candidate_set": scaffold.input["candidate_set"],
            "candidate_mask": scaffold.input["candidate_mask"],
            "cluster_stats": scaffold.input["cluster_stats"],
            "spectral_map": spectral,
        },
        outputs,
        name="v101_string_query_harmonic_attention",
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=2e-4), loss=loss, loss_weights=loss_weights)
    return model, loss_weights, (token_time, token_freq, token_count)


def _soft_binary_weights(y):
    y = np.asarray(y).reshape(-1) > 0.5
    pos = int(np.sum(y)); neg = len(y) - pos
    if not pos or not neg:
        raise V101Error(f"binary target lacks class pos={pos} neg={neg}")
    positive_weight = math.sqrt(neg / pos)
    w = np.ones(len(y), dtype=np.float32)
    w[y] = positive_weight
    w /= float(np.mean(w))
    return np.clip(w, 0.35, 4.0)


def _targets(slot_targets, pitch_targets, k):
    result = {f"string_{slot}": slot_targets[:, slot].reshape(-1, 1) for slot in range(SLOT_COUNT)}
    result.update({f"pitch_{slot}": pitch_targets[:, slot].reshape(-1, 1) for slot in range(SLOT_COUNT)})
    result["slot_count"] = (np.minimum(k, SLOT_COUNT).astype(np.float32) / float(SLOT_COUNT)).reshape(-1, 1)
    result.update({f"ge{stage}": (k >= stage).astype(np.float32).reshape(-1, 1) for stage in range(1, ORDINAL_STAGES + 1)})
    return result


def _sample_weights(slot_targets, pitch_mask, k):
    result = {f"string_{slot}": _soft_binary_weights(slot_targets[:, slot]) for slot in range(SLOT_COUNT)}
    for slot in range(SLOT_COUNT):
        mask = np.asarray(pitch_mask[:, slot], dtype=np.float32)
        prevalence = max(float(np.mean(mask)), 1e-6)
        active_weight = min(6.0, 1.0 / math.sqrt(prevalence))
        result[f"pitch_{slot}"] = mask * active_weight
    counts = np.bincount(np.minimum(k, SLOT_COUNT), minlength=SLOT_COUNT + 1).astype(np.float64)
    weights = np.ones(SLOT_COUNT + 1, dtype=np.float64)
    nz = counts > 0
    weights[nz] = np.sqrt(len(k) / ((SLOT_COUNT + 1) * counts[nz]))
    result["slot_count"] = np.clip(weights[np.minimum(k, SLOT_COUNT)], 0.35, 4.0).astype(np.float32)
    ordinal, _ = _conditional_sample_weights(k)
    result.update(ordinal)
    return result


def _inputs(cache, indices=None):
    if indices is None:
        return {
            "candidate_set": cache["sequence"],
            "candidate_mask": cache["mask"],
            "cluster_stats": cache["stats"],
            "spectral_map": cache["spectral"],
        }
    return {
        "candidate_set": cache["sequence"][indices],
        "candidate_mask": cache["mask"][indices],
        "cluster_stats": cache["stats"][indices],
        "spectral_map": cache["spectral"][indices],
    }


def _predict(model, inputs):
    raw = model.predict(inputs, batch_size=128, verbose=0)
    slots = np.stack([np.asarray(raw[f"string_{slot}"]).reshape(-1) for slot in range(SLOT_COUNT)], axis=1)
    pitches = np.stack([np.asarray(raw[f"pitch_{slot}"]).reshape(-1) for slot in range(SLOT_COUNT)], axis=1)
    conditionals = np.stack([np.asarray(raw[f"ge{stage}"]).reshape(-1) for stage in range(1, ORDINAL_STAGES + 1)], axis=1)
    return slots.astype(np.float64), pitches.astype(np.float64), conditionals.astype(np.float64)


def _poisson_binomial(slots):
    slots = np.clip(np.asarray(slots, dtype=np.float64), 0.0, 1.0)
    n = len(slots)
    dist = np.ones((n, 1), dtype=np.float64)
    for slot in range(SLOT_COUNT):
        p = slots[:, slot:slot + 1]
        updated = np.zeros((n, dist.shape[1] + 1), dtype=np.float64)
        updated[:, :-1] += dist * (1.0 - p)
        updated[:, 1:] += dist * p
        dist = updated
    return dist


def _decode(slots, conditionals, mode):
    if mode == "slot_poisson_binomial_argmax":
        return np.argmax(_poisson_binomial(slots), axis=1).astype(np.int32)
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
        raise ValueError(mode)
    return np.clip(np.floor(value + 0.5), 0, SLOT_COUNT).astype(np.int32)


def _pitch_report(targets, predictions, mask):
    truth = np.asarray(targets, dtype=np.float64) * PITCH_SCALE
    pred = np.asarray(predictions, dtype=np.float64) * PITCH_SCALE
    active = np.asarray(mask) > 0.5
    result = {"active_count": int(np.sum(active))}
    if np.any(active):
        error = np.abs(pred[active] - truth[active])
        result.update({
            "mae_semitones": float(np.mean(error)),
            "median_absolute_error_semitones": float(np.median(error)),
        })
    per_string = {}
    for slot in range(SLOT_COUNT):
        m = active[:, slot]
        per_string[str(slot)] = {
            "active_count": int(np.sum(m)),
            "mae_semitones": float(np.mean(np.abs(pred[m, slot] - truth[m, slot]))) if np.any(m) else None,
        }
    result["per_string"] = per_string
    return result


def train_eval(args):
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

    cache = _load_spectral_caches(args.cache_dir)
    indexed, train_split, validation = _dataset_split(args.dataset_dir)
    train_members = {t.annotation_member for t in train_split}
    validation_members = {t.annotation_member for t in validation}
    if train_members & validation_members:
        raise V101Error("train/validation leakage")
    if set(cache["track_members"]) != train_members:
        raise V101Error("V10 spectral caches do not exactly cover the full train split")

    pitch_targets, pitch_mask, pitch_diag = _derive_pitch_targets(cache, args.dataset_dir)
    fit_tracks, holdout_tracks, holdout_groups = _split_train_groups(train_split, args.seed)
    fit_set = {t.annotation_member for t in fit_tracks}
    holdout_set = {t.annotation_member for t in holdout_tracks}
    fit_idx = np.asarray([i for i, m in enumerate(cache["members"]) if str(m) in fit_set], dtype=np.int64)
    hold_idx = np.asarray([i for i, m in enumerate(cache["members"]) if str(m) in holdout_set], dtype=np.int64)
    k = np.minimum(cache["exact"], SLOT_COUNT)

    model, loss_weights, token_shape = _build_model()
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=2e-5),
    ]
    print(f"training V10.1 string-query attention fit={len(fit_idx)} holdout={len(hold_idx)} tracks={len(train_split)}")
    history = model.fit(
        _inputs(cache, fit_idx),
        _targets(cache["slot_targets"][fit_idx], pitch_targets[fit_idx], k[fit_idx]),
        sample_weight=_sample_weights(cache["slot_targets"][fit_idx], pitch_mask[fit_idx], k[fit_idx]),
        validation_data=(
            _inputs(cache, hold_idx),
            _targets(cache["slot_targets"][hold_idx], pitch_targets[hold_idx], k[hold_idx]),
            _sample_weights(cache["slot_targets"][hold_idx], pitch_mask[hold_idx], k[hold_idx]),
        ),
        epochs=args.epochs,
        batch_size=64,
        shuffle=True,
        callbacks=callbacks,
        verbose=2,
    )

    hold_slots, hold_pitch, hold_cond = _predict(model, _inputs(cache, hold_idx))
    sweep = []
    for mode in DECODE_MODES:
        pred = _decode(hold_slots, hold_cond, mode)
        metrics = _metrics(holdout_tracks, _cached_prediction_map(cache, hold_idx, pred))
        macro = 0.5 * (metrics["solo"]["f1"] + metrics["comp"]["f1"])
        sweep.append({
            "mode": mode,
            "macro_f1": float(macro),
            "metrics": metrics,
            "cardinality": _cardinality_report(k[hold_idx], pred),
        })
    best = max(sweep, key=lambda r: (r["macro_f1"], r["metrics"]["global"]["f1"], r["cardinality"]["accuracy"]))
    decode_mode = best["mode"]
    model.save_weights(args.output_dir / "v101-string-query-attention.weights.h5")

    locked12 = tuple(validation[:12])
    floor, v88_threshold, enc86, enc87, model88 = _load_frozen_stack(args)
    print("evaluating V10.1 once on historical locked12")
    score_streams, records, x88, out88 = _represent_full(locked12, args.base_model, floor, enc86, enc87, model88)
    clusters, fused, assignment, sequence, mask, stats, target, exact, truncated = _cluster_data(locked12, records, x88, out88)
    spectral = _spectral_maps_for_runtime(locked12, clusters, records)
    locked_cache = {"sequence": sequence, "mask": mask, "stats": stats, "spectral": spectral}
    slots, _, cond = _predict(model, _inputs(locked_cache))
    pred_k = _decode(slots, cond, decode_mode)
    metrics = _metrics(locked12, _prediction_map(clusters, records, fused, pred_k))
    v88 = _metrics(locked12, _retained_predictions(records, fused, v88_threshold))
    oracle_k = np.minimum(np.asarray(exact, dtype=np.int32), SLOT_COUNT)
    oracle = _metrics(locked12, _prediction_map(clusters, records, fused, oracle_k))
    slot_truth, slot_assignment = _slot_targets_for_runtime_clusters(locked12, clusters, records)

    v100_report = json.loads(args.v100_report.read_text())
    v100_locked = v100_report["locked12"]
    v100 = v100_locked["v100_metrics"]
    v91 = v100_locked["v91_metrics"]

    report = {
        "schema_version": 1,
        "architecture": {
            "name": "V10.1 six-string query cross-attention with pitch supervision",
            "frozen_base": "V8.4 + V8.6 + V8.7 + V8.8",
            "base_trainable": False,
            "v101_trainable_parameters": int(model.count_params()),
            "runtime_inputs_use_annotations": False,
            "string_labels_training_only": True,
            "pitch_labels_training_only": True,
            "offset_stream_executed": False,
            "offset_weights_modified": False,
            "spectral_map_shape": [TIME_FRAMES, SPECTRAL_BANDS, SPECTRAL_CHANNELS],
            "tf_token_grid": list(token_shape[:2]),
            "tf_token_count": token_shape[2],
            "string_query_count": SLOT_COUNT,
            "shared_cross_attention_heads": 4,
            "ordinal_decoder_consumes_string_evidence": True,
            "cluster_window_ms": CLUSTER_WINDOW_MS,
            "maximum_verification_delay_ms": (MAX_HORIZON + CLUSTER_WINDOW_SAMPLES) * 1000.0 / SAMPLE_RATE,
        },
        "configuration": {
            "seed": args.seed,
            "decode_mode": decode_mode,
            "decode_mode_selected_on_locked_validation": False,
            "decode_modes_fixed_before_locked_eval": list(DECODE_MODES),
            "epochs_ran": len(history.history["loss"]),
            "loss_weights": loss_weights,
            "matching_tolerance_ms": TOLERANCE_MS,
            "pitch_normalization": f"midi/{PITCH_SCALE}",
        },
        "data": {
            "indexed_track_count": len(indexed),
            "full_train_track_count": len(train_split),
            "full_validation_track_count": len(validation),
            "cached_cluster_count": len(cache["target"]),
            "cache_shard_count": len(cache["shard_paths"]),
            "fit_track_count": len(fit_tracks),
            "holdout_track_count": len(holdout_tracks),
            "fit_cluster_count": len(fit_idx),
            "holdout_cluster_count": len(hold_idx),
            "holdout_composition_groups": list(holdout_groups),
            "composition_group_leakage": bool({group_stem(t) for t in fit_tracks} & {group_stem(t) for t in holdout_tracks}),
            "pitch_supervision": pitch_diag,
            "locked_string_assignment": slot_assignment,
            "locked_truncated_cluster_count": int(np.sum(truncated > 0)),
        },
        "training_history": {key: [float(v) for v in values] for key, values in history.history.items()},
        "holdout_pitch": _pitch_report(pitch_targets[hold_idx], hold_pitch, pitch_mask[hold_idx]),
        "holdout_decode_calibration": {"best": best, "sweep": sweep},
        "locked12": {
            "frozen_v88_baseline_metrics": v88,
            "v91_metrics": v91,
            "v100_metrics": v100,
            "v101_metrics": metrics,
            "oracle_exact_cardinality_metrics": oracle,
            "v101_cardinality": _cardinality_report(oracle_k, pred_k),
            "v101_string_slots": _slot_report_local(slot_truth, slots),
            "candidate_ceiling": _candidate_ceiling(locked12, score_streams, floor),
        },
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "decode_mode": decode_mode,
        "parameters": int(model.count_params()),
        "token_shape": token_shape,
        "pitch_supervision": pitch_diag,
        "holdout_pitch": report["holdout_pitch"],
        "v91": v91,
        "v100": v100,
        "v101": metrics,
        "oracle": oracle,
        "cardinality": report["locked12"]["v101_cardinality"],
        "string_slots": report["locked12"]["v101_string_slots"],
    }, indent=2, sort_keys=True))
    return report


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("dataset_dir", nargs="?", type=Path, default=ROOT / "data" / "GuitarSet")
    p.add_argument("--base-model", type=Path, required=True)
    p.add_argument("--v86-weights", type=Path, required=True)
    p.add_argument("--v86-report", type=Path, required=True)
    p.add_argument("--v87-weights", type=Path, required=True)
    p.add_argument("--v87-report", type=Path, required=True)
    p.add_argument("--v88-weights", type=Path, required=True)
    p.add_argument("--v88-report", type=Path, required=True)
    p.add_argument("--v100-report", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED_V101)
    return p


def main(argv: Optional[Sequence[str]] = None):
    args = parser().parse_args(argv)
    train_eval(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
