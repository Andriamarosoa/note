"""V10.2 competitive source-time assignment for anonymous birth cardinality.

The V10.1 audit showed that its 78.24% all-cluster exact string-vector score
collapses to 13.83% on true K>=2 clusters, while about 20% of poly clusters
already have the correct thresholded string count but are then misdecoded by
the ordinal chain.  It also showed that high-K groups are often micro-sequences
spread across 20-35 ms rather than perfectly simultaneous attacks.

V10.2 therefore changes the decision representation rather than tuning a
threshold:

* preserve all 23 spectral time frames (frequency is compressed only);
* assign every time-frequency token competitively to six physical strings plus
  a background source (softmax across sources for every token);
* supervise each active string not only with occupancy and pitch, but also with
  its within-cluster birth-time distribution;
* derive an exact differentiable P(K=0..6) from the six string probabilities via
  a Poisson-binomial distribution;
* optimize that same structured distribution globally and again on K>=2 only;
* keep the old ordinal chain only as a low-weight auxiliary target and never use
  it for V10.2 locked decoding.

String, pitch and birth-time labels are training-only. Runtime inputs remain the
frozen audio-derived candidate/context features plus the causal V10 spectral
map. The 40 ms grouping protocol and maximum acoustic horizon are unchanged.
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
for _p in (ROOT, SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from causal_note.guitarset import ALLOWED_PLAYERS, SAMPLE_RATE, SLOT_COUNT, index_guitarset
from scripts.train_boundaries import group_stem
from scripts.train_v86_state_transition_proposals import MAX_HORIZON, TOLERANCE_MS
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
    FRAME_LENGTH,
    FRAME_STEP,
    PRE_SAMPLES,
    SPECTRAL_BANDS,
    SPECTRAL_CHANNELS,
    TIME_FRAMES,
    _cached_prediction_map,
    _load_spectral_caches,
    _spectral_maps_for_runtime,
)
from scripts import train_v101_string_query_attention as v101


DEFAULT_SEED_V102 = 10231
SOURCE_COUNT = SLOT_COUNT + 1
BACKGROUND_SOURCE = SLOT_COUNT
TOKEN_DIM = 96
TIME_TARGET_SIGMA_FRAMES = 1.5
DECODE_MODES = (
    "structured_pb_argmax",
    "structured_pb_expected_round",
    "slot_expected_round",
)

FRAME_CENTER_SAMPLES = (
    np.arange(TIME_FRAMES, dtype=np.float64) * FRAME_STEP
    + FRAME_LENGTH / 2.0
    - PRE_SAMPLES
)
FRAME_CENTER_MS = FRAME_CENTER_SAMPLES * 1000.0 / SAMPLE_RATE


class V102Error(RuntimeError):
    pass


def _nearest_cluster_for_event(
    onset: int,
    flat_samples: np.ndarray,
    flat_cids: np.ndarray,
) -> Optional[Tuple[int, int]]:
    """Exact nearest candidate-cluster lookup with V10.1 tie-breaking.

    V10.1 chooses min((distance, global_cluster_id)) among clusters with at least
    one candidate within LOCAL_RADIUS_SAMPLES.  Searching only samples in that
    radius is equivalent and avoids event x cluster quadratic work.
    """
    left = int(np.searchsorted(flat_samples, onset - LOCAL_RADIUS_SAMPLES, side="left"))
    right = int(np.searchsorted(flat_samples, onset + LOCAL_RADIUS_SAMPLES, side="right"))
    if right <= left:
        return None
    samples = flat_samples[left:right]
    cids = flat_cids[left:right]
    dist = np.abs(samples.astype(np.int64) - int(onset))
    best_dist = int(np.min(dist))
    best_cid = int(np.min(cids[dist == best_dist]))
    return best_dist, best_cid


def _time_distribution(relative_sample: float) -> np.ndarray:
    distance_frames = (FRAME_CENTER_SAMPLES - float(relative_sample)) / float(FRAME_STEP)
    weights = np.exp(-0.5 * (distance_frames / TIME_TARGET_SIGMA_FRAMES) ** 2)
    total = float(np.sum(weights))
    if not math.isfinite(total) or total <= 0.0:
        raise V102Error("invalid time-target normalization")
    return (weights / total).astype(np.float32)


def _derive_supervision(
    members: Sequence[str],
    candidate_samples: Sequence[np.ndarray],
    dataset_dir: Path,
    *,
    expected_slot_targets: Optional[np.ndarray] = None,
):
    """Assign GuitarSet note events to candidate clusters and build targets."""
    if len(members) != len(candidate_samples):
        raise V102Error("members/candidate_samples length mismatch")
    indexed = tuple(t for t in index_guitarset(dataset_dir) if t.player_id in ALLOWED_PLAYERS)
    by_member_track = {t.annotation_member: t for t in indexed}
    by_member_rows: Dict[str, List[int]] = defaultdict(list)
    for cid, member in enumerate(members):
        by_member_rows[str(member)].append(cid)

    n = len(members)
    pitch = np.zeros((n, SLOT_COUNT), dtype=np.float32)
    mask = np.zeros((n, SLOT_COUNT), dtype=np.float32)
    time_targets = np.zeros((n, SLOT_COUNT, TIME_FRAMES), dtype=np.float32)
    time_sample = np.full((n, SLOT_COUNT), np.nan, dtype=np.float32)

    assigned = 0
    unassigned = 0
    collisions = 0
    outside_frame_centers = 0
    distances: List[int] = []
    relative_samples: List[float] = []
    midi_values: List[float] = []

    for member, ids in by_member_rows.items():
        track = by_member_track.get(member)
        if track is None:
            raise V102Error(f"cache member missing from GuitarSet index: {member}")

        sample_parts = []
        cid_parts = []
        starts: Dict[int, int] = {}
        for cid in ids:
            samples = np.asarray(candidate_samples[cid], dtype=np.int32)
            if not len(samples):
                continue
            starts[cid] = int(np.min(samples))
            sample_parts.append(samples)
            cid_parts.append(np.full(len(samples), cid, dtype=np.int32))
        if not sample_parts:
            continue
        flat_samples = np.concatenate(sample_parts)
        flat_cids = np.concatenate(cid_parts)
        order = np.argsort(flat_samples, kind="stable")
        flat_samples = flat_samples[order]
        flat_cids = flat_cids[order]

        for slot, onset, midi in v101._pitch_events(track):
            nearest = _nearest_cluster_for_event(int(onset), flat_samples, flat_cids)
            if nearest is None:
                unassigned += 1
                continue
            dist, cid = nearest
            if mask[cid, slot] > 0.5:
                collisions += 1
                # Same-string double births inside 40 ms are not representable by
                # the six-slot factorization. Keep the closest-to-candidate event.
                old_dist = abs(float(time_sample[cid, slot]) - float(starts[cid]))
                if dist >= old_dist:
                    continue
            relative = float(onset - starts[cid])
            if relative < FRAME_CENTER_SAMPLES[0] or relative > FRAME_CENTER_SAMPLES[-1]:
                outside_frame_centers += 1
            pitch[cid, slot] = float(midi / v101.PITCH_SCALE)
            mask[cid, slot] = 1.0
            time_targets[cid, slot] = _time_distribution(relative)
            time_sample[cid, slot] = float(relative)
            assigned += 1
            distances.append(int(dist))
            relative_samples.append(relative)
            midi_values.append(float(midi))

    diag = {
        "assigned_events": assigned,
        "unassigned_events": unassigned,
        "assigned_fraction": assigned / (assigned + unassigned) if assigned + unassigned else None,
        "same_slot_collisions": collisions,
        "outside_frame_center_range": outside_frame_centers,
        "outside_frame_center_fraction": outside_frame_centers / assigned if assigned else None,
        "nearest_candidate_distance_median_ms": float(np.median(distances) * 1000.0 / SAMPLE_RATE) if distances else None,
        "nearest_candidate_distance_p90_ms": float(np.percentile(distances, 90) * 1000.0 / SAMPLE_RATE) if distances else None,
        "relative_birth_time_median_ms": float(np.median(relative_samples) * 1000.0 / SAMPLE_RATE) if relative_samples else None,
        "relative_birth_time_p10_ms": float(np.percentile(relative_samples, 10) * 1000.0 / SAMPLE_RATE) if relative_samples else None,
        "relative_birth_time_p90_ms": float(np.percentile(relative_samples, 90) * 1000.0 / SAMPLE_RATE) if relative_samples else None,
        "midi_min": min(midi_values) if midi_values else None,
        "midi_max": max(midi_values) if midi_values else None,
        "midi_mean": float(np.mean(midi_values)) if midi_values else None,
        "time_target_sigma_frames": TIME_TARGET_SIGMA_FRAMES,
        "frame_center_ms": [float(x) for x in FRAME_CENTER_MS],
        "indexing": "global_cluster_row",
    }

    if expected_slot_targets is not None:
        truth = np.asarray(expected_slot_targets, dtype=np.float32) > 0.5
        seen = mask > 0.5
        if truth.shape != seen.shape:
            raise V102Error("slot supervision shape mismatch")
        agreement = float(np.mean(truth == seen))
        active_coverage = float(np.mean(seen[truth])) if np.any(truth) else 1.0
        diag["slot_mask_agreement"] = agreement
        diag["active_slot_time_coverage"] = active_coverage
        if agreement < 0.999 or active_coverage < 0.995:
            raise V102Error(
                f"source-time assignment does not reproduce slot supervision: "
                f"agreement={agreement} active={active_coverage}"
            )
    return pitch, mask, time_targets, time_sample, diag


def _poisson_binomial_tensor(p):
    import tensorflow as tf

    p = tf.clip_by_value(p, 1e-6, 1.0 - 1e-6)
    batch = tf.shape(p)[0]
    dist = tf.ones((batch, 1), dtype=p.dtype)
    for slot in range(SLOT_COUNT):
        ps = p[:, slot:slot + 1]
        left = tf.concat([dist * (1.0 - ps), tf.zeros((batch, 1), dtype=p.dtype)], axis=1)
        right = tf.concat([tf.zeros((batch, 1), dtype=p.dtype), dist * ps], axis=1)
        dist = left + right
    return dist


def _build_model():
    try:
        import tensorflow as tf
        from tensorflow import keras
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required") from exc

    scaffold = _build_cluster_model()
    candidate_hidden = scaffold.get_layer("cluster_hidden2").output
    candidate_context = keras.layers.Dense(TOKEN_DIM, activation="relu", name="candidate_context")(candidate_hidden)

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

    # Preserve all 23 time frames. Only the frequency axis is compressed.
    x = keras.layers.Conv2D(32, (3, 5), strides=(1, 2), padding="same", activation="relu", name="st_conv1")(x)
    x = keras.layers.Conv2D(64, (3, 3), strides=(1, 2), padding="same", activation="relu", name="st_conv2")(x)
    x = keras.layers.Conv2D(TOKEN_DIM, (3, 3), strides=(1, 2), padding="same", activation="relu", name="st_conv3")(x)
    token_freq = int(math.ceil(math.ceil(math.ceil(SPECTRAL_BANDS / 2.0) / 2.0) / 2.0))
    token_count = TIME_FRAMES * token_freq
    tokens = keras.layers.Reshape((token_count, TOKEN_DIM), name="tf_tokens")(x)
    tokens = keras.layers.LayerNormalization(name="tf_token_norm")(tokens)
    tokens = keras.layers.Dense(TOKEN_DIM, activation="relu", name="tf_token_projection")(tokens)
    keys = keras.layers.Dense(TOKEN_DIM, use_bias=False, name="tf_assignment_keys")(tokens)

    source_scores = []
    source_queries = []
    for source in range(SOURCE_COUNT):
        query = keras.layers.Dense(TOKEN_DIM, activation="relu", name=f"source_{source}_query")(candidate_context)
        source_queries.append(query)
        score = keras.layers.Lambda(
            lambda z: tf.einsum("btd,bd->bt", z[0], z[1]) / math.sqrt(float(TOKEN_DIM)),
            name=f"source_{source}_score",
        )([keys, query])
        source_scores.append(score)
    score_stack = keras.layers.Lambda(lambda z: tf.stack(z, axis=-1), name="source_score_stack")(source_scores)
    assignment = keras.layers.Softmax(axis=-1, name="competitive_source_assignment")(score_stack)
    assignment_grid = keras.layers.Reshape((TIME_FRAMES, token_freq, SOURCE_COUNT), name="source_assignment_grid")(assignment)

    slot_features = []
    string_outputs = []
    pitch_outputs = []
    time_outputs = []
    for slot in range(SLOT_COUNT):
        weights = keras.layers.Lambda(lambda a, s=slot: a[:, :, s], name=f"string_{slot}_token_weights")(assignment)
        latent = keras.layers.Lambda(
            lambda z: tf.reduce_sum(z[0] * z[1][:, :, None], axis=1)
            / (tf.reduce_sum(z[1], axis=1, keepdims=True) + 1e-6),
            name=f"string_{slot}_source_pool",
        )([tokens, weights])
        feature = keras.layers.Concatenate(name=f"string_{slot}_fusion")([
            candidate_context,
            source_queries[slot],
            latent,
        ])
        feature = keras.layers.Dense(112, activation="relu", name=f"string_{slot}_hidden")(feature)
        feature = keras.layers.Dropout(0.10, name=f"string_{slot}_dropout")(feature)
        slot_features.append(feature)
        string_outputs.append(keras.layers.Dense(1, activation="sigmoid", name=f"string_{slot}")(feature))
        pitch_outputs.append(keras.layers.Dense(1, activation="sigmoid", name=f"pitch_{slot}")(feature))

        time_mass = keras.layers.Lambda(
            lambda a, s=slot: tf.reduce_sum(a[:, :, :, s], axis=2),
            name=f"string_{slot}_time_mass",
        )(assignment_grid)
        time_dist = keras.layers.Lambda(
            lambda t: t / (tf.reduce_sum(t, axis=1, keepdims=True) + 1e-6),
            name=f"time_{slot}",
        )(time_mass)
        time_outputs.append(time_dist)

    string_stack = keras.layers.Concatenate(name="string_birth_vector")(string_outputs)
    slot_count = keras.layers.Lambda(
        lambda t: tf.reduce_sum(t, axis=1, keepdims=True) / float(SLOT_COUNT),
        name="slot_count",
    )(string_stack)
    structured_count = keras.layers.Lambda(_poisson_binomial_tensor, name="structured_count")(string_stack)
    poly_count = keras.layers.Lambda(lambda t: tf.identity(t), name="poly_count")(structured_count)

    global_tokens = keras.layers.Concatenate(name="tf_global_pool")([
        keras.layers.GlobalAveragePooling1D(name="tf_global_average")(tokens),
        keras.layers.GlobalMaxPooling1D(name="tf_global_max")(tokens),
    ])
    all_sources = keras.layers.Concatenate(name="all_string_source_features")(slot_features)
    count_aux = keras.layers.Concatenate(name="ordinal_aux_fusion")([candidate_context, global_tokens, all_sources])
    count_aux = keras.layers.LayerNormalization(name="ordinal_aux_norm")(count_aux)
    count_aux = keras.layers.Dense(192, activation="relu", name="ordinal_aux_hidden1")(count_aux)
    count_aux = keras.layers.Dropout(0.15, name="ordinal_aux_dropout")(count_aux)
    count_aux = keras.layers.Dense(96, activation="relu", name="ordinal_aux_hidden2")(count_aux)
    ordinal_outputs = [
        keras.layers.Dense(1, activation="sigmoid", name=f"ge{stage}")(count_aux)
        for stage in range(1, ORDINAL_STAGES + 1)
    ]

    outputs = {f"string_{slot}": out for slot, out in enumerate(string_outputs)}
    outputs.update({f"pitch_{slot}": out for slot, out in enumerate(pitch_outputs)})
    outputs.update({f"time_{slot}": out for slot, out in enumerate(time_outputs)})
    outputs["slot_count"] = slot_count
    outputs["structured_count"] = structured_count
    outputs["poly_count"] = poly_count
    outputs.update({f"ge{stage}": out for stage, out in enumerate(ordinal_outputs, start=1)})

    loss = {f"string_{slot}": "binary_crossentropy" for slot in range(SLOT_COUNT)}
    loss.update({f"pitch_{slot}": "mse" for slot in range(SLOT_COUNT)})
    loss.update({f"time_{slot}": keras.losses.KLDivergence() for slot in range(SLOT_COUNT)})
    loss["slot_count"] = "mse"
    loss["structured_count"] = "categorical_crossentropy"
    loss["poly_count"] = "categorical_crossentropy"
    loss.update({f"ge{stage}": "binary_crossentropy" for stage in range(1, ORDINAL_STAGES + 1)})

    loss_weights = {f"string_{slot}": 0.60 for slot in range(SLOT_COUNT)}
    loss_weights.update({f"pitch_{slot}": 0.15 for slot in range(SLOT_COUNT)})
    loss_weights.update({f"time_{slot}": 0.35 for slot in range(SLOT_COUNT)})
    loss_weights["slot_count"] = 0.20
    loss_weights["structured_count"] = 1.30
    loss_weights["poly_count"] = 0.90
    loss_weights.update({f"ge{stage}": 0.15 * math.sqrt(stage) for stage in range(1, ORDINAL_STAGES + 1)})

    model = keras.Model(
        {
            "candidate_set": scaffold.input["candidate_set"],
            "candidate_mask": scaffold.input["candidate_mask"],
            "cluster_stats": scaffold.input["cluster_stats"],
            "spectral_map": spectral,
        },
        outputs,
        name="v102_competitive_source_time_assignment",
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=2e-4), loss=loss, loss_weights=loss_weights)
    return model, loss_weights, (TIME_FRAMES, token_freq, token_count)


def _count_weights(k: np.ndarray) -> np.ndarray:
    k = np.asarray(k, dtype=np.int32)
    counts = np.bincount(np.minimum(k, SLOT_COUNT), minlength=SLOT_COUNT + 1).astype(np.float64)
    weights = np.ones(SLOT_COUNT + 1, dtype=np.float64)
    nz = counts > 0
    weights[nz] = np.sqrt(len(k) / ((SLOT_COUNT + 1) * counts[nz]))
    out = np.clip(weights[np.minimum(k, SLOT_COUNT)], 0.35, 4.0)
    return (out / np.mean(out)).astype(np.float32)


def _poly_count_weights(k: np.ndarray) -> np.ndarray:
    k = np.asarray(k, dtype=np.int32)
    out = np.zeros(len(k), dtype=np.float64)
    active = k >= 2
    if not np.any(active):
        return out.astype(np.float32)
    counts = np.bincount(np.minimum(k[active], SLOT_COUNT), minlength=SLOT_COUNT + 1).astype(np.float64)
    for value in range(2, SLOT_COUNT + 1):
        if counts[value] > 0:
            out[k == value] = math.sqrt(float(np.sum(active)) / (5.0 * counts[value]))
    mean_active = float(np.mean(out[active]))
    if mean_active > 0:
        out[active] /= mean_active
    out[active] = np.clip(out[active], 0.35, 5.0)
    return out.astype(np.float32)


def _targets(slot_targets, pitch_targets, time_targets, k):
    k = np.minimum(np.asarray(k, dtype=np.int32), SLOT_COUNT)
    one_hot = np.eye(SLOT_COUNT + 1, dtype=np.float32)[k]
    result = {f"string_{slot}": slot_targets[:, slot].reshape(-1, 1) for slot in range(SLOT_COUNT)}
    result.update({f"pitch_{slot}": pitch_targets[:, slot].reshape(-1, 1) for slot in range(SLOT_COUNT)})
    result.update({f"time_{slot}": time_targets[:, slot, :] for slot in range(SLOT_COUNT)})
    result["slot_count"] = (k.astype(np.float32) / float(SLOT_COUNT)).reshape(-1, 1)
    result["structured_count"] = one_hot
    result["poly_count"] = one_hot
    result.update({f"ge{stage}": (k >= stage).astype(np.float32).reshape(-1, 1) for stage in range(1, ORDINAL_STAGES + 1)})
    return result


def _sample_weights(slot_targets, mask, k):
    result = {f"string_{slot}": v101._soft_binary_weights(slot_targets[:, slot]) for slot in range(SLOT_COUNT)}
    for slot in range(SLOT_COUNT):
        active = np.asarray(mask[:, slot], dtype=np.float32)
        prevalence = max(float(np.mean(active)), 1e-6)
        active_weight = min(6.0, 1.0 / math.sqrt(prevalence))
        result[f"pitch_{slot}"] = active * active_weight
        result[f"time_{slot}"] = active * active_weight
    cw = _count_weights(k)
    result["slot_count"] = cw
    result["structured_count"] = cw
    result["poly_count"] = _poly_count_weights(k)
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
    times = np.stack([np.asarray(raw[f"time_{slot}"]) for slot in range(SLOT_COUNT)], axis=1)
    count = np.asarray(raw["structured_count"], dtype=np.float64)
    conditionals = np.stack([np.asarray(raw[f"ge{stage}"]).reshape(-1) for stage in range(1, ORDINAL_STAGES + 1)], axis=1)
    return slots.astype(np.float64), pitches.astype(np.float64), times.astype(np.float64), count, conditionals.astype(np.float64)


def _decode(slots, count_probs, mode):
    if mode == "structured_pb_argmax":
        return np.argmax(count_probs, axis=1).astype(np.int32)
    if mode == "structured_pb_expected_round":
        expected = count_probs @ np.arange(SLOT_COUNT + 1, dtype=np.float64)
        return np.clip(np.floor(expected + 0.5), 0, SLOT_COUNT).astype(np.int32)
    if mode == "slot_expected_round":
        expected = np.sum(slots, axis=1)
        return np.clip(np.floor(expected + 0.5), 0, SLOT_COUNT).astype(np.int32)
    raise ValueError(mode)


def _time_report(time_targets, time_sample, time_predictions, mask):
    active = np.asarray(mask) > 0.5
    pred = np.asarray(time_predictions, dtype=np.float64)
    true_samples = np.asarray(time_sample, dtype=np.float64)
    expected_samples = np.sum(pred * FRAME_CENTER_SAMPLES[None, None, :], axis=2)
    result = {"active_count": int(np.sum(active))}
    if np.any(active):
        err_ms = np.abs(expected_samples[active] - true_samples[active]) * 1000.0 / SAMPLE_RATE
        result.update({
            "mean_absolute_error_ms": float(np.mean(err_ms)),
            "median_absolute_error_ms": float(np.median(err_ms)),
            "p90_absolute_error_ms": float(np.percentile(err_ms, 90)),
        })
        target = np.asarray(time_targets, dtype=np.float64)
        kl = np.sum(target[active] * (np.log(np.maximum(target[active], 1e-9)) - np.log(np.maximum(pred[active], 1e-9))), axis=1)
        result["mean_kl"] = float(np.mean(kl))
    per_string = {}
    for slot in range(SLOT_COUNT):
        m = active[:, slot]
        if np.any(m):
            err = np.abs(expected_samples[m, slot] - true_samples[m, slot]) * 1000.0 / SAMPLE_RATE
            per_string[str(slot)] = {
                "active_count": int(np.sum(m)),
                "mae_ms": float(np.mean(err)),
                "median_ms": float(np.median(err)),
            }
        else:
            per_string[str(slot)] = {"active_count": 0, "mae_ms": None, "median_ms": None}
    result["per_string"] = per_string
    return result


def _poly_by_k(k, pred):
    k = np.asarray(k, dtype=np.int32)
    pred = np.asarray(pred, dtype=np.int32)
    out = {}
    for value in range(2, SLOT_COUNT + 1):
        m = k == value
        out[str(value)] = {
            "clusters": int(np.sum(m)),
            "exact_accuracy": float(np.mean(pred[m] == value)) if np.any(m) else None,
            "mae": float(np.mean(np.abs(pred[m] - value))) if np.any(m) else None,
            "undercount_fraction": float(np.mean(pred[m] < value)) if np.any(m) else None,
            "overcount_fraction": float(np.mean(pred[m] > value)) if np.any(m) else None,
        }
    m = k >= 2
    out["poly"] = {
        "clusters": int(np.sum(m)),
        "exact_accuracy": float(np.mean(pred[m] == k[m])) if np.any(m) else None,
        "mae": float(np.mean(np.abs(pred[m] - k[m]))) if np.any(m) else None,
    }
    return out


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
        raise V102Error("train/validation leakage")
    if set(cache["track_members"]) != train_members:
        raise V102Error("V10 spectral caches do not exactly cover the full train split")

    candidate_samples, reconstruction = _reconstruct_candidates(cache)
    pitch_targets, time_mask, time_targets, time_sample, supervision_diag = _derive_supervision(
        [str(x) for x in cache["members"]],
        candidate_samples,
        args.dataset_dir,
        expected_slot_targets=cache["slot_targets"],
    )
    supervision_diag["cluster_reconstruction"] = reconstruction

    fit_tracks, holdout_tracks, holdout_groups = _split_train_groups(train_split, args.seed)
    fit_set = {t.annotation_member for t in fit_tracks}
    holdout_set = {t.annotation_member for t in holdout_tracks}
    fit_idx = np.asarray([i for i, m in enumerate(cache["members"]) if str(m) in fit_set], dtype=np.int64)
    hold_idx = np.asarray([i for i, m in enumerate(cache["members"]) if str(m) in holdout_set], dtype=np.int64)
    k = np.minimum(np.asarray(cache["exact"], dtype=np.int32), SLOT_COUNT)

    model, loss_weights, token_shape = _build_model()
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=2e-5),
    ]
    print(
        f"training V10.2 source-time assignment fit={len(fit_idx)} holdout={len(hold_idx)} "
        f"tracks={len(train_split)} poly_fit={int(np.sum(k[fit_idx] >= 2))}"
    )
    history = model.fit(
        _inputs(cache, fit_idx),
        _targets(cache["slot_targets"][fit_idx], pitch_targets[fit_idx], time_targets[fit_idx], k[fit_idx]),
        sample_weight=_sample_weights(cache["slot_targets"][fit_idx], time_mask[fit_idx], k[fit_idx]),
        validation_data=(
            _inputs(cache, hold_idx),
            _targets(cache["slot_targets"][hold_idx], pitch_targets[hold_idx], time_targets[hold_idx], k[hold_idx]),
            _sample_weights(cache["slot_targets"][hold_idx], time_mask[hold_idx], k[hold_idx]),
        ),
        epochs=args.epochs,
        batch_size=64,
        shuffle=True,
        callbacks=callbacks,
        verbose=2,
    )

    hold_slots, hold_pitch, hold_time, hold_count, _ = _predict(model, _inputs(cache, hold_idx))
    sweep = []
    for mode in DECODE_MODES:
        pred = _decode(hold_slots, hold_count, mode)
        metrics = v101._metrics(holdout_tracks, _cached_prediction_map(cache, hold_idx, pred))
        macro = 0.5 * (metrics["solo"]["f1"] + metrics["comp"]["f1"])
        sweep.append({
            "mode": mode,
            "macro_f1": float(macro),
            "metrics": metrics,
            "cardinality": _cardinality_report(k[hold_idx], pred),
            "poly_by_k": _poly_by_k(k[hold_idx], pred),
        })
    best = max(
        sweep,
        key=lambda r: (
            r["macro_f1"],
            r["poly_by_k"]["poly"]["exact_accuracy"],
            r["metrics"]["global"]["f1"],
        ),
    )
    decode_mode = best["mode"]
    model.save_weights(args.output_dir / "v102-source-time-assignment.weights.h5")

    locked12 = tuple(validation[:12])
    floor, v88_threshold, enc86, enc87, model88 = _load_frozen_stack(args)
    print("evaluating V10.2 once on historical locked12")
    score_streams, records, x88, out88 = _represent_full(locked12, args.base_model, floor, enc86, enc87, model88)
    clusters, fused, assignment, sequence, mask, stats, target, exact, truncated = _cluster_data(locked12, records, x88, out88)
    spectral = _spectral_maps_for_runtime(locked12, clusters, records)
    locked_cache = {"sequence": sequence, "mask": mask, "stats": stats, "spectral": spectral}
    locked_slots, locked_pitch, locked_time, locked_count, locked_cond = _predict(model, _inputs(locked_cache))
    pred_k = _decode(locked_slots, locked_count, decode_mode)
    metrics = v101._metrics(locked12, _prediction_map(clusters, records, fused, pred_k))
    v88 = v101._metrics(locked12, v101._retained_predictions(records, fused, v88_threshold))
    oracle_k = np.minimum(np.asarray(exact, dtype=np.int32), SLOT_COUNT)
    oracle = v101._metrics(locked12, _prediction_map(clusters, records, fused, oracle_k))
    slot_truth, slot_assignment = _slot_targets_for_runtime_clusters(locked12, clusters, records)

    locked_members = [str(cluster["member"]) for cluster in clusters]
    locked_candidate_samples = [
        np.asarray([int(records[i]["sample"]) for i in cluster["indices"]], dtype=np.int32)
        for cluster in clusters
    ]
    locked_pitch_targets, locked_time_mask, locked_time_targets, locked_time_sample, locked_supervision_diag = _derive_supervision(
        locked_members,
        locked_candidate_samples,
        args.dataset_dir,
        expected_slot_targets=slot_truth,
    )

    v101_report = json.loads(args.v101_report.read_text())
    v101_locked = v101_report["locked12"]

    report = {
        "schema_version": 1,
        "architecture": {
            "name": "V10.2 competitive six-string plus background source-time assignment",
            "frozen_base": "V8.4 + V8.6 + V8.7 + V8.8",
            "base_trainable": False,
            "v102_trainable_parameters": int(model.count_params()),
            "runtime_inputs_use_annotations": False,
            "string_labels_training_only": True,
            "pitch_labels_training_only": True,
            "birth_time_labels_training_only": True,
            "offset_stream_executed": False,
            "offset_weights_modified": False,
            "spectral_map_shape": [TIME_FRAMES, SPECTRAL_BANDS, SPECTRAL_CHANNELS],
            "source_count": SOURCE_COUNT,
            "background_source": True,
            "competition_axis": "source for every time-frequency token",
            "time_frames_preserved": TIME_FRAMES,
            "tf_token_grid": list(token_shape[:2]),
            "tf_token_count": token_shape[2],
            "count_distribution": "Poisson-binomial from six string probabilities",
            "poly_loss_reuses_same_structured_distribution": True,
            "ordinal_is_auxiliary_only": True,
            "cluster_window_ms": CLUSTER_WINDOW_MS,
            "maximum_verification_delay_ms": (MAX_HORIZON + CLUSTER_WINDOW_SAMPLES) * 1000.0 / SAMPLE_RATE,
        },
        "configuration": {
            "seed": args.seed,
            "decode_mode": decode_mode,
            "decode_mode_selected_on_locked_validation": False,
            "decode_modes_fixed_before_locked_eval": list(DECODE_MODES),
            "ordinal_decode_allowed": False,
            "epochs_ran": len(history.history["loss"]),
            "loss_weights": loss_weights,
            "matching_tolerance_ms": TOLERANCE_MS,
            "time_target_sigma_frames": TIME_TARGET_SIGMA_FRAMES,
            "pitch_normalization": f"midi/{v101.PITCH_SCALE}",
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
            "fit_poly_cluster_count": int(np.sum(k[fit_idx] >= 2)),
            "holdout_poly_cluster_count": int(np.sum(k[hold_idx] >= 2)),
            "holdout_composition_groups": list(holdout_groups),
            "composition_group_leakage": bool({group_stem(t) for t in fit_tracks} & {group_stem(t) for t in holdout_tracks}),
            "train_supervision": supervision_diag,
            "locked_supervision": locked_supervision_diag,
            "locked_string_assignment": slot_assignment,
            "locked_truncated_cluster_count": int(np.sum(truncated > 0)),
        },
        "training_history": {key: [float(v) for v in values] for key, values in history.history.items()},
        "holdout_pitch": v101._pitch_report(pitch_targets[hold_idx], hold_pitch, time_mask[hold_idx]),
        "holdout_time": _time_report(time_targets[hold_idx], time_sample[hold_idx], hold_time, time_mask[hold_idx]),
        "holdout_decode_calibration": {"best": best, "sweep": sweep},
        "locked12": {
            "frozen_v88_baseline_metrics": v88,
            "v91_metrics": v101_locked["v91_metrics"],
            "v100_metrics": v101_locked["v100_metrics"],
            "v101_metrics": v101_locked["v101_metrics"],
            "v102_metrics": metrics,
            "oracle_exact_cardinality_metrics": oracle,
            "v102_cardinality": _cardinality_report(oracle_k, pred_k),
            "v102_poly_by_k": _poly_by_k(oracle_k, pred_k),
            "v102_string_slots": v101._slot_report_local(slot_truth, locked_slots),
            "v102_pitch": v101._pitch_report(locked_pitch_targets, locked_pitch, locked_time_mask),
            "v102_time": _time_report(locked_time_targets, locked_time_sample, locked_time, locked_time_mask),
            "candidate_ceiling": v101._candidate_ceiling(locked12, score_streams, floor),
        },
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "decode_mode": decode_mode,
        "parameters": int(model.count_params()),
        "token_shape": token_shape,
        "train_supervision": supervision_diag,
        "holdout_time": report["holdout_time"],
        "v101": v101_locked["v101_metrics"],
        "v102": metrics,
        "oracle": oracle,
        "cardinality": report["locked12"]["v102_cardinality"],
        "poly_by_k": report["locked12"]["v102_poly_by_k"],
        "string_slots": report["locked12"]["v102_string_slots"],
        "locked_time": report["locked12"]["v102_time"],
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
    p.add_argument("--v101-report", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED_V102)
    return p


def main(argv: Optional[Sequence[str]] = None):
    args = parser().parse_args(argv)
    train_eval(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
