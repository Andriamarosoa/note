"""V23.1 categorical-cardinality conditioned dense transport decoder.

This is the implementation authorized by the V23 pre-audit.  It removes the
six independent Bernoulli existence decisions used by V19/V17.3 and makes K a
first-class categorical variable.

Architecture:
  23x64 dense conv latent
    -> categorical P(K=0..6)
    -> straight-through exact-K row mass
    -> differentiable capped transport over all 1472 dense cells
    -> K active object queries
    -> shared candidate / TF evidence decoder
    -> permutation-invariant set loss

The candidate memory is auxiliary evidence only.  It never defines object
identity.  No threshold is tuned and Locked12 is never indexed or evaluated.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for _p in (ROOT, SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from causal_note.guitarset import SLOT_COUNT
from scripts import train_v100_spectral_string_slots as v100
from scripts import train_v102_source_time_assignment as v102
from scripts import train_v130_causal_event_set_decoder as v130
from scripts import train_v171_controlled_assignment_ab as v171
from scripts import train_v172_mass_preserving_exchangeable as v172
from scripts import train_v176_shared_set_decoder as v176
from scripts import train_v190_dense_birth_centers as v190

DEFAULT_SEED = 16061
EVENT_QUERIES = SLOT_COUNT
CARDINALITY_CLASSES = EVENT_QUERIES + 1
MODEL_KEY = "v231_cardinality_transport"
PRED_KEY = "pred231_cardinality_transport"
BASE_ARM = "mass_permutation"
QUERY_DIM = 96
CENTER_CELLS = v100.TIME_FRAMES * v100.SPECTRAL_BANDS
EPS = 1e-6

# Fixed before any V23.1 training.  These are architectural weights, not tuned
# thresholds.  The categorical head is deliberately the dominant new signal.
CARDINALITY_WEIGHT = 1.00
CENTER_MAP_WEIGHT = 0.15
TRANSPORT_COVERAGE_WEIGHT = 0.35
SET_PRESENCE_WEIGHT = 0.20
SET_TIME_WEIGHT = 0.30
SET_CANDIDATE_WEIGHT = 0.25
TRANSPORT_ITERS = 6
TRANSPORT_TEMPERATURE = 0.65

_LAST_CENTER_TARGETS = None
_LAST_CENTER_ELIGIBLE = None

SURVIVAL_MATRIX = np.asarray(
    [[1.0 if q < k else 0.0 for q in range(EVENT_QUERIES)] for k in range(CARDINALITY_CLASSES)],
    dtype=np.float32,
)


class V231Error(RuntimeError):
    pass


def _input_by_name(model, name):
    return v190._input_by_name(model, name)


def _other_union(tf, assignment, q):
    return v190._other_union(tf, assignment, q)


def _capped_transport_tf(tf, scores, row_mass):
    """Differentiable row-normalized transport with repeated column capping.

    Forward row_mass is exactly binary via a straight-through categorical K
    gate, so total transported mass is exactly K.  Repeated column capping
    discourages two active objects from owning the same dense cell.
    """
    score = tf.cast(scores, tf.float32) / tf.constant(TRANSPORT_TEMPERATURE, tf.float32)
    mass = tf.cast(row_mass, tf.float32)
    z = tf.nn.softmax(score, axis=-1) * mass[:, :, None]
    for _ in range(TRANSPORT_ITERS):
        col = tf.reduce_sum(z, axis=1, keepdims=True)
        z = z * tf.minimum(1.0, 1.0 / (col + EPS))
        row = tf.reduce_sum(z, axis=2, keepdims=True)
        wanted = mass[:, :, None]
        z = tf.where(wanted > 0.0, z * wanted / (row + EPS), tf.zeros_like(z))
    return z


def _set_loss():
    import tensorflow as tf
    from tensorflow import keras

    perm = tf.constant(v172.PERMUTATION_MATRICES, dtype=tf.float32)

    class CardinalityConditionedSetLoss(keras.losses.Loss):
        def __init__(self):
            super().__init__(name="v231_cardinality_conditioned_set_loss")

        def call(self, y_true, y_pred):
            yt = tf.cast(y_true, tf.float32)
            yp = tf.cast(y_pred, tf.float32)
            truth_present = yt[:, :, 0]
            truth_valid = yt[:, :, v171.SET_VALID_OFFSET]
            truth_time = yt[:, :, v171.SET_TIME_OFFSET : v171.SET_CANDIDATE_OFFSET]
            truth_candidate = yt[:, :, v171.SET_CANDIDATE_OFFSET :]

            # event_set carries differentiable categorical survival P(K>q),
            # while runtime event_present_q is a hard exact-K decode.
            pred_present = tf.clip_by_value(yp[:, :, 0], 1e-6, 1.0 - 1e-6)
            pred_time = tf.clip_by_value(
                yp[:, :, v171.SET_TIME_OFFSET : v171.SET_CANDIDATE_OFFSET], 1e-7, 1.0
            )
            pred_candidate = tf.clip_by_value(yp[:, :, v171.SET_CANDIDATE_OFFSET :], 1e-7, 1.0)

            y = truth_present[:, None, :]
            p = pred_present[:, :, None]
            presence_cost = -(y * tf.math.log(p) + (1.0 - y) * tf.math.log(1.0 - p))
            time_cost = -tf.einsum("btd,bqd->bqt", truth_time, tf.math.log(pred_time))
            candidate_cost = -tf.einsum("btd,bqd->bqt", truth_candidate, tf.math.log(pred_candidate))
            detail = (truth_present * truth_valid)[:, None, :]
            pair = (
                tf.constant(SET_PRESENCE_WEIGHT, tf.float32) * presence_cost
                + tf.constant(SET_TIME_WEIGHT, tf.float32) * detail * time_cost
                + tf.constant(SET_CANDIDATE_WEIGHT, tf.float32) * detail * candidate_cost
            )
            scores = tf.einsum("bqt,rqt->br", pair, perm)
            return tf.reduce_min(scores, axis=1)

    return CardinalityConditionedSetLoss()


def _targets(cache, pitch_targets, string_time_targets, k, event_present, event_time, event_candidate):
    global _LAST_CENTER_TARGETS, _LAST_CENTER_ELIGIBLE
    out = v171._targets(cache, pitch_targets, string_time_targets, k, event_present, event_time, event_candidate)
    center, eligible, _ = v190._birth_center_targets(cache, pitch_targets, string_time_targets, k)
    kk = np.minimum(np.asarray(k, dtype=np.int32), EVENT_QUERIES)
    out["birth_center_map"] = center
    out["transport_coverage"] = center
    out["cardinality"] = np.eye(CARDINALITY_CLASSES, dtype=np.float32)[kk]
    _LAST_CENTER_TARGETS = center
    _LAST_CENTER_ELIGIBLE = eligible
    return out


def _sample_weights(cache, time_mask, k, event_present, event_valid):
    out = v171._sample_weights(cache, time_mask, k, event_present, event_valid)
    kk = np.minimum(np.asarray(k, dtype=np.int32), EVENT_QUERIES)
    timed = np.sum(np.asarray(time_mask, dtype=np.float32) > 0.5, axis=1).astype(np.int32)
    center_ok = ((kk > 0) & (timed == kk)).astype(np.float32)
    out["birth_center_map"] = center_ok
    out["transport_coverage"] = center_ok
    out["cardinality"] = v102._count_weights(kk).astype(np.float32)
    return out


def _build_model(spec):
    try:
        import tensorflow as tf
        from tensorflow import keras
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required") from exc

    base, _, token_shape = v102._build_model()
    candidate_context = base.get_layer("candidate_context").output
    tf_tokens = base.get_layer("tf_tokens").output
    candidate_set = _input_by_name(base, "candidate_set")
    candidate_mask = _input_by_name(base, "candidate_mask")
    spectral = _input_by_name(base, "spectral_map")

    # Shared 23x64 latent.  V23 pre-audit found useful hard-negative signal in
    # this representation; the treatment is therefore downstream structure.
    tc = np.linspace(-1.0, 1.0, v100.TIME_FRAMES, dtype=np.float32)[:, None]
    fc = np.linspace(-1.0, 1.0, v100.SPECTRAL_BANDS, dtype=np.float32)[None, :]
    coord = np.stack(
        [
            np.broadcast_to(tc, (v100.TIME_FRAMES, v100.SPECTRAL_BANDS)),
            np.broadcast_to(fc, (v100.TIME_FRAMES, v100.SPECTRAL_BANDS)),
        ],
        axis=-1,
    ).astype(np.float32)
    coord_const = tf.constant(coord, dtype=tf.float32)
    coords = keras.layers.Lambda(
        lambda s: tf.tile(coord_const[None, :, :, :], [tf.shape(s)[0], 1, 1, 1]),
        name="v231_dense_coordinates",
    )(spectral)
    dense = keras.layers.LayerNormalization(axis=-1, name="v231_dense_channel_norm")(spectral)
    dense = keras.layers.Concatenate(axis=-1, name="v231_dense_plus_coordinates")([dense, coords])
    dense = keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="v231_dense_conv1")(dense)
    dense = keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="v231_dense_conv2")(dense)
    dense_features = keras.layers.Conv2D(QUERY_DIM, (3, 3), padding="same", activation="relu", name="v231_dense_conv3")(dense)
    dense_flat = keras.layers.Reshape((CENTER_CELLS, QUERY_DIM), name="v231_dense_feature_flat")(dense_features)

    center4 = keras.layers.Conv2D(1, (1, 1), padding="same", name="v231_birth_center_logits")(dense_features)
    center_logits = keras.layers.Reshape((CENTER_CELLS,), name="v231_birth_center_logits_flat")(center4)
    center_map = keras.layers.Softmax(name="birth_center_map")(center_logits)

    # Explicit categorical cardinality head: no Poisson-binomial surrogate and
    # no six independent existence logits.
    avg = keras.layers.GlobalAveragePooling2D(name="v231_dense_global_average")(dense_features)
    mx = keras.layers.GlobalMaxPooling2D(name="v231_dense_global_max")(dense_features)
    card_h = keras.layers.Concatenate(name="v231_cardinality_context")([avg, mx, candidate_context])
    card_h = keras.layers.Dense(192, activation="relu", name="v231_cardinality_hidden1")(card_h)
    card_h = keras.layers.Dropout(0.08, name="v231_cardinality_dropout")(card_h)
    card_h = keras.layers.Dense(96, activation="relu", name="v231_cardinality_hidden2")(card_h)
    cardinality = keras.layers.Dense(CARDINALITY_CLASSES, activation="softmax", name="cardinality")(card_h)

    surv_const = tf.constant(SURVIVAL_MATRIX, dtype=tf.float32)
    soft_active = keras.layers.Lambda(
        lambda p: tf.linalg.matmul(p, surv_const), name="v231_soft_survival"
    )(cardinality)
    hard_k = keras.layers.Lambda(
        lambda p: tf.argmax(p, axis=-1, output_type=tf.int32), name="v231_hard_k"
    )(cardinality)
    hard_active = keras.layers.Lambda(
        lambda k: tf.cast(tf.range(EVENT_QUERIES, dtype=tf.int32)[None, :] < k[:, None], tf.float32),
        name="v231_hard_active",
    )(hard_k)
    # Forward = exactly q<K. Backward = categorical survival probabilities.
    row_mass = keras.layers.Lambda(
        lambda z: z[0] + tf.stop_gradient(z[1] - z[0]), name="v231_exact_k_straight_through_mass"
    )([soft_active, hard_active])

    # Six shared transport rows compete over all dense cells.  Row mass is 1
    # for exactly K rows and 0 for the rest; column capping implements soft
    # injective ownership rather than six unrelated Bernoulli proposals.
    qids = keras.layers.Lambda(
        lambda x: tf.tile(tf.range(EVENT_QUERIES, dtype=tf.int32)[None, :], [tf.shape(x)[0], 1]),
        name="v231_query_ids",
    )(candidate_context)
    qembed = keras.layers.Embedding(EVENT_QUERIES, QUERY_DIM, name="v231_query_embedding")(qids)
    gc = keras.layers.Dense(QUERY_DIM, activation="relu", name="v231_global_context")(candidate_context)
    gc = keras.layers.Lambda(
        lambda x: tf.tile(x[:, None, :], [1, EVENT_QUERIES, 1]), name="v231_global_context_broadcast"
    )(gc)
    query = keras.layers.LayerNormalization(name="v231_query_context_norm")(
        keras.layers.Add(name="v231_query_plus_context")([qembed, gc])
    )
    dkey = keras.layers.Dense(QUERY_DIM, use_bias=False, name="v231_dense_transport_keys")(dense_flat)
    qkey = keras.layers.TimeDistributed(
        keras.layers.Dense(QUERY_DIM, use_bias=False), name="v231_transport_query_keys"
    )(query)
    affinity = keras.layers.Lambda(
        lambda z: tf.einsum("bqd,bcd->bqc", z[0], z[1]) / math.sqrt(float(QUERY_DIM)),
        name="v231_transport_affinity",
    )([qkey, dkey])
    transport_scores = keras.layers.Lambda(
        lambda z: z[0] + z[1][:, None, :], name="v231_transport_scores"
    )([affinity, center_logits])
    transport = keras.layers.Lambda(
        lambda z: _capped_transport_tf(tf, z[0], z[1]), name="v231_transport_plan"
    )([transport_scores, row_mass])
    transport_row_mass = keras.layers.Lambda(
        lambda z: tf.reduce_sum(z, axis=2), name="transport_row_mass"
    )(transport)
    row_dist = keras.layers.Lambda(
        lambda z: tf.where(
            z[1][:, :, None] > 0.0,
            z[0] / (tf.reduce_sum(z[0], axis=2, keepdims=True) + EPS),
            tf.zeros_like(z[0]),
        ),
        name="v231_transport_row_distribution",
    )([transport, hard_active])
    transport_seed = keras.layers.Lambda(
        lambda z: tf.einsum("bqc,bcd->bqd", z[0], z[1]), name="v231_transport_seed"
    )([row_dist, dense_flat])
    coverage_raw = keras.layers.Lambda(
        lambda z: tf.reduce_sum(z, axis=1), name="v231_transport_coverage_raw"
    )(transport)
    transport_coverage = keras.layers.Lambda(
        lambda x: tf.where(
            tf.reduce_sum(x, axis=1, keepdims=True) > 0.0,
            x / (tf.reduce_sum(x, axis=1, keepdims=True) + EPS),
            tf.zeros_like(x),
        ),
        name="transport_coverage",
    )(coverage_raw)

    proposals = keras.layers.LayerNormalization(name="v231_transport_seed_norm")(
        keras.layers.Add(name="v231_transport_seed_plus_query")([transport_seed, query])
    )

    cand = keras.layers.TimeDistributed(keras.layers.LayerNormalization(), name="v231_candidate_norm")(candidate_set)
    cand = keras.layers.TimeDistributed(keras.layers.Dense(QUERY_DIM, activation="relu"), name="v231_candidate_hidden1")(cand)
    cand = keras.layers.TimeDistributed(keras.layers.Dense(QUERY_DIM, activation="relu"), name="v231_candidate_hidden2")(cand)
    cam = keras.layers.Lambda(
        lambda m: tf.tile(tf.cast(m[:, None, :] > 0.5, tf.bool), [1, EVENT_QUERIES, 1]),
        name="v231_candidate_attention_mask",
    )(candidate_mask)
    ca = keras.layers.MultiHeadAttention(
        num_heads=4, key_dim=24, dropout=0.05, name="v231_candidate_cross_attention"
    )(proposals, cand, attention_mask=cam)
    proposals = keras.layers.LayerNormalization(name="v231_candidate_cross_norm")(
        keras.layers.Add(name="v231_candidate_cross_residual")([proposals, ca])
    )
    ta = keras.layers.MultiHeadAttention(
        num_heads=4, key_dim=24, dropout=0.05, name="v231_tf_cross_attention"
    )(proposals, tf_tokens)
    proposals = keras.layers.LayerNormalization(name="v231_tf_cross_norm")(
        keras.layers.Add(name="v231_tf_cross_residual")([proposals, ta])
    )
    sa = keras.layers.MultiHeadAttention(
        num_heads=4, key_dim=24, dropout=0.05, name="v231_set_self_attention"
    )(proposals, proposals)
    proposals = keras.layers.LayerNormalization(name="v231_set_self_norm")(
        keras.layers.Add(name="v231_set_self_residual")([proposals, sa])
    )
    ff = keras.layers.Dense(192, activation="relu", name="v231_set_ff1")(proposals)
    ff = keras.layers.Dropout(0.08, name="v231_set_dropout")(ff)
    ff = keras.layers.Dense(QUERY_DIM, name="v231_set_ff2")(ff)
    proposals = keras.layers.LayerNormalization(name="v231_set_ff_norm")(
        keras.layers.Add(name="v231_set_ff_residual")([proposals, ff])
    )

    # Evidence ownership remains competitive, but low categorical survival
    # suppresses rows that K says are unlikely to exist.
    score_q = keras.layers.TimeDistributed(
        keras.layers.Dense(QUERY_DIM, use_bias=False), name="v231_shared_score_query"
    )(proposals)
    active_bias = keras.layers.Lambda(
        lambda a: tf.math.log(tf.clip_by_value(a, 1e-4, 1.0)), name="v231_soft_active_log_bias"
    )(soft_active)

    tf_keys = keras.layers.Dense(QUERY_DIM, use_bias=False, name="v231_tf_keys")(tf_tokens)
    tf_es = keras.layers.Lambda(
        lambda z: tf.einsum("btd,bqd->btq", z[0], z[1]) / math.sqrt(float(QUERY_DIM)) + z[2][:, None, :],
        name="v231_tf_event_scores",
    )([tf_keys, score_q, active_bias])
    tf_bg = keras.layers.Lambda(lambda x: tf.squeeze(x, axis=-1), name="v231_tf_background_score")(
        keras.layers.Dense(1, name="v231_tf_background_dense")(tf_tokens)
    )
    tf_scores = keras.layers.Concatenate(axis=-1, name="v231_tf_score_stack")(
        [tf_es, keras.layers.Lambda(lambda x: x[:, :, None], name="v231_tf_background_expand")(tf_bg)]
    )
    tf_assign = keras.layers.Softmax(axis=-1, name="v231_tf_competition")(tf_scores)

    cand_keys = keras.layers.TimeDistributed(
        keras.layers.Dense(QUERY_DIM, use_bias=False), name="v231_candidate_keys"
    )(cand)
    ce = keras.layers.Lambda(
        lambda z: tf.einsum("bcd,bqd->bcq", z[0], z[1]) / math.sqrt(float(QUERY_DIM)) + z[2][:, None, :],
        name="v231_candidate_event_scores",
    )([cand_keys, score_q, active_bias])
    cbg = keras.layers.Lambda(lambda x: tf.squeeze(x, axis=-1), name="v231_candidate_background_score")(
        keras.layers.TimeDistributed(keras.layers.Dense(1), name="v231_candidate_background_dense")(cand)
    )
    cscore = keras.layers.Concatenate(axis=-1, name="v231_candidate_score_stack")(
        [ce, keras.layers.Lambda(lambda x: x[:, :, None], name="v231_candidate_background_expand")(cbg)]
    )
    cand_assign = keras.layers.Softmax(axis=-1, name="v231_candidate_competition")(cscore)

    token_freq = int(token_shape[1])
    time_outputs, candidate_outputs = [], []
    for q in range(EVENT_QUERIES):
        own_tf = keras.layers.Lambda(lambda a, i=q: a[:, :, i], name=f"v231_event_{q}_tf_weights")(tf_assign)
        tf_grid = keras.layers.Reshape((v100.TIME_FRAMES, token_freq), name=f"v231_event_{q}_tf_grid")(own_tf)
        time_mass = keras.layers.Lambda(lambda a: tf.reduce_sum(a, axis=2), name=f"v231_event_{q}_time_mass")(tf_grid)
        time_dist = keras.layers.Lambda(
            lambda t: t / (tf.reduce_sum(t, axis=1, keepdims=True) + EPS), name=f"event_time_{q}"
        )(time_mass)
        time_outputs.append(time_dist)

        raw_c = keras.layers.Lambda(lambda a, i=q: a[:, :, i], name=f"v231_event_{q}_candidate_weights_raw")(cand_assign)
        own_c = keras.layers.Multiply(name=f"v231_event_{q}_candidate_weights")([raw_c, candidate_mask])
        cand_dist = keras.layers.Lambda(
            lambda w: w / (tf.reduce_sum(w, axis=1, keepdims=True) + EPS), name=f"event_candidate_{q}"
        )(own_c)
        candidate_outputs.append(cand_dist)

    outputs = {}
    for slot in range(SLOT_COUNT):
        outputs[f"string_{slot}"] = base.get_layer(f"string_{slot}").output
        outputs[f"pitch_{slot}"] = base.get_layer(f"pitch_{slot}").output
        outputs[f"time_{slot}"] = base.get_layer(f"time_{slot}").output

    set_slots = []
    for q in range(EVENT_QUERIES):
        runtime_present = keras.layers.Lambda(
            lambda a, i=q: a[:, i : i + 1], name=f"event_present_{q}"
        )(hard_active)
        train_present = keras.layers.Lambda(
            lambda a, i=q: a[:, i : i + 1], name=f"v231_train_present_{q}"
        )(soft_active)
        valid = keras.layers.Lambda(
            lambda x: tf.ones_like(x), name=f"v231_event_{q}_valid_placeholder"
        )(train_present)
        packed = keras.layers.Concatenate(name=f"v231_event_{q}_set_vector")(
            [train_present, valid, time_outputs[q], candidate_outputs[q]]
        )
        set_slots.append(packed)
        outputs[f"event_present_{q}"] = runtime_present
        outputs[f"event_time_{q}"] = time_outputs[q]
        outputs[f"event_candidate_{q}"] = candidate_outputs[q]

    outputs["event_set"] = keras.layers.Lambda(lambda xs: tf.stack(xs, axis=1), name="event_set")(set_slots)
    outputs["event_count_norm"] = keras.layers.Lambda(
        lambda a: tf.reduce_sum(a, axis=1, keepdims=True) / float(EVENT_QUERIES), name="event_count_norm"
    )(soft_active)
    outputs["birth_center_map"] = center_map
    outputs["transport_coverage"] = transport_coverage
    outputs["cardinality"] = cardinality
    outputs["transport_row_mass"] = transport_row_mass

    loss = {f"string_{s}": "binary_crossentropy" for s in range(SLOT_COUNT)}
    loss.update({f"pitch_{s}": "mse" for s in range(SLOT_COUNT)})
    loss.update({f"time_{s}": keras.losses.KLDivergence() for s in range(SLOT_COUNT)})
    loss["event_set"] = _set_loss()
    loss["event_count_norm"] = "mse"
    loss["birth_center_map"] = "categorical_crossentropy"
    loss["transport_coverage"] = "categorical_crossentropy"
    loss["cardinality"] = "categorical_crossentropy"

    lw = {f"string_{s}": 0.18 for s in range(SLOT_COUNT)}
    lw.update({f"pitch_{s}": 0.04 for s in range(SLOT_COUNT)})
    lw.update({f"time_{s}": 0.10 for s in range(SLOT_COUNT)})
    lw["event_set"] = 1.0
    lw["event_count_norm"] = 0.0
    lw["birth_center_map"] = CENTER_MAP_WEIGHT
    lw["transport_coverage"] = TRANSPORT_COVERAGE_WEIGHT
    lw["cardinality"] = CARDINALITY_WEIGHT

    model = keras.Model(base.inputs, outputs, name="v231_cardinality_conditioned_transport")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=2e-4), loss=loss, loss_weights=lw)
    return model, lw, token_shape


def _cardinality_diag(prob, true_k):
    p = np.asarray(prob, dtype=np.float64)
    k = np.asarray(true_k, dtype=np.int32)
    pred = np.argmax(p, axis=1).astype(np.int32)
    poly = k >= 2
    out = {
        "exact": float(np.mean(pred == k)),
        "poly_exact": float(np.mean(pred[poly] == k[poly])) if np.any(poly) else None,
        "mae": float(np.mean(np.abs(pred - k))),
        "under_rate": float(np.mean(pred < k)),
        "over_rate": float(np.mean(pred > k)),
        "mean_true_k": float(np.mean(k)),
        "mean_predicted_k": float(np.mean(pred)),
        "per_true_k": {},
    }
    for value in range(CARDINALITY_CLASSES):
        m = k == value
        out["per_true_k"][str(value)] = {
            "rows": int(np.sum(m)),
            "exact": float(np.mean(pred[m] == value)) if np.any(m) else None,
            "mean_predicted_k": float(np.mean(pred[m])) if np.any(m) else None,
        }
    return out


def _transport_mass_diag(row_mass, card_prob):
    mass = np.asarray(row_mass, dtype=np.float64)
    pred_k = np.argmax(np.asarray(card_prob), axis=1).astype(np.int32)
    decoded = np.sum(mass >= 0.5, axis=1).astype(np.int32)
    return {
        "rows": int(len(mass)),
        "exact_k_mass_decode_rate": float(np.mean(decoded == pred_k)),
        "max_absolute_total_mass_error": float(np.max(np.abs(np.sum(mass, axis=1) - pred_k))),
        "prefix_violation_rate": float(np.mean(np.any(np.diff(mass >= 0.5, axis=1).astype(np.int8) > 0, axis=1))),
        "mean_total_mass": float(np.mean(np.sum(mass, axis=1))),
        "mean_predicted_k": float(np.mean(pred_k)),
    }


def _postprocess(args, report, ctx, raw, center_diag, coverage_diag):
    # V190's postprocessor is reused only for the proven outer-clean reporting
    # and candidate-realization plumbing.  Its V19 protocol claims are then
    # explicitly replaced below by the V23.1 treatment definition.
    report = v190._postprocess(args, report, ctx, center_diag)
    v171._rename_report(report, v190.MODEL_KEY, MODEL_KEY)
    inherited = report.pop("v190")

    outer = np.asarray(ctx["outer_idx"], dtype=np.int64)
    true_k = np.asarray(ctx["k"], dtype=np.int32)[outer]
    card_diag = _cardinality_diag(np.asarray(raw["cardinality"]), true_k)
    mass_diag = _transport_mass_diag(np.asarray(raw["transport_row_mass"]), np.asarray(raw["cardinality"]))

    protocol = report["protocol"]
    for key in (
        "v190_only_scientific_treatment",
        "dense_center_anchor_selection",
        "v173_poisson_binomial_count_objective_unchanged",
        "v173_count_nll_weight",
        "runtime_count_decode_unchanged_from_v173",
        "mass_preserving_exchangeable_weights_unchanged",
    ):
        protocol.pop(key, None)
    protocol.update(
        {
            "v190_dense_birth_centers": False,
            "v231_cardinality_conditioned_transport": True,
            "v231_pre_audit_run": 34126377071,
            "v231_pre_audit_all_four_gates_passed": True,
            "v231_only_scientific_treatment": (
                "categorical P(K=0..6) controls exact-K straight-through capped dense transport; "
                "six independent Bernoulli existence decisions removed"
            ),
            "dense_center_grid": [int(v100.TIME_FRAMES), int(v100.SPECTRAL_BANDS)],
            "categorical_cardinality_head_exists": True,
            "categorical_cardinality_classes": CARDINALITY_CLASSES,
            "categorical_cardinality_loss": "categorical_crossentropy",
            "categorical_cardinality_loss_weight": CARDINALITY_WEIGHT,
            "categorical_cardinality_loss_weight_tuned": False,
            "six_independent_presence_bernoullis_exist": False,
            "poisson_binomial_cardinality_objective_exists": False,
            "runtime_cardinality_decode": "argmax categorical P(K=0..6)",
            "runtime_event_activation": "exact prefix q<K_pred; event_present is binary",
            "runtime_presence_threshold_tuned": False,
            "transport_type": "straight-through exact-K row mass + differentiable repeated column-capped transport",
            "transport_iterations": TRANSPORT_ITERS,
            "transport_temperature": TRANSPORT_TEMPERATURE,
            "transport_temperature_tuned": False,
            "transport_coverage_loss_weight": TRANSPORT_COVERAGE_WEIGHT,
            "transport_coverage_loss_weight_tuned": False,
            "dense_center_loss_weight": CENTER_MAP_WEIGHT,
            "dense_center_loss_weight_tuned": False,
            "raw_candidate_is_object_identity": False,
            "raw_candidate_role": "auxiliary realization/evidence memory only",
            "exact_720_truth_matching_unchanged": True,
            "historical_validation_or_locked12_indexed_or_evaluated": False,
        }
    )

    arch = dict(inherited.get("architecture", {}))
    arch.update(
        {
            "categorical_cardinality": card_diag,
            "transport_mass": mass_diag,
            "transport_coverage_diagnostics": coverage_diag,
            "birth_center_prior_diagnostics": center_diag,
            "transport_rows": EVENT_QUERIES,
            "transport_cells": CENTER_CELLS,
            "six_independent_presence_heads": 0,
        }
    )
    report["v231"] = {**inherited, "model_key": MODEL_KEY, "architecture": arch}

    npz_path = args.output_dir / f"predictions-fold-{args.outer_fold}.npz"
    with np.load(npz_path, allow_pickle=False) as z:
        data = {key: np.asarray(z[key]) for key in z.files}
    if v190.PRED_KEY not in data:
        raise V231Error(f"missing inherited prediction key {v190.PRED_KEY}")
    data[PRED_KEY] = data.pop(v190.PRED_KEY)
    data["cardinality_prob"] = np.asarray(raw["cardinality"], dtype=np.float32)
    data["categorical_k"] = np.argmax(np.asarray(raw["cardinality"]), axis=1).astype(np.int16)
    data["transport_row_mass"] = np.asarray(raw["transport_row_mass"], dtype=np.float32)
    np.savez_compressed(npz_path, **data)

    old_w = args.output_dir / f"v190-dense-birth-centers-fold-{args.outer_fold}.weights.h5"
    new_w = args.output_dir / f"v231-cardinality-transport-fold-{args.outer_fold}.weights.h5"
    if old_w.exists():
        old_w.replace(new_w)

    (args.output_dir / f"report-fold-{args.outer_fold}.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def train_fold(args):
    global _LAST_CENTER_TARGETS, _LAST_CENTER_ELIGIBLE
    if args.seed != DEFAULT_SEED:
        raise V231Error(f"V23.1 requires seed {DEFAULT_SEED}")
    if args.arm != BASE_ARM:
        raise V231Error(f"V23.1 only supports {BASE_ARM}")

    ctx = v172._fold_context(args)
    specs = [ctx["meta_spec"], ctx["final_spec"]]
    calls = {"count": 0}
    built = []

    def builder():
        i = calls["count"]
        if i >= 2:
            raise V231Error("unexpected model build")
        calls["count"] += 1
        model = _build_model(specs[i])
        built.append(model[0])
        return model

    old_build, old_targets, old_weights = v130._build_model, v130._targets, v130._sample_weights
    try:
        v130._build_model = builder
        v130._targets = _targets
        v130._sample_weights = _sample_weights
        report = v130.train_fold(args)
    finally:
        v130._build_model, v130._targets, v130._sample_weights = old_build, old_targets, old_weights

    if calls["count"] != 2 or _LAST_CENTER_TARGETS is None:
        raise V231Error("V23.1 build/target capture failed")

    outer = np.asarray(ctx["outer_idx"], dtype=np.int64)
    raw = built[-1].predict(v102._inputs(ctx["cache"], outer), batch_size=128, verbose=0)
    target = np.asarray(_LAST_CENTER_TARGETS)[outer]
    eligible = np.asarray(_LAST_CENTER_ELIGIBLE)[outer]
    kk = np.asarray(ctx["k"], dtype=np.int32)[outer]
    center_diag = v190._center_diagnostics(np.asarray(raw["birth_center_map"]), target, eligible, kk)
    coverage_diag = v190._center_diagnostics(np.asarray(raw["transport_coverage"]), target, eligible, kk)
    report = _postprocess(args, report, ctx, raw, center_diag, coverage_diag)

    g = report["strata"]["aggregate"][MODEL_KEY]["metrics"]["global"]
    card = report["v231"]["architecture"]["categorical_cardinality"]
    mass = report["v231"]["architecture"]["transport_mass"]
    print(
        json.dumps(
            {
                "outer": args.outer_fold,
                "selected_epochs": report["data"]["selected_epochs"],
                "v104_f1": report["strata"]["aggregate"]["v104"]["metrics"]["global"]["f1"],
                "v231_f1": g["f1"],
                "pred_ref": g["prediction_reference_ratio"],
                "categorical_k_exact": card["exact"],
                "categorical_k_poly_exact": card["poly_exact"],
                "exact_k_transport_mass": mass["exact_k_mass_decode_rate"],
                "coverage_top6_exact_poly": coverage_diag["top6_exact_center_coverage_poly"],
                "k2": report["per_true_k"]["2"][MODEL_KEY]["exact"],
                "k3": report["per_true_k"]["3"][MODEL_KEY]["exact"],
                "k4": report["per_true_k"]["4"][MODEL_KEY]["exact"],
                "k5": report["per_true_k"]["5"][MODEL_KEY]["exact"],
                "k6": report["per_true_k"]["6"][MODEL_KEY]["exact"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return report


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("dataset_dir", nargs="?", type=Path, default=ROOT / "data" / "GuitarSet")
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--baseline-eval-dir", type=Path, required=True)
    p.add_argument("--outer-fold", type=int, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--arm", choices=[BASE_ARM], default=BASE_ARM)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return p


def main(argv: Optional[Sequence[str]] = None):
    train_fold(parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
