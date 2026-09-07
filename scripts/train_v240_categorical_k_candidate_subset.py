"""V24.0 categorical-K + explicit candidate-subset decoder.

V23.2 established that categorical cardinality is trainable and finite, while
its dense transport almost never places the K events on the correct centers.
V17.7 established the complementary fact: once a causal candidate is selected,
identity/time realization is essentially solved, while candidate Bernoulli
objectness is a bad way to decide K.

V24 separates the decisions completely:

    dense acoustic evidence -> categorical P(K=0..6)
    causal candidate memory -> shared contextual ranker
    runtime                -> top-K valid candidates, K only from categorical head

Candidate scores never decide event count. Training uses a multi-positive
listwise subset objective plus permutation-invariant event-set supervision over
six differentiable without-replacement ranking slots. Runtime candidate
identity is hard, unique top-K. No threshold is tuned and Locked12 is never
indexed/evaluated.
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
from scripts import train_v101_string_query_attention as v101
from scripts import train_v102_source_time_assignment as v102
from scripts import train_v120_integrated_birth_source_time as v120
from scripts import train_v130_causal_event_set_decoder as v130
from scripts import train_v171_controlled_assignment_ab as v171
from scripts import train_v172_mass_preserving_exchangeable as v172
from scripts import train_v177_candidate_centric as v177
from scripts import train_v190_dense_birth_centers as v190
from scripts import train_v231_cardinality_transport as v231

DEFAULT_SEED = 16061
EVENT_QUERIES = SLOT_COUNT
CARDINALITY_CLASSES = EVENT_QUERIES + 1
MAX_CANDIDATES = v130.MAX_CANDIDATES
QUERY_DIM = 96
EPS = 1e-6
BASE_ARM = "mass_permutation"
MODEL_KEY = "v240_categorical_k_candidate_subset"
PRED_KEY = "pred240_categorical_k_candidate_subset"

CARDINALITY_WEIGHT = 1.00
CENTER_MAP_WEIGHT = 0.15
CANDIDATE_SUBSET_WEIGHT = 1.00

_LAST_CENTER_TARGETS = None
_LAST_CENTER_ELIGIBLE = None
_LAST_CANDIDATE_SUBSET = None

SURVIVAL_MATRIX = v231.SURVIVAL_MATRIX


class V240Error(RuntimeError):
    pass


def _input_by_name(model, name):
    return v190._input_by_name(model, name)


def _soft_rank_slots_tf(tf, logits, mask):
    """Differentiable six-step ranking; runtime still uses hard unique top-K."""
    valid = tf.cast(mask > 0.5, tf.float32)
    remaining = valid
    slots = []
    neg = tf.constant(-1e9, tf.float32)
    for _ in range(EVENT_QUERIES):
        score = tf.cast(logits, tf.float32) + tf.math.log(tf.maximum(remaining, 1e-5))
        score = tf.where(valid > 0.5, score, neg)
        p = tf.nn.softmax(score, axis=1) * valid
        p = p / (tf.reduce_sum(p, axis=1, keepdims=True) + EPS)
        slots.append(p)
        remaining = remaining * (1.0 - p)
    return tf.stack(slots, axis=1)


def _candidate_subset_loss():
    """Finite multi-positive listwise loss; K=0 rows explicitly return zero."""
    import tensorflow as tf
    from tensorflow import keras

    class CandidateSubsetLoss(keras.losses.Loss):
        def __init__(self):
            super().__init__(name="v240_candidate_subset_listwise_loss")

        def call(self, y_true, y_pred):
            y = tf.cast(y_true, tf.float32)
            p = tf.clip_by_value(tf.cast(y_pred, tf.float32), 1e-7, 1.0)
            npos = tf.reduce_sum(y, axis=1)
            nll = -tf.reduce_sum(y * tf.math.log(p), axis=1) / tf.maximum(npos, 1.0)
            return tf.where(npos > 0.0, nll, tf.zeros_like(nll))

    return CandidateSubsetLoss()


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

    # Preserve the V23 dense evidence + explicit categorical K treatment.
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
        name="v240_dense_coordinates",
    )(spectral)
    dense = keras.layers.LayerNormalization(axis=-1, name="v240_dense_channel_norm")(spectral)
    dense = keras.layers.Concatenate(axis=-1, name="v240_dense_plus_coordinates")([dense, coords])
    dense = keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="v240_dense_conv1")(dense)
    dense = keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="v240_dense_conv2")(dense)
    dense_features = keras.layers.Conv2D(QUERY_DIM, (3, 3), padding="same", activation="relu", name="v240_dense_conv3")(dense)

    center4 = keras.layers.Conv2D(1, (1, 1), padding="same", name="v240_birth_center_logits")(dense_features)
    center_logits = keras.layers.Reshape((v100.TIME_FRAMES * v100.SPECTRAL_BANDS,), name="v240_birth_center_logits_flat")(center4)
    center_map = keras.layers.Softmax(name="birth_center_map")(center_logits)

    avg = keras.layers.GlobalAveragePooling2D(name="v240_dense_global_average")(dense_features)
    mx = keras.layers.GlobalMaxPooling2D(name="v240_dense_global_max")(dense_features)
    card_h = keras.layers.Concatenate(name="v240_cardinality_context")([avg, mx, candidate_context])
    card_h = keras.layers.Dense(192, activation="relu", name="v240_cardinality_hidden1")(card_h)
    card_h = keras.layers.Dropout(0.08, name="v240_cardinality_dropout")(card_h)
    card_h = keras.layers.Dense(96, activation="relu", name="v240_cardinality_hidden2")(card_h)
    cardinality = keras.layers.Dense(CARDINALITY_CLASSES, activation="softmax", name="cardinality")(card_h)

    surv_const = tf.constant(SURVIVAL_MATRIX, dtype=tf.float32)
    soft_active = keras.layers.Lambda(lambda p: tf.linalg.matmul(p, surv_const), name="v240_soft_survival")(cardinality)
    hard_k = keras.layers.Lambda(lambda p: tf.argmax(p, axis=-1, output_type=tf.int32), name="v240_hard_k")(cardinality)
    hard_active = keras.layers.Lambda(
        lambda k: tf.cast(tf.range(EVENT_QUERIES, dtype=tf.int32)[None, :] < k[:, None], tf.float32),
        name="v240_hard_active",
    )(hard_k)

    # Candidate tokens are event identities, but never count decisions.
    cand = keras.layers.TimeDistributed(keras.layers.LayerNormalization(), name="v240_candidate_norm")(candidate_set)
    cand = keras.layers.TimeDistributed(keras.layers.Dense(QUERY_DIM, activation="relu"), name="v240_candidate_hidden1")(cand)
    cand = keras.layers.TimeDistributed(keras.layers.Dense(QUERY_DIM, activation="relu"), name="v240_candidate_hidden2")(cand)
    gc = keras.layers.Dense(QUERY_DIM, activation="relu", name="v240_candidate_global_context")(candidate_context)
    gc = keras.layers.Lambda(lambda x: tf.tile(x[:, None, :], [1, MAX_CANDIDATES, 1]), name="v240_candidate_global_context_broadcast")(gc)
    cand = keras.layers.LayerNormalization(name="v240_candidate_context_norm")(
        keras.layers.Add(name="v240_candidate_plus_context")([cand, gc])
    )

    tf_att = keras.layers.MultiHeadAttention(num_heads=4, key_dim=24, dropout=0.05, name="v240_candidate_tf_cross_attention")(cand, tf_tokens)
    cand = keras.layers.LayerNormalization(name="v240_candidate_tf_norm")(
        keras.layers.Add(name="v240_candidate_tf_residual")([cand, tf_att])
    )
    self_mask = keras.layers.Lambda(
        lambda m: tf.tile(tf.cast(m[:, None, :] > 0.5, tf.bool), [1, MAX_CANDIDATES, 1]),
        name="v240_candidate_self_attention_mask",
    )(candidate_mask)
    self_att = keras.layers.MultiHeadAttention(num_heads=4, key_dim=24, dropout=0.05, name="v240_candidate_self_attention")(
        cand, cand, attention_mask=self_mask
    )
    cand = keras.layers.LayerNormalization(name="v240_candidate_self_norm")(
        keras.layers.Add(name="v240_candidate_self_residual")([cand, self_att])
    )
    ff = keras.layers.TimeDistributed(keras.layers.Dense(192, activation="relu"), name="v240_candidate_ff1")(cand)
    ff = keras.layers.Dropout(0.08, name="v240_candidate_dropout")(ff)
    ff = keras.layers.TimeDistributed(keras.layers.Dense(QUERY_DIM), name="v240_candidate_ff2")(ff)
    cand = keras.layers.LayerNormalization(name="v240_candidate_ff_norm")(
        keras.layers.Add(name="v240_candidate_ff_residual")([cand, ff])
    )

    rank_hidden = keras.layers.TimeDistributed(keras.layers.Dense(64, activation="relu"), name="v240_shared_rank_hidden")(cand)
    rank_logits_3d = keras.layers.TimeDistributed(keras.layers.Dense(1), name="v240_shared_rank_logit")(rank_hidden)
    rank_logits = keras.layers.Lambda(lambda x: tf.squeeze(x, axis=-1), name="v240_candidate_rank_logits")(rank_logits_3d)
    masked_logits = keras.layers.Lambda(
        lambda z: tf.where(z[1] > 0.5, z[0], tf.cast(-1e9, z[0].dtype)),
        name="v240_masked_candidate_rank_logits",
    )([rank_logits, candidate_mask])
    rank_distribution = keras.layers.Softmax(axis=1, name="candidate_subset")(masked_logits)
    soft_slots = keras.layers.Lambda(
        lambda z: _soft_rank_slots_tf(tf, z[0], z[1]),
        name="v240_soft_without_replacement_slots",
    )([rank_logits, candidate_mask])

    # Candidate-conditioned time remains auxiliary; headline timing is the candidate timestamp.
    time_q = keras.layers.TimeDistributed(keras.layers.Dense(QUERY_DIM, use_bias=False), name="v240_candidate_time_query")(cand)
    time_k = keras.layers.Dense(QUERY_DIM, use_bias=False, name="v240_candidate_time_keys")(tf_tokens)
    time_scores = keras.layers.Lambda(
        lambda z: tf.einsum("bcd,btd->bct", z[0], z[1]) / math.sqrt(float(QUERY_DIM)),
        name="v240_candidate_tf_time_scores",
    )([time_q, time_k])
    token_freq = int(token_shape[1])
    time_grid = keras.layers.Reshape((MAX_CANDIDATES, v100.TIME_FRAMES, token_freq), name="v240_candidate_time_grid")(time_scores)
    time_mass = keras.layers.Lambda(lambda x: tf.reduce_logsumexp(x, axis=3), name="v240_candidate_time_mass")(time_grid)
    candidate_time = keras.layers.Softmax(axis=2, name="v240_candidate_time_distribution")(time_mass)
    soft_slot_time = keras.layers.Lambda(
        lambda z: tf.einsum("bqc,bct->bqt", z[0], z[1]),
        name="v240_soft_slot_time",
    )([soft_slots, candidate_time])

    # Hard runtime: one ranking, unique top-6 IDs, K exclusively from cardinality.
    top_scores, top_ids = keras.layers.Lambda(
        lambda x: tf.math.top_k(x, k=EVENT_QUERIES, sorted=True),
        name="v240_candidate_top6",
    )(masked_logits)
    top_valid = keras.layers.Lambda(lambda z: tf.gather(z[0], z[1], batch_dims=1), name="v240_top6_valid")([candidate_mask, top_ids])
    runtime_active = keras.layers.Multiply(name="v240_runtime_active_valid")([hard_active, top_valid])
    runtime_candidate = keras.layers.Lambda(
        lambda z: tf.one_hot(z[0], MAX_CANDIDATES, dtype=tf.float32) * z[1][:, :, None],
        name="v240_runtime_candidate_identity",
    )([top_ids, top_valid])
    runtime_time = keras.layers.Lambda(
        lambda z: tf.gather(z[0], z[1], batch_dims=1),
        name="v240_runtime_candidate_time",
    )([candidate_time, top_ids])

    outputs = {}
    for slot in range(SLOT_COUNT):
        outputs[f"string_{slot}"] = base.get_layer(f"string_{slot}").output
        outputs[f"pitch_{slot}"] = base.get_layer(f"pitch_{slot}").output
        outputs[f"time_{slot}"] = base.get_layer(f"time_{slot}").output

    set_slots = []
    for q in range(EVENT_QUERIES):
        train_present = keras.layers.Lambda(lambda x, i=q: x[:, i : i + 1], name=f"v240_train_present_{q}")(soft_active)
        valid_placeholder = keras.layers.Lambda(lambda x: tf.ones_like(x), name=f"v240_event_{q}_valid_placeholder")(train_present)
        train_time = keras.layers.Lambda(lambda x, i=q: x[:, i, :], name=f"v240_train_time_{q}")(soft_slot_time)
        train_candidate = keras.layers.Lambda(lambda x, i=q: x[:, i, :], name=f"v240_train_candidate_{q}")(soft_slots)
        set_slots.append(
            keras.layers.Concatenate(name=f"v240_event_{q}_set_vector")(
                [train_present, valid_placeholder, train_time, train_candidate]
            )
        )
        outputs[f"event_present_{q}"] = keras.layers.Lambda(lambda x, i=q: x[:, i : i + 1], name=f"event_present_{q}")(hard_active)
        outputs[f"event_time_{q}"] = keras.layers.Lambda(lambda x, i=q: x[:, i, :], name=f"event_time_{q}")(runtime_time)
        outputs[f"event_candidate_{q}"] = keras.layers.Lambda(lambda x, i=q: x[:, i, :], name=f"event_candidate_{q}")(runtime_candidate)

    outputs["event_set"] = keras.layers.Lambda(lambda xs: tf.stack(xs, axis=1), name="event_set")(set_slots)
    outputs["event_count_norm"] = keras.layers.Lambda(
        lambda a: tf.reduce_sum(a, axis=1, keepdims=True) / float(EVENT_QUERIES), name="event_count_norm"
    )(soft_active)
    outputs["birth_center_map"] = center_map
    outputs["cardinality"] = cardinality
    outputs["candidate_subset"] = rank_distribution
    outputs["candidate_selected_ids"] = top_ids
    outputs["candidate_selected_scores"] = top_scores
    outputs["candidate_runtime_active"] = runtime_active

    loss = {f"string_{s}": "binary_crossentropy" for s in range(SLOT_COUNT)}
    loss.update({f"pitch_{s}": "mse" for s in range(SLOT_COUNT)})
    loss.update({f"time_{s}": keras.losses.KLDivergence() for s in range(SLOT_COUNT)})
    loss["event_set"] = v231._set_loss()
    loss["event_count_norm"] = "mse"
    loss["birth_center_map"] = "categorical_crossentropy"
    loss["cardinality"] = "categorical_crossentropy"
    loss["candidate_subset"] = _candidate_subset_loss()

    lw = {f"string_{s}": 0.18 for s in range(SLOT_COUNT)}
    lw.update({f"pitch_{s}": 0.04 for s in range(SLOT_COUNT)})
    lw.update({f"time_{s}": 0.10 for s in range(SLOT_COUNT)})
    lw["event_set"] = 1.0
    lw["event_count_norm"] = 0.0
    lw["birth_center_map"] = CENTER_MAP_WEIGHT
    lw["cardinality"] = CARDINALITY_WEIGHT
    lw["candidate_subset"] = CANDIDATE_SUBSET_WEIGHT

    model = keras.Model(base.inputs, outputs, name="v240_categorical_k_candidate_subset")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=2e-4), loss=loss, loss_weights=lw)
    return model, lw, token_shape


def _targets(cache, pitch_targets, string_time_targets, k, event_present, event_time, event_candidate):
    global _LAST_CENTER_TARGETS, _LAST_CENTER_ELIGIBLE, _LAST_CANDIDATE_SUBSET
    out = v171._targets(cache, pitch_targets, string_time_targets, k, event_present, event_time, event_candidate)
    center, eligible, _ = v190._birth_center_targets(cache, pitch_targets, string_time_targets, k)
    kk = np.minimum(np.asarray(k, dtype=np.int32), EVENT_QUERIES)
    subset = np.clip(np.sum(np.asarray(event_candidate, dtype=np.float32), axis=1), 0.0, 1.0)
    out["birth_center_map"] = center
    out["cardinality"] = np.eye(CARDINALITY_CLASSES, dtype=np.float32)[kk]
    out["candidate_subset"] = subset
    _LAST_CENTER_TARGETS = center
    _LAST_CENTER_ELIGIBLE = eligible
    _LAST_CANDIDATE_SUBSET = subset
    return out


def _sample_weights(cache, time_mask, k, event_present, event_valid):
    out = v171._sample_weights(cache, time_mask, k, event_present, event_valid)
    kk = np.minimum(np.asarray(k, dtype=np.int32), EVENT_QUERIES)
    timed = np.sum(np.asarray(time_mask, dtype=np.float32) > 0.5, axis=1).astype(np.int32)
    center_ok = ((kk > 0) & (timed == kk)).astype(np.float32)
    out["birth_center_map"] = center_ok
    out["cardinality"] = v102._count_weights(kk).astype(np.float32)
    out["candidate_subset"] = (np.sum(np.asarray(event_valid, dtype=np.float32), axis=1) > 0.0).astype(np.float32)
    return out


def _decode_capture_factory(captures):
    def decode(model, inputs):
        raw = model.predict(inputs, batch_size=128, verbose=0)
        card = np.asarray(raw["cardinality"], dtype=np.float64)
        pred_k = np.argmax(card, axis=1).astype(np.int32)
        ids = np.asarray(raw["candidate_selected_ids"], dtype=np.int32)
        rank = np.asarray(raw["candidate_subset"], dtype=np.float64)
        mask = np.asarray(inputs["candidate_mask"], dtype=np.float64)
        valid = np.take_along_axis(mask, ids, axis=1) > 0.5
        q = np.arange(EVENT_QUERIES, dtype=np.int32)[None, :]
        active = (q < pred_k[:, None]) & valid
        selected = np.where(active, ids, -1).astype(np.int32)
        presence = active.astype(np.float64)
        time = np.stack([np.asarray(raw[f"event_time_{i}"], dtype=np.float64) for i in range(EVENT_QUERIES)], axis=1)
        candidate = np.stack([np.asarray(raw[f"event_candidate_{i}"], dtype=np.float64) for i in range(EVENT_QUERIES)], axis=1)
        pred = np.sum(active, axis=1).astype(np.int32)
        captures.append(
            {
                "cardinality": card,
                "candidate_rank_distribution": rank,
                "candidate_selected_ids": selected,
                "candidate_top6_ids": ids,
                "candidate_valid_count": np.sum(mask > 0.5, axis=1).astype(np.int32),
            }
        )
        return pred, presence, time, candidate, np.zeros(len(pred), dtype=bool)
    return decode


def _candidate_subset_diag(target, capture, true_k):
    target = np.asarray(target, dtype=np.float64) > 0.5
    selected = np.asarray(capture["candidate_selected_ids"], dtype=np.int32)
    k = np.asarray(true_k, dtype=np.int32)
    target_count = np.sum(target, axis=1).astype(np.int32)
    selected_count = np.sum(selected >= 0, axis=1).astype(np.int32)
    hits = np.zeros(len(k), dtype=np.int32)
    exact = np.zeros(len(k), dtype=bool)
    for i, ids in enumerate(selected):
        keep = ids[ids >= 0]
        if len(keep):
            hits[i] = int(np.sum(target[i, keep]))
        truth_ids = np.flatnonzero(target[i])
        exact[i] = len(keep) == len(truth_ids) and set(map(int, keep)) == set(map(int, truth_ids))
    pos = target_count > 0
    feasible = (k > 0) & (target_count == k)
    poly_feasible = (k >= 2) & (target_count == k)
    realized_exact_k = selected_count == k
    return {
        "rows": int(len(k)),
        "positive_rows": int(np.sum(pos)),
        "candidate_feasible_positive_rate": float(np.mean(target_count[k > 0] == k[k > 0])) if np.any(k > 0) else None,
        "candidate_feasible_poly_rate": float(np.mean(target_count[k >= 2] == k[k >= 2])) if np.any(k >= 2) else None,
        "mean_positive_candidate_recall": float(np.mean(hits[pos] / np.maximum(target_count[pos], 1))) if np.any(pos) else None,
        "exact_subset_feasible_rate": float(np.mean(exact[feasible])) if np.any(feasible) else None,
        "exact_subset_poly_feasible_rate": float(np.mean(exact[poly_feasible])) if np.any(poly_feasible) else None,
        "runtime_realized_exact_categorical_k_rate": float(np.mean(realized_exact_k)),
        "rows_candidate_count_below_predicted_k": int(np.sum(selected_count < k)),
        "selected_candidate_unique_rate": float(np.mean([len(x[x >= 0]) == len(np.unique(x[x >= 0])) for x in selected])),
    }


def _postprocess(args, report, ctx, capture, center_diag):
    # Reuse proven outer-clean reporting plumbing, then replace V19 claims and
    # frozen-ranking headline realization with direct selected candidates.
    report = v190._postprocess(args, report, ctx, center_diag)
    v171._rename_report(report, v190.MODEL_KEY, MODEL_KEY)
    inherited = report.pop("v190")

    outer = np.asarray(ctx["outer_idx"], dtype=np.int64)
    true_k = np.asarray(ctx["k"], dtype=np.int32)[outer]
    card_diag = v231._cardinality_diag(capture["cardinality"], true_k)
    subset_target = np.asarray(_LAST_CANDIDATE_SUBSET, dtype=np.float32)[outer]
    subset_diag = _candidate_subset_diag(subset_target, capture, true_k)

    cache = ctx["cache"]
    candidate_samples, reconstruction = v102._reconstruct_candidates(cache)
    selected_ids = np.asarray(capture["candidate_selected_ids"], dtype=np.int32)
    members = np.asarray(ctx["members"], dtype="U96")
    outer_members = members[outer]
    players = np.asarray([str(m).split("_", 1)[0] for m in outer_members], dtype="U2")
    modes = np.asarray([
        "comp" if str(m).endswith("_comp.jams") else "solo" if str(m).endswith("_solo.jams") else "other"
        for m in outer_members
    ], dtype="U8")
    by_member = {t.annotation_member: t for t in ctx["train_split"]}
    groups = np.asarray([v177.group_stem(by_member[str(m)]) for m in outer_members], dtype="U64")
    genres = np.asarray([v120._genre(g) for g in groups], dtype="U16")

    frozen_ranking_strata = {}
    for name, row in report.get("strata", {}).items():
        if not row or MODEL_KEY not in row:
            continue
        mask = v177._stratum_mask(name, outer_members, players, modes, genres)
        if mask is None or not np.any(mask):
            continue
        frozen_ranking_strata[name] = row[MODEL_KEY]["metrics"]
        pmap = v177._direct_prediction_map(cache, candidate_samples, outer, selected_ids, mask)
        member_set = {str(m) for m in outer_members[mask]}
        tracks = tuple(t for t in ctx["train_split"] if t.annotation_member in member_set)
        row[MODEL_KEY]["metrics"] = v101._metrics(tracks, pmap)

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
            "v240_categorical_k_candidate_subset": True,
            "v240_base_evidence": "V23 dense categorical K + V17.7 candidate identity evidence",
            "v240_only_scientific_treatment": "dense transport identity -> categorical-K controlled explicit candidate subset ranking",
            "v230_pre_audit_run": 34126377071,
            "categorical_cardinality_head_exists": True,
            "categorical_cardinality_classes": CARDINALITY_CLASSES,
            "categorical_cardinality_loss_weight": CARDINALITY_WEIGHT,
            "six_independent_presence_bernoullis_exist": False,
            "candidate_bernoulli_objectness_controls_count": False,
            "candidate_rank_controls_count": False,
            "runtime_cardinality_decode": "argmax categorical P(K=0..6)",
            "runtime_candidate_decode": "hard unique top-K valid causal candidates",
            "runtime_candidate_threshold_tuned": False,
            "dense_transport_used_for_event_identity": False,
            "transport_plan_exists_in_v240_graph": False,
            "candidate_subset_training": "multi-positive listwise + permutation-invariant soft without-replacement slots",
            "candidate_subset_loss_weight": CANDIDATE_SUBSET_WEIGHT,
            "candidate_subset_loss_weight_tuned": False,
            "dense_center_loss_weight": CENTER_MAP_WEIGHT,
            "dense_center_loss_weight_tuned": False,
            "raw_candidate_is_object_identity": True,
            "headline_candidate_realization": "direct selected candidate timestamps",
            "historical_validation_or_locked12_indexed_or_evaluated": False,
        }
    )

    direct_global = report["strata"]["aggregate"][MODEL_KEY]["metrics"]["global"]
    frozen_global = frozen_ranking_strata["aggregate"]["global"]
    arch = dict(inherited.get("architecture", {}))
    arch.update(
        {
            "categorical_cardinality": card_diag,
            "candidate_subset": subset_diag,
            "birth_center_prior_diagnostics": center_diag,
            "candidate_reconstruction": reconstruction,
            "candidate_ranker_controls_cardinality": False,
            "runtime_topk_unique_by_construction": True,
            "runtime_direct_global_f1": float(direct_global["f1"]),
            "same_realized_count_frozen_ranking_global_f1": float(frozen_global["f1"]),
            "direct_minus_frozen_ranking_global_f1": float(direct_global["f1"] - frozen_global["f1"]),
        }
    )
    report["v240"] = {
        **inherited,
        "model_key": MODEL_KEY,
        "architecture": arch,
        "same_realized_count_frozen_ranking_strata": frozen_ranking_strata,
    }

    npz_path = args.output_dir / f"predictions-fold-{args.outer_fold}.npz"
    with np.load(npz_path, allow_pickle=False) as z:
        data = {key: np.asarray(z[key]) for key in z.files}
    if v190.PRED_KEY not in data:
        raise V240Error(f"missing inherited prediction key {v190.PRED_KEY}")
    data[PRED_KEY] = data.pop(v190.PRED_KEY)
    data["cardinality_prob"] = np.asarray(capture["cardinality"], dtype=np.float32)
    data["categorical_k"] = np.argmax(capture["cardinality"], axis=1).astype(np.int16)
    data["candidate_rank_distribution"] = np.asarray(capture["candidate_rank_distribution"], dtype=np.float32)
    data["candidate_selected_ids"] = selected_ids.astype(np.int16)
    data["candidate_valid_count"] = np.asarray(capture["candidate_valid_count"], dtype=np.int16)
    data["candidate_subset_target"] = subset_target.astype(np.float32)
    np.savez_compressed(npz_path, **data)

    old_w = args.output_dir / f"v190-dense-birth-centers-fold-{args.outer_fold}.weights.h5"
    new_w = args.output_dir / f"v240-categorical-k-candidate-subset-fold-{args.outer_fold}.weights.h5"
    if old_w.exists():
        old_w.replace(new_w)
    (args.output_dir / f"report-fold-{args.outer_fold}.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def train_fold(args):
    global _LAST_CENTER_TARGETS
    if args.seed != DEFAULT_SEED:
        raise V240Error(f"V24.0 requires seed {DEFAULT_SEED}, got {args.seed}")
    if args.arm != BASE_ARM:
        raise V240Error(f"V24.0 only supports {BASE_ARM!r}")

    ctx = v172._fold_context(args)
    specs = [ctx["meta_spec"], ctx["final_spec"]]
    calls = {"count": 0}
    built = []
    captures = []

    def builder():
        i = calls["count"]
        if i >= 2:
            raise V240Error("unexpected model build")
        calls["count"] += 1
        result = _build_model(specs[i])
        built.append(result[0])
        return result

    old_build, old_targets, old_weights, old_decode = v130._build_model, v130._targets, v130._sample_weights, v130._decode
    try:
        v130._build_model = builder
        v130._targets = _targets
        v130._sample_weights = _sample_weights
        v130._decode = _decode_capture_factory(captures)
        report = v130.train_fold(args)
    finally:
        v130._build_model, v130._targets, v130._sample_weights, v130._decode = old_build, old_targets, old_weights, old_decode

    if calls["count"] != 2 or len(captures) != 2 or _LAST_CENTER_TARGETS is None:
        raise V240Error("V24.0 build/target/decode capture failed")

    outer = np.asarray(ctx["outer_idx"], dtype=np.int64)
    raw = built[-1].predict(v102._inputs(ctx["cache"], outer), batch_size=128, verbose=0)
    target = np.asarray(_LAST_CENTER_TARGETS)[outer]
    eligible = np.asarray(_LAST_CENTER_ELIGIBLE)[outer]
    kk = np.asarray(ctx["k"], dtype=np.int32)[outer]
    center_diag = v190._center_diagnostics(np.asarray(raw["birth_center_map"]), target, eligible, kk)
    report = _postprocess(args, report, ctx, captures[-1], center_diag)

    g = report["strata"]["aggregate"][MODEL_KEY]["metrics"]["global"]
    a = report["v240"]["architecture"]
    card = a["categorical_cardinality"]
    subset = a["candidate_subset"]
    print(json.dumps({
        "outer": args.outer_fold,
        "selected_epochs": report["data"]["selected_epochs"],
        "v104_f1": report["strata"]["aggregate"]["v104"]["metrics"]["global"]["f1"],
        "v240_direct_f1": g["f1"],
        "same_realized_count_frozen_ranking_f1": a["same_realized_count_frozen_ranking_global_f1"],
        "direct_minus_frozen_ranking_f1": a["direct_minus_frozen_ranking_global_f1"],
        "pred_ref": g["prediction_reference_ratio"],
        "categorical_k_exact": card["exact"],
        "categorical_k_poly_exact": card["poly_exact"],
        "candidate_subset_exact_poly_feasible": subset["exact_subset_poly_feasible_rate"],
        "candidate_subset_recall": subset["mean_positive_candidate_recall"],
        "runtime_realized_exact_k": subset["runtime_realized_exact_categorical_k_rate"],
    }, indent=2, sort_keys=True))
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
