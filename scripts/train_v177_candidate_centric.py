"""V17.7 candidate-centric birth-object decoder.

V17.6 removed all trainable q-specific heads, but six fixed anonymous seeds still
re-specialized (higher activity Gini, fewer effective slots) and global F1 only
moved +0.069 pp vs V17.3.  At the same time, sharing the proposal machinery
materially reduced raw candidate collisions.  V17.7 removes the remaining
artificial six-slot bottleneck entirely.

Architecture:
  * every one of the 48 causal candidate tokens is a potential birth object;
  * shared candidate/TF contextualization, self-attention and objectness heads;
  * no learned or fixed anonymous query/seed identities;
  * one birth-time distribution is predicted for every candidate object;
  * candidate ownership is intrinsic: object j realizes candidate j;
  * training uses an exact permutation-invariant one-to-one set DP over the six
    possible truth objects (2^6 states) while scanning the 48 candidate tokens;
  * V17.3 mass-preserving coefficients are retained: the no-object coefficient
    mass is divided across the available unmatched candidate tokens so each row
    has exactly the same total presence coefficient mass as V17.3;
  * exact Poisson-binomial true-K NLL remains at weight 0.35, now over candidate
    objectness probabilities rather than six artificial query probabilities;
  * runtime selects candidate objects directly at p>=0.5, capped at the physical
    guitar maximum six by top objectness.  No threshold tuning.

The six event_present/event_time/event_candidate outputs exposed by the Keras
model are reporting-only top-6 views of the candidate-object field.  They own no
trainable parameters and are never used by the V17.7 set loss.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
import re
import sys
from typing import Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for _p in (ROOT, SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from causal_note.guitarset import SLOT_COUNT
from scripts.train_boundaries import group_stem
from scripts.train_v90_structured_cluster_cardinality import MAX_CANDIDATES
from scripts.train_v100_spectral_string_slots import TIME_FRAMES
from scripts import train_v101_string_query_attention as v101
from scripts import train_v102_source_time_assignment as v102
from scripts import train_v120_integrated_birth_source_time as v120
from scripts import train_v130_causal_event_set_decoder as v130
from scripts import train_v171_controlled_assignment_ab as v171
from scripts import train_v172_mass_preserving_exchangeable as v172
from scripts import train_v173_poibin_count_consistency as v173

DEFAULT_SEED = 16061
EVENT_QUERIES = SLOT_COUNT
PRESENCE_THRESHOLD = 0.5
BASE_ARM = "mass_permutation"
MODEL_KEY = "v177_candidate_centric"
QUERY_DIM = 96
COUNT_NLL_WEIGHT = v173.COUNT_NLL_WEIGHT
SET_PRESENCE_WEIGHT = v172.SET_PRESENCE_WEIGHT
SET_TIME_WEIGHT = v172.SET_TIME_WEIGHT
SET_CANDIDATE_WEIGHT = v172.SET_CANDIDATE_WEIGHT
INF = 1e6
EPS = 1e-6


class V177Error(RuntimeError):
    pass


def _input_by_name(model, name):
    for tensor in model.inputs:
        if tensor.name.split(":", 1)[0] == name:
            return tensor
    raise V177Error(f"model input not found: {name}")


def _candidate_mass_weights_np(k: int, valid_count: int, spec: dict):
    """Return V17.3-equivalent object/no-object coefficient mass for candidate tokens."""
    k = int(k)
    valid_count = int(valid_count)
    ow = float(spec["mass"]["object_by_k"][k])
    old_nw = float(spec["mass"]["no_object_by_k"][k])
    old_total = float(k * ow + (EVENT_QUERIES - k) * old_nw)
    negative_total = float((EVENT_QUERIES - k) * old_nw)
    nw = negative_total / float(valid_count - k) if valid_count > k else 0.0
    new_total = float(k * ow + max(0, valid_count - k) * nw)
    return {
        "object_weight": ow,
        "candidate_no_object_weight": nw,
        "old_total_presence_coefficient_mass": old_total,
        "new_total_presence_coefficient_mass": new_total,
    }


def _poibin48_tf(tf, p):
    """Exact P(N=k), k=0..6, for up to 48 candidate Bernoullis.

    States above six are discarded, so entries 0..6 remain exact rather than
    making state six absorbing.
    """
    p = tf.clip_by_value(tf.cast(p, tf.float32), 0.0, 1.0 - 1e-6)
    batch = tf.shape(p)[0]
    dist = tf.concat(
        [tf.ones((batch, 1), tf.float32), tf.zeros((batch, EVENT_QUERIES), tf.float32)],
        axis=1,
    )
    for j in range(MAX_CANDIDATES):
        pj = p[:, j : j + 1]
        shifted = tf.concat([tf.zeros((batch, 1), tf.float32), dist[:, :-1]], axis=1)
        dist = dist * (1.0 - pj) + shifted * pj
    return dist


def _poibin48_np(p):
    x = np.clip(np.asarray(p, dtype=np.float64), 0.0, 1.0 - 1e-12)
    if x.ndim == 1:
        x = x[None, :]
    if x.shape[1] != MAX_CANDIDATES:
        raise V177Error(f"expected {MAX_CANDIDATES} candidate probabilities, got {x.shape}")
    dist = np.zeros((len(x), EVENT_QUERIES + 1), dtype=np.float64)
    dist[:, 0] = 1.0
    for j in range(MAX_CANDIDATES):
        pj = x[:, j : j + 1]
        shifted = np.concatenate([np.zeros((len(x), 1)), dist[:, :-1]], axis=1)
        dist = dist * (1.0 - pj) + shifted * pj
    return dist


def _candidate_set_loss(spec: dict):
    import tensorflow as tf
    from tensorflow import keras

    state_count = 1 << EVENT_QUERIES
    has_bit = np.zeros((EVENT_QUERIES, state_count), dtype=bool)
    predecessor = np.zeros((EVENT_QUERIES, state_count), dtype=np.int32)
    for t in range(EVENT_QUERIES):
        bit = 1 << t
        for s in range(state_count):
            has_bit[t, s] = bool(s & bit)
            predecessor[t, s] = s & ~bit
    has_bit_tf = tf.constant(has_bit[None, :, :], dtype=tf.bool)
    predecessor_tf = tf.constant(predecessor, dtype=tf.int32)
    state_bits = tf.constant([1 << t for t in range(EVENT_QUERIES)], dtype=tf.int32)
    obj_table = tf.constant(np.asarray(spec["mass"]["object_by_k"], dtype=np.float32))
    null_table = tf.constant(np.asarray(spec["mass"]["no_object_by_k"], dtype=np.float32))

    class CandidateCentricSetLoss(keras.losses.Loss):
        def __init__(self):
            super().__init__(name="v177_candidate_centric_set_loss")

        def call(self, y_true, y_pred):
            yt = tf.cast(y_true, tf.float32)
            yp = tf.cast(y_pred, tf.float32)
            truth_present = yt[:, :, 0]
            truth_valid = yt[:, :, v171.SET_VALID_OFFSET]
            truth_time = yt[:, :, v171.SET_TIME_OFFSET : v171.SET_CANDIDATE_OFFSET]
            truth_candidate = yt[:, :, v171.SET_CANDIDATE_OFFSET :]

            pred_present = tf.clip_by_value(yp[:, :, 0], 1e-6, 1.0 - 1e-6)
            pred_valid = tf.cast(yp[:, :, v171.SET_VALID_OFFSET] > 0.5, tf.float32)
            pred_time = tf.clip_by_value(
                yp[:, :, v171.SET_TIME_OFFSET : v171.SET_CANDIDATE_OFFSET], 1e-7, 1.0
            )

            true_k = tf.cast(tf.reduce_sum(truth_present, axis=1), tf.int32)
            valid_count = tf.reduce_sum(pred_valid, axis=1)
            obj_w = tf.gather(obj_table, true_k)
            old_null_w = tf.gather(null_table, true_k)
            negative_mass = (float(EVENT_QUERIES) - tf.cast(true_k, tf.float32)) * old_null_w
            negative_slots = tf.maximum(valid_count - tf.cast(true_k, tf.float32), 0.0)
            neg_w = tf.where(negative_slots > 0.0, negative_mass / tf.maximum(negative_slots, 1.0), 0.0)

            positive_presence = (
                tf.constant(SET_PRESENCE_WEIGHT, tf.float32)
                * obj_w[:, None, None]
                * (-tf.math.log(pred_present[:, :, None]))
            )
            negative_presence = (
                tf.constant(SET_PRESENCE_WEIGHT, tf.float32)
                * neg_w[:, None]
                * (-tf.math.log(1.0 - pred_present))
                * pred_valid
            )

            time_cost = -tf.einsum("btd,bcd->bct", truth_time, tf.math.log(pred_time))
            target_candidate_mass = tf.transpose(truth_candidate, [0, 2, 1])
            candidate_cost = -tf.math.log(tf.clip_by_value(target_candidate_mass, 1e-7, 1.0))
            detail = (truth_present * truth_valid)[:, None, :]
            match_cost = (
                positive_presence
                + detail
                * (
                    tf.constant(SET_TIME_WEIGHT, tf.float32) * time_cost
                    + tf.constant(SET_CANDIDATE_WEIGHT, tf.float32) * candidate_cost
                )
            )

            batch = tf.shape(yp)[0]
            dp = tf.concat(
                [tf.zeros((batch, 1), tf.float32), tf.fill((batch, state_count - 1), tf.constant(INF, tf.float32))],
                axis=1,
            )
            active_truth = truth_present > 0.5
            for j in range(MAX_CANDIDATES):
                valid_j = pred_valid[:, j] > 0.5
                unmatched = dp + negative_presence[:, j : j + 1]
                transitions = []
                for t in range(EVENT_QUERIES):
                    prev = tf.gather(dp, predecessor_tf[t], axis=1)
                    tr = prev + match_cost[:, j, t : t + 1]
                    allowed = (
                        has_bit_tf[:, t, :]
                        & active_truth[:, t : t + 1]
                        & valid_j[:, None]
                    )
                    transitions.append(tf.where(allowed, tr, tf.constant(INF, tf.float32)))
                matched = tf.reduce_min(tf.stack(transitions, axis=1), axis=1)
                dp = tf.minimum(unmatched, matched)

            target_state = tf.reduce_sum(
                tf.cast(active_truth, tf.int32) * state_bits[None, :], axis=1
            )
            batch_i = tf.range(batch, dtype=tf.int32)
            set_cost = tf.gather_nd(dp, tf.stack([batch_i, target_state], axis=1))

            count_dist = _poibin48_tf(tf, pred_present * pred_valid)
            true_prob = tf.gather_nd(count_dist, tf.stack([batch_i, true_k], axis=1))
            count_nll = -tf.math.log(tf.clip_by_value(true_prob, 1e-7, 1.0))
            return set_cost + tf.constant(COUNT_NLL_WEIGHT, tf.float32) * count_nll

    return CandidateCentricSetLoss()


def _build_model(spec: dict):
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

    # Candidate tokens are the object tokens.  No seed/query tensor exists.
    cand = keras.layers.TimeDistributed(
        keras.layers.LayerNormalization(), name="v177_candidate_norm"
    )(candidate_set)
    cand = keras.layers.TimeDistributed(
        keras.layers.Dense(QUERY_DIM, activation="relu"), name="v177_candidate_hidden1"
    )(cand)
    cand = keras.layers.TimeDistributed(
        keras.layers.Dense(QUERY_DIM, activation="relu"), name="v177_candidate_hidden2"
    )(cand)

    context = keras.layers.Dense(QUERY_DIM, activation="relu", name="v177_global_context")(candidate_context)
    context = keras.layers.Lambda(
        lambda x: tf.tile(x[:, None, :], [1, MAX_CANDIDATES, 1]),
        name="v177_global_context_broadcast",
    )(context)
    cand = keras.layers.Add(name="v177_candidate_plus_context")([cand, context])
    cand = keras.layers.LayerNormalization(name="v177_candidate_context_norm")(cand)

    tf_att = keras.layers.MultiHeadAttention(
        num_heads=4, key_dim=24, dropout=0.05, name="v177_candidate_tf_cross_attention"
    )(cand, tf_tokens)
    cand = keras.layers.Add(name="v177_candidate_tf_residual")([cand, tf_att])
    cand = keras.layers.LayerNormalization(name="v177_candidate_tf_norm")(cand)

    self_mask = keras.layers.Lambda(
        lambda m: tf.tile(tf.cast(m[:, None, :] > 0.5, tf.bool), [1, MAX_CANDIDATES, 1]),
        name="v177_candidate_self_attention_mask",
    )(candidate_mask)
    self_att = keras.layers.MultiHeadAttention(
        num_heads=4, key_dim=24, dropout=0.05, name="v177_candidate_self_attention"
    )(cand, cand, attention_mask=self_mask)
    cand = keras.layers.Add(name="v177_candidate_self_residual")([cand, self_att])
    cand = keras.layers.LayerNormalization(name="v177_candidate_self_norm")(cand)

    ff = keras.layers.TimeDistributed(
        keras.layers.Dense(192, activation="relu"), name="v177_candidate_ff1"
    )(cand)
    ff = keras.layers.Dropout(0.08, name="v177_candidate_dropout")(ff)
    ff = keras.layers.TimeDistributed(
        keras.layers.Dense(QUERY_DIM), name="v177_candidate_ff2"
    )(ff)
    cand = keras.layers.Add(name="v177_candidate_ff_residual")([cand, ff])
    cand = keras.layers.LayerNormalization(name="v177_candidate_ff_norm")(cand)

    object_hidden = keras.layers.TimeDistributed(
        keras.layers.Dense(64, activation="relu"), name="v177_shared_object_hidden"
    )(cand)
    object_raw = keras.layers.TimeDistributed(
        keras.layers.Dense(1, activation="sigmoid"), name="v177_shared_objectness"
    )(object_hidden)
    object_raw = keras.layers.Lambda(
        lambda x: tf.squeeze(x, axis=-1), name="v177_objectness_squeeze"
    )(object_raw)
    objectness = keras.layers.Multiply(name="candidate_objectness")([object_raw, candidate_mask])

    # Candidate-conditioned TF evidence -> one birth-time distribution per candidate.
    time_q = keras.layers.TimeDistributed(
        keras.layers.Dense(QUERY_DIM, use_bias=False), name="v177_time_query"
    )(cand)
    time_k = keras.layers.Dense(QUERY_DIM, use_bias=False, name="v177_time_keys")(tf_tokens)
    time_scores = keras.layers.Lambda(
        lambda z: tf.einsum("bcd,btd->bct", z[0], z[1]) / math.sqrt(float(QUERY_DIM)),
        name="v177_candidate_tf_time_scores",
    )([time_q, time_k])
    token_freq = int(token_shape[1])
    time_grid = keras.layers.Reshape(
        (MAX_CANDIDATES, TIME_FRAMES, token_freq), name="v177_candidate_time_grid"
    )(time_scores)
    time_mass = keras.layers.Lambda(
        lambda x: tf.reduce_logsumexp(x, axis=3), name="v177_candidate_time_mass"
    )(time_grid)
    candidate_time = keras.layers.Softmax(axis=2, name="candidate_time_distribution")(time_mass)

    eye_const = tf.constant(np.eye(MAX_CANDIDATES, dtype=np.float32))
    candidate_identity = keras.layers.Lambda(
        lambda m: tf.tile(eye_const[None, :, :], [tf.shape(m)[0], 1, 1]) * m[:, :, None],
        name="v177_candidate_identity_distribution",
    )(candidate_mask)

    packed = keras.layers.Concatenate(axis=2, name="event_set")([
        keras.layers.Lambda(lambda p: p[:, :, None], name="v177_objectness_expand")(objectness),
        keras.layers.Lambda(lambda m: m[:, :, None], name="v177_valid_expand")(candidate_mask),
        candidate_time,
        candidate_identity,
    ])

    # Reporting-only top-six view.  No trainable parameter lives downstream.
    masked_rank = keras.layers.Lambda(
        lambda z: tf.where(z[1] > 0.5, z[0], tf.cast(-1e9, z[0].dtype)),
        name="v177_reporting_masked_objectness",
    )([objectness, candidate_mask])
    top_scores, top_ids = keras.layers.Lambda(
        lambda x: tf.math.top_k(x, k=EVENT_QUERIES, sorted=True),
        name="candidate_top6",
    )(masked_rank)
    top_valid = keras.layers.Lambda(
        lambda z: tf.gather(z[0], z[1], batch_dims=1), name="v177_top6_valid"
    )([candidate_mask, top_ids])
    top_presence = keras.layers.Lambda(
        lambda z: tf.gather(z[0], z[1], batch_dims=1) * z[2], name="v177_top6_presence"
    )([objectness, top_ids, top_valid])
    top_time = keras.layers.Lambda(
        lambda z: tf.gather(z[0], z[1], batch_dims=1), name="v177_top6_time"
    )([candidate_time, top_ids])
    top_candidate = keras.layers.Lambda(
        lambda z: tf.one_hot(z[0], MAX_CANDIDATES, dtype=tf.float32) * z[1][:, :, None],
        name="v177_top6_candidate_identity",
    )([top_ids, top_valid])

    count_norm = keras.layers.Lambda(
        lambda p: tf.reduce_sum(p, axis=1, keepdims=True) / float(EVENT_QUERIES),
        name="event_count_norm",
    )(objectness)

    outputs = {}
    for slot in range(SLOT_COUNT):
        outputs[f"string_{slot}"] = base.get_layer(f"string_{slot}").output
        outputs[f"pitch_{slot}"] = base.get_layer(f"pitch_{slot}").output
        outputs[f"time_{slot}"] = base.get_layer(f"time_{slot}").output
    for q in range(EVENT_QUERIES):
        outputs[f"event_present_{q}"] = keras.layers.Lambda(
            lambda x, i=q: x[:, i : i + 1], name=f"event_present_{q}"
        )(top_presence)
        outputs[f"event_time_{q}"] = keras.layers.Lambda(
            lambda x, i=q: x[:, i, :], name=f"event_time_{q}"
        )(top_time)
        outputs[f"event_candidate_{q}"] = keras.layers.Lambda(
            lambda x, i=q: x[:, i, :], name=f"event_candidate_{q}"
        )(top_candidate)
    outputs["event_set"] = packed
    outputs["event_count_norm"] = count_norm
    outputs["candidate_objectness"] = objectness
    outputs["candidate_time_distribution"] = candidate_time
    outputs["candidate_selected_ids"] = top_ids
    outputs["candidate_selected_scores"] = top_scores

    loss = {f"string_{slot}": "binary_crossentropy" for slot in range(SLOT_COUNT)}
    loss.update({f"pitch_{slot}": "mse" for slot in range(SLOT_COUNT)})
    loss.update({f"time_{slot}": keras.losses.KLDivergence() for slot in range(SLOT_COUNT)})
    loss["event_set"] = _candidate_set_loss(spec)
    loss["event_count_norm"] = "mse"

    lw = {f"string_{slot}": 0.18 for slot in range(SLOT_COUNT)}
    lw.update({f"pitch_{slot}": 0.04 for slot in range(SLOT_COUNT)})
    lw.update({f"time_{slot}": 0.10 for slot in range(SLOT_COUNT)})
    lw["event_set"] = 1.0
    lw["event_count_norm"] = 0.0

    model = keras.Model(base.inputs, outputs, name="v177_candidate_centric_decoder")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=2e-4),
        loss=loss,
        loss_weights=lw,
    )
    return model, lw, token_shape


def _decode_capture_factory(captures):
    def decode(model, inputs):
        raw = model.predict(inputs, batch_size=128, verbose=0)
        presence = np.stack(
            [np.asarray(raw[f"event_present_{q}"], dtype=np.float64).reshape(-1) for q in range(EVENT_QUERIES)],
            axis=1,
        )
        time = np.stack(
            [np.asarray(raw[f"event_time_{q}"], dtype=np.float64) for q in range(EVENT_QUERIES)],
            axis=1,
        )
        candidate = np.stack(
            [np.asarray(raw[f"event_candidate_{q}"], dtype=np.float64) for q in range(EVENT_QUERIES)],
            axis=1,
        )
        obj = np.asarray(raw["candidate_objectness"], dtype=np.float64)
        ids = np.asarray(raw["candidate_selected_ids"], dtype=np.int32)
        mask = np.asarray(inputs["candidate_mask"], dtype=np.float64)
        valid = np.take_along_axis(mask, ids, axis=1) > 0.5
        active = (presence >= PRESENCE_THRESHOLD) & valid
        selected = np.where(active, ids, -1).astype(np.int32)
        pred = np.sum(active, axis=1).astype(np.int32)
        prefix_violation = np.zeros(len(pred), dtype=bool)  # top-k view is sorted by construction
        captures.append({
            "candidate_objectness": obj,
            "candidate_mask": mask,
            "selected_candidate_ids": selected,
            "selected_candidate_scores": np.take_along_axis(obj, ids, axis=1),
        })
        return pred, presence, time, candidate, prefix_violation
    return decode


def _direct_prediction_map(cache, candidate_samples, outer_idx, selected_ids, local_mask):
    retained = defaultdict(list)
    local_rows = np.flatnonzero(np.asarray(local_mask, dtype=bool))
    for local in local_rows:
        global_idx = int(outer_idx[local])
        member = str(cache["members"][global_idx])
        retained[member]  # preserve zero-prediction members
        samples = np.asarray(candidate_samples[global_idx], dtype=np.int32)
        for candidate_id in np.asarray(selected_ids[local], dtype=np.int32):
            if candidate_id < 0:
                continue
            if candidate_id >= len(samples):
                raise V177Error(
                    f"candidate id {candidate_id} outside reconstructed candidate count {len(samples)}"
                )
            retained[member].append(int(samples[candidate_id]))
    return {member: tuple(sorted(values)) for member, values in retained.items()}


def _stratum_mask(name: str, outer_members, players, modes, genres):
    if name == "aggregate":
        return np.ones(len(outer_members), dtype=bool)
    if name == "comp":
        return modes == "comp"
    if name == "solo":
        return modes == "solo"
    if name == "player00_comp":
        return (players == "00") & (modes == "comp")
    if name == "player00_solo":
        return (players == "00") & (modes == "solo")
    if name == "player00_rock_comp":
        return (players == "00") & (modes == "comp") & (genres == "Rock")
    m = re.fullmatch(r"player([0-9]{2})", name)
    if m:
        return players == m.group(1)
    return None


def _postprocess(args, report, ctx, capture):
    # Reuse all proven V17.3 row/protocol/cardinality plumbing, then replace the
    # model's event-realization metrics with the candidate objects it actually selected.
    report = v173._postprocess(args, report, ctx)
    v171._rename_report(report, v173.MODEL_KEY, MODEL_KEY)
    inherited = report.pop("v173")
    inherited.pop("final_model_presence_gradient_mass", None)
    inherited.pop("final_model_presence_gradient_mass_error", None)

    report["protocol"].update({
        "v177_candidate_centric": True,
        "v177_base_version": "V17.3 objective / V10.2 evidence backbone",
        "v177_only_architecture_change": "six anonymous proposal slots -> causal candidate tokens as birth objects",
        "fixed_or_trainable_anonymous_seed_count": 0,
        "candidate_object_token_count": MAX_CANDIDATES,
        "candidate_ownership_intrinsic_one_to_one": True,
        "candidate_set_dp_states": 1 << EVENT_QUERIES,
        "candidate_set_dp_permutation_invariant": True,
        "v173_mass_coefficient_preservation": True,
        "v173_poisson_binomial_count_objective_weight_unchanged": True,
        "v173_count_nll_weight": COUNT_NLL_WEIGHT,
        "poisson_binomial_domain": "up to 48 masked candidate objectness Bernoullis; exact states K=0..6",
        "runtime_candidate_realization": "direct candidate objects with p>=0.5, top-6 cap",
        "headline_candidate_realization_changed_from_v173": True,
        "runtime_presence_threshold": PRESENCE_THRESHOLD,
        "runtime_presence_threshold_tuned": False,
        "categorical_cardinality_head_exists": False,
        "v175_injective_ownership_used": False,
        "historical_validation_or_locked12_indexed_or_evaluated": False,
    })

    outer_idx = np.asarray(ctx["outer_idx"], dtype=np.int64)
    if len(capture["candidate_objectness"]) != len(outer_idx):
        raise V177Error("captured candidate-object rows do not match outer fold")
    cache = ctx["cache"]
    candidate_samples, reconstruction = v102._reconstruct_candidates(cache)
    selected_ids = np.asarray(capture["selected_candidate_ids"], dtype=np.int32)
    objectness = np.asarray(capture["candidate_objectness"], dtype=np.float64)
    candidate_mask = np.asarray(capture["candidate_mask"], dtype=np.float64)

    members = np.asarray(ctx["members"], dtype="U96")
    outer_members = members[outer_idx]
    players = np.asarray([str(m).split("_", 1)[0] for m in outer_members], dtype="U2")
    modes = np.asarray([
        "comp" if str(m).endswith("_comp.jams") else "solo" if str(m).endswith("_solo.jams") else "other"
        for m in outer_members
    ], dtype="U8")
    by_member = {t.annotation_member: t for t in ctx["train_split"]}
    groups = np.asarray([group_stem(by_member[str(m)]) for m in outer_members], dtype="U64")
    genres = np.asarray([v120._genre(g) for g in groups], dtype="U16")

    count_only_strata = {}
    for name, row in report.get("strata", {}).items():
        if not row or MODEL_KEY not in row:
            continue
        mask = _stratum_mask(name, outer_members, players, modes, genres)
        if mask is None or not np.any(mask):
            continue
        count_only_strata[name] = row[MODEL_KEY]["metrics"]
        pmap = _direct_prediction_map(cache, candidate_samples, outer_idx, selected_ids, mask)
        member_set = {str(m) for m in outer_members[mask]}
        tracks = tuple(t for t in ctx["train_split"] if t.annotation_member in member_set)
        row[MODEL_KEY]["metrics"] = v101._metrics(tracks, pmap)

    hard_count = np.sum(selected_ids >= 0, axis=1).astype(np.int32)
    true_k = np.asarray(ctx["k"], dtype=np.int32)[outer_idx]
    valid_count = np.sum(candidate_mask > 0.5, axis=1).astype(np.int32)
    unique_ok = []
    for ids in selected_ids:
        keep = ids[ids >= 0]
        unique_ok.append(len(keep) == len(np.unique(keep)))

    mass_proof = {}
    for value in range(EVENT_QUERIES + 1):
        rows = true_k == value
        counts = valid_count[rows]
        if not len(counts):
            continue
        probes = [_candidate_mass_weights_np(value, int(c), ctx["final_spec"]) for c in np.unique(counts)]
        mass_proof[str(value)] = {
            "valid_candidate_count_min": int(np.min(counts)),
            "valid_candidate_count_max": int(np.max(counts)),
            "max_abs_mass_error": float(max(abs(p["new_total_presence_coefficient_mass"] - p["old_total_presence_coefficient_mass"]) for p in probes)),
        }

    direct_global = report["strata"]["aggregate"][MODEL_KEY]["metrics"]["global"]
    count_global = count_only_strata["aggregate"]["global"]
    architecture = {
        "candidate_tokens_are_object_tokens": True,
        "anonymous_seed_count": 0,
        "candidate_token_permutation_equivariant_by_construction": True,
        "outer_valid_candidate_count_mean": float(np.mean(valid_count)),
        "outer_valid_candidate_count_min": int(np.min(valid_count)),
        "outer_valid_candidate_count_max": int(np.max(valid_count)),
        "outer_soft_object_count": float(np.mean(np.sum(objectness, axis=1))),
        "outer_hard_object_count": float(np.mean(hard_count)),
        "outer_true_count": float(np.mean(true_k)),
        "selected_candidate_ids_unique_rate": float(np.mean(unique_ok)),
        "direct_global_f1": float(direct_global["f1"]),
        "count_only_frozen_ranking_global_f1": float(count_global["f1"]),
        "direct_minus_count_only_global_f1": float(direct_global["f1"] - count_global["f1"]),
        "candidate_reconstruction": reconstruction,
        "mass_preservation_by_true_k": mass_proof,
    }
    report["v177"] = {
        **inherited,
        "model_key": MODEL_KEY,
        "architecture": architecture,
        "count_only_strata": count_only_strata,
    }

    npz_path = args.output_dir / f"predictions-fold-{args.outer_fold}.npz"
    with np.load(npz_path, allow_pickle=False) as z:
        data = {key: np.asarray(z[key]) for key in z.files}
    old_pred = "pred173_poibin"
    if old_pred not in data:
        raise V177Error(f"missing inherited prediction key {old_pred}")
    data["pred177_candidate_centric"] = data.pop(old_pred)
    data["candidate_objectness"] = objectness.astype(np.float32)
    data["candidate_selected_ids"] = selected_ids.astype(np.int16)
    data["candidate_selected_scores"] = np.asarray(capture["selected_candidate_scores"], dtype=np.float32)
    data["candidate_valid_count"] = valid_count.astype(np.int16)
    np.savez_compressed(npz_path, **data)

    old_w = args.output_dir / f"v173-poibin-fold-{args.outer_fold}.weights.h5"
    new_w = args.output_dir / f"v177-candidate-centric-fold-{args.outer_fold}.weights.h5"
    if old_w.exists():
        old_w.replace(new_w)

    report_path = args.output_dir / f"report-fold-{args.outer_fold}.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def train_fold(args):
    if args.seed != DEFAULT_SEED:
        raise V177Error(f"V17.7 requires seed {DEFAULT_SEED}, got {args.seed}")
    if args.arm != BASE_ARM:
        raise V177Error(f"V17.7 only supports base arm {BASE_ARM!r}")

    ctx = v172._fold_context(args)
    stage_specs = [ctx["meta_spec"], ctx["final_spec"]]
    calls = {"count": 0}
    captures = []

    def builder():
        i = calls["count"]
        if i >= 2:
            raise V177Error(f"unexpected model build call {i + 1}")
        calls["count"] += 1
        return _build_model(stage_specs[i])

    old_build = v130._build_model
    old_targets = v130._targets
    old_weights = v130._sample_weights
    old_decode = v130._decode
    try:
        v130._build_model = builder
        v130._targets = v171._targets
        v130._sample_weights = v171._sample_weights
        v130._decode = _decode_capture_factory(captures)
        report = v130.train_fold(args)
    finally:
        v130._build_model = old_build
        v130._targets = old_targets
        v130._sample_weights = old_weights
        v130._decode = old_decode

    if calls["count"] != 2:
        raise V177Error(f"expected exactly 2 model builds, got {calls['count']}")
    if len(captures) != 2:
        raise V177Error(f"expected meta+outer decode captures, got {len(captures)}")

    report = _postprocess(args, report, ctx, captures[-1])
    g = report["strata"]["aggregate"][MODEL_KEY]["metrics"]["global"]
    card = report["strata"]["aggregate"][MODEL_KEY]["cardinality"]
    arch = report["v177"]["architecture"]
    print(json.dumps({
        "outer": args.outer_fold,
        "selected_epochs": report["data"]["selected_epochs"],
        "v104_f1": report["strata"]["aggregate"]["v104"]["metrics"]["global"]["f1"],
        "v177_direct_f1": g["f1"],
        "v177_count_only_f1": arch["count_only_frozen_ranking_global_f1"],
        "direct_minus_count_only_f1": arch["direct_minus_count_only_global_f1"],
        "pred_ref": g["prediction_reference_ratio"],
        "poly_exact": card.get("poly_cluster_accuracy", card.get("poly_accuracy", card.get("poly_exact_accuracy"))),
        "soft_object_count": arch["outer_soft_object_count"],
        "hard_object_count": arch["outer_hard_object_count"],
        "true_count": arch["outer_true_count"],
        "selected_unique_rate": arch["selected_candidate_ids_unique_rate"],
        "k2": report["per_true_k"]["2"][MODEL_KEY]["exact"],
        "k3": report["per_true_k"]["3"][MODEL_KEY]["exact"],
        "k4": report["per_true_k"]["4"][MODEL_KEY]["exact"],
        "k5": report["per_true_k"]["5"][MODEL_KEY]["exact"],
        "k6": report["per_true_k"]["6"][MODEL_KEY]["exact"],
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
