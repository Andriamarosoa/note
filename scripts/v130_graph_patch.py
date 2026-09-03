"""Audited graph fix for V13.0.

The initial V13 preflight failed before training because one Lambda captured the
external KerasTensor ``candidate_mask`` instead of receiving it as an explicit
input.  TensorFlow 2.15 cannot build a TypeSpec for that closure.  This module
provides the same V13 builder with that single graph-wiring correction.
"""
from __future__ import annotations

import math
import numpy as np

from scripts import train_v130_causal_event_set_decoder as v130
from scripts import train_v102_source_time_assignment as v102
from scripts.train_v100_spectral_string_slots import TIME_FRAMES
from causal_note.guitarset import SLOT_COUNT


def build_model_fixed():
    try:
        import tensorflow as tf
        from tensorflow import keras
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required") from exc

    base, _, token_shape = v102._build_model()
    candidate_context = base.get_layer("candidate_context").output
    tf_tokens = base.get_layer("tf_tokens").output
    candidate_set = v130._input_by_name(base, "candidate_set")
    candidate_mask = v130._input_by_name(base, "candidate_mask")

    cand = keras.layers.TimeDistributed(keras.layers.LayerNormalization(), name="event_candidate_norm")(candidate_set)
    cand = keras.layers.TimeDistributed(keras.layers.Dense(v130.QUERY_DIM, activation="relu"), name="event_candidate_hidden1")(cand)
    cand = keras.layers.TimeDistributed(keras.layers.Dense(v130.QUERY_DIM, activation="relu"), name="event_candidate_hidden2")(cand)
    cand_keys = keras.layers.TimeDistributed(
        keras.layers.Dense(v130.QUERY_DIM, use_bias=False), name="event_candidate_keys"
    )(cand)
    tf_keys = keras.layers.Dense(v130.QUERY_DIM, use_bias=False, name="event_tf_keys")(tf_tokens)

    queries = [
        keras.layers.Dense(v130.QUERY_DIM, activation="relu", name=f"event_{q}_query")(candidate_context)
        for q in range(v130.EVENT_QUERIES)
    ]

    tf_scores = [
        keras.layers.Lambda(
            lambda z: tf.einsum("btd,bd->bt", z[0], z[1]) / math.sqrt(float(v130.QUERY_DIM)),
            name=f"event_{q}_tf_score",
        )([tf_keys, queries[q]])
        for q in range(v130.EVENT_QUERIES)
    ]
    tf_background = keras.layers.Lambda(lambda x: tf.squeeze(x, axis=-1), name="event_tf_background_score")(
        keras.layers.Dense(1, name="event_tf_background_dense")(tf_tokens)
    )
    tf_score_stack = keras.layers.Lambda(
        lambda xs: tf.stack(xs, axis=-1), name="event_tf_score_stack"
    )(tf_scores + [tf_background])
    tf_assignment = keras.layers.Softmax(axis=-1, name="event_tf_competition")(tf_score_stack)

    cand_scores = [
        keras.layers.Lambda(
            lambda z: tf.einsum("bcd,bd->bc", z[0], z[1]) / math.sqrt(float(v130.QUERY_DIM)),
            name=f"event_{q}_candidate_score",
        )([cand_keys, queries[q]])
        for q in range(v130.EVENT_QUERIES)
    ]
    cand_background = keras.layers.Lambda(
        lambda x: tf.squeeze(x, axis=-1), name="event_candidate_background_score"
    )(
        keras.layers.TimeDistributed(
            keras.layers.Dense(1), name="event_candidate_background_dense"
        )(cand)
    )
    cand_score_stack = keras.layers.Lambda(
        lambda xs: tf.stack(xs, axis=-1), name="event_candidate_score_stack"
    )(cand_scores + [cand_background])
    cand_assignment = keras.layers.Softmax(axis=-1, name="event_candidate_competition")(cand_score_stack)

    token_freq = int(token_shape[1])
    tf_grid = keras.layers.Reshape(
        (TIME_FRAMES, token_freq, v130.EVENT_QUERIES + 1), name="event_tf_assignment_grid"
    )(tf_assignment)

    presence_outputs = []
    time_outputs = []
    candidate_outputs = []
    for q in range(v130.EVENT_QUERIES):
        spec_w = keras.layers.Lambda(
            lambda a, i=q: a[:, :, i], name=f"event_{q}_tf_weights"
        )(tf_assignment)
        spec_latent = keras.layers.Lambda(
            lambda z: tf.reduce_sum(z[0] * z[1][:, :, None], axis=1)
            / (tf.reduce_sum(z[1], axis=1, keepdims=True) + 1e-6),
            name=f"event_{q}_tf_pool",
        )([tf_tokens, spec_w])

        raw_cand_w = keras.layers.Lambda(
            lambda a, i=q: a[:, :, i], name=f"event_{q}_raw_candidate_weights"
        )(cand_assignment)
        cand_w = keras.layers.Multiply(name=f"event_{q}_masked_candidate_weights")([
            raw_cand_w, candidate_mask
        ])
        cand_latent = keras.layers.Lambda(
            lambda z: tf.reduce_sum(z[0] * z[1][:, :, None], axis=1)
            / (tf.reduce_sum(z[1], axis=1, keepdims=True) + 1e-6),
            name=f"event_{q}_candidate_pool",
        )([cand, cand_w])
        cand_dist = keras.layers.Lambda(
            lambda w: w / (tf.reduce_sum(w, axis=1, keepdims=True) + 1e-6),
            name=f"event_candidate_{q}",
        )(cand_w)

        time_mass = keras.layers.Lambda(
            lambda a, i=q: tf.reduce_sum(a[:, :, :, i], axis=2),
            name=f"event_{q}_time_mass",
        )(tf_grid)
        time_dist = keras.layers.Lambda(
            lambda t: t / (tf.reduce_sum(t, axis=1, keepdims=True) + 1e-6),
            name=f"event_time_{q}",
        )(time_mass)

        spec_mass = keras.layers.Lambda(
            lambda w: tf.reduce_mean(w, axis=1, keepdims=True), name=f"event_{q}_tf_mass"
        )(spec_w)
        # Graph fix: candidate_mask is an explicit Lambda input, not a closure.
        cand_mass = keras.layers.Lambda(
            lambda z: tf.reduce_sum(z[0], axis=1, keepdims=True)
            / (tf.reduce_sum(z[1], axis=1, keepdims=True) + 1e-6),
            name=f"event_{q}_candidate_mass",
        )([cand_w, candidate_mask])

        feature = keras.layers.Concatenate(name=f"event_{q}_feature")([
            candidate_context, queries[q], spec_latent, cand_latent, spec_mass, cand_mass
        ])
        feature = keras.layers.LayerNormalization(name=f"event_{q}_feature_norm")(feature)
        feature = keras.layers.Dense(
            128, activation="relu", kernel_regularizer=keras.regularizers.l2(1.5e-3),
            name=f"event_{q}_hidden1",
        )(feature)
        feature = keras.layers.Dropout(0.08, name=f"event_{q}_dropout")(feature)
        feature = keras.layers.Dense(64, activation="relu", name=f"event_{q}_hidden2")(feature)
        present = keras.layers.Dense(1, activation="sigmoid", name=f"event_present_{q}")(feature)

        presence_outputs.append(present)
        time_outputs.append(time_dist)
        candidate_outputs.append(cand_dist)

    presence_vector = keras.layers.Concatenate(name="event_presence_vector")(presence_outputs)
    count_norm = keras.layers.Lambda(
        lambda p: tf.reduce_sum(p, axis=1, keepdims=True) / float(v130.EVENT_QUERIES),
        name="event_count_norm",
    )(presence_vector)

    outputs = {}
    for slot in range(SLOT_COUNT):
        outputs[f"string_{slot}"] = base.get_layer(f"string_{slot}").output
        outputs[f"pitch_{slot}"] = base.get_layer(f"pitch_{slot}").output
        outputs[f"time_{slot}"] = base.get_layer(f"time_{slot}").output
    for q in range(v130.EVENT_QUERIES):
        outputs[f"event_present_{q}"] = presence_outputs[q]
        outputs[f"event_time_{q}"] = time_outputs[q]
        outputs[f"event_candidate_{q}"] = candidate_outputs[q]
    outputs["event_count_norm"] = count_norm

    loss = {f"string_{slot}": "binary_crossentropy" for slot in range(SLOT_COUNT)}
    loss.update({f"pitch_{slot}": "mse" for slot in range(SLOT_COUNT)})
    loss.update({f"time_{slot}": keras.losses.KLDivergence() for slot in range(SLOT_COUNT)})
    for q in range(v130.EVENT_QUERIES):
        loss[f"event_present_{q}"] = "binary_crossentropy"
        loss[f"event_time_{q}"] = keras.losses.KLDivergence()
        loss[f"event_candidate_{q}"] = "categorical_crossentropy"
    loss["event_count_norm"] = "mse"

    loss_weights = {f"string_{slot}": 0.18 for slot in range(SLOT_COUNT)}
    loss_weights.update({f"pitch_{slot}": 0.04 for slot in range(SLOT_COUNT)})
    loss_weights.update({f"time_{slot}": 0.10 for slot in range(SLOT_COUNT)})
    for q in range(v130.EVENT_QUERIES):
        loss_weights[f"event_present_{q}"] = 1.0
        loss_weights[f"event_time_{q}"] = 0.30
        loss_weights[f"event_candidate_{q}"] = 0.25
    loss_weights["event_count_norm"] = 0.35

    model = keras.Model(base.inputs, outputs, name="v130_causal_anonymous_event_set")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=2e-4),
        loss=loss,
        loss_weights=loss_weights,
    )
    return model, loss_weights, token_shape


def apply():
    v130._build_model = build_model_fixed
    return v130
