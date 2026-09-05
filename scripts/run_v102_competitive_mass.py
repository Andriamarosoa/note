"""V10.2 execution wrapper with assignment-mass-aware source occupancy.

The base V10.2 design normalizes each source's token pool.  This wrapper also
feeds the fraction of TF mass won by each source into its occupancy head, so a
source cannot look active merely because a tiny assigned subset has a plausible
spectral signature.  It also uses a shape-safe differentiable Poisson-binomial
implementation.  All data protocol, losses, calibration and locked evaluation
remain those of train_v102_source_time_assignment.
"""
from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import train_v102_source_time_assignment as v


def _poisson_binomial_tensor_fixed(p):
    import tensorflow as tf

    p = tf.clip_by_value(p, 1e-6, 1.0 - 1e-6)
    dist = tf.ones_like(p[:, :1])
    for slot in range(v.SLOT_COUNT):
        ps = p[:, slot:slot + 1]
        zero = tf.zeros_like(ps)
        left = tf.concat([dist * (1.0 - ps), zero], axis=1)
        right = tf.concat([zero, dist * ps], axis=1)
        dist = left + right
    return dist


def _build_model_mass_aware():
    try:
        import tensorflow as tf
        from tensorflow import keras
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required") from exc

    scaffold = v._build_cluster_model()
    candidate_hidden = scaffold.get_layer("cluster_hidden2").output
    candidate_context = keras.layers.Dense(v.TOKEN_DIM, activation="relu", name="candidate_context")(candidate_hidden)

    spectral = keras.Input((v.TIME_FRAMES, v.SPECTRAL_BANDS, v.SPECTRAL_CHANNELS), name="spectral_map")
    x = keras.layers.LayerNormalization(axis=-1, name="spectral_channel_norm")(spectral)

    time_coord = np.linspace(-1.0, 1.0, v.TIME_FRAMES, dtype=np.float32)[:, None]
    freq_coord = np.linspace(-1.0, 1.0, v.SPECTRAL_BANDS, dtype=np.float32)[None, :]
    coord_grid = np.stack(
        (
            np.broadcast_to(time_coord, (v.TIME_FRAMES, v.SPECTRAL_BANDS)),
            np.broadcast_to(freq_coord, (v.TIME_FRAMES, v.SPECTRAL_BANDS)),
        ),
        axis=-1,
    ).astype(np.float32)
    coord_const = tf.constant(coord_grid, dtype=tf.float32)
    coords = keras.layers.Lambda(
        lambda t: tf.tile(coord_const[None, :, :, :], [tf.shape(t)[0], 1, 1, 1]),
        output_shape=(v.TIME_FRAMES, v.SPECTRAL_BANDS, 2),
        name="absolute_tf_coordinates",
    )(spectral)
    x = keras.layers.Concatenate(axis=-1, name="spectral_plus_coordinates")([x, coords])
    x = keras.layers.Conv2D(32, (3, 5), strides=(1, 2), padding="same", activation="relu", name="st_conv1")(x)
    x = keras.layers.Conv2D(64, (3, 3), strides=(1, 2), padding="same", activation="relu", name="st_conv2")(x)
    x = keras.layers.Conv2D(v.TOKEN_DIM, (3, 3), strides=(1, 2), padding="same", activation="relu", name="st_conv3")(x)
    token_freq = int(math.ceil(math.ceil(math.ceil(v.SPECTRAL_BANDS / 2.0) / 2.0) / 2.0))
    token_count = v.TIME_FRAMES * token_freq
    tokens = keras.layers.Reshape((token_count, v.TOKEN_DIM), name="tf_tokens")(x)
    tokens = keras.layers.LayerNormalization(name="tf_token_norm")(tokens)
    tokens = keras.layers.Dense(v.TOKEN_DIM, activation="relu", name="tf_token_projection")(tokens)
    keys = keras.layers.Dense(v.TOKEN_DIM, use_bias=False, name="tf_assignment_keys")(tokens)

    source_scores = []
    source_queries = []
    for source in range(v.SOURCE_COUNT):
        query = keras.layers.Dense(v.TOKEN_DIM, activation="relu", name=f"source_{source}_query")(candidate_context)
        source_queries.append(query)
        score = keras.layers.Lambda(
            lambda z: tf.einsum("btd,bd->bt", z[0], z[1]) / math.sqrt(float(v.TOKEN_DIM)),
            name=f"source_{source}_score",
        )([keys, query])
        source_scores.append(score)
    score_stack = keras.layers.Lambda(lambda z: tf.stack(z, axis=-1), name="source_score_stack")(source_scores)
    assignment = keras.layers.Softmax(axis=-1, name="competitive_source_assignment")(score_stack)
    assignment_grid = keras.layers.Reshape(
        (v.TIME_FRAMES, token_freq, v.SOURCE_COUNT),
        name="source_assignment_grid",
    )(assignment)

    slot_features = []
    string_outputs = []
    pitch_outputs = []
    time_outputs = []
    for slot in range(v.SLOT_COUNT):
        weights = keras.layers.Lambda(lambda a, s=slot: a[:, :, s], name=f"string_{slot}_token_weights")(assignment)
        mass = keras.layers.Lambda(
            lambda w: tf.reduce_mean(w, axis=1, keepdims=True),
            name=f"string_{slot}_assignment_mass",
        )(weights)
        latent = keras.layers.Lambda(
            lambda z: tf.reduce_sum(z[0] * z[1][:, :, None], axis=1)
            / (tf.reduce_sum(z[1], axis=1, keepdims=True) + 1e-6),
            name=f"string_{slot}_source_pool",
        )([tokens, weights])
        feature = keras.layers.Concatenate(name=f"string_{slot}_fusion")([
            candidate_context,
            source_queries[slot],
            latent,
            mass,
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
            lambda t: t / (tf.reduce_sum(t, axis=1, keepdims=True) + 1e-7),
            name=f"time_{slot}",
        )(time_mass)
        time_outputs.append(time_dist)

    string_stack = keras.layers.Concatenate(name="string_birth_vector")(string_outputs)
    slot_count = keras.layers.Lambda(
        lambda t: tf.reduce_sum(t, axis=1, keepdims=True) / float(v.SLOT_COUNT),
        name="slot_count",
    )(string_stack)
    structured_count = keras.layers.Lambda(_poisson_binomial_tensor_fixed, name="structured_count")(string_stack)
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
        for stage in range(1, v.ORDINAL_STAGES + 1)
    ]

    outputs = {f"string_{slot}": out for slot, out in enumerate(string_outputs)}
    outputs.update({f"pitch_{slot}": out for slot, out in enumerate(pitch_outputs)})
    outputs.update({f"time_{slot}": out for slot, out in enumerate(time_outputs)})
    outputs["slot_count"] = slot_count
    outputs["structured_count"] = structured_count
    outputs["poly_count"] = poly_count
    outputs.update({f"ge{stage}": out for stage, out in enumerate(ordinal_outputs, start=1)})

    loss = {f"string_{slot}": "binary_crossentropy" for slot in range(v.SLOT_COUNT)}
    loss.update({f"pitch_{slot}": "mse" for slot in range(v.SLOT_COUNT)})
    loss.update({f"time_{slot}": keras.losses.KLDivergence() for slot in range(v.SLOT_COUNT)})
    loss["slot_count"] = "mse"
    loss["structured_count"] = "categorical_crossentropy"
    loss["poly_count"] = "categorical_crossentropy"
    loss.update({f"ge{stage}": "binary_crossentropy" for stage in range(1, v.ORDINAL_STAGES + 1)})

    loss_weights = {f"string_{slot}": 0.60 for slot in range(v.SLOT_COUNT)}
    loss_weights.update({f"pitch_{slot}": 0.15 for slot in range(v.SLOT_COUNT)})
    loss_weights.update({f"time_{slot}": 0.35 for slot in range(v.SLOT_COUNT)})
    loss_weights["slot_count"] = 0.20
    loss_weights["structured_count"] = 1.30
    loss_weights["poly_count"] = 0.90
    loss_weights.update({f"ge{stage}": 0.15 * math.sqrt(stage) for stage in range(1, v.ORDINAL_STAGES + 1)})

    model = keras.Model(
        {
            "candidate_set": scaffold.input["candidate_set"],
            "candidate_mask": scaffold.input["candidate_mask"],
            "cluster_stats": scaffold.input["cluster_stats"],
            "spectral_map": spectral,
        },
        outputs,
        name="v102_competitive_source_time_assignment_mass_aware",
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=2e-4), loss=loss, loss_weights=loss_weights)
    return model, loss_weights, (v.TIME_FRAMES, token_freq, token_count)


v._poisson_binomial_tensor = _poisson_binomial_tensor_fixed
v._build_model = _build_model_mass_aware


if __name__ == "__main__":
    raise SystemExit(v.main())
