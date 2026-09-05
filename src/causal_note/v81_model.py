"""V8.1 causal burst training wrapper around the V8 stream backbone.

The deployable stream model remains fully causal and keeps the same four public
score tensors. V8.1 changes what is optimized during training: each reference
burst owns a causal response bag, and the model is rewarded for producing one
sparse, early pulse anywhere inside that bag instead of being forced to peak at
one exact annotation sample.
"""
from __future__ import annotations

from .v8_model import build_v8_stream_model
from .v81_targets import (
    DEFAULT_OFFSET_HORIZON_SAMPLES,
    DEFAULT_ONSET_HORIZON_SAMPLES,
)


def _load_tensorflow():
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is required to build V8.1; install requirements-train.txt"
        ) from exc
    return tf


def build_v81_bag_model(
    *,
    filters: int = 32,
    kernel_size: int = 5,
    dilation_rates=(1, 2, 4, 8, 16, 32, 64, 128, 256, 512),
    onset_horizon_samples: int = DEFAULT_ONSET_HORIZON_SAMPLES,
    offset_horizon_samples: int = DEFAULT_OFFSET_HORIZON_SAMPLES,
):
    """Build the V8.1 MIL-style training model and nested stream model.

    Outputs per boundary kind:
    - ``*_bag_presence``: maximum hazard in the allowed causal response bag;
    - ``*_mass``: mean hazard density. A perfect single-sample pulse of height
      one has target density ``1 / bag_length``;
    - ``*_delay``: hazard-weighted normalized response delay, minimized only on
      positive bags;
    - ``*_count``: hazard-attended anonymous burst cardinality 1/2/3+.
    """
    for name, value in (
        ("onset_horizon_samples", onset_horizon_samples),
        ("offset_horizon_samples", offset_horizon_samples),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be an integer > 0")

    tf = _load_tensorflow()
    keras = tf.keras
    stream = build_v8_stream_model(
        filters=filters,
        kernel_size=kernel_size,
        dilation_rates=dilation_rates,
        name="causal_boundary_v81_stream",
    )
    receptive_field = int(stream.receptive_field)
    maximum_horizon = max(onset_horizon_samples, offset_horizon_samples)
    context_samples = receptive_field + maximum_horizon

    audio = keras.Input(
        shape=(context_samples, 1),
        dtype=tf.float32,
        name="audio_bag_context",
    )
    sequence = stream(audio)
    response_start = receptive_field - 1

    outputs = {}
    epsilon = tf.keras.backend.epsilon()

    for kind, horizon in (
        ("onset", onset_horizon_samples),
        ("offset", offset_horizon_samples),
    ):
        length = horizon + 1
        presence_sequence = sequence[f"{kind}_presence"]
        count_sequence = sequence[f"{kind}_multiplicity"]

        presence = keras.layers.Lambda(
            lambda tensor, start=response_start, stop=response_start + length: tensor[:, start:stop, :],
            name=f"{kind}_response_presence",
        )(presence_sequence)
        count = keras.layers.Lambda(
            lambda tensor, start=response_start, stop=response_start + length: tensor[:, start:stop, :],
            name=f"{kind}_response_count",
        )(count_sequence)

        outputs[f"{kind}_bag_presence"] = keras.layers.GlobalMaxPooling1D(
            name=f"{kind}_bag_presence"
        )(presence)

        outputs[f"{kind}_mass"] = keras.layers.Lambda(
            lambda tensor: tf.reduce_mean(tensor, axis=1),
            name=f"{kind}_mass",
        )(presence)

        delay_axis = tf.reshape(
            tf.cast(tf.range(length), tf.float32) / float(horizon),
            (1, length, 1),
        )
        outputs[f"{kind}_delay"] = keras.layers.Lambda(
            lambda tensor, axis=delay_axis, eps=epsilon: (
                tf.reduce_sum(tensor * axis, axis=1)
                / (tf.reduce_sum(tensor, axis=1) + eps)
            ),
            name=f"{kind}_delay",
        )(presence)

        outputs[f"{kind}_count"] = keras.layers.Lambda(
            lambda tensors, eps=epsilon: (
                tf.reduce_sum(tensors[0] * tensors[1], axis=1)
                / (tf.reduce_sum(tensors[0], axis=1) + eps)
            ),
            name=f"{kind}_count",
        )([presence, count])

    model = keras.Model(audio, outputs, name="causal_boundary_v81_bag")
    model.receptive_field = receptive_field
    model.maximum_horizon_samples = maximum_horizon
    model.context_samples = context_samples
    model.stream_model_name = stream.name
    model.onset_horizon_samples = onset_horizon_samples
    model.offset_horizon_samples = offset_horizon_samples
    return model


__all__ = ["build_v81_bag_model"]
