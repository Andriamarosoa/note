"""V8.3 split-task causal boundary architecture.

V8.3 keeps a short shared acoustic stem, then gives onset and offset independent
causal towers.  The default split happens after dilations 1/2/4/8; both towers
continue through 16/32/64/128/256/512.  A V8.1 stream checkpoint can be copied
into this topology exactly: shared blocks keep their source weights, each
private tower receives a copy of the corresponding later V8.1 blocks, and both
private fusion projections receive the original fusion projection.  Before any
training update the four V8.3 stream outputs therefore match V8.1 numerically.
"""
from __future__ import annotations

from typing import Iterable, Tuple

from .v8_model import (
    DEFAULT_DILATION_RATES,
    MULTIPLICITY_CLASSES,
    calculate_receptive_field,
)
from .v81_targets import (
    DEFAULT_OFFSET_HORIZON_SAMPLES,
    DEFAULT_ONSET_HORIZON_SAMPLES,
)

DEFAULT_SPLIT_AFTER = 4


def _load_tensorflow():
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is required to build V8.3; install requirements-train.txt"
        ) from exc
    return tf


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be an integer > 0")
    return value


def _validated_rates(values: Iterable[int]) -> Tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError("dilation_rates must be a non-empty iterable")
    try:
        rates = tuple(values)
    except TypeError as exc:
        raise ValueError("dilation_rates must be a non-empty iterable") from exc
    if len(rates) < 2:
        raise ValueError("V8.3 requires at least two dilation blocks")
    for rate in rates:
        _positive_integer("each dilation rate", rate)
    return rates


def _residual_block(
    keras,
    hidden,
    *,
    filters: int,
    kernel_size: int,
    dilation: int,
    feature_name: str,
    gate_name: str,
    mix_name: str,
    add_name: str,
):
    features = keras.layers.Conv1D(
        filters,
        kernel_size,
        dilation_rate=dilation,
        padding="causal",
        activation="tanh",
        name=feature_name,
    )(hidden)
    gate = keras.layers.Conv1D(
        filters,
        kernel_size,
        dilation_rate=dilation,
        padding="causal",
        activation="sigmoid",
        name=gate_name,
    )(hidden)
    gated = keras.layers.Multiply(name=add_name.replace("residual_add", "gated"))(
        [features, gate]
    )
    mixed = keras.layers.Conv1D(
        filters,
        1,
        padding="same",
        name=mix_name,
    )(gated)
    return keras.layers.Add(name=add_name)([hidden, mixed])


def build_v83_stream_model(
    *,
    filters: int = 32,
    kernel_size: int = 5,
    dilation_rates: Iterable[int] = DEFAULT_DILATION_RATES,
    split_after: int = DEFAULT_SPLIT_AFTER,
    name: str = "causal_boundary_v83_stream",
):
    """Build the deployable V8.3 stream model.

    The first ``split_after`` residual blocks are shared.  Remaining blocks and
    the fusion projection are duplicated for onset and offset.  Output names and
    shapes remain identical to V8/V8.1.
    """
    channels = _positive_integer("filters", filters)
    kernel = _positive_integer("kernel_size", kernel_size)
    rates = _validated_rates(dilation_rates)
    if isinstance(split_after, bool) or not isinstance(split_after, int):
        raise ValueError("split_after must be an integer")
    if split_after <= 0 or split_after >= len(rates):
        raise ValueError("split_after must leave at least one shared and one private block")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("name must be a non-empty string")

    tf = _load_tensorflow()
    keras = tf.keras
    audio = keras.Input(shape=(None, 1), dtype=tf.float32, name="audio")

    transient = keras.layers.Conv1D(
        channels,
        9,
        padding="causal",
        activation="relu",
        name="v8_transient_conv",
    )(audio)
    shared = keras.layers.Conv1D(
        channels,
        1,
        padding="same",
        activation="relu",
        name="v8_input_projection",
    )(audio)

    for index, rate in enumerate(rates[:split_after], start=1):
        shared = _residual_block(
            keras,
            shared,
            filters=channels,
            kernel_size=kernel,
            dilation=rate,
            feature_name=f"v8_feature_{index}",
            gate_name=f"v8_gate_{index}",
            mix_name=f"v8_mix_{index}",
            add_name=f"v8_residual_add_{index}",
        )

    private = {}
    for kind in ("onset", "offset"):
        hidden = shared
        for index, rate in enumerate(rates[split_after:], start=split_after + 1):
            hidden = _residual_block(
                keras,
                hidden,
                filters=channels,
                kernel_size=kernel,
                dilation=rate,
                feature_name=f"v83_{kind}_feature_{index}",
                gate_name=f"v83_{kind}_gate_{index}",
                mix_name=f"v83_{kind}_mix_{index}",
                add_name=f"v83_{kind}_residual_add_{index}",
            )
        fused = keras.layers.Concatenate(name=f"v83_{kind}_fusion")(
            [hidden, transient]
        )
        private[kind] = keras.layers.Conv1D(
            channels,
            1,
            padding="same",
            activation="relu",
            name=f"v83_{kind}_fusion_projection",
        )(fused)

    outputs = {
        "onset_presence": keras.layers.Conv1D(
            1, 1, activation="sigmoid", name="onset_presence_sequence"
        )(private["onset"]),
        "offset_presence": keras.layers.Conv1D(
            1, 1, activation="sigmoid", name="offset_presence_sequence"
        )(private["offset"]),
        "onset_multiplicity": keras.layers.Conv1D(
            MULTIPLICITY_CLASSES,
            1,
            activation="softmax",
            name="onset_multiplicity_sequence",
        )(private["onset"]),
        "offset_multiplicity": keras.layers.Conv1D(
            MULTIPLICITY_CLASSES,
            1,
            activation="softmax",
            name="offset_multiplicity_sequence",
        )(private["offset"]),
    }

    model = keras.Model(audio, outputs, name=name.strip())
    model.receptive_field = max(9, calculate_receptive_field(kernel, rates))
    model.anonymous_boundaries = True
    model.hierarchical_cardinality = True
    model.split_tasks = True
    model.split_after = split_after
    model.dilation_rates = rates
    return model


def initialize_v83_stream_from_v81(source, target, *, split_after: int = DEFAULT_SPLIT_AFTER):
    """Copy a V8.1 stream into V8.3 while preserving its initial function."""
    if int(getattr(target, "split_after", split_after)) != split_after:
        raise ValueError("target split_after does not match requested initialization")

    for name in ("v8_transient_conv", "v8_input_projection"):
        target.get_layer(name).set_weights(source.get_layer(name).get_weights())

    rates = tuple(getattr(target, "dilation_rates", DEFAULT_DILATION_RATES))
    for index in range(1, split_after + 1):
        for stem in ("feature", "gate", "mix"):
            source_name = f"v8_{stem}_{index}"
            target.get_layer(source_name).set_weights(source.get_layer(source_name).get_weights())

    for kind in ("onset", "offset"):
        for index in range(split_after + 1, len(rates) + 1):
            for stem in ("feature", "gate", "mix"):
                source_name = f"v8_{stem}_{index}"
                target_name = f"v83_{kind}_{stem}_{index}"
                target.get_layer(target_name).set_weights(
                    source.get_layer(source_name).get_weights()
                )
        target.get_layer(f"v83_{kind}_fusion_projection").set_weights(
            source.get_layer("v8_fusion_projection").get_weights()
        )

    for name in (
        "onset_presence_sequence",
        "offset_presence_sequence",
        "onset_multiplicity_sequence",
        "offset_multiplicity_sequence",
    ):
        target.get_layer(name).set_weights(source.get_layer(name).get_weights())
    return target


def build_v83_bag_model(
    *,
    filters: int = 32,
    kernel_size: int = 5,
    dilation_rates: Iterable[int] = DEFAULT_DILATION_RATES,
    split_after: int = DEFAULT_SPLIT_AFTER,
    onset_horizon_samples: int = DEFAULT_ONSET_HORIZON_SAMPLES,
    offset_horizon_samples: int = DEFAULT_OFFSET_HORIZON_SAMPLES,
):
    """Build the V8.1 bag objective around the V8.3 stream topology."""
    for name, value in (
        ("onset_horizon_samples", onset_horizon_samples),
        ("offset_horizon_samples", offset_horizon_samples),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be an integer > 0")

    tf = _load_tensorflow()
    keras = tf.keras
    stream = build_v83_stream_model(
        filters=filters,
        kernel_size=kernel_size,
        dilation_rates=dilation_rates,
        split_after=split_after,
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

    model = keras.Model(audio, outputs, name="causal_boundary_v83_bag")
    model.receptive_field = receptive_field
    model.maximum_horizon_samples = maximum_horizon
    model.context_samples = context_samples
    model.stream_model_name = stream.name
    model.onset_horizon_samples = onset_horizon_samples
    model.offset_horizon_samples = offset_horizon_samples
    model.split_after = split_after
    return model


__all__ = [
    "DEFAULT_SPLIT_AFTER",
    "build_v83_bag_model",
    "build_v83_stream_model",
    "initialize_v83_stream_from_v81",
]
