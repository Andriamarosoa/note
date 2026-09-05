"""V8 hierarchical anonymous boundary model.

The dominant V7 failure was false boundary presence. V8 therefore separates:
1. whether a boundary exists at sample t;
2. its anonymous multiplicity, conditioned on presence.

String/slot identity is not predicted.
"""
from typing import Iterable, Tuple

MULTIPLICITY_CLASSES = 3  # 1, 2, 3+
DEFAULT_DILATION_RATES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
STREAM_OUTPUT_NAMES = (
    "onset_presence",
    "offset_presence",
    "onset_multiplicity",
    "offset_multiplicity",
)


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be an integer > 0")
    return value


def _validated_dilation_rates(values: Iterable[int]) -> Tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError("dilation_rates must be a non-empty iterable")
    try:
        rates = tuple(values)
    except TypeError as exc:
        raise ValueError("dilation_rates must be a non-empty iterable") from exc
    if not rates:
        raise ValueError("dilation_rates must not be empty")
    for rate in rates:
        _positive_integer("each dilation rate", rate)
    return rates


def calculate_receptive_field(
    kernel_size: int = 5,
    dilation_rates: Iterable[int] = DEFAULT_DILATION_RATES,
) -> int:
    kernel = _positive_integer("kernel_size", kernel_size)
    rates = _validated_dilation_rates(dilation_rates)
    return 1 + (kernel - 1) * sum(rates)


def _load_tensorflow():
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is required to build V8; install requirements-train.txt"
        ) from exc
    return tf


def _residual_gated_block(
    keras,
    hidden,
    *,
    filters: int,
    kernel_size: int,
    dilation: int,
    index: int,
):
    features = keras.layers.Conv1D(
        filters,
        kernel_size,
        dilation_rate=dilation,
        padding="causal",
        activation="tanh",
        name=f"v8_feature_{index}",
    )(hidden)
    gate = keras.layers.Conv1D(
        filters,
        kernel_size,
        dilation_rate=dilation,
        padding="causal",
        activation="sigmoid",
        name=f"v8_gate_{index}",
    )(hidden)
    gated = keras.layers.Multiply(name=f"v8_gated_{index}")([features, gate])
    mixed = keras.layers.Conv1D(
        filters,
        1,
        padding="same",
        name=f"v8_mix_{index}",
    )(gated)
    return keras.layers.Add(name=f"v8_residual_add_{index}")([hidden, mixed])


def build_v8_stream_model(
    *,
    filters: int = 32,
    kernel_size: int = 5,
    dilation_rates: Iterable[int] = DEFAULT_DILATION_RATES,
    name: str = "causal_boundary_v8_stream",
):
    """Return the causal sequence model used by future live inference."""

    channels = _positive_integer("filters", filters)
    kernel = _positive_integer("kernel_size", kernel_size)
    rates = _validated_dilation_rates(dilation_rates)
    if not isinstance(name, str) or not name.strip():
        raise ValueError("name must be a non-empty string")

    tf = _load_tensorflow()
    keras = tf.keras
    audio = keras.Input(shape=(None, 1), dtype=tf.float32, name="audio")

    # Short branch keeps local attack/transient evidence explicit.
    transient = keras.layers.Conv1D(
        channels,
        9,
        padding="causal",
        activation="relu",
        name="v8_transient_conv",
    )(audio)

    hidden = keras.layers.Conv1D(
        channels,
        1,
        padding="same",
        activation="relu",
        name="v8_input_projection",
    )(audio)
    for index, rate in enumerate(rates, start=1):
        hidden = _residual_gated_block(
            keras,
            hidden,
            filters=channels,
            kernel_size=kernel,
            dilation=rate,
            index=index,
        )

    fused = keras.layers.Concatenate(name="v8_fusion")([hidden, transient])
    fused = keras.layers.Conv1D(
        channels,
        1,
        padding="same",
        activation="relu",
        name="v8_fusion_projection",
    )(fused)

    outputs = {
        "onset_presence": keras.layers.Conv1D(
            1, 1, activation="sigmoid", name="onset_presence_sequence"
        )(fused),
        "offset_presence": keras.layers.Conv1D(
            1, 1, activation="sigmoid", name="offset_presence_sequence"
        )(fused),
        "onset_multiplicity": keras.layers.Conv1D(
            MULTIPLICITY_CLASSES,
            1,
            activation="softmax",
            name="onset_multiplicity_sequence",
        )(fused),
        "offset_multiplicity": keras.layers.Conv1D(
            MULTIPLICITY_CLASSES,
            1,
            activation="softmax",
            name="offset_multiplicity_sequence",
        )(fused),
    }
    model = keras.Model(audio, outputs, name=name.strip())
    model.receptive_field = max(9, calculate_receptive_field(kernel, rates))
    model.anonymous_boundaries = True
    model.hierarchical_cardinality = True
    return model


def build_v8_point_model(**kwargs):
    """Return a fixed-context training model scoring only query sample t.

    The causal stream model is embedded as one nested layer, so the trained
    stream model can be extracted and saved directly after point-query training.
    """

    tf = _load_tensorflow()
    keras = tf.keras
    stream = build_v8_stream_model(**kwargs)
    receptive_field = int(stream.receptive_field)
    audio = keras.Input(
        shape=(receptive_field, 1),
        dtype=tf.float32,
        name="audio_context",
    )
    sequence_outputs = stream(audio)
    outputs = {}
    for output_name in STREAM_OUTPUT_NAMES:
        channels = 1 if output_name.endswith("presence") else MULTIPLICITY_CLASSES
        last = keras.layers.Cropping1D(
            cropping=(receptive_field - 1, 0),
            name=f"{output_name}_last_sample",
        )(sequence_outputs[output_name])
        outputs[output_name] = keras.layers.Reshape(
            (channels,),
            name=output_name,
        )(last)

    model = keras.Model(audio, outputs, name="causal_boundary_v8_point")
    model.receptive_field = receptive_field
    model.stream_model_name = stream.name
    model.anonymous_boundaries = True
    model.hierarchical_cardinality = True
    return model


__all__ = [
    "DEFAULT_DILATION_RATES",
    "MULTIPLICITY_CLASSES",
    "STREAM_OUTPUT_NAMES",
    "build_v8_point_model",
    "build_v8_stream_model",
    "calculate_receptive_field",
]
