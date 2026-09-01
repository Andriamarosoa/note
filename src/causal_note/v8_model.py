"""V8 anonymous-cardinality causal boundary model.

V8 removes string/slot identity from the neural output. The model predicts only
how many onsets and offsets occur at each sample. Association to opaque event
ids is handled later by deterministic runtime logic.
"""
from typing import Iterable, Tuple

COUNT_CLASSES = 4  # 0, 1, 2, 3+
OUTPUT_NAMES = ("onset_count", "offset_count")
DEFAULT_DILATION_RATES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)


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


def _residual_block(
    keras,
    hidden,
    *,
    filters: int,
    kernel_size: int,
    dilation: int,
    index: int,
):
    residual = hidden
    if int(hidden.shape[-1]) != filters:
        residual = keras.layers.Conv1D(
            filters,
            1,
            padding="same",
            name=f"v8_residual_proj_{index}",
        )(residual)
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
    return keras.layers.Add(name=f"v8_residual_add_{index}")([residual, mixed])


def build_v8_cardinality_model(
    *,
    filters: int = 32,
    kernel_size: int = 5,
    dilation_rates: Iterable[int] = DEFAULT_DILATION_RATES,
    name: str = "causal_boundary_v8",
):
    """Build the first V8 anonymous cardinality model.

    The output classes are 0, 1, 2 and 3+ boundaries independently for onset
    and offset. The network remains strictly causal and keeps the V7 temporal
    budget while replacing the plain stack with gated residual blocks and a
    short transient branch.
    """

    channel_count = _positive_integer("filters", filters)
    kernel = _positive_integer("kernel_size", kernel_size)
    rates = _validated_dilation_rates(dilation_rates)
    if not isinstance(name, str) or not name.strip():
        raise ValueError("name must be a non-empty string")

    tf = _load_tensorflow()
    keras = tf.keras
    audio = keras.Input(shape=(None, 1), dtype=tf.float32, name="audio")

    transient = keras.layers.Conv1D(
        channel_count,
        9,
        padding="causal",
        activation="relu",
        name="v8_transient_conv",
    )(audio)
    hidden = keras.layers.Conv1D(
        channel_count,
        1,
        padding="same",
        activation="relu",
        name="v8_input_projection",
    )(audio)
    for index, rate in enumerate(rates, start=1):
        hidden = _residual_block(
            keras,
            hidden,
            filters=channel_count,
            kernel_size=kernel,
            dilation=rate,
            index=index,
        )

    fused = keras.layers.Concatenate(name="v8_fusion")([hidden, transient])
    fused = keras.layers.Conv1D(
        channel_count,
        1,
        padding="same",
        activation="relu",
        name="v8_fusion_projection",
    )(fused)
    outputs = {
        output_name: keras.layers.Conv1D(
            COUNT_CLASSES,
            1,
            padding="same",
            activation="softmax",
            name=output_name,
        )(fused)
        for output_name in OUTPUT_NAMES
    }

    model = keras.Model(audio, outputs, name=name.strip())
    model.receptive_field = max(9, calculate_receptive_field(kernel, rates))
    model.count_classes = COUNT_CLASSES
    model.anonymous_boundaries = True
    return model


__all__ = [
    "COUNT_CLASSES",
    "DEFAULT_DILATION_RATES",
    "OUTPUT_NAMES",
    "build_v8_cardinality_model",
    "calculate_receptive_field",
]
