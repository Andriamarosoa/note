"""Compact causal neural network for sample-wise note boundaries.

TensorFlow is intentionally imported only by :func:`build_causal_boundary_model`.
Importing this module therefore remains possible in the live inference package
when the optional training dependency is not installed.
"""

from typing import Iterable, Tuple


EVENT_SLOTS = 6
OUTPUT_NAMES = ("onset", "offset")
DEFAULT_DILATION_RATES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be an integer > 0")
    return value


def _validated_dilation_rates(values: Iterable[int]) -> Tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError("dilation_rates must be a non-empty iterable of integers > 0")
    try:
        rates = tuple(values)
    except TypeError as exc:
        raise ValueError(
            "dilation_rates must be a non-empty iterable of integers > 0"
        ) from exc
    if not rates:
        raise ValueError("dilation_rates must not be empty")
    for rate in rates:
        _positive_integer("each dilation rate", rate)
    return rates


def calculate_receptive_field(
    kernel_size: int = 5,
    dilation_rates: Iterable[int] = DEFAULT_DILATION_RATES,
) -> int:
    """Return the number of input samples visible to one output sample.

    This formula applies to the stride-one convolutional stack built below.
    The two one-sample output heads do not enlarge the receptive field.
    """

    kernel = _positive_integer("kernel_size", kernel_size)
    rates = _validated_dilation_rates(dilation_rates)
    return 1 + (kernel - 1) * sum(rates)


def _load_tensorflow():
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is required to build the neural model; install "
            "requirements-train.txt"
        ) from exc
    return tf


def build_causal_boundary_model(
    *,
    filters: int = 24,
    kernel_size: int = 5,
    dilation_rates: Iterable[int] = DEFAULT_DILATION_RATES,
    name: str = "causal_boundary_model",
):
    """Build a mono-audio model with onset and offset slot outputs.

    Input shape is ``(batch, time, 1)``.  Every output has shape
    ``(batch, time, 6)``.  All convolutions use causal padding, so an output at
    sample ``t`` can only depend on audio at or before ``t``.
    """

    channel_count = _positive_integer("filters", filters)
    kernel = _positive_integer("kernel_size", kernel_size)
    rates = _validated_dilation_rates(dilation_rates)
    if not isinstance(name, str) or not name.strip():
        raise ValueError("name must be a non-empty string")

    tf = _load_tensorflow()
    keras = tf.keras

    audio = keras.Input(shape=(None, 1), dtype=tf.float32, name="audio")
    hidden = audio
    for index, rate in enumerate(rates):
        hidden = keras.layers.Conv1D(
            filters=channel_count,
            kernel_size=kernel,
            dilation_rate=rate,
            padding="causal",
            activation="relu",
            name=f"causal_conv_{index + 1}",
        )(hidden)

    outputs = {
        output_name: keras.layers.Conv1D(
            filters=EVENT_SLOTS,
            kernel_size=1,
            padding="causal",
            activation="sigmoid",
            name=output_name,
        )(hidden)
        for output_name in OUTPUT_NAMES
    }
    model = keras.Model(inputs=audio, outputs=outputs, name=name.strip())
    model.receptive_field = calculate_receptive_field(kernel, rates)
    model.event_slots = EVENT_SLOTS
    return model


__all__ = [
    "DEFAULT_DILATION_RATES",
    "EVENT_SLOTS",
    "OUTPUT_NAMES",
    "build_causal_boundary_model",
    "calculate_receptive_field",
]
