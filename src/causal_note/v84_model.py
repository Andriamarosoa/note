"""V8.4 fully independent onset/offset causal streams.

V8.3 proved that splitting only the deep residual tower is insufficient because
onset replay still changes the shared transient/stem representation and because
onset-selected crops also become offset training data.  V8.4 removes both
couplings from deployment: onset and offset each own a complete V8/V8.1 stream
with no shared trainable variables.  Both streams can be initialized exactly
from one frozen V8.1 checkpoint, preserving the source function before
continuation.
"""
from __future__ import annotations

from .v8_model import build_v8_stream_model


def _load_tensorflow():
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is required to build V8.4; install requirements-train.txt"
        ) from exc
    return tf


def build_v84_stream_model(*, filters: int = 32, kernel_size: int = 5):
    tf = _load_tensorflow()
    keras = tf.keras
    audio = keras.Input(shape=(None, 1), dtype=tf.float32, name="audio")

    onset_stream = build_v8_stream_model(
        filters=filters,
        kernel_size=kernel_size,
        name="v84_onset_stream",
    )
    offset_stream = build_v8_stream_model(
        filters=filters,
        kernel_size=kernel_size,
        name="v84_offset_stream",
    )
    onset = onset_stream(audio)
    offset = offset_stream(audio)
    outputs = {
        "onset_presence": onset["onset_presence"],
        "offset_presence": offset["offset_presence"],
        "onset_multiplicity": onset["onset_multiplicity"],
        "offset_multiplicity": offset["offset_multiplicity"],
    }
    model = keras.Model(audio, outputs, name="causal_boundary_v84_stream")
    model.receptive_field = int(onset_stream.receptive_field)
    model.anonymous_boundaries = True
    model.hierarchical_cardinality = True
    model.independent_tasks = True
    model.onset_stream_name = onset_stream.name
    model.offset_stream_name = offset_stream.name
    return model


def initialize_v84_stream_from_v81(source, target):
    onset_stream = target.get_layer("v84_onset_stream")
    offset_stream = target.get_layer("v84_offset_stream")
    source_weights = source.get_weights()
    onset_stream.set_weights(source_weights)
    offset_stream.set_weights(source_weights)
    return target


def assemble_v84_stream(onset_source, offset_source, *, filters: int = 32):
    """Create a deployable independent stream from separately trained sources."""
    target = build_v84_stream_model(filters=filters)
    target.get_layer("v84_onset_stream").set_weights(onset_source.get_weights())
    target.get_layer("v84_offset_stream").set_weights(offset_source.get_weights())
    return target


__all__ = [
    "assemble_v84_stream",
    "build_v84_stream_model",
    "initialize_v84_stream_from_v81",
]
