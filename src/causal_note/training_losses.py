"""Serializable training-only losses for NOTE boundary models."""

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(
    package="causal_note",
    name="elementwise_binary_crossentropy_v1",
)
def elementwise_binary_crossentropy_v1(y_true, y_pred):
    """Return BCE per boundary slot so Keras weights before reducing slots."""

    return tf.keras.backend.binary_crossentropy(y_true, y_pred)


__all__ = ["elementwise_binary_crossentropy_v1"]
