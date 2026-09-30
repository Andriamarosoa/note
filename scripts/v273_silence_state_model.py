"""Opt-in eight-state native model; existing seven-state runs stay reproducible."""
import numpy as np
from scripts.train_v250_count_only import build_model as build_count_model
from scripts.v273_spectral_normalization import SEED, DROPOUT_SEED


def build_model(seed=SEED, dropout_seed=DROPOUT_SEED):
    import tensorflow as tf
    base = build_count_model('categorical', seed, time_frames=31,
        spectral_normalization='fixed_scale', count_dropout_seed=dropout_seed,
        ownership_context=True, spectral_channels=4)
    hidden = base.get_layer('v240_cardinality_hidden2').output
    output = tf.keras.layers.Dense(8, activation='softmax', name='count_state',
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed+9901))(hidden)
    model = tf.keras.Model(base.inputs, output, name='v273_silence_count_states')
    model.compile(optimizer=tf.keras.optimizers.Adam(2e-4),
        loss='sparse_categorical_crossentropy',
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='state_exact')])
    return model


def require_training_population(states, trainable, fit, validation):
    """No eight-class training on a dataset that contains no verified silence."""
    from causal_note.count_states import encode_states
    states, trainable = np.asarray(states), np.asarray(trainable)
    if states.ndim != 1 or trainable.shape != states.shape or trainable.dtype != np.bool_:
        raise ValueError('aligned state labels and boolean eligibility required')
    encode_states(states[trainable])
    result = []
    for name, ids in (('fit', fit), ('validation', validation)):
        ids = np.asarray(ids, np.int64)
        chosen = ids[trainable[ids]]
        if not len(chosen) or not np.any(states[chosen] == -1) or not np.any(states[chosen] == 0):
            raise ValueError(f'{name} needs real labelled silence and sounding K=0 examples')
        result.append(chosen)
    if np.intersect1d(*result).size:
        raise ValueError('fit and validation overlap')
    return tuple(result)
