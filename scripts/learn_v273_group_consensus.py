"""One learned utility per available verdict, using all full-group evidence.

The shared group network and parameter count are unchanged. Equal destinations
pool their logits by a count-normalized mean before the final probability.
The categorical arm also models OTHER: no changing proposal has the true K.
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf

from scripts.learn_v273_group127 import JointGroupCritic, event_loss
from scripts.v273_group127_contract import choose_groups

MODES = ('pooled_bce', 'pooled_ce')


def consensus_outputs(raw_logits, proposals, baseline_logits, base, mode):
    if mode not in MODES:
        raise ValueError('unknown consensus mode')
    proposals = tf.cast(proposals, tf.int32)
    base = tf.cast(base, tf.int32)
    membership = tf.one_hot(proposals, 7)
    counts = tf.reduce_sum(membership, axis=1)
    pooled = tf.reduce_sum(membership*raw_logits[:, :, None], axis=1)/tf.maximum(counts, 1.)
    available = (counts > 0) & (tf.range(7)[None] != base[:, None])
    # OTHER has a fixed zero logit (reference level), not an extra parameter.
    categorical_logits = tf.concat([tf.where(available, pooled, -1.e9),
                                    tf.zeros_like(pooled[:, :1])], axis=1)
    if mode == 'pooled_ce':
        distribution = tf.nn.softmax(categorical_logits, axis=1)
        conditional_class = distribution[:, :7]
        other = distribution[:, 7]
    else:
        conditional_class = tf.sigmoid(pooled)*tf.cast(available, tf.float32)
        other = tf.zeros_like(pooled[:, 0])  # undefined in BCE; do not claim a simplex
    conditional = tf.gather(conditional_class, proposals, batch_dims=1)
    baseline_correct = tf.sigmoid(baseline_logits)
    changing = proposals != base[:, None]
    active = tf.cast(changing, tf.float32)
    correction = (1-baseline_correct[:, None])*conditional*active
    regression = baseline_correct[:, None]*active
    return dict(baseline_logits=baseline_logits,
        conditional_logits=tf.gather(pooled, proposals, batch_dims=1),
        categorical_logits=categorical_logits, available=available,
        pooled_class_logits=pooled, baseline_correct=baseline_correct,
        conditional_correct=conditional, conditional_other=other,
        class_probability=(1-baseline_correct[:, None])*conditional_class
            + tf.one_hot(base, 7)*baseline_correct[:, None],
        other_probability=(1-baseline_correct)*other,
        outcome_probability=tf.stack([correction, regression, 1-correction-regression], -1),
        expected_gain=correction-regression, changing=changing)


class GroupConsensus(JointGroupCritic):
    def __init__(self, mode, **kwargs):
        super().__init__(**kwargs)
        if mode not in MODES:
            raise ValueError('unknown consensus mode')
        self.mode = mode

    def call(self, x, training=False):
        raw = super().call(x, training=training)
        base = tf.argmax(x['baseline'], axis=1, output_type=tf.int32)
        return consensus_outputs(raw['conditional_logits'], x['proposal'],
                                 raw['baseline_logits'], base, self.mode)


def consensus_loss(out, truth, base, proposals, mode):
    if mode == 'pooled_bce':
        # Exactly the old per-event averaged BCE; only the logit pooling changes.
        return event_loss(out, truth, base, proposals)
    if mode != 'pooled_ce':
        raise ValueError('unknown consensus mode')
    y = tf.cast(truth, tf.int32)
    correct_base = tf.cast(y == tf.cast(base, tf.int32), tf.float32)
    bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=correct_base, logits=out['baseline_logits'])
    has_truth = tf.gather(out['available'], y, batch_dims=1)
    target = tf.where(has_truth, y, tf.fill(tf.shape(y), 7))
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target, logits=out['categorical_logits'])
    ce = (1-correct_base)*ce
    return tf.reduce_mean(bce+ce), tf.reduce_mean(bce), tf.reduce_mean(ce)


def predict(model, inputs):
    keys = ('baseline_correct', 'conditional_correct', 'outcome_probability', 'expected_gain',
            'class_probability', 'other_probability', 'pooled_class_logits')
    chunks = []
    for start in range(0, len(inputs['baseline']), 192):
        out = model({k: v[start:start+192] for k, v in inputs.items()}, training=False)
        chunks.append({k: out[k].numpy() for k in keys})
    return {k: np.concatenate([o[k] for o in chunks]) for k in keys}


def train_consensus(train, truth, base, held, mode, epochs=30, seed=27402):
    tf.keras.utils.set_random_seed(seed)
    model = GroupConsensus(mode)
    optimizer = tf.keras.optimizers.Adam(.002)
    dataset = tf.data.Dataset.from_tensor_slices((train, np.asarray(truth, np.int32), np.asarray(base, np.int32)))
    dataset = dataset.shuffle(len(truth), seed=seed, reshuffle_each_iteration=True).batch(192)
    history = []
    for epoch in range(epochs):
        losses = []
        for inputs, y, b in dataset:
            with tf.GradientTape() as tape:
                out = model(inputs, training=True)
                loss, baseline_loss, conditional_loss = consensus_loss(out, y, b, inputs['proposal'], mode)
            gradients = tape.gradient(loss, model.trainable_variables)
            if any(g is None for g in gradients):
                raise ValueError('disconnected parameter')
            for g in gradients:
                tf.debugging.assert_all_finite(g, 'nonfinite gradient')
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            losses.append([float(loss), float(baseline_loss), float(conditional_loss)])
        if epoch in (0, epochs//2, epochs-1):
            mean = np.mean(losses, axis=0)
            history.append(dict(epoch=epoch+1, loss=float(mean[0]), baseline_loss=float(mean[1]), conditional_loss=float(mean[2])))
    out = predict(model, held)
    prediction, selection = choose_groups(out['expected_gain'], held['proposal'], np.argmax(held['baseline'], 1))
    return prediction, selection, out, history, model
