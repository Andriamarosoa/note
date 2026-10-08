"""Coherent seven-class probabilities for the existing subset-attention router.

The same event has one probability of baseline correctness. A destination's
expected Exact-K advantage is P(Y=destination)-P(Y=baseline). KEEP has zero
advantage; ties preserve the baseline. K0/K1 are training classes, not enabled
output destinations in this controlled experiment.
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf


def choose_legal_class(probability, base):
    """No labels, tuned threshold, or independently estimated KEEP odds."""
    p, base = np.asarray(probability), np.asarray(base, int)
    if p.shape != (len(base), 7) or not np.isin(base, (2, 3, 4)).all():
        raise ValueError("seven-class probability/baseline contract")
    if not np.isfinite(p).all() or np.any(p < 0) or not np.allclose(p.sum(1), 1., atol=1e-5):
        raise ValueError("invalid probability distribution")
    proposal = 2+np.argmax(p[:, 2:], axis=1)
    gain = p[np.arange(len(base)), proposal]-p[np.arange(len(base)), base]
    return np.where(gain > 0, proposal, base)


class CoherentCombinationArbiter(tf.keras.Model):
    def __init__(self, width=32, **kwargs):
        super().__init__(**kwargs)
        self.subset_embed = tf.keras.layers.Dense(width, activation="gelu")
        self.subset_gate = tf.keras.layers.Dense(1)
        self.score_hidden = tf.keras.layers.Dense(width, activation="gelu")
        self.target_logit = tf.keras.layers.Dense(1)
        self.baseline_hidden = tf.keras.layers.Dense(width, activation="gelu")
        self.baseline_and_low_logits = tf.keras.layers.Dense(3)

    def call(self, x, training=False):
        combo = tf.cast(x["subset_features"], tf.float32)
        valid = tf.cast(x["subset_mask"], tf.bool)
        base = tf.cast(x["baseline"], tf.float32)
        context = tf.cast(x["context"], tf.float32)
        metadata = tf.cast(x["keep_heads"], tf.float32)
        n = tf.shape(combo)[0]
        if combo.shape[1:] != (5, 64, 21):
            raise ValueError("expected repaired 5x64x21 descriptors")
        target = tf.broadcast_to(tf.eye(5)[None], [n, 5, 5])
        bsub = tf.broadcast_to(base[:, None, None], [n, 5, 64, 7])
        tsub = tf.broadcast_to(target[:, :, None], [n, 5, 64, 5])
        csub = tf.broadcast_to(context[:, None, None], [n, 5, 64, tf.shape(context)[1]])
        state = self.subset_embed(tf.concat([combo, bsub, tsub, csub], axis=-1))
        scores = tf.squeeze(self.subset_gate(state), axis=-1)
        attention = tf.nn.softmax(tf.where(valid, scores, -1e9), axis=2)
        attention = tf.where(valid, attention, 0.)
        attention /= tf.maximum(tf.reduce_sum(attention, axis=2, keepdims=True), 1e-9)
        pooled = tf.einsum("bks,bksd->bkd", attention, state)
        selected = tf.einsum("bks,bksd->bkd", attention, combo)
        ctarget = tf.broadcast_to(context[:, None], [n, 5, tf.shape(context)[1]])
        btarget = tf.broadcast_to(base[:, None], [n, 5, 7])
        hidden = self.score_hidden(tf.concat([pooled, selected, ctarget, btarget, target], -1))
        target_logits = tf.squeeze(self.target_logit(hidden), -1)
        legal = tf.reduce_any(valid, axis=-1)
        maximum = tf.reduce_max(tf.where(legal, target_logits, -1e9), axis=1, keepdims=True)
        mean = tf.reduce_sum(tf.where(legal, target_logits, 0.), axis=1, keepdims=True)/4.
        row = self.baseline_hidden(tf.concat([base, context, metadata, maximum, mean], -1))
        low_and_base = self.baseline_and_low_logits(row)
        poly_logits = tf.where(base[:, 2:] > .5, low_and_base[:, 2, None], target_logits)
        class_logits = tf.concat([low_and_base[:, :2], poly_logits], axis=1)
        probability = tf.nn.softmax(class_logits, axis=1)
        baseline_correct = tf.reduce_sum(probability*base, axis=1)
        # A single coherent quantity replaces five independent regression risks.
        regression = tf.broadcast_to(baseline_correct[:, None], [n, 5])
        advantage = probability[:, 2:]-regression
        advantage = tf.where(legal, advantage, -1e9)
        return dict(class_logits=class_logits, class_probability=probability,
            candidate_correct=probability[:, 2:], candidate_regress=regression,
            keep_correct=baseline_correct,
            unavailable_class_probability=tf.reduce_sum(probability[:, :2], axis=1),
            action_logits=tf.concat([advantage, tf.zeros([n, 1])], axis=1),
            subset_attention=attention, selected_subset_features=selected)


def train_coherent(train, truth, base, held, epochs=30, seed=27402):
    tf.keras.utils.set_random_seed(seed)
    model = CoherentCombinationArbiter(width=32)
    dataset = tf.data.Dataset.from_tensor_slices((train, np.asarray(truth, np.int32)))
    dataset = dataset.shuffle(len(truth), seed=seed, reshuffle_each_iteration=True).batch(192)
    optimizer = tf.keras.optimizers.Adam(.002)
    history = []
    for epoch in range(epochs):
        losses = []
        for inputs, labels in dataset:
            with tf.GradientTape() as tape:
                out = model(inputs, training=True)
                # Proper categorical likelihood; no balanced pseudo-probabilities
                # and no conflicting auxiliary definition of KEEP correctness.
                loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=labels, logits=out["class_logits"]))
            grads = tape.gradient(loss, model.trainable_variables)
            if any(g is None for g in grads):
                raise ValueError("disconnected trainable variable")
            for g in grads:
                tf.debugging.assert_all_finite(g, "nonfinite gradient")
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            losses.append(float(loss))
        if epoch in (0, epochs//2, epochs-1):
            history.append(dict(epoch=epoch+1, loss=float(np.mean(losses))))
    out = model(held, training=False)
    held_base = np.argmax(held["baseline"], axis=1)
    prediction = choose_legal_class(out["class_probability"].numpy(), held_base)
    return prediction, out, history, model
