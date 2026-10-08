"""Shared nonlinear critic trained on each complete group's joint outcome.

Every event has one baseline-correctness probability, shared by all changing
groups. Conditional correction probabilities are group-specific. This yields
normalized correction/regression/neutral probabilities and no individual veto.
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf

from scripts.v273_group127_contract import DIRECT_DIM, choose_groups


class JointGroupCritic(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.group_hidden = tf.keras.layers.Dense(64, activation="gelu")
        self.group_interaction = tf.keras.layers.Dense(32, activation="gelu")
        self.conditional_logit = tf.keras.layers.Dense(1)
        self.baseline_hidden = tf.keras.layers.Dense(32, activation="gelu")
        self.baseline_logit = tf.keras.layers.Dense(1)

    def call(self, x, training=False):
        features = tf.cast(x["group_features"], tf.float32)
        base = tf.cast(x["baseline"], tf.float32)
        context = tf.cast(x["context"], tf.float32)
        proposals = tf.cast(x["proposal"], tf.int32)
        votes = tf.cast(x["member_votes"], tf.float32)
        if features.shape[1:] != (127, 64):
            raise ValueError("127 full-group descriptors of dimension 64 required")
        n = tf.shape(features)[0]
        changing = proposals != tf.argmax(base, axis=1, output_type=tf.int32)[:, None]
        cgroup = tf.broadcast_to(context[:, None], [n, 127, tf.shape(context)[1]])
        bgroup = tf.broadcast_to(base[:, None], [n, 127, 7])
        state = self.group_hidden(tf.concat([features, cgroup, bgroup, tf.one_hot(proposals, 7)], -1))
        state = self.group_interaction(state)
        conditional_logits = tf.squeeze(self.conditional_logit(state), -1)
        active = tf.cast(changing, tf.float32)
        audit_summary = tf.reduce_sum(features[..., DIRECT_DIM:]*active[:, :, None], axis=1)/tf.maximum(tf.reduce_sum(active, axis=1, keepdims=True), 1.)
        row = self.baseline_hidden(tf.concat([context, base, tf.reshape(votes, [n, 35]), audit_summary], -1))
        baseline_logits = tf.squeeze(self.baseline_logit(row), -1)
        baseline_correct = tf.sigmoid(baseline_logits)
        conditional = tf.sigmoid(conditional_logits)
        correction = (1-baseline_correct[:, None])*conditional*active
        regression = baseline_correct[:, None]*active
        neutral = 1-correction-regression
        return dict(baseline_logits=baseline_logits, conditional_logits=conditional_logits,
            baseline_correct=baseline_correct, conditional_correct=conditional,
            outcome_probability=tf.stack([correction, regression, neutral], -1),
            expected_gain=correction-regression, changing=changing)


def event_loss(out, truth, base, proposals):
    """Proper factorized likelihood, averaging groups WITHIN each event.

No class weighting. An event does not become 127 independent observations.
For an unchanged group, the joint outcome is analytically neutral.
"""
    y = tf.cast(truth, tf.int32); b = tf.cast(base, tf.int32)
    correct_base = tf.cast(y == b, tf.float32)
    baseline_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=correct_base, logits=out["baseline_logits"])
    target = tf.cast(tf.cast(proposals, tf.int32) == y[:, None], tf.float32)
    ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=target, logits=out["conditional_logits"])
    change = tf.cast(out["changing"], tf.float32)
    group_loss = tf.reduce_sum(ce*change, 1)/tf.maximum(tf.reduce_sum(change, 1), 1.)
    conditional_loss = (1-correct_base)*group_loss
    return tf.reduce_mean(baseline_loss+conditional_loss), tf.reduce_mean(baseline_loss), tf.reduce_mean(conditional_loss)


def predict(model, inputs):
    outputs = []
    for start in range(0, len(inputs["baseline"]), 192):
        out = model({k: v[start:start+192] for k, v in inputs.items()}, training=False)
        outputs.append({k: out[k].numpy() for k in
                        ("baseline_correct", "conditional_correct", "outcome_probability", "expected_gain")})
    return {k: np.concatenate([o[k] for o in outputs]) for k in outputs[0]}


def train_groups(train, truth, base, held, epochs=30, seed=27402):
    tf.keras.utils.set_random_seed(seed)
    model = JointGroupCritic()
    optimizer = tf.keras.optimizers.Adam(.002)
    dataset = tf.data.Dataset.from_tensor_slices((train, np.asarray(truth, np.int32), np.asarray(base, np.int32)))
    dataset = dataset.shuffle(len(truth), seed=seed, reshuffle_each_iteration=True).batch(192)
    history = []
    for epoch in range(epochs):
        losses = []
        for inputs, y, b in dataset:
            with tf.GradientTape() as tape:
                out = model(inputs, training=True)
                loss, baseline_loss, conditional_loss = event_loss(out, y, b, inputs["proposal"])
            gradients = tape.gradient(loss, model.trainable_variables)
            if any(g is None for g in gradients):
                raise ValueError("disconnected parameter")
            for g in gradients:
                tf.debugging.assert_all_finite(g, "nonfinite gradient")
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            losses.append([float(loss), float(baseline_loss), float(conditional_loss)])
        if epoch in (0, epochs//2, epochs-1):
            mean = np.mean(losses, axis=0)
            history.append(dict(epoch=epoch+1, loss=float(mean[0]), baseline_loss=float(mean[1]), conditional_group_loss=float(mean[2])))
    out = predict(model, held)
    prediction, selection = choose_groups(out["expected_gain"], held["proposal"], np.argmax(held["baseline"], 1))
    return prediction, selection, out, history, model
