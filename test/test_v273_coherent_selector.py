"""Real TensorFlow gradients and a common probability space for all actions."""
import unittest

import numpy as np
import tensorflow as tf

from scripts.learn_v273_coherent_selector import (
    CoherentCombinationArbiter, choose_legal_class, train_coherent,
)
from scripts.v273_selector_contract import direct_options, fallback_metadata


class CoherentSelectorTests(unittest.TestCase):
    def fixture(self):
        rng = np.random.default_rng(733)
        base = np.resize([2, 3, 4], 18)
        p = rng.dirichlet(np.ones(5), size=(18, 6)).astype(np.float32)
        options, mask, identity = direct_options(p, base)
        audits = rng.uniform(0, 1, (18, 5, 64, 4)).astype(np.float32)
        features = np.concatenate([options, audits, identity, audits], -1)*mask[..., None]
        x = dict(subset_features=features, subset_mask=mask,
            baseline=np.eye(7, dtype=np.float32)[base],
            context=rng.normal(size=(18, 43)).astype(np.float32),
            keep_heads=fallback_metadata(p, base))
        y = np.resize(np.arange(7), 18).astype(np.int32)
        return x, y, base

    def test_one_probability_of_baseline_correctness(self):
        x, _, base = self.fixture()
        out = CoherentCombinationArbiter()(x)
        p = out["class_probability"].numpy()
        np.testing.assert_allclose(p.sum(axis=1), 1., atol=1e-6)
        expected = p[np.arange(len(base)), base]
        np.testing.assert_allclose(out["keep_correct"].numpy(), expected, atol=1e-7)
        np.testing.assert_allclose(out["candidate_regress"].numpy(),
                                   np.broadcast_to(expected[:, None], (len(base), 5)), atol=1e-7)
        np.testing.assert_allclose(out["candidate_correct"].numpy(), p[:, 2:], atol=1e-7)

    def test_multiclass_counterexample_and_keep_ties(self):
        p = np.array([[0., 0., .3, .4, .2, .06, .04]])
        np.testing.assert_array_equal(choose_legal_class(p, np.array([3])), [3])
        p = np.array([[0., 0., .4, .4, .1, .06, .04]])
        np.testing.assert_array_equal(choose_legal_class(p, np.array([3])), [3])
        p = np.array([[.8, .1, .02, .04, .01, .02, .01]])
        np.testing.assert_array_equal(choose_legal_class(p, np.array([3])), [3])

    def test_all_layers_have_finite_gradients_and_audits_affect_decision(self):
        tf.keras.utils.set_random_seed(27402)
        x, y, _ = self.fixture()
        model = CoherentCombinationArbiter()
        features = tf.Variable(x["subset_features"])
        inputs = dict(x, subset_features=features)
        with tf.GradientTape(persistent=True) as tape:
            out = model(inputs)
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=y, logits=out["class_logits"]))
        grads = tape.gradient(loss, model.trainable_variables)
        self.assertTrue(all(g is not None and bool(tf.reduce_all(tf.math.is_finite(g))) for g in grads))
        feature_grad = tape.gradient(loss, features).numpy()
        self.assertGreater(float(np.abs(feature_grad[..., 6:10]).sum()), 0.)
        self.assertGreater(float(np.abs(feature_grad[..., 17:21]).sum()), 0.)

    def test_real_training_and_legal_predictions(self):
        x, y, b = self.fixture()
        prediction, out, history, model = train_coherent(x, y, b, x, epochs=2)
        self.assertTrue(np.isin(prediction, (2, 3, 4, 5, 6)).all())
        self.assertTrue(np.isfinite(out["class_probability"].numpy()).all())
        self.assertEqual(history[-1]["epoch"], 2)
        self.assertGreater(model.count_params(), 0)


if __name__ == "__main__":
    unittest.main()
