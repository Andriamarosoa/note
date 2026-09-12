from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from scripts import train_v280_balanced_internal as g
from test.test_v280_nested_outer import synthetic_arrays


class WeightingTests(unittest.TestCase):
    def test_group_mass_and_mean_are_normalized_using_only_fit_labels(self):
        a = synthetic_arrays()
        roles, spec = g.weighting(a, "balanced")
        k = a["target_cardinality"][roles["fit"]]
        w = g.batch_weights(a, roles["fit"], spec)["cardinality"]
        self.assertAlmostEqual(float(w.mean()), 1)
        self.assertAlmostEqual(float(w[k < 2].sum()), len(k) / 2)
        self.assertAlmostEqual(float(w[k >= 2].sum()), len(k) / 2)
        a["target_cardinality"][np.concatenate([roles["outer"], roles["validation"]])] = 6
        self.assertEqual(g.weighting(a, "balanced")[1], spec)

    def test_control_and_auxiliary_weights_are_preserved_without_mutation(self):
        a = synthetic_arrays()
        a["weight_string_birth"][6] = 0
        original = copy.deepcopy(a)
        roles, control = g.weighting(a, "control")
        _, balanced = g.weighting(a, "balanced")
        left = g.batch_weights(a, roles["fit"], control)
        right = g.batch_weights(a, roles["fit"], balanced)
        np.testing.assert_array_equal(left["cardinality"], np.ones(len(roles["fit"])))
        self.assertFalse(np.array_equal(left["cardinality"], right["cardinality"]))
        for name in g.e.LOSS_WEIGHTS:
            if name != "cardinality":
                np.testing.assert_array_equal(left[name], right[name])
        for name in a:
            np.testing.assert_array_equal(a[name], original[name])

    def test_invalid_arm_or_missing_fit_group_is_rejected(self):
        a = synthetic_arrays()
        with self.assertRaisesRegex(g.BalancedError, "unknown"):
            g.weighting(a, "other")
        roles, _ = g.weighting(a, "control")
        a["target_cardinality"][roles["fit"]] = 1
        with self.assertRaisesRegex(g.BalancedError, "both fit groups"):
            g.weighting(a, "balanced")

    def test_head_audit_rejects_changed_identity_or_incoherent_residual(self):
        a = synthetic_arrays()
        val = g.f.partitions(a["row_fold"], a["member"], 0)["validation"]
        strings = np.tile(np.linspace(.1, .9, 6), (len(val), 1)).astype(np.float32)
        pb = g.model_code.poisson_binomial_numpy(strings).astype(np.float32)
        logits = np.zeros_like(pb)
        prediction = {"global_index": a["global_index"][val].copy(), "member": a["member"][val],
            "k": a["target_cardinality"][val], "string_birth": strings, "poibin": pb,
            "residual_logits": logits, "probability": g.model_code.residual_cardinality_numpy(pb, logits).astype(np.float32)}
        g.verify_prediction(prediction, a)
        changed = copy.deepcopy(prediction)
        changed["global_index"][0] = 0
        with self.assertRaisesRegex(g.BalancedError, "identities"):
            g.verify_prediction(changed, a)
        changed = copy.deepcopy(prediction)
        changed["residual_logits"][:, 0] += 2
        with self.assertRaisesRegex(g.BalancedError, "residual/count"):
            g.verify_prediction(changed, a)


@unittest.skipUnless(importlib.util.find_spec("tensorflow"), "TensorFlow gates run in Actions before training")
class TensorFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import tensorflow as tf
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(4)

    def test_weighted_crossentropy_and_gradient_match_two_group_means(self):
        import tensorflow as tf
        a = synthetic_arrays()
        roles, spec = g.weighting(a, "balanced")
        k = a["target_cardinality"][roles["fit"]]
        w = g.batch_weights(a, roles["fit"], spec)["cardinality"]
        logits = tf.Variable(np.random.default_rng(2).normal(size=(len(k), 7)).astype(np.float32))
        with tf.GradientTape(persistent=True) as tape:
            per_row = tf.keras.losses.sparse_categorical_crossentropy(k, logits, from_logits=True)
            actual = tf.reduce_mean(per_row * w)
            expected = .5 * tf.reduce_mean(tf.boolean_mask(per_row, k < 2)) + .5 * tf.reduce_mean(tf.boolean_mask(per_row, k >= 2))
        np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=1e-6)
        np.testing.assert_allclose(tape.gradient(actual, logits).numpy(), tape.gradient(expected, logits).numpy(), rtol=1e-6, atol=1e-7)

    def test_both_arms_resume_finish_and_compare_without_outer_feature_access(self):
        a = synthetic_arrays()
        data = np.random.default_rng(10).uniform(0, 1, (15, *g.e.FEATURE_SHAPE)).astype(np.float16)
        accessed = []

        class TrackedFeatures:
            def __getitem__(self, rows):
                accessed.extend(np.asarray(rows).tolist())
                return data[rows]

        with TemporaryDirectory() as temp:
            root = Path(temp)
            with patch.object(g, "load_data", return_value=(TrackedFeatures(), a, {})):
                for arm in g.ARMS:
                    for chunk in (1, 2):
                        g.train_chunk(SimpleNamespace(prepared_dir=root, arm=arm, chunk=chunk,
                            previous_dir=root / f"v280-g-{arm}-1" if chunk == 2 else None,
                            output_dir=root / f"v280-g-{arm}-{chunk}"))
                    state = g.state_from(root / f"v280-g-{arm}-2", a, arm, 2)
                    self.assertEqual(state["optimizer_updates"], 12)
                    self.assertEqual(state["epochs_completed"], 12)
                self.assertTrue(set(accessed).isdisjoint({0, 1, 2}))
                for row in range(3, 15):
                    self.assertEqual(accessed.count(row), 24)
                g.compare(SimpleNamespace(prepared_dir=root, input_dir=root, output_dir=root / "comparison"))
                result = g.f.read_json(root / "comparison/comparison.json")
                self.assertEqual(result["status"], "complete")
                self.assertFalse(result["outer_evaluation_launched"])
                self.assertEqual(result["arms"]["control"]["initial_all_weights_sha256"],
                                 result["arms"]["balanced"]["initial_all_weights_sha256"])
                with self.assertRaisesRegex(g.BalancedError, "arm"):
                    g.state_from(root / "v280-g-control-1", a, "balanced", 1)


if __name__ == "__main__":
    unittest.main()
