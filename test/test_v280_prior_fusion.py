from __future__ import annotations

import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from scripts import train_v280_prior_fusion as h
from test.test_v280_nested_outer import synthetic_arrays


class PriorFusionTests(unittest.TestCase):
    def test_inverse_weighting_recovers_known_posterior_without_labels(self):
        p = h.normalize(np.random.default_rng(17).uniform(.01, 1, (12, 7)))
        spec = {"nonpoly_weight": .57, "poly_weight": 4.08}
        weights = np.asarray([.57] * 2 + [4.08] * 5)
        q = h.normalize(p * weights)
        np.testing.assert_allclose(h.correct_prior(q, spec), p, rtol=1e-14)
        np.testing.assert_allclose(h.fuse(p, q, spec), p, rtol=1e-14)

    def test_fit_only_weights_and_explicit_second_partition(self):
        arrays = synthetic_arrays()
        roles = h.roles(arrays)
        self.assertEqual(set(arrays["row_fold"][roles["fit"]]), {1, 3, 4})
        self.assertTrue(np.all(arrays["row_fold"][roles["validation"]] == 2))
        before = h.weighting(arrays, "balanced")
        weights = h.g.batch_weights(arrays, roles["fit"], before)["cardinality"]
        k = arrays["target_cardinality"][roles["fit"]]
        self.assertAlmostEqual(float(weights[k >= 2].sum()), len(k) / 2)
        self.assertAlmostEqual(float(weights[k < 2].sum()), len(k) / 2)
        arrays["target_cardinality"][np.concatenate([roles["outer"], roles["validation"]])] = 6
        self.assertEqual(h.weighting(arrays, "balanced"), before)

    def test_gate_rejects_poly_gain_that_loses_global_or_adds_false_poly(self):
        control = {"poly_correct": 100, "correct": 900, "false_poly_rows": 30}
        self.assertFalse(h.passes_gate(control, control))
        self.assertTrue(h.passes_gate(control, {**control, "poly_correct": 101}))
        self.assertFalse(h.passes_gate(control, {**control, "poly_correct": 110, "correct": 899}))
        self.assertFalse(h.passes_gate(control, {**control, "poly_correct": 110, "false_poly_rows": 31}))

    def test_invalid_posteriors_and_weights_are_rejected(self):
        for p in (np.zeros((1, 7)), np.full((1, 7), np.nan), -np.ones((1, 7))):
            with self.assertRaises(h.FusionError):
                h.normalize(p)
        with self.assertRaises(h.FusionError):
            h.correct_prior(np.ones((1, 7)), {"nonpoly_weight": 0, "poly_weight": 1})
        with self.assertRaises(h.FusionError):
            h.fuse(np.ones((1, 7)), np.ones((2, 7)), {"nonpoly_weight": 1, "poly_weight": 1})


@unittest.skipUnless(importlib.util.find_spec("tensorflow"), "TensorFlow isolation gate runs in Actions")
class TensorFlowFreezeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import tensorflow as tf
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(4)

    def test_two_fresh_arms_freeze_before_one_validation_pass_and_never_use_outer(self):
        arrays = synthetic_arrays()
        raw = np.random.default_rng(11).uniform(0, 1, (15, *h.e.FEATURE_SHAPE)).astype(np.float16)
        accessed = []

        class TrackedFeatures:
            freeze_path = None

            def __getitem__(self, rows):
                rows = np.asarray(rows)
                if np.any(np.isin(rows, [6, 7, 8])):
                    if self.freeze_path is None or not self.freeze_path.is_file():
                        raise AssertionError("validation indexed before both states were frozen")
                accessed.extend(rows.tolist())
                return raw[rows]

        tracked = TrackedFeatures()
        with TemporaryDirectory() as temp:
            root = Path(temp)
            with patch.object(h, "load_data", return_value=(tracked, arrays, {})):
                for arm in h.g.ARMS:
                    h.train(SimpleNamespace(prepared_dir=root, arm=arm, output_dir=root / arm))
                    state = h.validate_state(root / arm, arrays, arm)
                    self.assertEqual(state["optimizer_updates"], 2)
                    self.assertEqual(state["epochs_completed"], 2)
                fit = h.roles(arrays)["fit"]
                self.assertEqual(set(accessed), set(fit))
                self.assertTrue(all(accessed.count(int(row)) == 4 for row in fit))
                accessed.clear()
                tracked.freeze_path = root / "result/freeze-before-validation.json"
                h.evaluate(SimpleNamespace(prepared_dir=root, input_dir=root, output_dir=root / "result"))
                self.assertEqual(sorted(accessed), [6, 6, 7, 7, 8, 8])
                report = h.f.read_json(root / "result/comparison.json")
                self.assertFalse(report["reference_promoted"])
                self.assertEqual(report["validation_inference_passes_per_arm"], 1)
                self.assertEqual(report["parameters_fusion"], 220804)
                self.assertEqual(report["states"]["control"]["initial_all_weights_sha256"],
                                 report["states"]["balanced"]["initial_all_weights_sha256"])
                # Saved predictions independently reproduce the frozen fusion and every metric.
                c = h.f.read_npz(root / "result/control-validation.npz")
                b = h.f.read_npz(root / "result/balanced-validation.npz")
                fusion = h.f.read_npz(root / "result/fusion-validation.npz")
                np.testing.assert_array_equal(fusion["global_index"], arrays["global_index"][[6, 7, 8]])
                np.testing.assert_array_equal(fusion["probability"], h.fuse(c["probability"], b["probability"], report["states"]["balanced"]["weighting"]))
                for name, p in (("control", c), ("balanced", b), ("fusion", fusion)):
                    actual = h.score(p["k"], p["probability"])
                    expected = report["metrics"][name]
                    for key in actual:
                        if key in ("nll", "poly_nll", "brier"):
                            self.assertAlmostEqual(actual[key], expected[key], places=12)
                        else:
                            self.assertEqual(actual[key], expected[key])
                # A mismatched arm must fail before another inference starts.
                state = h.f.read_json(root / "balanced/state.json")
                state["arm"] = "control"
                h.e.smoke._atomic_json(root / "balanced/state.json", state)
                accessed.clear()
                with self.assertRaisesRegex(h.FusionError, "arm"):
                    h.evaluate(SimpleNamespace(prepared_dir=root, input_dir=root, output_dir=root / "bad"))
                self.assertFalse(accessed)


if __name__ == "__main__":
    unittest.main()
