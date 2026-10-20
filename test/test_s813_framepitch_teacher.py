"""Contract tests; synthetic fixtures never substitute for GuitarSet scores."""
from copy import deepcopy
import unittest

import numpy as np
from threadpoolctl import threadpool_limits

from causal_note.s813_framepitch_teacher import (
    FEATURE_DIM, fit_teacher, sample_training_rows, specialist_region,
)


def fixture(name, comp, seed, frames=18):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 2, (frames, 49, FEATURE_DIM)).astype(np.float16)
    # A deliberately learnable acoustic variable, with positives also in bass.
    y = X[:, :, 0] > 1.3
    return dict(name=name, composition=comp, X=X, y=y)


class FramePitchTeacherTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = [fixture("a", "TRAIN_A", 11), fixture("b", "TRAIN_B", 12)]
        cls.cal = [fixture("c", "INNER_CAL", 13)]
        cls.teacher = fit_teacher(cls.train, cls.cal, max_points=700)
        cls.X = fixture("unseen", "OUTER", 14, frames=13)["X"]

    def test_future_suffix_and_partition_invariance(self):
        X = self.X.copy()
        before = X.copy()
        whole = self.teacher.predict_proba(X)
        changed = X.copy()
        changed[6:] = 100  # Future frames must never alter earlier decisions.
        np.testing.assert_array_equal(whole[:6], self.teacher.predict_proba(changed)[:6])
        partitioned = np.concatenate([self.teacher.predict_proba(X[:4]),
                                      self.teacher.predict_proba(X[4:9]),
                                      self.teacher.predict_proba(X[9:])])
        np.testing.assert_array_equal(whole, partitioned)
        np.testing.assert_array_equal(X, before)
        self.assertEqual(self.teacher.predict(X).shape, (13, 49))
        self.assertEqual(self.teacher.predict(X).dtype, np.dtype(bool))
        self.assertFalse(whole.flags.writeable)

    def test_retraining_one_head_does_not_modify_other_head_or_teacher(self):
        changed = deepcopy(self.teacher)
        flat = self.X.astype(np.float32).reshape(-1, FEATURE_DIM)
        original = self.teacher.predict_proba(self.X).copy()
        specialist_before = changed.heads["bass_harmonic"].probabilities(flat)
        general_before = changed.heads["general"].probabilities(flat)
        # Replace the fit of one owned head with a deliberately opposite task.
        labels = (flat[:, 0] < 1.3).astype(np.uint8)
        with threadpool_limits(limits=1):
            changed.heads["general"].model.fit(flat, labels)
        general_after = changed.heads["general"].probabilities(flat)
        self.assertGreater(float(np.max(np.abs(general_before-general_after))), .1)
        np.testing.assert_array_equal(
            specialist_before, changed.heads["bass_harmonic"].probabilities(flat))
        np.testing.assert_array_equal(original, self.teacher.predict_proba(self.X))

    def test_prior_weights_restore_true_rare_positive_population(self):
        row = fixture("rare", "FIT", 23, frames=30)
        row["y"][:] = False
        row["y"].ravel()[::100] = True
        _, y, weights, proof = sample_training_rows([row], "general", max_points=100)
        true_prior = float(row["y"].mean())
        sampled_prior = float(y.mean())
        weighted_prior = float((weights*y).sum()/weights.sum())
        self.assertGreater(sampled_prior, 5*true_prior)
        self.assertAlmostEqual(weighted_prior, true_prior, places=12)
        self.assertAlmostEqual(proof["weighted_positive_prior"], true_prior, places=12)
        self.assertIsNone(proof["class_weight"])

    def test_calibration_cannot_share_composition_or_recording(self):
        overlap = fixture("different_recording", "TRAIN_A", 30)
        with self.assertRaisesRegex(ValueError, "disjoint"):
            fit_teacher(self.train, [overlap], max_points=100)
        overlap = fixture("a", "DIFFERENT_COMP", 31)
        with self.assertRaisesRegex(ValueError, "disjoint"):
            fit_teacher(self.train, [overlap], max_points=100)

    def test_only_declared_selector_and_private_metadata(self):
        meta = self.teacher.metadata
        self.assertEqual(len(meta["calibration_results"]), 9)
        self.assertEqual(meta["calibration_compositions"], ["INNER_CAL"])
        self.assertEqual(meta["train_compositions"], ["TRAIN_A", "TRAIN_B"])
        self.assertFalse(meta["supervised_inputs"])
        meta["heads"].append("foreign")
        meta["selected"]["threshold"] = 0
        self.assertEqual(len(self.teacher.metadata["heads"]), 2)
        self.assertNotEqual(self.teacher.metadata["selected"]["threshold"], 0)
        with self.assertRaises(TypeError):
            self.teacher.heads["foreign"] = self.teacher.heads["general"]

    def test_region_uses_current_acoustics_and_rejects_s800_abi(self):
        X = np.zeros((2, 49, FEATURE_DIM), np.float32)
        region = specialist_region(X)
        self.assertTrue(region[:, :8].all())
        self.assertFalse(region[:, 8:].any())
        X[1, 10, [126, 134, 142]] = 2  # parent at MIDI50, upper at MIDI62
        region = specialist_region(X)
        self.assertFalse(region[0, 22])
        self.assertTrue(region[1, 22])
        with self.assertRaisesRegex(ValueError, "222"):
            self.teacher.predict(np.zeros((3, 49, 23), np.float32))
        X[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            self.teacher.predict(X)


if __name__ == "__main__":
    unittest.main()
