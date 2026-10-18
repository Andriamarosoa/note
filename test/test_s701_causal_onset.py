"""S701 frozen-ABI, past-only and shadow candidate regression checks."""
import unittest
import numpy as np
from scripts.s701_train_causal_onset_shadow import (
    onset_features, true_onset_rows, training_rows, rescue, select_on_held,
)

class S701Tests(unittest.TestCase):
    def fixture(self, n=80):
        rng = np.random.default_rng(701)
        feat = rng.random((n, 49, 126), dtype=np.float32)
        post = rng.random((n, 49, 3), dtype=np.float32)
        k = rng.random((n, 7), dtype=np.float32)
        k /= k.sum(axis=1, keepdims=True)
        return feat, post, k

    def test_exact_feature_abi_and_causality(self):
        feat, post, k = self.fixture()
        a = onset_features(feat, post, k)
        self.assertEqual(a.shape, (80, 49, 39))
        feat[46:] = 50
        post[46:] = 1
        k[46:] = 0
        b = onset_features(feat, post, k)
        np.testing.assert_array_equal(a[:46], b[:46])

    def test_only_original_onsets_may_be_rescued(self):
        _, post, _ = self.fixture(n=22)
        old = post.copy()
        class Selector:
            onset = .54
            active_on = .48
        score = np.ones((22, 49), np.float32)
        out = rescue(post, score, "head_0.85_floor_0.20", Selector())
        np.testing.assert_array_equal(out[:, :, 1:], old[:, :, 1:])
        np.testing.assert_array_equal(
            out[:, :, 0][old[:, :, 0] >= .54],
            old[:, :, 0][old[:, :, 0] >= .54])
        self.assertTrue(np.all(out[:, :, 0] >= old[:, :, 0]))

    def test_held_gating_vetoes_false_positives(self):
        baseline = dict(onset_F1=.60, K_frames_correct=1000,
                        tp=200, fp=70)
        scores = {"native": baseline}
        for name in ("head_0.60_floor_0.10", "head_0.75_floor_0.10",
                     "head_0.85_floor_0.10", "head_0.92_floor_0.10",
                     "head_0.75_floor_0.20", "head_0.85_floor_0.20",
                     "head_0.92_floor_0.20"):
            scores[name] = dict(onset_F1=.61, K_frames_correct=999,
                                tp=205, fp=90)
        groups = {"a": {k: dict(onset_F1=v["onset_F1"]) for k, v in scores.items()}}
        selected, _, _ = select_on_held(scores, groups)
        self.assertEqual(selected, "native")

    def test_targets_include_only_observed_nonfuture_frames(self):
        ends = np.arange(1, 51, dtype=np.int64) * 256
        n = len(ends)
        onset = ends[10] / 44100.
        targets = true_onset_rows(ends, [dict(pitch=40, time=onset)], n)
        self.assertEqual(int(targets[:10, 0].sum()), 0)
        self.assertEqual(int(targets[:, 1:].sum()), 0)
        self.assertGreater(int(targets[:, 0].sum()), 0)

    def test_train_targets_never_enter_source_features(self):
        feat, post, k = self.fixture(n=100)
        x = onset_features(feat, post, k)
        ends = np.arange(1, 101, dtype=np.int64) * 256
        notes = [dict(pitch=41, time=.05)]
        a = training_rows(x, post, ends, notes, 701)
        self.assertEqual(a[0].shape[1], 39)
        self.assertTrue(set(np.unique(a[1])).issubset({0, 1}))

if __name__ == "__main__":
    unittest.main()
