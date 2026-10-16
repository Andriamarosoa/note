"""Training plumbing smoke tests, synthetic pairs only: not model validation."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from causal_note.harmonic_expert import FEATURE_NAMES, HarmonicGenuineOnsetHead, N_FEATURES
from scripts.train_live_harmonic_expert import train


class HarmonicTrainingTests(unittest.TestCase):
    def save(self, path, labels, groups, seed):
        y = np.asarray(labels, np.int8)
        rng = np.random.default_rng(seed)
        X = rng.normal(0, 1, (len(y), N_FEATURES)).astype(np.float32)
        X[:, 0] += 2. * y
        np.savez(path, X=X, y=y, groups=np.asarray(groups),
                 feature_names=np.asarray(FEATURE_NAMES))

    def test_training_saves_loadable_head_and_calibrates_only_on_dev(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            ytr = np.tile([0, 1], 180)
            ydev = np.tile([0, 1], 70)
            self.save(folder / "train.npz", ytr,
                      np.repeat(["train1", "train2", "train3"], 120), 10)
            self.save(folder / "dev.npz", ydev,
                      np.repeat(["dev1", "dev2"], 70), 20)
            out = folder / "head.npz"
            report = train(folder / "train.npz", folder / "dev.npz", out)
            self.assertTrue(out.exists())
            self.assertTrue(HarmonicGenuineOnsetHead(out).trained)
            self.assertGreaterEqual(report["genuine_dev_recall"], .99)

    def test_rejects_group_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            ytr = np.tile([0, 1], 180)
            ydev = np.tile([0, 1], 70)
            self.save(folder / "train.npz", ytr,
                      np.repeat(["train1", "train2", "train3"], 120), 10)
            self.save(folder / "dev.npz", ydev,
                      np.repeat(["train1", "dev2"], 70), 20)
            with self.assertRaisesRegex(ValueError, "overlap"):
                train(folder / "train.npz", folder / "dev.npz", folder / "bad.npz")


if __name__ == "__main__":
    unittest.main()
