"""Behavioral checks for the new count-only spectral interface."""
import unittest
import numpy as np
import tensorflow as tf

from scripts.train_v273_raw_spectral_ab import (
    ARMS, CONV, SEED, extend_anchor, freeze_and_compile, nested_base, paired,
)
from scripts.train_v273_group_gate_ab import build_model, weight_hash


class RawSpectralTest(unittest.TestCase):
    def test_anchor_replay_capacity_gradient_and_freeze(self):
        anchor = build_model("learned_gate", SEED)
        rng = np.random.default_rng(22)
        inputs = {t.name.split(":")[0]: rng.uniform(.1, 1., (2, *t.shape[1:])).astype(np.float32)
                  for t in anchor.inputs}
        inputs["candidate_mask"][:] = 1.
        inputs["spectral_map"] *= 8.
        expected = np.asarray(anchor(inputs, training=False))
        hashes, sizes = [], []
        for arm in ARMS:
            model = extend_anchor(anchor, arm)
            hashes.append(weight_hash(model)); sizes.append(model.count_params())
            np.testing.assert_allclose(model(inputs, training=False), expected, rtol=2e-5, atol=2e-6)
            frozen = freeze_and_compile(model)
            conv = nested_base(model).get_layer(CONV)
            self.assertEqual(np.count_nonzero(conv.get_weights()[0][:, :, 5:, :]), 0)
            loss = model.train_on_batch(inputs, np.asarray([2, 4], np.int32))
            self.assertTrue(np.isfinite(loss))
            self.assertGreater(np.linalg.norm(conv.get_weights()[0][:, :, 5:, :]), 0.)
            for name, before in frozen.items():
                for a, b in zip(before, nested_base(model).get_layer(name).get_weights()):
                    np.testing.assert_array_equal(a, b)
        self.assertEqual(hashes[0], hashes[1])
        self.assertEqual(sizes[0], sizes[1])

    def test_added_raw_path_retains_information_lost_by_channel_centering(self):
        anchor = build_model("learned_gate", SEED)
        rng = np.random.default_rng(9)
        x = rng.uniform(.2, 3., (2, 31, 64, 3)).astype(np.float32)
        evidence = {}
        for arm in ARMS:
            model = extend_anchor(anchor, arm)
            base = nested_base(model)
            sp = next(t for t in base.inputs if t.name.startswith("spectral_map"))
            probe = tf.keras.Model(sp, base.get_layer("v273_extra_spectral_evidence").output)
            evidence[arm] = (np.asarray(probe(x)), np.asarray(probe(x + 2.)))
        # Float32 cancellation is amplified where three channels nearly agree.
        np.testing.assert_allclose(*evidence["normalized_duplicate"], atol=2e-5)
        np.testing.assert_allclose(evidence["raw"][1] - evidence["raw"][0], 2./12., atol=1e-6)

    def test_paired_accounting(self):
        y = np.asarray([0, 1, 2, 3, 4, 5, 6])
        a = np.asarray([0, 1, 1, 3, 3, 4, 5])
        b = np.asarray([1, 1, 2, 2, 4, 4, 5])
        q = paired(y, a, b)
        self.assertEqual(q["global"], {"corrected": 2, "regressed": 2, "net": 0})
        self.assertEqual(q["poly"]["net"], 1)
        self.assertEqual(q["low"]["net"], -1)


if __name__ == "__main__":
    unittest.main()
