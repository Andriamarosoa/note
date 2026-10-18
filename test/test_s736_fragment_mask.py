"""S736: structural, temporal, differentiability and non-regression contracts.

Synthetic tests only; they do NOT certify improvements on GuitarSet.
"""
import unittest

import numpy as np
import torch

from causal_note.s736_fragment_mask import (
    MASKS, BLOCK_SAMPLES, BLOCK_MS, FEATURE_DIM,
    FragmentMaskRouter, block_features, all_mask_offers,
    subset_descriptors, subset_class_compatibility,
    apply_fragment_masks, validate_decision_grid,
)


class FragmentCombinationTests(unittest.TestCase):
    def setUp(self):
        self.post = np.zeros((3, 49, 3), dtype=np.float32)
        self.post[:, :, 0] = .10
        self.post[:, :, 1] = .70
        self.post[:, :, 2] = .20
        self.heads = np.zeros((3, 49, 6), dtype=np.float32)
        self.heads[:, :, 0] = .10
        self.heads[0, 0, 1] = .80
        self.heads[1, 1, 2] = .75
        self.heads[2, 2, 5] = .90
        self.eligible = np.zeros((3, 49), dtype=bool)
        self.eligible[0, 0] = self.eligible[1, 1] = \
            self.eligible[2, 2] = True
        self.context = np.zeros((3, 49, 16), np.float32)
        self.context[0, 0, 4] = .8
        self.k = np.zeros((3, 7), np.float32)
        self.k[:, 2] = 1.

    def test_all_32_combinations_and_single_shared_mask_each_block(self):
        self.assertEqual(len(MASKS), 32)
        self.assertEqual(len(set(MASKS)), 32)
        self.assertTrue(all(m & 1 for m in MASKS))
        self.assertAlmostEqual(BLOCK_MS, 1000 * 1024 / 44100)
        masks = np.asarray((3, 5, 33), dtype=np.uint8)
        revised = apply_fragment_masks(
            self.post, self.heads, self.eligible, masks)
        self.assertAlmostEqual(float(revised[0, 0, 0]), .80)
        self.assertAlmostEqual(float(revised[1, 1, 0]), .75)
        self.assertAlmostEqual(float(revised[2, 2, 0]), .90)
        self.assertTrue(np.array_equal(
            revised[:, :, 1:], self.post[:, :, 1:]))
        self.assertTrue(np.array_equal(
            revised[:, :, 0][~self.eligible],
            self.post[:, :, 0][~self.eligible]))

    def test_only_causal_four_hop_block(self):
        validate_decision_grid([BLOCK_SAMPLES * j for j in (1,2,3)])
        with self.assertRaises(ValueError):
            validate_decision_grid([BLOCK_SAMPLES, BLOCK_SAMPLES * 3])
        first = block_features(
            self.heads, self.context, self.eligible, self.k)
        modified = self.heads.copy()
        modified[2, 0, 1] = .99
        again = block_features(
            modified, self.context, self.eligible, self.k)
        self.assertEqual(first.shape, (3, FEATURE_DIM))
        np.testing.assert_array_equal(first[:2], again[:2])

    def test_k_conditional_masks_do_not_use_true_k(self):
        available = torch.ones(2, 6, dtype=torch.bool)
        legal = torch.ones(2, 6, 7, dtype=torch.bool)
        legal[:, 1:5, 0] = False
        valid = subset_class_compatibility(available, legal)
        self.assertEqual(tuple(valid.shape), (2, 32, 7))
        self.assertTrue(bool(valid[:, 0, :].all()))
        self.assertFalse(bool(valid[:, 1, 0].any()))  # H0+H1, K0
        self.assertTrue(bool(valid[:, 1, 2].all()))
        self.assertTrue(bool(valid[:, 16, 0].all()))  # H0+H5, K0

    def test_audited_neural_selector_gradient_and_keep(self):
        b = 3
        x = torch.from_numpy(block_features(
            self.heads, self.context, self.eligible, self.k))
        d = torch.from_numpy(subset_descriptors(
            self.post, self.heads, self.eligible, self.context))
        audit = torch.full((b, 32, 7, 3), .1)
        k = torch.from_numpy(self.k)
        valid = subset_class_compatibility(
            torch.ones(b, 6, dtype=torch.bool),
            torch.ones(b, 6, 7, dtype=torch.bool))
        model = FragmentMaskRouter()
        result = model(x, d, audit, k, valid)
        self.assertEqual(tuple(result["action_logits"].shape), (b, 33))
        self.assertEqual(tuple(result["class_selection_weights"].shape),
                         (b, 32, 7))
        self.assertEqual(tuple(result["p_correction"].shape), (b, 32))
        self.assertEqual(tuple(result["p_regression"].shape), (b, 32))
        self.assertTrue(bool(torch.isfinite(result["action_logits"]).all()))
        self.assertTrue(torch.allclose(
            result["class_selection_weights"].sum(dim=1),
            torch.ones((b, 7)), atol=1e-5))
        # Real gradient must reach both class weighting and risk estimation.
        loss = result["action_logits"][:, 2].mean() + \
            result["action_logits"][:, 0].mean()
        loss.backward()
        self.assertIsNotNone(model.k_gate.weight.grad)
        self.assertIsNotNone(model.risk[0].weight.grad)
        self.assertIsNotNone(model.keep_head[0].weight.grad)
        self.assertGreater(float(model.k_gate.weight.grad.abs().sum()), 0.)
        self.assertGreater(float(model.risk[0].weight.grad.abs().sum()), 0.)


if __name__ == "__main__":
    unittest.main()
