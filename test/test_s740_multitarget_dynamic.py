"""S740 architecture tests: real tensors, causal fragments, no fake promotion."""
import unittest
import numpy as np
import torch

from causal_note.s736_fragment_mask import MASKS, block_features
from causal_note.s740_multitarget_dynamic import (
    TARGETS, SixDomainFragmentRouters, build_offers, select_offers,
    target_descriptors, posterior_target, offset_instance_candidates,
    apply_offset_masks, pitch_identity_candidates, exact_k_candidates,
    apply_domain, PromotionGate,
)

class S740DomainTests(unittest.TestCase):
    def setUp(self):
        self.t = 8
        self.post = np.full((self.t, 49, 3), .2, np.float32)
        self.native = np.full_like(self.post, .3)
        self.context = np.zeros((self.t, 49, 16), np.float32)
        self.k = np.tile(np.array([.1, .7, .2, 0, 0, 0, 0], np.float32),
                         (self.t, 1))
        self.eligible = np.ones((self.t, 49), bool)

    def test_six_independent_trainable_routers_and_faithful_port(self):
        model = SixDomainFragmentRouters()
        self.assertEqual(tuple(model.routers), TARGETS)
        ids = [model.routers[key].head_proj[0].weight.data_ptr()
               for key in TARGETS]
        self.assertEqual(len(set(ids)), len(TARGETS))
        self.assertTrue(all(model.routers[t].attention.num_heads == 3
                            for t in TARGETS))
        x = torch.from_numpy(block_features(
            np.full((self.t, 49, 6), .2, np.float32),
            self.context, self.eligible, self.k))
        d = torch.zeros((self.t, 32, 12))
        a = torch.zeros((self.t, 32, 7, 3))
        c = torch.ones((self.t, 32, 7), dtype=torch.bool)
        result = model("pitch_identity", x, d, a,
                       torch.from_numpy(self.k), c)
        self.assertEqual(result["action_logits"].shape, (self.t, 33))
        self.assertEqual(result["class_selection_weights"].shape,
                         (self.t, 32, 7))
        with self.assertRaises(RuntimeError):
            model.select_masks(
                "pitch_identity", x.numpy(), d.numpy(), a.numpy(),
                self.k, c.numpy(), np.zeros(37), np.ones(37))

    def test_real_32_subsets_change_independently_by_fragment(self):
        base = np.full((self.t, 49), .25, np.float32)
        heads = np.repeat(base[..., None], 6, axis=-1)
        heads[:, :, 1] = .8
        heads[:, :, 2] = .6
        options = build_offers(base, heads, self.eligible)
        self.assertEqual(options.shape, (self.t, 32, 49))
        self.assertEqual(len(MASKS), 32)
        masks = np.array([1, 3, 5, 1, 63, 9, 1, 3], np.uint8)
        result = select_offers(base, heads, self.eligible, masks)
        np.testing.assert_array_equal(result[0], base[0])
        self.assertAlmostEqual(result[1, 0], .8, places=5)
        self.assertAlmostEqual(result[2, 0], .6, places=5)
        self.assertAlmostEqual(result[4, 0], .8, places=5)
        self.assertEqual(target_descriptors(
            base, heads, self.eligible).shape, (self.t, 32, 12))
        with self.assertRaises(ValueError):
            select_offers(base, heads, self.eligible, np.array([3]))

    def test_sustain_and_release_each_preserve_the_other_channels(self):
        for target, ch in (("sustain", 1), ("release", 2)):
            base, heads = posterior_target(
                target, self.post, self.native, self.context, self.k)
            new = apply_domain(target, base, heads, self.eligible,
                               np.array([1, 3, 1, 63, 3, 1, 5, 3]))
            self.assertEqual(new.shape, (self.t, 49))
            np.testing.assert_array_equal(new[0], base[0])
            newpost = self.post.copy()
            newpost[:, :, ch] = new
            for other in set((0, 1, 2)) - {ch}:
                np.testing.assert_array_equal(
                    newpost[:, :, other], self.post[:, :, other])
        with self.assertRaises(ValueError):
            posterior_target("offset", self.post, self.native,
                             self.context, self.k)

    def test_offset_uses_true_note_instances_not_pitch_aggregate(self):
        # Two separate retriggers of the same E2; their hazards stay distinct.
        frames = [np.array([3, 7], int), np.array([11, 15], int)]
        plan = dict(
            notes=[dict(pitch=40, start=0), dict(pitch=40, start=9)],
            frames=frames,
            probabilities=[
                np.array([.2, .4], np.float32),
                np.array([.7, .8], np.float32)
            ],
            onsets=[dict(type="note_on", pitch=40),
                    dict(type="note_on", pitch=40)]
        )
        post = np.full((16, 49, 3), .2, np.float32)
        ctx = np.zeros((16, 49, 16), np.float32)
        k = np.tile(self.k[0], (16, 1))
        candidates = offset_instance_candidates(plan, post, ctx, k)
        self.assertEqual([x.shape for x in candidates], [(2, 6), (2, 6)])
        masks = np.array([1, 3, 1, 31], np.uint8)
        chosen = apply_offset_masks(plan, candidates, masks)
        self.assertAlmostEqual(chosen["probabilities"][0][0], .2, places=6)
        self.assertNotAlmostEqual(chosen["probabilities"][0][1], .4, places=4)
        self.assertAlmostEqual(chosen["probabilities"][1][0], .7, places=6)
        np.testing.assert_array_equal(plan["probabilities"][0], np.array([.2, .4], np.float32))
        # A changed future note must not alter an earlier note candidate.
        updated = post.copy()
        updated[15:, :, :] = .9
        trial = offset_instance_candidates(plan, updated, ctx, k)
        np.testing.assert_array_equal(candidates[0], trial[0])
        with self.assertRaises(ValueError):
            apply_domain("offset", self.post[:, :, 0],
                         np.ones((8, 49, 6)), self.eligible,
                         np.ones(8, np.uint8))

    def test_pitch_identity_has_five_real_event_actions(self):
        base = np.zeros((self.t, 49, 5), np.float32)
        native = np.zeros_like(base)
        harmonic = np.zeros_like(base)
        base[..., 0] = 1
        native[..., 0] = .7
        native[..., 1] = .3
        harmonic[..., 0] = .6
        harmonic[..., 3] = .4
        original, h = pitch_identity_candidates(
            base, native, harmonic, self.post, self.context, self.k)
        self.assertEqual(h.shape, (self.t, 49, 5, 6))
        selection = apply_domain(
            "pitch_identity", original, h,
            np.ones_like(original, bool),
            np.array([1, 3, 5, 7, 9, 11, 13, 63]))
        np.testing.assert_array_equal(selection[0], original[0])
        np.testing.assert_allclose(selection.sum(-1), 1, atol=1e-6)
        self.assertGreater(selection[1, 0, 1], 0)
        self.assertGreater(selection[-1, 0, 3], 0)
        altered = self.post.copy()
        altered[-1] = .95
        _, hx = pitch_identity_candidates(
            base, native, harmonic, altered, self.context, self.k)
        np.testing.assert_array_equal(h[:-1], hx[:-1])

    def test_exact_k_requires_five_valid_distinct_source_inputs(self):
        independent = np.repeat(self.k[..., None], 5, axis=-1)
        independent[:, :, 0] = self.k[:, ::-1]
        origin, heads = exact_k_candidates(self.k, independent)
        choices = apply_domain(
            "exact_k", origin, heads,
            np.ones_like(origin, bool),
            np.array([1, 3, 1, 3, 63, 1, 3, 3]))
        np.testing.assert_array_equal(choices[0], origin[0])
        np.testing.assert_allclose(choices.sum(-1), 1, atol=1e-6)
        with self.assertRaises(ValueError):
            exact_k_candidates(self.k, independent[:, :, :4])

    def test_gate_needs_six_strict_over_80_and_actual_decoder(self):
        keys = ("onset_f1", "exact_k", "active_exact_k",
                "offset_accuracy", "full_note_f1", "frame_pitch_f1")
        old = {key: .75 for key in keys}
        good = {key: .81 for key in keys}
        def gate(actual, **kw):
            return PromotionGate(old, actual, 0, 0, True, True, True, **kw)
        self.assertTrue(gate(good).above_80())
        self.assertFalse(gate(dict(good, full_note_f1=.79)).above_80())
        self.assertFalse(PromotionGate(old, good, 1, 0,
                                       True, True, True).safe())
        self.assertFalse(PromotionGate(old, good, 0, 0,
                                       False, True, True).above_80())
        with self.assertRaises(ValueError):
            gate({"onset_f1": .9}).safe()


if __name__ == "__main__":
    unittest.main()
