from __future__ import annotations

import importlib.util
import unittest

import numpy as np

from causal_note.v280_causal_cqt import CausalCQTConfig
from scripts import train_v280_harmonic_count as v


class V280CountMathTests(unittest.TestCase):
    def test_parameter_budget_is_fixed_before_graph_execution(self):
        self.assertEqual(v.expected_parameter_count(harmonic=True), 110_402)
        self.assertEqual(v.expected_parameter_count(harmonic=False), 67_298)
        self.assertLess(v.expected_parameter_count(harmonic=True), 300_000)

    def test_string_fret_grid_covers_standard_guitar_and_preserves_unisons(self):
        config = CausalCQTConfig()
        indices = v.string_fret_pitch_indices(config)
        self.assertEqual(indices.shape, (6, 20))
        self.assertEqual(indices[0, 0], 0)
        self.assertEqual(indices[-1, -1], 43)
        midi_64_index = 64 - int(config.input_min_midi)
        self.assertGreaterEqual(int(np.sum(indices == midi_64_index)), 2)

    def test_poisson_binomial_deterministic_strings_emit_exact_k(self):
        q = np.zeros((7, 6), dtype=np.float64)
        for k in range(7):
            q[k, :k] = 1.0
        probability = v.poisson_binomial_numpy(q)
        np.testing.assert_allclose(probability, np.eye(7), rtol=0.0, atol=0.0)

    def test_poisson_binomial_matches_fair_binomial_and_expectation(self):
        q = np.full((1, 6), 0.5, dtype=np.float64)
        probability = v.poisson_binomial_numpy(q)
        expected = np.asarray([[1, 6, 15, 20, 15, 6, 1]], dtype=np.float64) / 64.0
        np.testing.assert_allclose(probability, expected, rtol=0.0, atol=1e-15)
        support = np.arange(7, dtype=np.float64)
        self.assertAlmostEqual(float((probability @ support)[0]), float(np.sum(q)), places=12)

    def test_zero_residual_preserves_non_degenerate_poisson_binomial(self):
        q = np.asarray([[0.1, 0.2, 0.35, 0.5, 0.65, 0.8]], dtype=np.float64)
        base = v.poisson_binomial_numpy(q)
        final = v.residual_cardinality_numpy(base, np.zeros_like(base))
        np.testing.assert_allclose(final, base, rtol=1e-12, atol=1e-12)

    def test_residual_can_express_dependent_string_correction(self):
        q = np.full((1, 6), 0.5, dtype=np.float64)
        base = v.poisson_binomial_numpy(q)
        residual = np.zeros_like(base)
        residual[:, 4] = 4.0
        final = v.residual_cardinality_numpy(base, residual)
        self.assertEqual(int(np.argmax(final, axis=1)[0]), 4)
        np.testing.assert_allclose(final.sum(axis=1), np.ones(1), atol=1e-12)

    def test_synthetic_targets_are_physically_consistent(self):
        config = CausalCQTConfig()
        targets = v.synthetic_targets(14, config, v.SEED)
        k = targets["cardinality"]
        np.testing.assert_array_equal(targets["poibin_cardinality"], k)
        np.testing.assert_array_equal(targets["string_birth"].sum(axis=1), k)
        np.testing.assert_array_equal(targets["string_fret_onset"].sum(axis=(1, 2)), k)
        self.assertTrue(np.all(targets["pitch_onset"].sum(axis=1) <= k))

    def test_invalid_probability_contract_is_rejected(self):
        with self.assertRaises(v.V280ModelError):
            v.poisson_binomial_numpy(np.zeros((2, 5)))
        with self.assertRaises(v.V280ModelError):
            v.poisson_binomial_numpy(np.full((2, 6), 1.1))
        with self.assertRaises(v.V280ModelError):
            v.residual_cardinality_numpy(np.zeros((2, 7)), np.zeros((3, 7)))


@unittest.skipUnless(importlib.util.find_spec("tensorflow"), "TensorFlow graph tests run on the Mac preflight")
class V280CountGraphTests(unittest.TestCase):
    def test_output_shapes_parameter_gate_and_harmonic_ablation(self):
        config = CausalCQTConfig()
        harmonic = v.build_model(config, seed=v.SEED, harmonic=True)
        ablation = v.build_model(config, seed=v.SEED, harmonic=False)
        expected = {
            "cardinality": (None, 7),
            "string_birth": (None, 6),
            "string_fret_onset": (None, 6, 20),
            "pitch_onset": (None, 44),
            "poibin_cardinality": (None, 7),
        }
        self.assertEqual({key: tuple(value.shape) for key, value in harmonic.output.items()}, expected)
        self.assertEqual(harmonic.count_params(), v.expected_parameter_count(harmonic=True))
        self.assertEqual(ablation.count_params(), v.expected_parameter_count(harmonic=False))
        harmonic_names = {layer.name for layer in harmonic.layers}
        ablation_names = {layer.name for layer in ablation.layers}
        self.assertIn("v280_trunk1_harmonic_aggregate", harmonic_names)
        self.assertNotIn("v280_trunk1_harmonic_aggregate", ablation_names)

    def test_forward_probabilities_and_one_gradient_step_are_finite(self):
        config = CausalCQTConfig()
        model = v.build_model(config, seed=v.SEED, harmonic=True)
        rng = np.random.default_rng(v.SEED)
        features = np.abs(
            rng.normal(
                0.0,
                0.3,
                (7, config.cluster_frames, len(config.center_frequencies_hz), 3),
            )
        ).astype(np.float32)
        targets = v.synthetic_targets(7, config, v.SEED + 1)
        before = model(features, training=False)
        np.testing.assert_allclose(before["cardinality"], before["poibin_cardinality"], atol=1e-6)
        for name in ("cardinality", "poibin_cardinality"):
            probability = np.asarray(before[name])
            self.assertTrue(np.isfinite(probability).all())
            np.testing.assert_allclose(probability.sum(axis=1), 1.0, atol=1e-5)
        loss = model.train_on_batch(features, targets, return_dict=True)
        self.assertTrue(np.isfinite(np.asarray(list(loss.values()), dtype=np.float64)).all())


if __name__ == "__main__":
    unittest.main()
