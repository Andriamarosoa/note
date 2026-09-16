import importlib.util
import unittest

import numpy as np

from scripts import v273_decay_native_inputs as v


def exponential_crop(slope=-20.0):
    time = (np.arange(v.v100.TIME_FRAMES) - v.PRE_FRAMES + 1) * (
        v.v100.FRAME_STEP / v.v100.SAMPLE_RATE
    )
    power = np.broadcast_to(np.exp(1.0 + slope * time)[None, :, None], (2, 23, 64)).copy()
    crop = np.zeros((2, 23, 64, 3), dtype=np.float32)
    crop[..., 0] = np.log1p(power)
    return crop


class NativeDecayFeatureTests(unittest.TestCase):
    def test_fit_windows_are_strictly_before_cluster_and_observation_ends_at_40ms(self):
        ends = (np.arange(v.v100.TIME_FRAMES) * v.v100.FRAME_STEP
                + v.v100.FRAME_LENGTH - v.v100.PRE_SAMPLES)
        self.assertEqual(v.PRE_FRAMES, 9)
        self.assertLess(int(ends[v.PRE_FRAMES - 1]), 0)
        self.assertEqual(int(ends[-1]), 1764)

    def test_uninterrupted_exponential_has_no_new_attack(self):
        features, audit = v.native_decay_features(exponential_crop())
        self.assertTrue(audit["reliable"].all())
        np.testing.assert_allclose(audit["power_slope_per_second"], -20, atol=1e-5)
        np.testing.assert_allclose(features, 0, atol=1e-6)

    def test_new_attack_is_detected_without_refitting_on_post_cluster_evidence(self):
        crop = exponential_crop()
        baseline, first = v.native_decay_features(crop)
        changed = crop.copy()
        changed[:, v.PRE_FRAMES:, 10, 0] += 0.7
        features, second = v.native_decay_features(changed)
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])
        np.testing.assert_allclose(features[:, [10, 74]] - baseline[:, [10, 74]], 0.7, atol=1e-6)
        unchanged = np.ones(v.FEATURE_DIM, dtype=bool)
        unchanged[[10, 74]] = False
        np.testing.assert_array_equal(features[:, unchanged], baseline[:, unchanged])

    def test_silence_flat_rising_and_saturated_history_are_not_decay(self):
        crops = [np.zeros((2, 23, 64, 3)), exponential_crop(0), exponential_crop(20)]
        saturated = exponential_crop()
        saturated[:, 0, :, 0] = 12
        crops.append(saturated)
        for crop in crops:
            with self.subTest(kind=float(crop[0, 0, 0, 0])):
                features, audit = v.native_decay_features(crop)
                self.assertFalse(audit["reliable"].any())
                np.testing.assert_array_equal(features, 0)
                self.assertTrue(np.isfinite(features).all())

    def test_float16_cache_still_detects_a_reattack(self):
        crop = exponential_crop()
        crop[:, v.PRE_FRAMES:, :, 0] += 0.2
        features, audit = v.native_decay_features(crop.astype(np.float16))
        self.assertTrue(audit["reliable"].all())
        np.testing.assert_allclose(features, 0.2, atol=0.004)

    def test_rejects_v28_cqt_schema_and_invalid_maps(self):
        for crop in (np.zeros((1, 24, 238, 3)), np.full((1, 23, 64, 3), np.nan),
                     np.full((1, 23, 64, 3), -1), np.full((1, 23, 64, 3), 13)):
            with self.assertRaises(ValueError):
                v.native_decay_features(crop)

    def test_control_and_decay_preserve_all_four_original_inputs(self):
        crop = exponential_crop()
        crop[:, v.PRE_FRAMES:, :, 0] += 0.2
        cache = {"spectral": crop, "sequence": np.ones((2, 48, 3)),
                 "mask": np.ones((2, 48)), "stats": np.ones((2, 8))}
        indices = np.array([1, 0])
        control = v.native_inputs(cache, indices, "control")
        decay = v.native_inputs(cache, indices, "decay")
        for key in ("spectral_map", "candidate_set", "candidate_mask", "cluster_stats"):
            np.testing.assert_array_equal(control[key], decay[key])
        np.testing.assert_array_equal(control["decay_evidence"], 0)
        self.assertGreater(float(decay["decay_evidence"].max()), 0)
        with self.assertRaises(KeyError):
            v.native_inputs({k: a for k, a in cache.items() if k != "stats"}, indices, "control")


@unittest.skipUnless(importlib.util.find_spec("tensorflow"), "requires original TensorFlow 2.15.1")
class NativeNetworkParityTests(unittest.TestCase):
    @staticmethod
    def inputs(original):
        rng = np.random.default_rng(71)
        inputs = {}
        for tensor in original.inputs:
            name = tensor.name.split(":")[0]
            shape = (3, *[int(d) for d in tensor.shape[1:]])
            inputs[name] = rng.uniform(0.1, 1.0, size=shape).astype(np.float32)
        inputs["candidate_mask"][:] = 1
        return inputs

    def test_all_three_original_heads_have_exact_initial_parity_and_learnable_evidence(self):
        import tensorflow as tf
        self.assertEqual(tf.__version__, "2.15.1")
        for component in v.COMPONENTS:
            with self.subTest(component=component):
                extended, original = v.build_count_model(component, 7241)
                inputs = self.inputs(original)
                before = original(inputs, training=False).numpy()
                zeros = np.zeros((3, v.FEATURE_DIM), dtype=np.float32)
                evidence = np.full_like(zeros, 0.2)
                control = {**inputs, "decay_evidence": zeros}
                treatment = {**inputs, "decay_evidence": evidence}
                np.testing.assert_array_equal(extended(control, training=False).numpy(), before)
                np.testing.assert_array_equal(extended(treatment, training=False).numpy(), before)
                for layer in original.layers:
                    self.assertIs(extended.get_layer(layer.name), layer)
                projection = extended.get_layer("native_decay_projection").kernel
                labels = np.array([0, 1, 2], dtype=np.int32)
                extended.train_on_batch(control, labels)
                np.testing.assert_array_equal(projection.numpy(), 0)
                # Even after training, the zero-evidence branch exactly matches
                # the same current legacy weights, rather than merely matching K.
                np.testing.assert_array_equal(extended(control, training=False).numpy(),
                                              original(inputs, training=False).numpy())
                extended.train_on_batch(treatment, labels)
                self.assertTrue(np.any(projection.numpy() != 0))
                self.assertGreater(float(np.max(np.abs(
                    extended(treatment, training=False).numpy()
                    - extended(control, training=False).numpy()))), 1e-8)


if __name__ == "__main__":
    unittest.main()
