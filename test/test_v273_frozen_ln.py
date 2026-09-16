import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts import train_v273_frozen_ln as repair
from scripts import train_v273_native_paired as paired
from scripts import v273_decay_native_inputs as native


class SelectionTests(unittest.TestCase):
    @staticmethod
    def record(poly=40, correct=80, poly_over=20, over=30, strength=0.5, teacher=0.0):
        return {'metrics': {'rows': 100, 'poly_rows': 60, 'poly_correct': poly,
                            'correct': correct, 'poly_over': poly_over, 'over': over},
                'strength': strength, 'teacher_weight': teacher}

    def test_rejects_more_overcounting_or_global_damage_even_if_poly_improves(self):
        base = self.record()
        for bad in [self.record(poly=42, poly_over=21), self.record(poly=42, over=31),
                    self.record(poly=42, correct=79), self.record(poly=40, correct=82)]:
            self.assertEqual(repair.select_inner({'control': base, 'candidate': bad}), 'control')

    def test_selects_inner_improvement_and_conservative_tie(self):
        records = {'control': self.record(), 'small': self.record(poly=41, strength=0.25),
                   'large': self.record(poly=41, strength=1.0)}
        self.assertEqual(repair.select_inner(records), 'small')
        records['better'] = self.record(poly=42, strength=1.0)
        self.assertEqual(repair.select_inner(records), 'better')

    def test_rejects_changed_population(self):
        wrong = self.record(poly=41)
        wrong['metrics']['rows'] = 101
        with self.assertRaises(RuntimeError):
            repair.select_inner({'control': self.record(), 'wrong': wrong})

    def test_cannot_claim_gain_against_numerically_weaker_replay(self):
        control = self.record(poly=39)
        control['archived_metrics'] = self.record(poly=40)['metrics']
        self.assertEqual(repair.select_inner({'control': control, 'candidate': self.record(poly=40)}), 'control')


@unittest.skipUnless(importlib.util.find_spec('tensorflow'), 'TensorFlow 2.15.1 integration test runs in CI')
class FrozenNetworkTests(unittest.TestCase):
    def test_only_projection_learns_zero_input_is_unchanged_and_export_reloads(self):
        import tensorflow as tf
        self.assertEqual(tf.__version__, '2.15.1')
        for component in native.COMPONENTS:
            with self.subTest(component=component), tempfile.TemporaryDirectory() as temporary:
                model, original = native.build_count_model(component, 4137)
                rng = np.random.default_rng(89)
                inputs = {tensor.name.split(':')[0]: rng.uniform(
                    0.1, 1, size=(4, *map(int, tensor.shape[1:]))).astype(np.float32)
                          for tensor in original.inputs}
                inputs['candidate_mask'][:] = 1
                evidence = np.ones((4, native.FEATURE_DIM), dtype=np.float32)*0.2
                evidence[:2] = 0
                x = {**inputs, 'decay_evidence': evidence}
                baseline = model(x, training=False).numpy()
                context_model = tf.keras.Model(original.inputs, original.get_layer('v240_cardinality_context').output)
                context = context_model(inputs, training=False).numpy()
                before = paired.weights_digest(original)
                head = repair.freeze_and_build_head(model, original, component)
                head.compile(optimizer=tf.keras.optimizers.Adam(2e-4), loss='categorical_crossentropy')
                head_inputs = {'frozen_context': context, 'decay_evidence': evidence}
                expected = head(head_inputs, training=False).numpy()
                np.testing.assert_allclose(expected, baseline, atol=2e-6)
                classes = int(head.output_shape[-1])
                for _ in range(3):
                    head.train_on_batch(head_inputs, np.eye(classes, dtype=np.float32)[[0, 1, 2, 3]])
                self.assertEqual(paired.weights_digest(original), before)
                self.assertTrue(np.any(model.get_layer('native_decay_projection').kernel.numpy()))
                after = model(x, training=False).numpy()
                np.testing.assert_array_equal(after[:2], baseline[:2])
                self.assertGreater(float(np.max(np.abs(after[2:]-baseline[2:]))), 1e-7)
                np.testing.assert_allclose(head(head_inputs, training=False).numpy(), after, atol=2e-6)
                path = Path(temporary)/'model.weights.h5'
                model.save_weights(path)
                restored, _ = native.build_count_model(component, 4137)
                restored.load_weights(path)
                np.testing.assert_array_equal(restored(x, training=False).numpy(), after)


if __name__ == '__main__':
    unittest.main()
