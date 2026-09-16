from __future__ import annotations

import copy
import importlib.util
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from scripts import train_v280_moderate_final as trial
from test.test_v280_nested_outer import synthetic_arrays


class ModerateTests(unittest.TestCase):
    def test_fit_groups_receive_35_and_65_percent_without_held_label_dependence(self):
        arrays = synthetic_arrays()
        roles, spec = trial.weighting(arrays)
        k = arrays['target_cardinality'][roles['fit']]
        weights = trial.g.batch_weights(arrays, roles['fit'], spec)['cardinality']
        self.assertAlmostEqual(float(weights[k >= 2].sum()), .35 * len(k), places=6)
        self.assertAlmostEqual(float(weights[k < 2].sum()), .65 * len(k), places=6)
        self.assertAlmostEqual(float(weights.mean()), 1, places=6)
        arrays['target_cardinality'][np.concatenate([roles['outer'], roles['validation']])] = 6
        self.assertEqual(trial.weighting(arrays)[1], spec)

    def test_auxiliary_masks_are_preserved_and_inputs_not_mutated(self):
        arrays = synthetic_arrays()
        arrays['weight_string_birth'][7] = 0
        original = copy.deepcopy(arrays)
        roles, spec = trial.weighting(arrays)
        modified = trial.g.batch_weights(arrays, roles['fit'], spec)
        control = trial.g.batch_weights(arrays, roles['fit'], trial.g.weighting(arrays, 'control')[1])
        for name in trial.e.LOSS_WEIGHTS:
            if name != 'cardinality':
                np.testing.assert_array_equal(modified[name], control[name])
        for name in arrays:
            np.testing.assert_array_equal(arrays[name], original[name])

    def test_any_weak_uncertain_or_harmful_result_returns_to_v273(self):
        base = {'rows': 10000, 'poly_rows': 1000, 'poly_correct': 400, 'correct': 8000,
                'confusion_matrix_true_by_predicted': [[0] * 7 for _ in range(7)]}
        candidate = {**base, 'poly_correct': 450}
        good = {'lower_95_pp': .1}
        self.assertTrue(trial.decide(base, candidate, good)['passed'])
        cases = [({**candidate, 'poly_correct': 449}, good),
                 (candidate, {'lower_95_pp': 0}),
                 ({**candidate, 'correct': 7999}, good)]
        more_fp = copy.deepcopy(candidate)
        more_fp['confusion_matrix_true_by_predicted'][1][2] = 1
        cases.append((more_fp, good))
        for metrics, bootstrap in cases:
            decision = trial.decide(base, metrics, bootstrap)
            self.assertEqual(decision['decision'], 'stop_v28_return_to_v273')
            self.assertFalse(decision['further_v28_weight_search'])
            self.assertFalse(decision['automatic_next_training'])

    def test_false_poly_counts_only_true_zero_or_one_predicted_as_two_or_more(self):
        confusion = np.zeros((7, 7), dtype=int)
        confusion[0, 1] = 100
        confusion[0, 2] = 3
        confusion[1, 6] = 4
        confusion[2, 3] = 50
        self.assertEqual(trial.false_poly({'confusion_matrix_true_by_predicted': confusion.tolist()}), 7)


@unittest.skipUnless(importlib.util.find_spec('tensorflow'), 'TensorFlow gates run in Actions')
class TensorFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import tensorflow as tf
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(4)

    def test_keras_weighted_loss_and_gradient_match_35_65_group_means(self):
        import tensorflow as tf
        arrays = synthetic_arrays()
        roles, spec = trial.weighting(arrays)
        k = arrays['target_cardinality'][roles['fit']]
        weights = trial.g.batch_weights(arrays, roles['fit'], spec)['cardinality']
        logits = tf.Variable(np.random.default_rng(22).normal(size=(len(k), 7)).astype(np.float32))
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        with tf.GradientTape(persistent=True) as tape:
            actual = loss(k, logits, sample_weight=weights)
            per_row = tf.keras.losses.sparse_categorical_crossentropy(k, logits, from_logits=True)
            expected = .65 * tf.reduce_mean(tf.boolean_mask(per_row, k < 2)) + .35 * tf.reduce_mean(tf.boolean_mask(per_row, k >= 2))
        np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=1e-6)
        np.testing.assert_allclose(tape.gradient(actual, logits).numpy(), tape.gradient(expected, logits).numpy(), rtol=1e-6, atol=1e-7)

    def test_frozen_control_complete_resume_pairing_and_final_decision(self):
        arrays = synthetic_arrays()
        raw = np.random.default_rng(13).uniform(0, 1, (15, *trial.e.FEATURE_SHAPE)).astype(np.float16)
        accessed = []

        class Features:
            def __getitem__(self, rows):
                accessed.extend(np.asarray(rows).tolist())
                return raw[rows]

        with TemporaryDirectory() as temp:
            root = Path(temp)
            with patch.object(trial, 'load_data', return_value=(Features(), arrays, {})), \
                 patch.object(trial.g, 'load_data', return_value=(Features(), arrays, {})):
                with patch.dict(os.environ, {'GITHUB_SHA': trial.CONTROL_SOURCE['head_sha'],
                                              'GITHUB_RUN_ID': str(trial.CONTROL_SOURCE['run_id'])}):
                    for chunk in (1, 2):
                        trial.g.train_chunk(SimpleNamespace(prepared_dir=root, arm='control', chunk=chunk,
                            previous_dir=root / 'control-1' if chunk == 2 else None, output_dir=root / f'control-{chunk}'))
                control = trial.f.read_json(root / 'control-2/state.json')
                reference_dir = root / 'reference'
                reference_dir.mkdir()
                trial.e.smoke._atomic_json(reference_dir / 'comparison.json',
                    {'status': 'complete', 'contract': trial.g.CONTRACT, 'arms': {'control': control}})
                digest = trial.e.smoke._sha256_file(reference_dir / 'comparison.json')
                with patch.dict(trial.CONTROL_SOURCE, {'comparison_json_sha256': digest}):
                    for chunk in (1, 2):
                        trial.train_chunk(SimpleNamespace(prepared_dir=root, chunk=chunk,
                            previous_dir=root / 'moderate-1' if chunk == 2 else None,
                            output_dir=root / f'moderate-{chunk}'))
                    state = trial.state_from(root / 'moderate-2', arrays, 2)
                    self.assertEqual(state['epochs_completed'], 12)
                    self.assertEqual(state['optimizer_updates'], 12)
                    self.assertFalse(set(accessed) & {0, 1, 2})
                    for row in range(3, 15):
                        self.assertEqual(accessed.count(row), 24)
                    arguments = dict(prepared_dir=root, control_dir=root / 'control-2',
                        control_comparison_dir=reference_dir, candidate_dir=root / 'moderate-2')
                    trial.compare(SimpleNamespace(**arguments, output_dir=root / 'result'))
                    result = trial.f.read_json(root / 'result/comparison.json')
                    self.assertEqual(result['control']['initial_all_weights_sha256'], result['moderate']['initial_all_weights_sha256'])
                    self.assertFalse(result['outer_evaluation_launched'])
                    self.assertEqual(result['decision']['official_reference'], 'V27.3')
                    state['initial_all_weights_sha256'] = 'different'
                    trial.e.smoke._atomic_json(root / 'moderate-2/state.json', state)
                    with self.assertRaisesRegex(trial.FinalTrialError, 'unpaired'):
                        trial.compare(SimpleNamespace(**arguments, output_dir=root / 'bad'))
                    with self.assertRaisesRegex(trial.FinalTrialError, 'contract'):
                        trial.state_from(root / 'control-1', arrays, 1)
                    (reference_dir / 'comparison.json').write_text('{}')
                    with self.assertRaisesRegex(trial.FinalTrialError, 'comparison changed'):
                        trial.load_control(root / 'control-2', reference_dir, arrays)


if __name__ == '__main__':
    unittest.main()
