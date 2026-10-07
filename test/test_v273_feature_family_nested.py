import unittest
from unittest.mock import patch

import numpy as np

from scripts.audit_v273_feature_family_nested import (
    ALL_FEATURES, NAMES, choose_on_fit, decisions, fit_model, inner_models,
    score, validate_split,
)
from scripts.audit_v273_residual_feature_family_lofo import counts


def sample(folds):
    f = np.repeat(folds, 6)
    n = len(f)
    X = np.zeros((n, len(ALL_FEATURES)))
    X[:, 0] = np.arange(n)
    return {'X': X, 'y': np.tile([2, 3, 4, 2, 3, 4], len(folds)), 'fold': f,
            'row_id': np.arange(n) + 100 * min(folds),
            'recording': np.array([f'{q:02d}_Jazz1-120-C_comp.jams' for q in f])}


class NestedFamilyTests(unittest.TestCase):
    def test_inner_models_exclude_each_validation_fold_and_recording(self):
        fit = sample([0, 1, 2])
        def train(X, y, name):
            ids = X[:, 0].astype(int)
            return {'ids': ids, 'train_folds': set(fit['fold'][ids])}
        def predict(X, state, name):
            ids = X[:, 0].astype(int)
            self.assertFalse(set(ids) & set(state['ids']))
            self.assertFalse(set(fit['fold'][ids]) & state['train_folds'])
            return np.full(len(ids), .4)
        with patch('scripts.audit_v273_feature_family_nested.fit_model', side_effect=train), \
             patch('scripts.audit_v273_feature_family_nested.score', side_effect=predict):
            p, states = inner_models(fit)
        np.testing.assert_array_equal(p, np.full((4, 18), .4))
        self.assertEqual(len(states), 4)

    def test_scope_and_recording_overlap_are_rejected(self):
        fit, val = sample([0, 1, 2]), sample([4])
        validate_split(fit, val, 4)
        wrong = dict(fit, fold=np.full(len(fit['y']), 3))
        with self.assertRaisesRegex(RuntimeError, 'excluded'):
            inner_models(wrong)
        with self.assertRaisesRegex(RuntimeError, 'excluded'):
            validate_split(fit, val, 3)
        val['recording'][:] = fit['recording'][0]
        with self.assertRaisesRegex(RuntimeError, 'recording overlap'):
            validate_split(fit, val, 4)

    def test_selection_abstains_and_uses_no_val_labels(self):
        fit, val = sample([0, 1, 2]), sample([4])
        p = np.tile(np.where(fit['y'] == 2, .8, .2), (4, 1))
        choices, diagnostic = choose_on_fit(fit, p)
        self.assertEqual(choices['fixed050'], {'family': 'base', 'threshold': .5})
        self.assertEqual(choices['robust_family']['family'], 'base')
        changed = dict(val, y=np.full(len(val['y']), 3))
        pv = np.full((4, len(val['y'])), .8)
        original = decisions(val, pv, choices)
        mutated = decisions(changed, pv, choices)
        self.assertNotEqual(original, mutated)
        self.assertEqual(choose_on_fit(fit, p), (choices, diagnostic))
        bad = np.tile(np.where(fit['y'] == 3, 1., 0.), (4, 1))
        abstain, _ = choose_on_fit(fit, bad)
        self.assertTrue(all(v is None for v in abstain.values()))
        self.assertTrue(all(v['actions'] == 0 for v in decisions(val, pv, abstain).values()))

    def test_fixed_threshold_boundary_and_other_k_accounting(self):
        result = counts([{'true_k': k} for k in (2, 3, 4, 2)], [.5, .5, .8, .499], .5)
        self.assertEqual(result, {'actions': 3, 'corrections': 1, 'regressions': 1, 'other_k': 1, 'net': 0})

    def test_portable_models_support_every_frozen_family(self):
        from scripts.audit_v273_feature_family_nested import columns
        from scripts.audit_v273_residual_feature_family_lofo import classifier
        rng = np.random.default_rng(61007)
        fit = sample([0, 1, 2]); fit['X'][:] = rng.normal(size=fit['X'].shape)
        for name in NAMES:
            X = fit['X'][:, columns(name)]
            state = fit_model(X, fit['y'], name)
            query = rng.normal(size=(7, len(columns(name))))
            actual = score(query, state, name)
            train = np.isin(fit['y'], (2, 3))
            model = classifier().fit(np.array(X[train], order='C'), (fit['y'][train] == 2).astype(int))
            np.testing.assert_allclose(actual, model.predict_proba(query)[:, 1], rtol=0, atol=1e-15)
            self.assertEqual(state['scale_n_samples_seen'], 12)
            self.assertTrue(np.isfinite(actual).all())
            self.assertTrue(np.all((actual >= 0) & (actual <= 1)))


if __name__ == '__main__':
    unittest.main()
