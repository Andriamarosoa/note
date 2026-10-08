import unittest
import numpy as np
from scipy.optimize import approx_fprime
from scripts.v273_regression_loops import (
    binary_objective, conditional_objective, available, decode, fit, predict,
    group_context, select_policy, counts,
)


class RegressionLoopTests(unittest.TestCase):
    def test_joint_loss_analytic_gradient(self):
        rng = np.random.default_rng(91); n = 11
        z = rng.normal(size=n); x = rng.normal(size=(n, 3)); target = rng.integers(0, 2, n)
        theta = np.array([.7, -.2, .1, -.15, .25])
        args = (z, x, target)
        numerical = approx_fprime(theta, lambda t: binary_objective(t, *args)[0], 1e-6)
        np.testing.assert_allclose(numerical, binary_objective(theta, *args)[1], atol=5e-7)
        logits = rng.normal(size=(n, 7)); x = rng.normal(size=(n, 7, 3))
        av = np.zeros((n, 7), bool); av[:, [2, 4]] = True
        target = np.array([2, 4, 7, 7, 2, 4, 4, 7, 2, 7, 4])
        args = (logits, x, av, target, rng.integers(0, 2, n))
        numerical = approx_fprime(theta, lambda t: conditional_objective(t, *args)[0], 1e-6)
        np.testing.assert_allclose(numerical, conditional_objective(theta, *args)[1], atol=5e-7)

    def test_fit_ignores_excluded_labels_and_inference_uses_no_labels(self):
        rng = np.random.default_rng(14); n = 35
        b = np.full(n, 2); props = np.tile([3, 4], (n, 1))
        data = dict(truth=rng.choice([2, 3, 4, 5], n), baseline=b, z=rng.normal(size=n),
                    logits=rng.normal(size=(n, 7)), available=available(props, b))
        xr = rng.normal(size=(n, 2)); xq = rng.normal(size=(n, 7, 2)); train = np.arange(22); test = np.arange(22, n)
        model = fit(data, xr, xq, train, 'group_audit')
        changed = dict(data, truth=data['truth'].copy()); changed['truth'][test] = 6
        self.assertEqual(model, fit(changed, xr, xq, train, 'group_audit'))
        no_labels = {k: v for k, v in data.items() if k != 'truth'}
        r, cp, other, logits = predict(no_labels, xr, xq, test, model)
        np.testing.assert_allclose(cp.sum(1)+other, 1)
        self.assertTrue((cp[:, [0, 1, 5, 6]] == 0).all())

    def test_full_group_available_without_singleton_veto(self):
        props = np.full((1, 255), 2); props[:, 2] = 3  # S1+S2; neither singleton proposes 3.
        b = np.array([2]); audits = np.zeros((1, 255, 9)); audits[0, 2, 3] = .8
        gc, _ = group_context(props, audits, b)
        self.assertEqual(gc[0, 3, 3], .8)
        self.assertEqual(gc[0, 3, 12], 1.)
        self.assertEqual(gc[0, 3, 13], 1.)
        cp = np.zeros((1, 7)); cp[0, 2] = .1; cp[0, 3] = .8
        logits = np.zeros((1, 7)); logits[0, 6] = 1000  # Unsupported destination must stay masked.
        self.assertEqual(decode(b, available(props, b), np.array([.1]), cp, logits, 1)[0], 3)
        self.assertEqual(decode(b, available(props, b), np.array([.8]), cp, logits, 1)[0], 2)

    def test_inner_selection_rejects_zero_regression_collapse(self):
        b = np.full(20, 2); y = np.r_[np.full(10, 3), np.full(10, 2)]
        original = np.full(20, 3)
        good = original.copy(); good[10:18] = 2
        collapse = b.copy(); sacrifice = good.copy(); sacrifice[:5] = 2
        matrix = np.column_stack([original, collapse, sacrifice, good])
        chosen, proof = select_policy(y, b, matrix, original)
        self.assertEqual(chosen, 3)
        self.assertEqual(proof['chosen'], counts(y, b, good))
        self.assertEqual(proof['chosen']['regressions'], 2)

    def test_identity_of_successes_is_a_distinct_constraint(self):
        from scripts.route_v273_regression_loops import source_search
        b = np.repeat([2, 3, 4], 16); y = b.copy(); original = b.copy(); alternative = b.copy()
        for i, source in enumerate([2, 3, 4]):
            start = 16*i; y[start:start+8] = source+1
            original[start:start+4] = source+1
            original[start+8:start+16] = source+1
            alternative[start+4:start+8] = source+1
            alternative[start+8] = source+1
        matrix = np.column_stack([original, alternative])
        unprotected, _ = source_search(y, b, original, matrix, False)
        protected, _ = source_search(y, b, original, matrix, True)
        np.testing.assert_array_equal(unprotected, [1, 1, 1])
        np.testing.assert_array_equal(protected, [0, 0, 0])


if __name__ == '__main__': unittest.main()
