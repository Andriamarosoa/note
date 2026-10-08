"""Checks for label-blind diagnostic interventions and paired accounting."""
import unittest
import numpy as np

from scripts.audit_v273_k34_coherent import derangement, perturb, decode, flow


class DiagnosticTests(unittest.TestCase):
    def test_donors_are_within_class_nonself_and_reproducible(self):
        b = np.array([2, 3, 4, 2, 3, 4, 2, 3, 4])
        ids = np.array([91, 25, 5, 43, 66, 2, 17, 1, 55])
        d = derangement(b, ids, 27403)
        np.testing.assert_array_equal(b[d], b)
        np.testing.assert_array_equal(np.sort(d), np.arange(len(b)))
        self.assertTrue(np.all(d != np.arange(len(b))))
        np.testing.assert_array_equal(d, derangement(b, ids, 27403))
        order = np.array([8, 6, 4, 2, 0, 7, 5, 3, 1])
        d2 = derangement(b[order], ids[order], 27403)
        by_id = dict(zip(ids, ids[d]))
        self.assertEqual(by_id, dict(zip(ids[order], ids[order][d2])))

    def test_interventions_change_only_declared_blocks_and_no_source_mutation(self):
        rng = np.random.default_rng(81)
        original = dict(context=rng.normal(size=(6, 4)).astype(np.float32),
            subset_features=rng.uniform(0, 2, (6, 5, 64, 21)).astype(np.float32),
            subset_mask=np.ones((6, 5, 64), bool), baseline=np.eye(7)[[2]*6],
            keep_heads=rng.uniform(size=(6, 14)).astype(np.float32))
        saved = {k: v.copy() for k, v in original.items()}
        names = ['spectral__a', 'birth__a', 'damping__a', 'persistence__a']
        donor = np.roll(np.arange(6), 1)
        out = perturb(original, 'spectral', donor, names)
        np.testing.assert_array_equal(out['context'][:, 0], original['context'][donor, 0])
        np.testing.assert_array_equal(out['context'][:, 1:], original['context'][:, 1:])
        for name, columns in [('direct_votes', list(range(6))), ('audit_rates', [7,8,9,18,19,20]), ('identity_control', list(range(10,17)))]:
            out = perturb(original, name, donor, names)
            np.testing.assert_array_equal(out['subset_features'][..., columns], original['subset_features'][donor][..., columns])
            other = [j for j in range(21) if j not in columns]
            np.testing.assert_array_equal(out['subset_features'][..., other], original['subset_features'][..., other])
            np.testing.assert_array_equal(out['context'], original['context'])
        out = perturb(original, 'count_reference_size', None, names)
        np.testing.assert_allclose(np.expm1(out['subset_features'][..., [6,17]]), np.expm1(original['subset_features'][..., [6,17]])*2/3, rtol=1e-6)
        other = [j for j in range(21) if j not in [6,17]]
        np.testing.assert_array_equal(out['subset_features'][..., other], original['subset_features'][..., other])
        for key in saved:
            np.testing.assert_array_equal(original[key], saved[key])

    def test_decision_ties_low_classes_and_neutral_accounting(self):
        q = np.array([[.8,0,.10,.05,.03,.01,.01], [0,0,.4,.4,.1,.1,0], [0,0,.1,.7,.1,.1,0]])
        base = np.array([3,3,2]); y = np.array([0,3,3])
        pred = decode(q, base)
        np.testing.assert_array_equal(pred, [2,3,3])
        result = flow(y, base, pred, q, np.ones(3, bool))
        self.assertEqual((result['rows'],result['corrections'],result['regressions'],result['neutral'],result['net']), (2,1,0,1,1))
        self.assertAlmostEqual(result['probability_expected_net'], .65)


if __name__ == '__main__':
    unittest.main()
