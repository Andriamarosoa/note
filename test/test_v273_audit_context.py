import unittest
import numpy as np

from scripts.v273_audit_context import audit_context, LOCAL_COLUMNS


class AuditContextTests(unittest.TestCase):
    def test_global_ablation_preserves_original_memory_and_other_inputs(self):
        rng = np.random.default_rng(17)
        d = dict(member_votes=rng.random((3, 8, 5)),
                 group_features=rng.random((3, 255, 70)).astype(np.float32))
        original = d['group_features'].copy()
        global_inputs, evidence = audit_context(d, 'global')
        np.testing.assert_array_equal(d['group_features'], original)
        np.testing.assert_array_equal(global_inputs['group_features'][..., :61], original[..., :61])
        self.assertTrue(np.all(global_inputs['group_features'][..., [61+j for j in LOCAL_COLUMNS]] == 0))
        self.assertEqual(evidence['rows_with_local_evidence'], 0)
        local_inputs, evidence = audit_context(d, 'local')
        np.testing.assert_array_equal(local_inputs['group_features'], original)
        self.assertEqual(evidence['rows_with_local_evidence'], 3)
        self.assertEqual(evidence['local_values_nonzero'], 3*255*5)

    def test_invalid_mode_fails_instead_of_silently_disabling_memory(self):
        with self.assertRaises(ValueError):
            audit_context({}, 'missing')


if __name__ == '__main__':
    unittest.main()
