"""Guards against evaluating the wrong rows or reversing intervention effects."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.audit_v273_fold3_ablation import effect, load_fold_cache, metrics
from scripts.train_v100_spectral_string_slots import CACHE_SCHEMA_VERSION


class Fold3AuditTests(unittest.TestCase):
    def test_subset_keeps_global_indices_and_shard_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for shard, (members, values) in enumerate([
                    (['a', 'b', 'a'], [10, 11, 12]), (['b', 'a'], [13, 14])]):
                arrays = {key: np.array(values) for key in ('spectral', 'sequence', 'mask', 'stats', 'exact')}
                np.savez(root/f'v100-spectral-shard-{shard:02d}.npz',
                         schema_version=[CACHE_SCHEMA_VERSION], members=members, **arrays)
            cache, indices, total = load_fold_cache(root, {'a': 3, 'b': 1})
            self.assertEqual(total, 5)
            np.testing.assert_array_equal(indices, [0, 2, 4])
            np.testing.assert_array_equal(cache['spectral'], [10, 12, 14])
            np.testing.assert_array_equal(cache['members'], ['a', 'a', 'a'])

    def test_wrong_to_wrong_is_not_counted_as_corrected(self):
        k = np.array([2, 2, 3, 2])
        before, after = np.array([3, 2, 4, 2]), np.array([2, 3, 2, 2])
        actual = effect(k, before, after, np.ones(4, bool))
        self.assertEqual(actual, {'rows': 4, 'changed': 3, 'corrected': 1,
                                  'regressed': 1, 'still_over': 1, 'still_under': 1})
        subset = effect(k, before, after, np.array([True, False, False, False]))
        self.assertEqual(subset['corrected'], 1)
        self.assertEqual(subset['regressed'], 0)

    def test_poly_counts_exclude_zero_and_one(self):
        actual = metrics(np.array([0, 1, 2, 3]), np.array([1, 1, 3, 3]))
        self.assertEqual(actual, {'rows': 4, 'correct': 2, 'over': 2, 'under': 0,
                                  'poly_rows': 2, 'poly_correct': 1, 'poly_over': 1, 'poly_under': 0})


if __name__ == '__main__':
    unittest.main()
