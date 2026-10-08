import unittest
from unittest.mock import patch

from scripts.build_v273_head_history_registry import collect, role, tracked


class HistoryPreservationTests(unittest.TestCase):
    def test_learning_helpers_and_numerical_proofs_are_not_dropped(self):
        for path in ['scripts/learn_v273_catalogue.py', 'scripts/v273_probability_calibration.py',
                     'analysis/evidence/example.csv', 'analysis/evidence/predictions.npz',
                     'test/test_v273_catalogue_neural.py', '.github/workflows/v273-recording-calibration.yml']:
            self.assertTrue(tracked(path), path)
        self.assertEqual(role('scripts/learn_v273_catalogue.py'), 'predictor_or_neural_candidate')

    def test_inventory_keeps_each_variant_and_blob_provenance(self):
        def fake_git(*args):
            if args[0] == 'rev-parse': return 'commit-'+args[1]
            if args[0] == 'ls-tree':
                return 'a'*40+' scripts/learn_v273_catalogue.py\n'+'b'*40+' analysis/evidence/negative-result.csv'
            raise AssertionError(args)
        with patch('scripts.build_v273_head_history_registry.git', side_effect=fake_git), \
             patch('scripts.build_v273_head_history_registry.historical_fix_commits', return_value=[]):
            rows, missing, _ = collect(['old', 'new'])
        self.assertEqual(len(rows), 4)
        self.assertEqual(len({r['head_id'] for r in rows}), 4)
        self.assertEqual(missing, [])
        self.assertTrue(all(r['source_blob_sha'] for r in rows))
        self.assertTrue(all(not r['selection_eligible'] for r in rows))


if __name__ == '__main__':
    unittest.main()
