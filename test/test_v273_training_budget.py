import unittest
import numpy as np
from scripts.test_v273_training_budget import validate_optimizer_snapshot,interpret_budget


class TrainingBudgetTest(unittest.TestCase):
    def test_missing_or_reset_optimizer_is_rejected(self):
        for state in ([np.array(2712)],[np.array(0),np.zeros(2)]):
            with self.assertRaises(RuntimeError):
                validate_optimizer_snapshot(state,2712)
        validate_optimizer_snapshot([np.array(2712),np.zeros(2),np.zeros(2)],2712)

    def test_better_fit_alone_is_not_a_validation_success(self):
        self.assertEqual(interpret_budget(100,-10),'better_fit_does_not_improve_internal_validation')
        self.assertEqual(interpret_budget(100,0),'better_fit_does_not_improve_internal_validation')

    def test_internal_improvement_has_a_separate_verdict(self):
        self.assertEqual(interpret_budget(100,10),'more_training_improves_seen_and_internal_validation')
        self.assertEqual(interpret_budget(-1,10),'internal_gain_without_better_aggregate_fit')

    def test_paired_audit_counts_corrections_and_regressions(self):
        from scripts.summarize_v273_training_budget import paired_change
        before=dict(global_index=np.arange(4),k=np.array([1,2,2,3]),
                    member=np.array(['00_BN2-166-Ab_comp.jams']*4),predicted=np.array([0,1,2,3]))
        after=dict(before,predicted=np.array([1,2,1,3]))
        change=paired_change(before,after)
        self.assertEqual(change['all']['delta'],1)
        self.assertEqual(change['poly']['corrected'],1)
        self.assertEqual(change['poly']['regressed'],1)
        self.assertEqual(change['poly']['delta'],0)
        self.assertEqual(change['by_k']['1']['delta'],1)
        after['global_index']=np.array([1,0,2,3])
        with self.assertRaises(AssertionError):
            paired_change(before,after)

    def test_changed_counts_cannot_pass_metric_verification(self):
        from scripts.summarize_v273_training_budget import check_metrics
        check_metrics({'correct':3,'nll':0.1},{'correct':3,'nll':0.1+1e-12})
        with self.assertRaises(RuntimeError):
            check_metrics({'correct':3},{'correct':4})


if __name__=='__main__':
    unittest.main()
