import unittest
import numpy as np
from scripts.audit_v273_low_k import decisions, transitions, event_context, spectral_probe


class LowKAuditTest(unittest.TestCase):
    def test_scope_uses_truth_even_when_prediction_is_four_or_more(self):
        r = decisions(np.array([0,1,2,3,4,6]), np.array([2,1,5,4,5,6]))
        self.assertEqual((r['rows'],r['correct'],r['over']), (4,1,3))
        self.assertEqual(r['confusion'][2][5],1)

    def test_removing_overcount_can_create_undercount(self):
        r = transitions(np.array([2,2,0,3]), np.array([3,4,0,1]), np.array([1,2,1,4]))
        self.assertEqual((r['removed_over'],r['over_to_correct'],r['over_to_under']), (2,1,1))
        self.assertEqual((r['new_over'],r['correct_to_over'],r['under_to_over']), (2,1,1))
        self.assertEqual(r['net_over'],0)

    def test_context_separates_neighbor_birth_and_old_sustain(self):
        context, checks = event_context(np.array([10000,12000]),
            [np.array([10000]),np.array([12000])],
            [(10000,10100,0),(12200,12500,1),(7000,12500,2)],np.array([1,1]))
        self.assertEqual(context['foreign_births_added_tail'][0],1)
        self.assertEqual(context['own_births_added_tail'][0],0)
        self.assertEqual(context['carried_notes_at_start'][0],1)
        self.assertEqual((checks['assigned'],checks['unassigned']),(2,1))

    def test_own_late_birth_is_not_neighbor_contamination(self):
        c, _ = event_context(np.array([10000]), [np.array([10000,11764])],
                             [(12200,13000,0)],np.array([1]))
        self.assertEqual(c['own_births_added_tail'][0],1)
        self.assertEqual(c['foreign_births_added_tail'][0],0)

    def test_label_mismatch_blocks_context_audit(self):
        with self.assertRaises(AssertionError):
            event_context(np.array([10000]),[np.array([10000])],[(10000,11000,0)],np.array([0]))

    def test_tail_interventions_preserve_prefix_and_do_not_mutate(self):
        x=np.arange(2*31*4*3,dtype=np.float32).reshape(2,31,4,3)
        original=x.copy()
        for kind in ('tail_zero','tail_hold'):
            z=spectral_probe(x,kind)
            np.testing.assert_array_equal(z[:,:23],original[:,:23])
        np.testing.assert_array_equal(spectral_probe(x,'tail_hold')[:,23:],np.repeat(x[:,22:23],8,axis=1))
        np.testing.assert_array_equal(spectral_probe(x,'head_zero')[:,8:],original[:,8:])
        np.testing.assert_array_equal(x,original)


if __name__=='__main__':
    unittest.main()
