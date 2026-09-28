import unittest
import numpy as np
from scripts.audit_v273_acoustic_residual import ownership, raw_metrics, category_masks


class AcousticResidualTests(unittest.TestCase):
    def test_audible_attack_can_be_eligible_for_two_groups_but_owned_once(self):
        assigned,k,eligible,d=ownership([np.array([10000]),np.array([11200])],np.array([10600]))
        np.testing.assert_array_equal(assigned,[0])
        np.testing.assert_array_equal(k,[1,0])
        np.testing.assert_array_equal(eligible[:,0],[True,True])
        self.assertEqual(int(d[1,0]),600)

    def test_unassigned_annotation_is_not_a_birth_of_either_group(self):
        assigned,k,eligible,_=ownership([np.array([10000]),np.array([13000])],np.array([11500]))
        np.testing.assert_array_equal(assigned,[-1])
        np.testing.assert_array_equal(k,[0,0])
        self.assertFalse(eligible.any())

    def test_nonzero_quiet_signal_is_not_digital_silence(self):
        quiet=raw_metrics(np.ones(4096)*1e-5)
        zero=raw_metrics(np.zeros(4096))
        self.assertFalse(quiet['digitally_silent'])
        self.assertTrue(zero['digitally_silent'])
        self.assertAlmostEqual(quiet['rms_dbfs'],-100)

    def test_categories_overlap_and_never_count_k4_as_low_k(self):
        masks=category_masks(np.array([0,2,4]),np.array([1,3,5]),np.array([1,3,5]),
                            np.array([1,3,5]),np.array([1,0,1]),np.array([1,3,5]))
        self.assertEqual(int(masks['local_eligible_equals_prediction'].sum()),2)
        self.assertEqual(int(masks['active_note_count_equals_prediction'].sum()),2)
        self.assertFalse(any(v[-1] for v in masks.values()))


if __name__=='__main__':unittest.main()
