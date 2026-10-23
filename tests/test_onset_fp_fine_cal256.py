"""Pre-registered native PCM256 CAL-only fine-grid experiment #145.

This exercise tests the previously missing resolution close to zero, without
modifying trained models, the 80 original audio tracks, or GT access in live.
"""
import inspect
import unittest
import numpy as np
from onset256.acoustics import HOP,SR
from scripts.recalibrate_onset_fp_track_sampling256 import (
    FINE_CUTS,MODES,HYPOTHESIS_ISSUE,HISTORIC_69,HISTORIC_FOUR,run,
)

class RecalibrateCALGridTests(unittest.TestCase):
    def test_grid_has_small_nonzero_choices_and_does_not_omit_old_boundaries(self):
        self.assertEqual(HOP,256)
        self.assertEqual(SR,44100)
        self.assertEqual(HYPOTHESIS_ISSUE,145)
        self.assertEqual(MODES,("track_uniform","track_class"))
        self.assertEqual(FINE_CUTS[0],0.0)
        self.assertEqual(FINE_CUTS[1],.00001)
        self.assertIn(.0007,FINE_CUTS)
        self.assertIn(.05,FINE_CUTS)
        self.assertIn(.99,FINE_CUTS)
        self.assertEqual(len(FINE_CUTS),len(set(FINE_CUTS)))
        self.assertEqual(list(FINE_CUTS),sorted(FINE_CUTS))
        self.assertEqual(HISTORIC_69,dict(tp=8127,fp=4519,fn=3140))
        self.assertEqual(HISTORIC_FOUR,dict(tp=8160,fp=4605,fn=3107))
    def test_exact_historic_CAL_guard_and_no_silent_fit_or_outer_label_use(self):
        s=inspect.getsource(run)
        self.assertIn("MAX_CAL_TP_LOSS_FRACTION*original_tp",s)
        self.assertIn("result["removed_fp"]>=MIN_CAL_FP_REMOVAL",s)
        self.assertIn("result["f1"]>original["f1"]",s)
        self.assertIn("joblib.load(model_path)",s)
        self.assertNotIn(".fit(",s)
        self.assertIn("source_fold["selected_cut"]!=0.",s)
        self.assertIn("source_trained_checkpoint_sha256",s)
        self.assertIn("digest(dst)!=sha",s)
        self.assertIn("OUTER_truth_not_used_in_this_script=True",s)
    def test_cut_zero_is_default_but_fine_granularity_admits_no_tp_loss(self):
        from scripts.train_onset_fp_discriminator256 import (
            MIN_CAL_FP_REMOVAL,MAX_CAL_TP_LOSS_FRACTION)
        self.assertEqual(MIN_CAL_FP_REMOVAL,5)
        self.assertAlmostEqual(MAX_CAL_TP_LOSS_FRACTION,.025)
        # Example: ten true positives are kept, six false positives cut.
        cut=.0007
        probs=np.array([.8]*10+[.0001]*6+[.9]*10,np.float32)
        truth=np.array([1]*10+[0]*6+[0]*10)
        loss_tp=int(np.sum((probs<cut)&(truth==1)))
        loss_fp=int(np.sum((probs<cut)&(truth==0)))
        self.assertEqual(loss_tp,0)
        self.assertEqual(loss_fp,6)
        self.assertTrue(loss_fp>=MIN_CAL_FP_REMOVAL)
        self.assertTrue(loss_tp<=.025*int(sum(truth)))
if __name__=="__main__":
    unittest.main()
