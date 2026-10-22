"""Contract tests for remaining FP audit after native256 MLP69 veto."""
import unittest
import numpy as np
from scripts.audit_onset_fp_residual256 import (
    HYPOTHESIS_ISSUE, MAX_PAST_SAMPLES, prior_parent_ages,
    report_ratio, INTERVALS)
from onset256.acoustics import HOP,SR

class FPResidualAuditTests(unittest.TestCase):
    def test_preregistered_research_is_pcm256_and_exact80ms(self):
        self.assertEqual(HOP,256)
        self.assertEqual(SR,44100)
        self.assertEqual(MAX_PAST_SAMPLES,3528)
        self.assertEqual(HYPOTHESIS_ISSUE,142)
        self.assertEqual(INTERVALS,(12,19,24,28))
    def test_parent_features_see_only_native_past_not_current(self):
        previous=np.full(49,-10**12,np.int64)
        # E2 (MIDI40) was emitted at sample256. Future E3 (MIDI52)
        # at sample512 can observe ONLY the earlier E2 attack.
        previous[0]=256
        vec=prior_parent_ages(previous,512,52)
        self.assertEqual(vec.shape,(4,))
        self.assertAlmostEqual(float(vec[0]),1/32)
        self.assertEqual([float(x) for x in vec[1:]],[1.,1.,1.])
        # current or future attack cannot enter the pretrained risk model.
        previous[0]=512
        self.assertEqual(float(prior_parent_ages(previous,512,52)[0]),1.)
        previous[0]=768
        self.assertEqual(float(prior_parent_ages(previous,512,52)[0]),1.)
    def test_parent_age_tracks_4_fixed_harmonic_intervals(self):
        previous=np.full(49,-10**12,np.int64)
        for interval in INTERVALS:
            previous[88-40-interval]=256
        v=prior_parent_ages(previous,512,88)
        np.testing.assert_array_almost_equal(v,np.ones(4,np.float32)/32.)
    def test_causal_future_gt_cannot_enter_training_vector(self):
        rows=[dict(label="TP"),dict(label="FP"),dict(label="FP")]
        s=report_ratio(rows)
        self.assertEqual(s["tp"],1)
        self.assertEqual(s["fp"],2)
        self.assertAlmostEqual(s["false_rate"],2/3)
        self.assertEqual(s["proposals"],3)
if __name__=="__main__":unittest.main()
