"""S846: negative tests for known failure modes in an AI guitar audio demo.

All test arrays are NON-MUSICAL software fixtures, not synthetic guitar,
training data, or purported audio demonstrations.
"""
import unittest
import numpy as np
from scripts.s845_make_real_audio_ab_demo256 import audit_transposed_audio


class TranspositionAudioGateTests(unittest.TestCase):
    def test_reject_stationary_high_frequency_output_unrelated_to_guitar(self):
        sr=44100
        seconds=3.5
        x=np.arange(round(sr*seconds))/sr
        # Both sides below are synthetic TEST FIXTURES only, never audio demos.
        source=np.sin(2*np.pi*196*x)*(.25+.3*(1.+np.sin(2*np.pi*1.3*x)))
        noise=np.sin(2*np.pi*13950*x)*.25
        a=audit_transposed_audio(source.astype(np.float32),noise.astype(np.float32))
        self.assertFalse(a["passed"])
        self.assertGreater(a["spectral_median_ratio"],7.)
        self.assertGreater(a["estimated"]["energy_above_10khz_fraction"],.45)
        self.assertGreater(a["source_envelope_cv"],a["estimated_envelope_cv"])

    def test_reject_silent_predictions(self):
        source=np.ones(44100,np.float32)*.1
        with self.assertRaises(ValueError):
            audit_transposed_audio(source,np.zeros_like(source))

    def test_reject_mismatched_or_too_short_audio(self):
        src=np.ones(10000,np.float32)*.1
        with self.assertRaises(ValueError):
            audit_transposed_audio(src,np.zeros(9999,np.float32))
        with self.assertRaises(ValueError):
            audit_transposed_audio(src[:300],src[:300])

    def test_gate_acceptance_is_not_pitch_accuracy_criterion(self):
        # If signal is unchanged it passes the basic *noise* audit.
        # This deliberately shows why a PASS CANNOT prove +12 transposition.
        t=np.arange(44100*3,dtype=np.float64)/44100
        source=np.sin(2*np.pi*200*t).astype(np.float32)
        a=audit_transposed_audio(source,source)
        self.assertTrue(a["passed"])
        self.assertAlmostEqual(a["spectral_median_ratio"],1.,places=9)


if __name__=="__main__":
    unittest.main()
