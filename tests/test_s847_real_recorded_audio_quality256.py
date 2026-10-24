"""S847 audible recorded-bank sanity. Array signals below are test fixtures,
not a music demo, training audio, synthetic guitar, or authenticity proof.
"""
import unittest
import numpy as np
from scripts.s847_play_real_recorded_E2_E3_reference256 import true_capture_profile


class S847CapturedNoteSanityTests(unittest.TestCase):
    def test_detect_high_frequency_whistle_not_a_good_guitar_sample(self):
        n=44100
        t=np.arange(n,dtype=np.float64)/44100
        high=(np.sin(2*np.pi*13900*t)*.02).astype(np.float32)
        low=(np.sin(2*np.pi*165*t)*.02).astype(np.float32)
        a=true_capture_profile(high)
        b=true_capture_profile(low)
        self.assertGreater(a["energy_above_10khz"],.9)
        self.assertLess(b["energy_above_10khz"],.001)
        self.assertGreater(a["spectral_median_hz"],10000.)
        self.assertLess(b["spectral_median_hz"],300.)

    def test_reject_silent_and_too_short(self):
        with self.assertRaises(ValueError):
            true_capture_profile(np.zeros(256*4,np.float32))
        with self.assertRaises(ValueError):
            true_capture_profile(np.zeros(256*20,np.float32))

if __name__=="__main__":
    unittest.main()
