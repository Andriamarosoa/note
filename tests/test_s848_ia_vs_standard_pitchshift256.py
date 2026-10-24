"""S848 quality comparison tests, NOT synthesized musical training data."""
import unittest
import numpy as np
from scripts.s848_ia_vs_standard_pitchshift256 import (
    spec_profile,envelope_similarity,estimate_note_histogram,SHIFT,
    DECISION_THRESHOLD)


class S848ComparisonToolsTests(unittest.TestCase):
    def test_same_source_and_ai_has_fixed_pcm256_hops(self):
        self.assertEqual(SHIFT,12)
        self.assertTrue(0<DECISION_THRESHOLD<1)

    def test_profile_rejects_silence(self):
        info=spec_profile(np.zeros(44100,np.float32))
        self.assertTrue(info["silent"])
        self.assertIsNone(info["median_hz"])

    def test_spectral_reference_can_distinguish_low_and_ultrahigh(self):
        t=np.arange(44100,dtype=np.float64)/44100
        x=np.sin(2*np.pi*165*t).astype(np.float32)
        y=np.sin(2*np.pi*15000*t).astype(np.float32)
        a=spec_profile(x)
        b=spec_profile(y)
        self.assertLess(a["median_hz"],200)
        self.assertGreater(b["median_hz"],14000)
        self.assertLess(a["over_10khz"],.01)
        self.assertGreater(b["over_10khz"],.99)

    def test_envelope_metrics_will_not_treat_silence_as_pitch_accuracy(self):
        x=np.zeros(256*40,np.float32)
        self.assertIsNone(envelope_similarity(x,x))
        p=np.tile(np.array([1,2,3,4],np.float32).repeat(256),10)
        self.assertAlmostEqual(envelope_similarity(p,p),1.,places=8)

    def test_bank_output_stats_count_silence_and_top_midi_choices(self):
        stats=estimate_note_histogram([52,None,52,64,None],{52:1,64:2})
        self.assertEqual(stats["played_hops"],3)
        self.assertEqual(stats["silence_hops"],2)
        self.assertEqual(stats["played_note_histogram"]["52"],2)

if __name__=="__main__":
    unittest.main()
