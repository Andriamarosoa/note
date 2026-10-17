"""S649 physical unique harmonic onset contrast and causal protection tests."""
import unittest
import numpy as np
from causal_note.strict_live import SR,HOP
from scripts.s649_exclusive_harmonic_attack import (
    exclusive_harmonic_attack, matrices,PATTERNS, WIDTH
)
from scripts.s640_learned_overlap_rescue import FEATURE_NAMES
from scripts.s641_fundamental_consistency import ADDITIONAL_NAMES


class ExclusiveHarmonicSupportTests(unittest.TestCase):
    def make_pcm(self):
        t=np.arange(20000)/SR
        f0=440*2**((52-69)/12)
        return np.asarray(
            .12*np.sin(2*np.pi*f0*t)+.07*np.sin(2*np.pi*2*f0*t)+
            .05*np.sin(2*np.pi*3*f0*t),np.float32)

    def test_duplicate_octave_competitor_removes_shared_harmonics(self):
        wave=self.make_pcm()
        native=np.zeros((49,3),np.float32)
        alone=np.zeros(49,bool)
        with_high=alone.copy()
        with_high[52-40+12]=True
        clean=exclusive_harmonic_attack(wave,17000,52-40,native,alone)
        competing=exclusive_harmonic_attack(wave,17000,52-40,native,with_high)
        self.assertEqual(clean.shape,(24,))
        self.assertTrue(np.isfinite(clean).all())
        self.assertGreater(float(clean[0]),float(competing[0]))
        self.assertLess(float(clean[1]),float(competing[1]))
        self.assertGreater(float(competing[6]),0.)
        self.assertLess(float(competing[6]),1.)

    def test_causality_held_when_future_pcm_changes(self):
        wave=self.make_pcm()
        altered=wave.copy()
        altered[18000:]*=.01
        native=np.zeros((49,3),np.float32)
        mask=np.zeros(49,bool)
        for sample in (8500,16000,18000):
            a=exclusive_harmonic_attack(wave,sample,4,native,mask)
            b=exclusive_harmonic_attack(altered,sample,4,native,mask)
            np.testing.assert_array_equal(a,b)

    def test_all_5_selection_policies_keep_original_feature_baseline(self):
        f=np.zeros(len(FEATURE_NAMES),np.float32)
        p=np.zeros(len(ADDITIONAL_NAMES),np.float32)
        n=np.zeros(WIDTH,np.float32)
        e=np.zeros(24,np.float32)
        result=matrices(f,p,n,e)
        self.assertEqual(set(PATTERNS),set(result))
        self.assertEqual(len(result["control41"]),len(FEATURE_NAMES))
        self.assertEqual(len(result["exclusive41"]),len(FEATURE_NAMES)+24)
        self.assertEqual(len(result["exclusive65"]),len(FEATURE_NAMES)+len(ADDITIONAL_NAMES)+24)
        self.assertEqual(len(result["exclusive_temporal65"]),len(FEATURE_NAMES)+len(ADDITIONAL_NAMES)+WIDTH+24)
        with self.assertRaises(ValueError):
            exclusive_harmonic_attack(self.make_pcm(),0,4,
                                      np.zeros((49,3),np.float32),np.zeros(49,bool))


if __name__=="__main__":
    unittest.main()
