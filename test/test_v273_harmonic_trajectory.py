"""Synthetic controls for the harmonic trajectory *proxy*.

Known sinusoidal source timing here is used ONLY to test signal-measurement
behavior; no synthetic label truth enters the real GuitarSet extractor.
"""
from __future__ import annotations
import unittest
import numpy as np
from scripts.extract_v273_energy_flow import SAMPLE_RATE
from scripts.extract_v273_harmonic_trajectory import (
    trajectory_features, _frames, features_from_spectrogram,
    HARMONIC_BANK, DETUNED_BANK
)

def tone(f0,start_sec,stop_sec,amplitude=0.25,length=1.25):
    t=np.arange(int(round(length*SAMPLE_RATE)),dtype=float)/SAMPLE_RATE
    env=np.minimum(1.0,np.maximum(0.0,(t-start_sec)/0.007))
    env*=np.minimum(1.0,np.maximum(0.0,(stop_sec-t)/0.009))
    signal=np.zeros_like(t)
    for h in range(1,6):
        signal+=np.sin(2*np.pi*f0*h*t+0.18*h)/(h**1.25)
    return signal*env*amplitude

class HarmonicTrajectoryTests(unittest.TestCase):
    def test_normalized_amplitude_and_silence(self):
        x=tone(220.,.1,1.1)
        a=trajectory_features(x,int(.62*SAMPLE_RATE))
        b=trajectory_features(.5*x,int(.62*SAMPLE_RATE))
        self.assertEqual(set(a),set(b))
        self.assertTrue(np.isfinite(list(a.values())).all())
        self.assertAlmostEqual(a["source__novelty_l1"],b["source__novelty_l1"],places=5)
        z=trajectory_features(np.zeros_like(x),int(.62*SAMPLE_RATE))
        self.assertTrue(np.isfinite(list(z.values())).all())

    def test_new_harmonic_source_onset(self):
        center=int(.62*SAMPLE_RATE)
        existing=tone(220.,.10,1.1)
        added=tone(440.,.62,1.1,amplitude=.5)
        base=trajectory_features(existing,center)
        mixed=trajectory_features(existing+added,center)
        self.assertGreater(mixed["source__novelty_l1"],base["source__novelty_l1"]+.04)
        self.assertGreater(mixed["birth__novel_mass"],base["birth__novel_mass"]+.02)

    def test_muted_source_has_temporal_death(self):
        center=int(.62*SAMPLE_RATE)
        constant=tone(220.,.10,1.1)+tone(440.,.10,1.1)
        muted=tone(220.,.10,1.1)+tone(440.,.10,.62)
        steady=trajectory_features(constant,center)
        cutoff=trajectory_features(muted,center)
        self.assertGreater(cutoff["damping__negative_flow"],steady["damping__negative_flow"]+.01)

    def test_wrong_harmonic_bank_is_distinct(self):
        center=int(.62*SAMPLE_RATE)
        x=tone(220.,.10,1.1)+tone(440.,.62,1.1,amplitude=.5)
        spec,t=_frames(x,center)
        good=features_from_spectrogram(spec,t,HARMONIC_BANK)
        wrong=features_from_spectrogram(spec,t,DETUNED_BANK)
        self.assertNotAlmostEqual(good["source__novelty_l1"],wrong["source__novelty_l1"],places=5)

if __name__=="__main__":unittest.main()
