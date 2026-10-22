"""Structural, causal and per-axis PCM1024 tests independent of FIT80 scores."""
import unittest
from unittest.mock import patch
import numpy as np
from onset1024 import runtime
from onset1024.acoustics import HOP,SR
class FakeNative:
    def predict(self,feat):
        return np.zeros((49,3),np.float32),np.ones(7,np.float32)/7
class FakeTemporal:
    def __init__(self,*a):self.calls=0
    def step(self,x):
        assert x.shape==(49,288)
        self.calls+=1
        return np.zeros((49,3),np.float32),np.ones(7,np.float32)/7
class Native1024(unittest.TestCase):
    def test_true_inference_on_every_1024_samples(self):
        with patch.object(runtime,"IncrementalS708TemporalHead",FakeTemporal):
            s=runtime.Stream(FakeNative(),object(),1.0)
            decisions=[]
            for i in range(4):
                decisions.extend(s.push(np.zeros(1024,np.float32),start_sample=i*1024))
            self.assertEqual([d.sample for d in decisions],[1024,512,768,1024])
            self.assertEqual(s.calls,4)
            self.assertEqual(s.temporal.calls,4)
            with self.assertRaises(ValueError):
                s.push(np.zeros(1024,np.float32),start_sample=0)
    def test_independent_decoder_emits_correct_axis(self):
        d=runtime.OnsetDecoder()
        attack=np.zeros((49,3),np.float32)
        attack[0]=[.95,.9,.01]
        a=d.step(attack,1024)
        release=np.zeros((49,3),np.float32)
        release[0]=[.05,.1,.95]
        b=d.step(release,512)
        self.assertEqual([(e.kind,e.pitch,e.sample) for e in a.intents],[("note_on",40,1024)])
        self.assertEqual(b.intents,())

    def test_cadence(self):
        self.assertEqual(HOP,1024)
        self.assertAlmostEqual(HOP/SR*1000,23.2199546485,places=7)
if __name__=="__main__":unittest.main()
