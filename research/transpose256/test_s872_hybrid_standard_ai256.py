"""Structural unit tests only. Fake arrays are NOT musical benchmark data."""
import unittest
import numpy as np
import torch
from s872_hybrid_standard_ai256 import (
    GranularDown256,NeuralGate256,StandardPlusAI256,HOP)

class TestIndependentHybridStandardAI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(872)
    def test_exactly_256_and_no_hold(self):
        player=StandardPlusAI256(NeuralGate256().eval())
        pcm=np.zeros(HOP,np.float32)
        self.assertEqual(len(player.push(pcm,0)),HOP)
        for length in (128,1024):
            with self.assertRaises(ValueError):
                player.push(np.zeros(length,np.float32),HOP)
        with self.assertRaises(ValueError):
            player.push(pcm,HOP*2)
        self.assertEqual(len(player.push(pcm,HOP)),HOP)

    def test_true_past_only_with_and_without_ai(self):
        rng=np.random.default_rng(872)
        for size0,size1 in ((2048,4096),(4096,8192)):
            for enabled in (True,False):
                x=rng.normal(0,.02,14*HOP).astype(np.float32)
                y=x.copy()
                y[7*HOP:]=rng.normal(0,.03,7*HOP)
                def process(data):
                    s=StandardPlusAI256(NeuralGate256().eval(),
                      ai_enabled=enabled,grain_fast=size0,grain_quality=size1)
                    return np.concatenate([s.push(data[i:i+HOP],i)
                                          for i in range(0,len(data),HOP)])
                np.testing.assert_array_equal(process(x)[:7*HOP],
                                             process(y)[:7*HOP])

    def test_exact_fallback_is_reference_standard(self):
        rng=np.random.default_rng(3)
        x=rng.normal(0,.01,16*HOP).astype(np.float32)
        for grain in (4096,8192):
            s=StandardPlusAI256(NeuralGate256().eval(),
                 ai_enabled=False,grain_fast=2048,grain_quality=grain)
            out=np.concatenate([s.push(x[i:i+HOP],i) for i in range(0,len(x),HOP)])
            ref=GranularDown256(grain)
            truth=np.concatenate([ref.push(x[i:i+HOP]) for i in range(0,len(x),HOP)])
            np.testing.assert_array_equal(out,truth)

    def test_rnn_256_stream_equals_unrolled(self):
        m=NeuralGate256().eval()
        x=torch.randn(1,9,HOP)*.02
        a=torch.randn_like(x)*.02;b=torch.randn_like(x)*.02
        with torch.no_grad():
            full,*_=m(x,a,b)
            state=blend=None;parts=[]
            for t in range(9):
                z,state,blend,_,_=m(x[:,t:t+1],a[:,t:t+1],
                                   b[:,t:t+1],state,blend)
                parts.append(z)
        torch.testing.assert_close(full,torch.cat(parts,1),atol=1e-6,rtol=1e-6)

    def test_silence_stays_silent(self):
        stream=StandardPlusAI256(NeuralGate256().eval())
        for i in range(12):
            self.assertEqual(np.max(abs(stream.push(np.zeros(HOP,np.float32),i*HOP))),0)

if __name__=="__main__":
    unittest.main()
