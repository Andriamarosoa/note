"""Contract tests with synthetic fixtures; NOT musical benchmarking."""
import unittest
import numpy as np
import torch
from scipy.signal import lfilter
from s872_hybrid_standard_ai256 import GranularDown256
from s876_neural_eq256 import (NeuralEQ256,BandCorrectionStream256,
                              BAND_FILTERS,HOP)
class NeuralEQContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(875)
    def test_exact_pcm256_no_queue(self):
        s=BandCorrectionStream256(NeuralEQ256().eval())
        x=np.zeros(256,np.float32)
        self.assertEqual(s.push(x,0).shape,(HOP,))
        for n in (128,1024):
            with self.assertRaises(ValueError):
                s.push(np.zeros(n,np.float32),HOP)
        with self.assertRaises(ValueError):s.push(x,2*HOP)
        self.assertEqual(s.push(x,HOP).shape,(HOP,))
    def test_no_future_block_dependency(self):
        r=np.random.default_rng(7)
        a=r.normal(0,.025,24*256).astype(np.float32)
        b=a.copy();b[11*256:]=r.normal(0,.04,13*256)
        net=NeuralEQ256().eval()
        def run(x):
            s=BandCorrectionStream256(net)
            return np.concatenate([s.push(x[i:i+256],i)
                                   for i in range(0,len(x),256)])
        np.testing.assert_array_equal(run(a)[:11*256],run(b)[:11*256])
    def test_silence_and_exact_standard_fallback(self):
        r=np.random.default_rng(8);x=r.normal(0,.04,23*256).astype(np.float32)
        net=NeuralEQ256().eval()
        s=BandCorrectionStream256(net,ai_enabled=False)
        y=np.concatenate([s.push(x[i:i+256],i)
                           for i in range(0,len(x),256)])
        ref=GranularDown256(8192)
        expected=np.concatenate([ref.push(x[i:i+256])
                                 for i in range(0,len(x),256)])
        np.testing.assert_array_equal(y,expected)
        z=BandCorrectionStream256(net)
        for i in range(14):
            self.assertEqual(float(np.max(abs(z.push(np.zeros(256,np.float32),i*256)))),0)
    def test_neutral_eq_perfect_reconstruction(self):
        r=np.random.default_rng(2);x=r.normal(0,.01,17*256).astype(np.float32)
        net=NeuralEQ256().eval()
        stream=BandCorrectionStream256(net)
        y=np.concatenate([stream.push(x[i:i+256],i)
                          for i in range(0,len(x),256)])
        std=GranularDown256(8192)
        baseline=np.concatenate([std.push(x[i:i+256]) for i in range(0,len(x),256)])
        np.testing.assert_allclose(y,baseline,rtol=1e-6,atol=1e-6)
    def test_streaming_and_full_sequence_are_identical(self):
        r=np.random.default_rng(6);x=r.normal(0,.04,20*256).astype(np.float32)
        net=NeuralEQ256().eval()
        std=GranularDown256(8192)
        q=np.concatenate([std.push(x[i:i+256]) for i in range(0,len(x),256)])
        l=[lfilter(f,[1],q).astype(np.float32) for f in BAND_FILTERS]
        bands=np.stack([l[0],l[1]-l[0],l[2]-l[1],q-l[2]])
        xb=torch.from_numpy(x.reshape(1,-1,256))
        qb=torch.from_numpy(q.reshape(1,-1,256))
        bb=torch.from_numpy(bands.reshape(4,-1,256).transpose(1,0,2)[None].copy())
        with torch.no_grad():bulk,*_=net(xb,qb,bb)
        s=BandCorrectionStream256(net)
        stream=np.concatenate([s.push(x[i:i+256],i)
                               for i in range(0,len(x),256)])
        np.testing.assert_array_equal(stream,bulk.reshape(-1).numpy())
if __name__=='__main__':
    unittest.main()
