"""Structural tests: synthetic test arrays only, NOT musical performance scores."""
import unittest
import numpy as np
import torch
from s863_live_shadow256 import NeuralOctave256,NeuralPCM256Stream,HOP,TAPS

class CausalPCM256(unittest.TestCase):
    def test_exact_256_contiguous_no_future(self):
        torch.manual_seed(863)
        m=NeuralOctave256().eval()
        rng=np.random.default_rng(863)
        a=rng.normal(0,.05,HOP).astype(np.float32)
        b=rng.normal(0,.05,HOP).astype(np.float32)
        s=NeuralPCM256Stream(m)
        first=s.push(a,0)
        self.assertEqual(first.shape,(256,))
        with self.assertRaises(ValueError):s.push(np.zeros(1024,np.float32),256)
        with self.assertRaises(ValueError):s.push(np.zeros(128,np.float32),256)
        with self.assertRaises(ValueError):s.push(np.zeros(256,np.float32),512)
        first_again=NeuralPCM256Stream(m).push(a,0)
        np.testing.assert_array_equal(first,first_again)
        y=s.push(b,256)
        self.assertTrue(np.isfinite(y).all())

    def test_current_pcm_cannot_depend_on_next_block(self):
        torch.manual_seed(863)
        m=NeuralOctave256().eval()
        x=torch.randn(1,512+512)*.02
        y=x.clone()
        y[:,-256:]=torch.randn(1,256)
        with torch.inference_mode():
            a=m.compute_last_256(x[:,0:768])
            b=m.compute_last_256(y[:,0:768])
        torch.testing.assert_close(a,b,atol=0,rtol=0)

if __name__=="__main__":
    unittest.main()
