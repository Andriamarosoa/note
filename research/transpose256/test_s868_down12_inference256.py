"""PCM256 contract tests; synthetic fixtures are NOT musical validation."""
import unittest
import numpy as np
import torch
from s868_down12_inference256 import (
    NeuralDownOctave256,DownOctaveStream256,
    RecurrentDownOctave256,RecurrentDownStream
)

class DownOctaveShadowContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_tcn_exact_256_and_continuity(self):
        torch.manual_seed(868)
        s=DownOctaveStream256(NeuralDownOctave256())
        x=np.zeros(256,np.float32)
        self.assertEqual(s.push(x,0).shape,(256,))
        for length in (128,1024):
            with self.assertRaises(ValueError):
                s.push(np.zeros(length,np.float32),256)
        with self.assertRaises(ValueError):
            s.push(x,512)
        self.assertEqual(s.push(x,256).shape,(256,))

    def test_tcn_future_cannot_change_past_output(self):
        torch.manual_seed(868)
        m=NeuralDownOctave256("tcn_gate").eval()
        a=torch.randn(1,2048+512)*.02
        b=a.clone()
        b[:,-256:]=torch.randn(1,256)
        with torch.no_grad():
            x=m(a)[:,:-256]
            y=m(b)[:,:-256]
        torch.testing.assert_close(x,y,rtol=0,atol=0)

    def test_recurrent_block_and_state(self):
        torch.manual_seed(870)
        m=RecurrentDownOctave256().eval()
        stream=RecurrentDownStream(m)
        x=np.zeros(256,np.float32)
        self.assertEqual(stream.push(x,0).shape,(256,))
        with self.assertRaises(ValueError):
            stream.push(np.zeros(1024,np.float32),256)
        with self.assertRaises(ValueError):
            stream.push(x,512)
        with torch.no_grad():
            audio=torch.randn(1,5,256)
            whole,_=m(audio)
            state=None;pieces=[]
            for j in range(5):
                output,state=m(audio[:,j:j+1],state)
                pieces.append(output)
        torch.testing.assert_close(whole,torch.cat(pieces,1),atol=1e-6,rtol=1e-6)

if __name__=="__main__":
    unittest.main()
