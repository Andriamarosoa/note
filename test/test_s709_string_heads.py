import unittest
import numpy as np
import torch
from causal_note.s709_string_heads import StringTemporalHeads
from causal_note.s705_temporal_heads import predict_track,TemporalHeadStream


class StringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2);torch.manual_seed(709)
        cls.model=StringTemporalHeads(np.zeros(288),np.ones(288),width=12).eval()
        cls.x=np.random.default_rng(709).normal(size=(45,49,288)).astype(np.float16)

    def test_string_normalization_and_count_mean(self):
        with torch.inference_mode():
            p,k,s=self.model(torch.tensor(self.x[:9].astype(np.float32))[None],return_strings=True)
        sp=s.softmax(-1);kp=k.softmax(-1)
        np.testing.assert_allclose(sp.sum(-1).numpy(),1,atol=1e-6)
        np.testing.assert_allclose(kp.sum(-1).numpy(),1,atol=1e-6)
        expected=(kp*torch.arange(7)).sum(-1)
        np.testing.assert_allclose(expected.numpy(),(1-sp[:,:,:,0]).sum(-1).numpy(),atol=1e-6)
        self.assertEqual(tuple(p.shape),(1,9,49,3))

    def test_causal_prefix_and_incremental_parity(self):
        p,k=predict_track(self.model,self.x,chunk=16)
        pp,kk=predict_track(self.model,self.x[:21])
        np.testing.assert_allclose(p[:21],pp,atol=1e-6)
        np.testing.assert_allclose(k[:21],kk,atol=1e-6)
        stream=TemporalHeadStream(self.model)
        for i,x in enumerate(self.x):
            sp,sk=stream.step(x)
            np.testing.assert_allclose(p[i],sp,atol=1e-6)
            np.testing.assert_allclose(k[i],sk,atol=1e-6)


if __name__=='__main__':unittest.main()
