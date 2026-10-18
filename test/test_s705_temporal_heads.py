import unittest
import numpy as np
import torch
from causal_note.s705_temporal_heads import TemporalHeads,TemporalHeadStream,predict_track,CONTEXT


class TemporalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(705)
        cls.model=TemporalHeads(np.zeros(192),np.ones(192),width=12).eval()
        cls.x=np.random.default_rng(705).normal(size=(79,49,192)).astype(np.float16)

    def test_future_cannot_modify_prefix(self):
        p,k=predict_track(self.model,self.x)
        changed=self.x.copy();changed[31:]*=-7
        pp,kk=predict_track(self.model,changed)
        np.testing.assert_allclose(p[:31],pp[:31],atol=1e-6)
        np.testing.assert_allclose(k[:31],kk[:31],atol=1e-6)

    def test_chunking_equals_incremental_bounded_stream(self):
        p,k=predict_track(self.model,self.x,chunk=17)
        full,fullk=predict_track(self.model,self.x,chunk=200)
        np.testing.assert_allclose(p,full,atol=1e-6)
        np.testing.assert_allclose(k,fullk,atol=1e-6)
        stream=TemporalHeadStream(self.model)
        for i in range(len(self.x)):
            pp,kk=stream.step(self.x[i])
            np.testing.assert_allclose(pp,p[i],atol=1e-6)
            np.testing.assert_allclose(kk,k[i],atol=1e-6)
        self.assertEqual(len(stream.history),CONTEXT+1)

    def test_masked_training_prefix_matches_true_start(self):
        with torch.inference_mode():
            x=torch.tensor(self.x[:10].astype(np.float32))[None]
            p,k=self.model(x)
            padded=torch.cat((torch.zeros((1,20,49,192)),x),1)
            mask=torch.cat((torch.zeros((1,20)),torch.ones((1,10))),1)
            pp,kk=self.model(padded,mask)
        np.testing.assert_allclose(p.numpy(),pp[:,20:].numpy(),atol=2e-6)
        np.testing.assert_allclose(k.numpy(),kk[:,20:].numpy(),atol=2e-6)


if __name__=='__main__':
    unittest.main()
