import unittest
import numpy as np
import torch
from causal_note.s705_temporal_heads import TemporalHeads,TemporalHeadStream,predict_track
from causal_note.s711_context_heads import ContextTemporalHeads,transition_loss

class ContextTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(711);torch.set_num_threads(1)
        self.base=TemporalHeads(np.zeros(288),np.ones(288)).eval()
        self.model=ContextTemporalHeads(np.zeros(288),np.ones(288)).eval()
        self.model.warm_encoder(self.base.state_dict())

    def test_warm_start_preserves_frozen_model(self):
        x=torch.randn(1,43,49,288)
        with torch.no_grad():a,ak=self.base(x);b,bk=self.model(x)
        torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)
        torch.testing.assert_close(ak,bk,rtol=1e-5,atol=1e-6)

    def test_future_frames_cannot_change_prefix(self):
        with torch.no_grad():
            self.model.attention_residual.weight.copy_(torch.eye(48)*.1)
            self.model.pitch[0].weight[:,55:].normal_(0,.01)
        x=torch.randn(1,63,49,288);changed=x.clone();changed[:,38:]+=8
        with torch.no_grad():a,ak=self.model(x);b,bk=self.model(changed)
        torch.testing.assert_close(a[:,:38],b[:,:38],rtol=1e-5,atol=1e-6)
        torch.testing.assert_close(ak[:,:38],bk[:,:38],rtol=1e-5,atol=1e-6)

    def test_incremental_matches_chunked_attention(self):
        with torch.no_grad():self.model.attention_residual.weight.copy_(torch.eye(48)*.1)
        x=np.random.default_rng(711).normal(size=(39,49,288)).astype(np.float32)
        p,k=predict_track(self.model,x,chunk=13)
        stream=TemporalHeadStream(self.model)
        result=[stream.step(frame) for frame in x]
        np.testing.assert_allclose(np.stack([r[0] for r in result]),p,atol=2e-6,rtol=1e-5)
        np.testing.assert_allclose(np.stack([r[1] for r in result]),k,atol=2e-6,rtol=1e-5)

    def test_transition_objective_penalizes_missing_boundaries(self):
        active=torch.tensor([0.,1.,1.,0.])[None,:,None].expand(1,4,49)
        logits=torch.full((1,4,49,3),-12.)
        logits[:,:, :,1]=torch.where(active>0,12.,-12.)
        logits[:,1,:,0]=12.;logits[:,3,:,2]=12.
        good=transition_loss(logits,active)
        bad=logits.clone();bad[:,1,:,0]=-12.;bad[:,3,:,2]=-12.
        wrong=transition_loss(bad,active)
        self.assertLess(float(good.mean()),.001)
        self.assertGreater(float(wrong.mean()),1.)

if __name__=="__main__":unittest.main()
