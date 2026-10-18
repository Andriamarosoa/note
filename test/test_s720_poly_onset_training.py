"""S720 independent tests: FIT-only polyphonic Onset loss and six-metric guard."""
import unittest
import torch
from scripts.s704_joint_metrics import SCORES
from scripts.s720_poly_onset_training import onset_weights,training_loss,guards


class S720TrainingTests(unittest.TestCase):
    def setUp(self):
        self.y=torch.zeros((1,3,49,3),dtype=torch.uint8)
        self.y[0,1,0,0]=1
        self.y[0,1,35,0]=1
        self.y[0,2,0,0]=1
        self.k=torch.tensor([[1,3,1]],dtype=torch.int32)

    def test_polyphonic_only_positives_reweighted_and_background_untouched(self):
        w=onset_weights(self.y,self.k,poly_factor=3.,bass_factor=1.5)
        self.assertEqual(w.shape,self.y.shape)
        self.assertAlmostEqual(float(w[0,1,0,0]),4.)
        self.assertAlmostEqual(float(w[0,1,35,0]),3.)
        self.assertEqual(float(w[0,1,2,0]),1.)
        self.assertEqual(float(w[0,2,0,0]),1.)
        self.assertEqual(float(w[0,1,0,2]),1.)
        self.assertEqual(float(w[0,1,0,1]),1.6)
        self.assertTrue(torch.all(w>=1))

    def test_control_matches_original_s706_bce_objective(self):
        import torch.nn.functional as F
        logits=torch.randn(2,4,49,3,requires_grad=True)
        y=(torch.rand(2,4,49,3)>.8).float()
        k=torch.randint(0,7,(2,4))
        count=torch.randn(2,4,7,requires_grad=True)
        mask=torch.ones((2,4))
        got=training_loss(logits,count,y,k,mask,poly_factor=1.,bass_factor=1.)
        pitch=F.binary_cross_entropy_with_logits(logits,y,
                     pos_weight=torch.tensor((10.,2.,8.)),reduction='none').mean((-1,-2))
        ce=F.cross_entropy(count.flatten(0,1),k.flatten(),reduction='none').reshape_as(k)
        consistency=(logits[:,:,:,1].sigmoid().sum(-1)-k.float()).square()
        expected=((pitch+.25*ce+.008*consistency)*mask).sum()/mask.sum()
        torch.testing.assert_close(got,expected,rtol=0,atol=1e-6)
        got.backward()
        self.assertIsNotNone(logits.grad)
        self.assertIsNotNone(count.grad)

    def test_weighted_loss_changes_only_fit_positive_gradient(self):
        x=torch.full((1,3,49,3),-.5,requires_grad=True)
        klogits=torch.zeros((1,3,7),requires_grad=True)
        mask=torch.ones((1,3))
        control=training_loss(x,klogits,self.y,self.k,mask)
        weighted=training_loss(x,klogits,self.y,self.k,mask,
                                poly_factor=3.,bass_factor=1.5)
        self.assertGreater(float(weighted),float(control))
        self.assertTrue(torch.isfinite(weighted))

    def test_six_scores_or_correct_offset_loss_blocks_candidate(self):
        b={k:.5 for k in SCORES}
        t={k:.6 for k in SCORES}
        t['offset_accuracy']=.49
        b['tp']=100;t['tp']=120
        self.assertFalse(guards(b,t,dict(lost_long=0,lost_correct_offsets=0)))
        t['offset_accuracy']=.6
        self.assertFalse(guards(b,t,dict(lost_long=0,lost_correct_offsets=1)))
        self.assertTrue(guards(b,t,dict(lost_long=0,lost_correct_offsets=0)))


if __name__=="__main__":
    unittest.main()
