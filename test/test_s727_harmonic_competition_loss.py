"""S727 hard harmonic competition loss does not erase real E2->E3 starts."""
import unittest
import torch
from causal_note.s727_harmonic_competition_loss import (
    harmonic_parent_mask,onset_competition_weights,competition_loss)
from scripts.s720_poly_onset_training import training_loss


class S727HarmonicCompetitionTests(unittest.TestCase):
    def setUp(self):
        self.y=torch.zeros((1,5,49,3))
        self.y[0,:,0,1]=1  # E2 sustains all five decision frames.
        self.y[0,3,12,0]=1 # at frame 3 REAL E3 pinched while E2 ringing
        self.y[0,3:,12,1]=1

    def test_true_E3_new_onset_not_treated_as_hard_negative(self):
        m=harmonic_parent_mask(self.y)
        self.assertEqual(float(m[0,3,12]),1.)
        f=onset_competition_weights(self.y,2.)
        self.assertEqual(float(f[0,2,12,0]),2.)  # E2 2nd harmonic, fake E3
        self.assertAlmostEqual(float(f[0,3,12,0]),1.1,places=5)
        self.assertEqual(float(f[0,3,12,1]),1.) # no offset/sustain distortion

    def test_unrelated_pitches_never_reweighted(self):
        w=onset_competition_weights(self.y,3.)
        self.assertEqual(float(w[0,1,13,0]),1.)
        self.assertEqual(float(w[0,1,1,0]),1.)
        self.assertEqual(float(w[0,1,31,0]),1.)
        self.assertEqual(float(w[0,1,19,0]),3.)

    def test_unweighted_matches_original_S706_exactly(self):
        torch.manual_seed(727)
        logits=torch.randn((1,5,49,3))
        kk=torch.randn((1,5,7))
        count=torch.randint(0,4,(1,5))
        mask=torch.ones((1,5))
        a=competition_loss(logits,kk,self.y,count,mask,strength=1.)
        b=training_loss(logits,kk,self.y,count,mask)
        torch.testing.assert_close(a,b,atol=1e-7,rtol=0.)

    def test_gradient_changes_for_fake_harmonic_but_not_ground_truth_true(self):
        logits=torch.zeros((1,5,49,3),requires_grad=True)
        k_logits=torch.zeros((1,5,7),requires_grad=True)
        k=torch.zeros((1,5),dtype=torch.long)
        mask=torch.ones((1,5))
        l=competition_loss(logits,k_logits,self.y,k,mask,strength=2.)
        l.backward()
        # An independently plucked E3 has negative dL/dlogit; the fake
        # E3 while E2 sustains should have positive derivative.
        self.assertLess(float(logits.grad[0,3,12,0]),0.)
        self.assertGreater(float(logits.grad[0,2,12,0]),0.)

    def test_invalid_harmonic_input_shape_or_strength(self):
        with self.assertRaises(ValueError):
            harmonic_parent_mask(torch.zeros((1,2,49,2)))
        with self.assertRaises(ValueError):
            onset_competition_weights(self.y,0.9)


if __name__=="__main__":
    unittest.main()
