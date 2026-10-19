"""S741 real-input K evidence contracts, no oracle or future frame."""
import unittest
import numpy as np
from causal_note.s741_exactk_sources import (
    continuous_count_distribution, frozen_acoustic_exactk_heads)
from causal_note.s740_multitarget_dynamic import apply_domain


class ExactKSourceTests(unittest.TestCase):
    def setUp(self):
        self.t=5
        self.k=np.tile(np.array([.1,.6,.3,0,0,0,0],np.float32),(self.t,1))
        self.native=np.tile(np.array([.2,.3,.5,0,0,0,0],np.float32),(self.t,1))
        self.post=np.full((self.t,49,3),.10,np.float32)
        self.ends=np.arange(1,6)*1024
        self.events=[
            dict(type="note_on",pitch=40,available_sample=1024),
            dict(type="note_off",pitch=40,available_sample=3072)
        ]

    def test_real_five_heads_and_32_k_variants(self):
        base,heads=frozen_acoustic_exactk_heads(
            self.k,self.native,self.post,self.events,self.ends)
        self.assertEqual(heads.shape,(self.t,7,6))
        np.testing.assert_array_equal(base,self.k)
        np.testing.assert_array_equal(heads[:,:,0],self.k)
        np.testing.assert_array_equal(heads[:,:,1],self.native)
        np.testing.assert_array_equal(heads[1,:,2],self.k[0])
        selection=apply_domain(
            "exact_k",base,heads,np.ones_like(base,bool),
            np.array([1,3,5,7,63],np.uint8))
        np.testing.assert_array_equal(selection[0],base[0])
        np.testing.assert_allclose(selection.sum(-1),1,atol=1e-6)

    def test_unseen_future_cannot_change_prior_k_evidence(self):
        a=frozen_acoustic_exactk_heads(
            self.k,self.native,self.post,self.events,self.ends)[1]
        future=self.post.copy()
        future[-1,:,:]=1.
        b=frozen_acoustic_exactk_heads(
            self.k,self.native,future,self.events,self.ends)[1]
        np.testing.assert_array_equal(a[:-1],b[:-1])
        # No annotation of the true K enters any head.
        self.assertEqual(continuous_count_distribution(
            np.array([0.,6.],np.float32)).shape,(2,7))

    def test_reject_unormalized_native(self):
        invalid=self.native.copy()
        invalid[0,:]=0
        with self.assertRaises(ValueError):
            frozen_acoustic_exactk_heads(
                self.k,invalid,self.post,self.events,self.ends)


if __name__=="__main__":
    unittest.main()
