"""Smoke tests for same-checkpoint neural head and combination ablations."""
import unittest
import numpy as np
import tensorflow as tf

from scripts.audit_v273_neural_head_ablations import (
    masking, mask_drop, observe, summarize, INTERVENTIONS, ATTACKS
)
from scripts.evaluate_v273_neural_history_mix import (
    build_candidate_logits, inputs, HEADS, H
)
from scripts.learn_v273_neural_head_selector import ClassConditionalAuditSelector


class RealNeuralAblationTests(unittest.TestCase):
    def test_head_fallback_must_survive(self):
        b=np.array([2,2,3,4])
        P=np.full((4,6,5),.2,np.float32)
        logits,active,kind,support=build_candidate_logits(P,b)
        x=inputs(logits,active,kind,support,np.zeros((4,H,21),np.float32),
                 np.zeros((4,3),np.float32),b)
        with self.assertRaises(AssertionError):
            masking(x,(1,2,3))
        solo=masking(x,(0,))
        self.assertTrue(np.all(solo["head_mask"][:,0]))
        self.assertFalse(np.any(solo["head_mask"][:,1:]))
        self.assertTrue(np.all(solo["head_k_mask"][:,0,:]))
        self.assertTrue(np.all(~solo["head_k_mask"][:,1:,:]))

    def test_actual_network_forward_pass_is_not_a_direct_vote(self):
        tf.keras.utils.set_random_seed(27402)
        b=np.array([2,2,3,3,4,4],int)
        rng=np.random.default_rng(174)
        P=rng.uniform(.05,1.,size=(6,6,5)).astype("float32")
        P/=P.sum(axis=2,keepdims=True)
        logits,active,kind,support=build_candidate_logits(P,b)
        val=inputs(logits,active,kind,support,
                   np.zeros((6,H,21),np.float32),
                   rng.normal(size=(6,5)).astype("float32"),b)
        net=ClassConditionalAuditSelector(hidden=24)
        reference=observe(net,val,b)
        self.assertEqual(reference.shape,(6,))
        self.assertTrue(np.isin(reference,range(7)).all())
        # A head unavailable for this event may not influence its output.
        drop_C23=observe(net,mask_drop(val,(6,)),b)
        self.assertTrue(np.array_equal(reference[b!=2],drop_C23[b!=2]))
        # Every faithful K2->K3 subset includes H0.
        for free in range(64):
            keep=(0,)+tuple(i for j,i in enumerate((1,2,3,4,5,6))
                             if free & (1<<j))
            p=observe(net,masking(val,keep),b)
            self.assertEqual(len(p),len(b))
        self.assertEqual(len(ATTACKS),4)
        self.assertIn("drop_H2_lifecycle",INTERVENTIONS)

    def test_rescued_K2_to_K1_measures_correct_outcome(self):
        y=np.array([2,2,2,2,3,4,1],int)
        base=np.array([2,2,2,2,2,3,2],int)
        original=np.array([1,1,2,3,3,3,1],int)
        new=np.array([2,1,1,2,2,4,2],int)
        s=summarize(y,base,original,new)
        self.assertEqual(s["regressions_K2_to_K1_original"],2)
        self.assertEqual(s["regressions_K2_to_K1_after_ablation"],2)
        self.assertEqual(s["K2_to_K1_regressions_rescued_to_correct_K2"],1)
        self.assertEqual(s["K2_to_K1_regressions_newly_created"],1)
        self.assertEqual(s["net_vs_unablated_neural"],
                         int(np.sum(new==y)-np.sum(original==y)))
        self.assertEqual(s["by_true_K"]["2"]["n"],4)


if __name__=="__main__":
    unittest.main()
