"""Unit contract only; these tests DO NOT validate GuitarSet performance."""
import unittest
import torch
from causal_note.s733_one_target_selector import (
    OneTargetClassConditionalSelector,SingleTargetContract,
    target_loss,propose_single_channel)

class SpecialistTests(unittest.TestCase):
    def input(self):
        b,h=5,4
        scores=torch.rand(b,h)
        audit=torch.rand(b,h,21)
        types=torch.zeros(b,h,4);types[:,:,1]=1.
        mask=torch.ones(b,h,dtype=torch.bool);mask[0,3]=False
        k_mask=torch.ones(b,h,7,dtype=torch.bool)
        k_mask[:,2,5:]=False;k_mask[0,3,:]=False
        context=torch.rand(b,16)
        k_probs=torch.softmax(torch.randn(b,7),dim=-1)
        baseline=torch.rand(b)
        return scores,audit,types,mask,k_mask,context,k_probs,baseline

    def test_class_conditional_weights_and_gradients(self):
        rows=self.input()
        model=OneTargetClassConditionalSelector()
        out=model(*rows)
        self.assertEqual(tuple(out["class_selection_weights"].shape),(5,5,7))
        self.assertTrue(torch.all(out["class_selection_weights"][0,3]==0))
        self.assertTrue(torch.all(out["class_selection_weights"][:,2,5:]==0))
        self.assertTrue(torch.allclose(out["class_selection_weights"].sum(1),
                                       torch.ones(5,7),atol=1e-6))
        self.assertTrue(torch.allclose(out["selection_weights"].sum(1),
                                       torch.ones(5),atol=1e-6))
        loss=target_loss(out,torch.tensor([0,1,2,1,0]),
                         torch.rand(5,4,2),rows[3])
        loss.backward()
        self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in model.parameters()))

    def test_sixth_head_reads_soft_frozen_exact_k_individually(self):
        rows=list(self.input())
        model=OneTargetClassConditionalSelector()
        model.eval()
        with torch.no_grad():
            first=model(*rows)
            other=[t.clone() for t in rows]
            other[6][:]=torch.tensor([.01,.04,.08,.09,.75,.02,.01])
            changed=model(*other)
        self.assertEqual(tuple(first["k_linked_onset"].shape),(5,))
        self.assertTrue(((first["k_linked_onset"]>=0)&
                         (first["k_linked_onset"]<=1)).all())
        self.assertEqual(tuple(first["class_selection_weights"].shape),(5,5,7))
        self.assertGreater(float(torch.abs(
            first["k_linked_onset"]-changed["k_linked_onset"]).sum()),1e-9)
        # H5 is not forbidden at K0 by a mistaken global-K prediction.
        self.assertTrue(torch.all(
            first["class_selection_weights"][:,-1,0]>0))

    def test_K_linked_onset_has_trainable_gradient(self):
        rows=self.input()
        model=OneTargetClassConditionalSelector()
        output=model(*rows)
        truth=torch.tensor([1.,0.,1.,0.,0.])
        loss=torch.nn.functional.binary_cross_entropy(
            output["k_linked_onset"],truth)
        loss.backward()
        self.assertTrue(any(p.grad is not None and
                         float(p.grad.abs().sum())>0
                         for p in model.k_linked_onset.parameters()))

    def test_edits_only_selected_pitch_and_target(self):
        out=OneTargetClassConditionalSelector()(*self.input())
        post=torch.rand(5,49,3)
        pitches=torch.tensor([40,52,64,76,88])
        out["action_logits"]=torch.tensor([
            [0.,4.,0.],[0.,0.,4.],[4.,0.,0.],
            [0.,5.,0.],[0.,0.,5.]])
        changed=propose_single_channel(post,pitches,out,"onset")
        self.assertTrue(torch.equal(post[:,:,1:],changed[:,:,1:]))
        for i in range(5):
            not_selected=torch.ones(49,dtype=torch.bool)
            not_selected[pitches[i]-40]=False
            self.assertTrue(torch.equal(post[i,not_selected,0],
                                         changed[i,not_selected,0]))

    def test_require_frozen_baseline_and_real_oof(self):
        rows=list(self.input());rows[3][0,0]=False
        with self.assertRaises(ValueError):
            OneTargetClassConditionalSelector()(*rows)
        self.assertEqual(
            SingleTargetContract("onset","a"*64,True).validate().target,
            "onset")
        with self.assertRaises(ValueError):
            SingleTargetContract("all","a"*64,True).validate()
        with self.assertRaises(ValueError):
            SingleTargetContract("onset","a"*64,False).validate()

if __name__=="__main__":unittest.main()
