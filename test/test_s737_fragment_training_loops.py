"""Tests for real FIT-only S737 learned 32-combination training contracts."""
import unittest
import numpy as np
import torch

from causal_note.s736_fragment_mask import (
    MASKS, all_mask_offers, apply_fragment_masks)
from scripts.s737_train_fragment_onset_loops import (
    supervision, training_audit, audit_all_fit, structural_mask,
    model_data, ONSET_THRESHOLD, REFERENCE_NAME)
from causal_note.strict_live import SR

class TrainingAudit(unittest.TestCase):
    def example(self, name, outcomes):
        n=len(outcomes)
        return dict(
            source=name,features=np.zeros((n,37),np.float32),
            descriptors=np.zeros((n,32,12),np.float32),
            k=np.tile(np.array([0,1,0,0,0,0,0],np.float32),(n,1)),
            target=np.zeros(n,np.int64),
            audit_true_k=np.ones(n,np.int32),
            risk=np.tile(np.asarray(outcomes,dtype=np.float32)
                         .reshape(n,1,2),(1,32,1)),
            coverage=np.ones((n,32),np.float32),
        )

    def test_32_mask_and_h0_fallback(self):
        self.assertEqual(len(MASKS),32)
        self.assertTrue(all(m & 1 for m in MASKS))
        self.assertEqual(MASKS[-1],63)
        mask=structural_mask(4)
        self.assertEqual(tuple(mask.shape),(4,32,7))
        self.assertTrue(bool(mask[:,0,:].all()))
        self.assertFalse(bool(mask[:,1:,0].any()))
        self.assertTrue(bool(mask[:,:,1:].all()))

    def test_train_risk_is_other_composition_only(self):
        left=self.example("left",[[1,0],[1,0],[1,0]])
        right=self.example("right",[[0,1],[0,1],[0,1]])
        result=training_audit([left,right],0)
        self.assertLess(float(result[0,1,0]),1e-9)
        self.assertGreater(float(result[0,1,1]),0.)
        left["risk"][:]=0
        unchanged=training_audit([left,right],0)
        self.assertTrue(np.array_equal(result,unchanged))
        self.assertTrue(np.isfinite(audit_all_fit([left,right])).all())

    def test_train_model_abi(self):
        a=self.example("train_a",[[0,1],[1,0]])
        b=self.example("train_b",[[0,1],[0,0]])
        x,d,profile,k,mask,label,risk=model_data(
            a,training_audit([a,b],0),
            np.zeros(37,np.float32),np.ones(37,np.float32))
        self.assertEqual(tuple(d.shape),(2,32,12))
        self.assertEqual(tuple(profile.shape),(2,32,7,3))
        self.assertEqual(tuple(mask.shape),(2,32,7))
        self.assertEqual(tuple(risk.shape),(2,32,2))

    def test_candidate_supervision_onset_only(self):
        t=4
        post=np.zeros((t,49,3),np.float32)
        post[:,:,0]=.1
        post[:,:,1]=.7
        post[:,:,2]=.3
        heads=np.zeros((t,49,6),np.float32)
        heads[:,:,0]=.1
        heads[0,0,1]=.9
        eligible=np.zeros((t,49),bool)
        eligible[0,0]=True
        feature=np.zeros((t,37),np.float32)
        context=np.zeros((t,32,12),np.float32)
        k=np.tile(np.array([0,1,0,0,0,0,0],np.float32),(t,1))
        end=np.arange(1,t+1,dtype=np.int64)*1024
        track=dict(item=dict(decision_ends=end,spec=dict(composition="a")),
                   post=post,head=heads,eligible=eligible,features=feature,
                   descriptors=context,k=k)
        true=[dict(pitch=40,time=float(end[0]/SR),duration=.045)]
        sample=supervision(track,true,737)
        self.assertGreater(sample["rescuable"],0)
        self.assertEqual(sample["risk"].shape[1:],(32,2))
        # No extra permission to alter Sustain/Release.
        masks=np.ones(t,np.uint8)
        masks[0]=3
        altered=apply_fragment_masks(post,heads,eligible,masks)
        self.assertTrue(np.array_equal(post[:,:,1:],altered[:,:,1:]))
        self.assertGreater(float(altered[0,0,0]),float(post[0,0,0]))
        self.assertEqual(REFERENCE_NAME,"S708_frozen")

if __name__=="__main__":
    unittest.main()
