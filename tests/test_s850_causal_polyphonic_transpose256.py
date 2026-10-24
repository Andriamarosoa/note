"""S850 invariants. Arrays are PROGRAMMING FIXTURES, not music or scores."""
from __future__ import annotations
import unittest
import numpy as np
import torch
from onset256.s850_causal_polyphonic_transposer256 import (
    NeuralPolyphonicTranspose256,CausalPolyphonicTransposeStream256,
    shifted_truth_only_offline,real_note_loss,
    train_only_weights,hard_negative_rank,HOP)
from scripts.s850_prepare_real_mic_longcontext256 import (
    target_mask,sample_sequences,split_of,PLAYERS)
from scripts.s850_train_real_causal_poly_note256 import metrics,select_threshold

class LongCausalActualGuitarModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(850)

    def model(self):
        return NeuralPolyphonicTranspose256(hidden=32)

    def test_one_new_256_samples_per_decision_no_future_hold_or_midi(self):
        stream=CausalPolyphonicTransposeStream256(self.model(),shift=12)
        result=stream.push(np.zeros(256,np.float32),start_sample=0)
        self.assertEqual(result.sample_end,256)
        self.assertEqual(result.midi_probabilities.shape,(49,))
        self.assertEqual(result.source_probabilities.shape,(49,))
        self.assertEqual(result.hold_samples,0)
        self.assertEqual(result.future_mic_samples,0)
        self.assertFalse(hasattr(result,'midi'))
        self.assertEqual(stream.push(np.ones(256,np.float32),start_sample=256).sample_end,512)
        with self.assertRaises(ValueError):
            stream.push(np.zeros(1024,np.float32),start_sample=512)

    def test_unrolled_only_past_equal_streaming(self):
        net=self.model().eval()
        inputs=torch.randn((1,12,256))*.02
        with torch.inference_mode():
            full,_=net(inputs,torch.tensor([12]))
            state=None
            outputs=[]
            for i in range(12):
                row,state=net(inputs[:,i:i+1,:],torch.tensor([12]),state)
                outputs.append(row['target_logits'])
        torch.testing.assert_close(full['target_logits'],torch.cat(outputs,dim=1),
                                    atol=1e-6,rtol=1e-6)

    def test_real_polyphonic_targets_and_shift_leave_out_of_range(self):
        gt=torch.zeros((2,3,49))
        gt[:,:,0]=1.
        gt[:,:,12]=1.
        pos=shifted_truth_only_offline(gt,torch.tensor([12,-12]))
        self.assertEqual(int(pos[0,:,12].sum()),3)
        self.assertEqual(int(pos[0,:,24].sum()),3)
        self.assertEqual(int(pos[1,:,0].sum()),3)
        self.assertEqual(int(pos[1].sum()),3)
        self.assertFalse(bool(pos[1,:,48].any()))

    def test_rank_loss_updates_weight_on_genuine_gt_cases(self):
        net=self.model()
        x=torch.randn((3,5,256))*.02
        gt=torch.zeros((3,5,49))
        gt[:,:,0]=1
        gt[0,:,12]=1 # genuine two-active-note supervision fixture
        weights,_=train_only_weights(gt)
        loss,parts=real_note_loss(net,x,gt,torch.tensor([12,0,-12]),
                                  weights,ranking=.75)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(parts['ranking']),0)
        loss.backward()
        self.assertGreater(float(net.shift_transport[0].weight.grad.abs().sum()),0)
        self.assertGreater(float(net.original_notes.weight.grad.abs().sum()),0)

    def test_real_note_overlaps_framed_only_now(self):
        starts=np.arange(8)*256
        notes=[(.003,.035,40,5),(.006,.021,52,4)]
        got=target_mask(notes,starts)
        self.assertEqual(got.shape,(8,49))
        self.assertEqual(got[1,0],1)
        self.assertEqual(got[1,12],1)
        self.assertEqual(got[-1,0],0)

    def test_long_32_hop_real_array_geometry_no_live_lookahead(self):
        raw=np.zeros(256*120,dtype=np.int16)
        notes=[(.04,.35,40,5),(.08,.29,52,4)]
        pcm,gt=sample_sequences(raw,notes,seed=850,count=7,hops=32)
        self.assertEqual(pcm.shape,(7,32,256))
        self.assertEqual(gt.shape,(7,32,49))
        self.assertGreater(int(gt[:,:,0].sum()),0)
        self.assertGreater(int(gt[:,:,12].sum()),0)

    def test_cal_only_threshold_not_outer_relabeling(self):
        scores=np.array([.05,.08,.61,.95])
        target=np.array([0,0,1,1],bool)
        threshold=select_threshold(scores,target)
        m=metrics(target,scores,threshold)
        self.assertEqual((m['tp'],m['fp'],m['fn']),(2,0,0))
        self.assertTrue(0<threshold<1)

    def test_player_splits_are_disjoint(self):
        self.assertEqual(split_of('00_BN1-test_comp'),'train')
        self.assertEqual(split_of('04_BN1-test_comp'),'cal')
        self.assertEqual(split_of('05_BN1-test_comp'),'outer')
        a=[x for values in PLAYERS.values() for x in values]
        self.assertEqual(len(a),len(set(a)))

if __name__=="__main__":
    unittest.main()
