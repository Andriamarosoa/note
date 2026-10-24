"""S844 regression audit: all fixtures are NON-MUSICAL test signals only."""
import unittest
import numpy as np
import torch

from onset256.real_audio_transpose_head256 import (
    HOP,NeuralAcousticTransposeHead256,reconstruction_loss)
from scripts.s843_train_real_hex_transpose256 import real_blocks
from scripts.s843_evaluate_real_hex_reconstruction256 import (
    best_cut_cal_only,measure_one,f1_counts)


class S844RareGuitarNotes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(844)

    def test_train_only_low_threshold_resolution_below_original_005(self):
        probs=np.array([.01,.012,.023,.027,.035,.045])
        actual=np.array([0,0,0,0,1,1],bool)
        cut=best_cut_cal_only(probs,actual)
        self.assertLess(cut,.05)
        self.assertGreater(cut,.027)
        self.assertEqual(f1_counts(probs>=cut,actual)["tp"],2)
        self.assertEqual(f1_counts(probs>=cut,actual)["fp"],0)

    def test_no_positive_held_out_is_not_false_music_f1_zero(self):
        result=measure_one("E2_E3",np.array([.1,.2,.8]),
                           np.array([.4,.5]),np.array([0,1,1],bool),
                           np.array([0,0],bool))
        self.assertEqual(result["status"],"UNASSESSABLE_NO_POSITIVE_OUTER")
        self.assertIsNone(result["f1"])
        self.assertEqual(result["outer_positive"],0)

    def test_more_rare_onset_context_sampled_than_small_random_prototype(self):
        # Audio is all-zero TEST FIXTURE, NEVER actual training music.
        length=int(.9*44100)
        signal=np.zeros(length,np.float32)
        six=np.zeros((length,6),np.float32)
        notes=[(.10,.45,40,5),(.15,.40,52,4)]
        pcm,hex_audio,gt,st=real_blocks(signal,six,notes,seed=844,
                                          sequences=16,seq_hops=8)
        self.assertEqual(pcm.shape,(16,8,256))
        self.assertEqual(hex_audio.shape,(16,8,6,256))
        self.assertTrue((gt[...,0]>.5).any())
        self.assertTrue((gt[...,12]>.5).any())
        self.assertGreater(int(((gt[...,0]>.5)&(gt[...,12]>.5)).sum()),0)

    def test_train_only_weighted_recon_produces_stronger_true_E2_gradient(self):
        net=NeuralAcousticTransposeHead256(hidden=32,shift_dim=8)
        x=torch.zeros((2,2,256))
        six=torch.zeros((2,2,6,256))
        notes=torch.zeros((2,2,49))
        notes[:,:,0]=1
        activity=torch.zeros((2,2,6));activity[:,:,5]=1
        plain,_,_=reconstruction_loss(net,x,six,notes,activity)
        weights=torch.ones(49);weights[0]=20
        rare,_,_=reconstruction_loss(net,x,six,notes,activity,
                                     note_positive_weight=weights)
        self.assertTrue(torch.isfinite(rare))
        self.assertGreater(float(rare),float(plain))
        rare.backward()
        self.assertGreater(float(net.source_note.weight.grad.abs().sum()),0)
        bad=torch.ones(49)*31
        with self.assertRaises(ValueError):
            reconstruction_loss(net,x,six,notes,activity,
                                note_positive_weight=bad)

    def test_no_ghost_overlap_in_near_simultaneous_nonoverlapping_notes(self):
        signal=np.zeros(int(.65*44100),np.float32)
        six=np.zeros((len(signal),6),np.float32)
        notes=[(.10,.15,40,5),(.17,.29,52,4)]
        pcm,hex_audio,gt,string=real_blocks(signal,six,notes,seed=844,
                                             sequences=12,seq_hops=8)
        self.assertFalse(bool(((gt[...,0]>.5)&(gt[...,12]>.5)).any()))


if __name__=="__main__":
    unittest.main()
