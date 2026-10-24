"""S847 software invariants: artificial arrays are TEST FIXTURES only.

Never treat random numbers here as guitar audio or musical validation.
"""
import tempfile
from pathlib import Path
import unittest
import numpy as np
import torch
from onset256.real_recorded_transposer256 import (
    HOP,NeuralShiftSelector256,RecordedAudioNeuralTransposeStream256,
    make_recorded_bank,shift_truth_for_supervision_only,supervised_loss)
from scripts.s847_train_real_recorded_transposer256 import (
    best_cut_cal_only,binary_metrics_from_real_gt,train_only_weights)


class S847RecordedAudioNeuralShiftTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(847)

    def test_exact_one_hop_and_genuine_exemplar_bytes_not_generated(self):
        model=NeuralShiftSelector256(hidden=32)
        physical=np.linspace(-.3,.3,256*10,dtype=np.float32)
        bank=make_recorded_bank([physical],[52],["verified_track_A"])
        stream=RecordedAudioNeuralTransposeStream256(
            model,bank,shift=12,threshold=0.)
        first=stream.push(np.zeros(256,np.float32),start_sample=0)
        np.testing.assert_array_equal(first.physical_recorded_wave256,physical[:256])
        self.assertEqual(first.note,52)
        self.assertEqual(first.end_sample,256)
        second=stream.push(np.ones(256,np.float32),start_sample=256)
        np.testing.assert_array_equal(second.physical_recorded_wave256,physical[256:512])
        self.assertFalse(hasattr(first,"midi"))
        self.assertEqual(stream.added_hold_samples,0)
        self.assertEqual(stream.future_microphone_samples,0)

    def test_no_future_audio_changes_a_past_playback_or_note_score(self):
        physical=np.random.default_rng(847).normal(size=256*12).astype(np.float32)
        net=NeuralShiftSelector256(hidden=32).eval()
        a=RecordedAudioNeuralTransposeStream256(net,{52:physical},threshold=0.)
        b=RecordedAudioNeuralTransposeStream256(net,{52:physical},threshold=0.)
        v=a.push(np.zeros(256,np.float32),start_sample=0)
        w=b.push(np.zeros(256,np.float32),start_sample=0)
        np.testing.assert_array_equal(v.physical_recorded_wave256,
                                      w.physical_recorded_wave256)
        np.testing.assert_array_equal(v.learned_shift_probabilities,
                                      w.learned_shift_probabilities)
        a.push(np.ones(256,np.float32),start_sample=256)
        b.push(np.zeros(256,np.float32),start_sample=256)
        np.testing.assert_array_equal(v.learned_shift_probabilities,
                                      w.learned_shift_probabilities)

    def test_reject_wrong_frames_and_mismatched_timestamp(self):
        bank={52:np.ones(256*8,np.float32)}
        stream=RecordedAudioNeuralTransposeStream256(
            NeuralShiftSelector256(hidden=32),bank,threshold=0.)
        for n in (128,512,1024,8192):
            with self.assertRaises(ValueError):
                stream.push(np.zeros(n,np.float32),start_sample=0)
        with self.assertRaises(ValueError):
            stream.push(np.zeros(256,np.float64),start_sample=0)
        with self.assertRaises(ValueError):
            stream.push(np.zeros(256,np.float32),start_sample=256)
        self.assertEqual(stream.received,0)

    def test_true_octave_gt_only_in_training_and_neural_gradient(self):
        net=NeuralShiftSelector256(hidden=32)
        source=torch.zeros((3,4,49))
        source[:,:,0]=1.
        offs=torch.tensor([12,12,12],dtype=torch.int64)
        expected=shift_truth_for_supervision_only(source,offs)
        self.assertEqual(int(expected[:,:,12].sum()),12)
        self.assertEqual(int(expected[:,:,0].sum()),0)
        x=torch.randn((3,4,256))*.02
        weights=torch.ones(49);weights[0]=4.;weights[12]=4.
        loss,_=supervised_loss(net,x,source,offs,weights)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(float(net.shift_embedding.weight.grad.abs().sum()),0)
        self.assertGreater(float(net.transport[0].weight.grad.abs().sum()),0)

    def test_real_gt_split_calibration_does_not_select_outer(self):
        cal=dict(truth=np.zeros((2,2,49),np.float32))
        outer=dict(truth=np.zeros((2,2,49),np.float32))
        cal["truth"][0,:,0]=1.
        outer["truth"][1,:,0]=1.
        score_cal=np.zeros((2,2,49))
        score_outer=np.zeros((2,2,49))
        score_cal[0,:,12]=.8
        score_cal[1,:,12]=.2
        score_outer[1,:,12]=.7
        metric=binary_metrics_from_real_gt(cal,outer,12,score_cal,score_outer,52)
        self.assertEqual(metric["tp"],2)
        self.assertEqual(metric["fp"],0)
        self.assertEqual(metric["fn"],0)
        self.assertTrue(metric["cut_from_cal"]>.2)

    def test_a_real_clip_is_not_an_ai_generated_reconstruction(self):
        physical=np.random.default_rng(99).normal(0,.005,256*20).astype(np.float32)
        bank=make_recorded_bank([physical],[52],["real_capture"])
        np.testing.assert_array_equal(bank[52],physical)
        with self.assertRaises(ValueError):
            make_recorded_bank([np.zeros(200,np.float32)],[52],["not_valid"])
        with self.assertRaises(ValueError):
            make_recorded_bank([physical],[96],["not_valid"])


if __name__=="__main__":
    unittest.main()
