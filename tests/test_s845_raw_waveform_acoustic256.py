"""S845 raw-waveform preservation guards. Synthetic arrays are unit test
fixtures only, NOT examples of real guitar performance, sound or F1 results.
"""
import unittest
import numpy as np
import torch
from onset256.real_audio_transpose_head256 import (
    HOP,NeuralAcousticTransposeHead256,
    CausalRealAudioTransposeStream256,reconstruction_loss)


class S845PhysicalWaveformGuards(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(845)
        torch.set_num_threads(1)

    def test_real_raw_256_samples_can_influence_6_stems_directly(self):
        model=NeuralAcousticTransposeHead256(hidden=32,shift_dim=8)
        latent=torch.randn(2,32)
        x=torch.zeros(2,256)
        y=torch.ones(2,256)*.08
        first,_,_=model.reconstruct(latent,x)
        second,_,_=model.reconstruct(latent,y)
        self.assertEqual(first.shape,(2,6,256))
        self.assertGreater(float((first-second).abs().sum()),0)

    def test_no_silence_collapse_when_string_confidence_low(self):
        model=NeuralAcousticTransposeHead256(hidden=32,shift_dim=8)
        with torch.no_grad():
            model.stem_presence.weight.zero_()
            model.stem_presence.bias.fill_(-25)
        h=torch.randn(1,32)
        genuine_lookalike=torch.randn(1,256)*.01
        out,presence,_=model.reconstruct(h,genuine_lookalike)
        self.assertTrue(bool((presence < -20).all()))
        self.assertGreater(float(out.abs().mean()),1e-7)

    def test_real_hex_supervised_reconstruction_updates_local_filters(self):
        model=NeuralAcousticTransposeHead256(hidden=32,shift_dim=8)
        mic=torch.randn((2,3,256))*.02
        target=torch.randn((2,3,6,256))*.02
        labels=torch.zeros((2,3,49));labels[:,:,0]=1
        strings=torch.zeros((2,3,6));strings[:,:,5]=1
        weights=torch.ones(49);weights[0]=3.
        loss,info,_=reconstruction_loss(model,mic,target,labels,strings,
                                       note_positive_weight=weights)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(float(model.local_string_filters[0].weight.grad.abs().sum()),0)
        self.assertGreater(float(model.string_condition.weight.grad.abs().sum()),0)

    def test_strict_256_no_future_midi_and_causal_recurrence(self):
        model=NeuralAcousticTransposeHead256(hidden=32,shift_dim=8)
        stream=CausalRealAudioTransposeStream256(model)
        a=stream.push(np.ones(256,np.float32)*.01,start_sample=0)
        self.assertEqual(a.sample,256)
        self.assertEqual(a.strings.shape,(6,256))
        self.assertEqual(a.plain_probabilities.shape,(49,))
        self.assertEqual(a.note_probabilities.shape,(49,))
        self.assertFalse(hasattr(a,'midi'))
        with self.assertRaises(ValueError):
            stream.push(np.ones(1024,np.float32),start_sample=256)
        with self.assertRaises(ValueError):
            stream.push(np.ones(256,np.float32),start_sample=512)
        b=stream.push(np.zeros(256,np.float32),start_sample=256)
        self.assertEqual(b.sample,512)

    def test_transposed_branch_does_not_claim_true_recorded_future_waveform(self):
        model=NeuralAcousticTransposeHead256(hidden=32,shift_dim=8)
        x=torch.randn(1,256)*.001
        h=model.encode_hop(x,model.initial_state(1))
        decoded=model.from_latent(h,views=(-12,12),original_pcm=x)
        self.assertEqual(decoded["stems"].shape,(1,6,256))
        self.assertEqual(len(decoded["transposed_acoustic_estimates"]),2)
        self.assertEqual(decoded["transposed_acoustic_estimates"][0].shape,(1,6,256))
        with self.assertRaises(ValueError):
            model.reconstruct(h,torch.randn(1,1024))


if __name__=="__main__":
    unittest.main()
