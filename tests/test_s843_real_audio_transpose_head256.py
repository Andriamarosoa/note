"""S843 neural genuine-audio reconstruction checks (software-only fixtures).

Synthetic random/zero tensors here are strictly shape/gradient TEST FIXTURES,
NEVER 'real guitar' training data, music scores or validation evidence.
"""
from __future__ import annotations
import io
import json
import tempfile
import unittest
import zipfile

import numpy as np
import torch
from scipy.io import wavfile

from onset256.real_audio_transpose_head256 import (
    HOP,STRINGS,PITCHES,ABI,NeuralAcousticTransposeHead256,
    CausalRealAudioTransposeStream256,reconstruction_loss,real_shift_pair_loss)
from scripts.s843_train_real_hex_transpose256 import (
    OFFICIAL,TRAIN_PLAYERS,CAL_PLAYERS,OUTER_PLAYERS,
    read_jams_note_midi,wav_in_zip,real_blocks,target_masks,
    pair_pool,player_split,authenticate)


class S843ReconstructionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(843)

    def model(self):
        return NeuralAcousticTransposeHead256(hidden=32,shift_dim=8)

    def test_pcm256_contract_and_reconstructed_real_acoustic_stream_shapes(self):
        stream=CausalRealAudioTransposeStream256(self.model())
        a=stream.push(np.zeros(HOP,np.float32),start_sample=0)
        self.assertEqual(a.sample,256)
        self.assertEqual(a.strings.shape,(6,256))
        self.assertEqual(a.mic_reconstruction.shape,(256,))
        self.assertEqual(a.note_probabilities.shape,(49,))
        self.assertEqual(a.plain_probabilities.shape,(49,))
        self.assertEqual(a.view_weights.shape,(3,))
        self.assertEqual(len(a.transposed_acoustic_estimates),2)
        self.assertEqual(a.transposed_acoustic_estimates[0].shape,(6,256))
        self.assertAlmostEqual(float(a.view_weights.sum()),1.,places=5)
        self.assertFalse(hasattr(a,"intents"))
        self.assertFalse(hasattr(a,"midi"))
        self.assertEqual(stream.push(np.ones(HOP,np.float32),start_sample=256).sample,512)

    def test_reject_invalid_block_gaps_or_nonfloat32(self):
        stream=CausalRealAudioTransposeStream256(self.model())
        for n in (128,255,512,1024,8192):
            with self.subTest(n=n),self.assertRaises(ValueError):
                stream.push(np.zeros(n,np.float32),start_sample=0)
        with self.assertRaises(ValueError):
            stream.push(np.zeros(256,np.float32),start_sample=256)
        with self.assertRaises(ValueError):
            stream.push(np.zeros(256,np.float64),start_sample=0)
        with self.assertRaises(ValueError):
            stream.push(np.full(256,np.nan,np.float32),start_sample=0)
        self.assertEqual(stream.received,0)

    def test_future_pcm_cannot_change_any_past_prediction_or_audio(self):
        model=self.model()
        stream_a=CausalRealAudioTransposeStream256(model)
        stream_b=CausalRealAudioTransposeStream256(model)
        x=np.random.default_rng(42).normal(size=256).astype(np.float32)*.01
        ax=stream_a.push(x,start_sample=0)
        bx=stream_b.push(x,start_sample=0)
        np.testing.assert_array_equal(ax.strings,bx.strings)
        np.testing.assert_array_equal(ax.note_probabilities,bx.note_probabilities)
        stream_a.push(np.zeros(HOP,np.float32),start_sample=HOP)
        stream_b.push(np.ones(HOP,np.float32),start_sample=HOP)
        np.testing.assert_array_equal(ax.strings,bx.strings)

    def test_chunkwise_equal_two_step_unroll_with_past_recurrent_state(self):
        model=self.model().eval()
        data=torch.randn((1,3,256))*.01
        with torch.inference_mode():
            whole,_=model(data)
            state=None
            each=[]
            for j in range(3):
                o,state=model(data[:,j:j+1,:],state)
                each.append(o["note_logits"])
        torch.testing.assert_close(whole["note_logits"],torch.cat(each,dim=1),
                                   atol=1e-6,rtol=1e-6)

    def test_shift_embeddings_learn_not_formulaic_frequency_bins(self):
        net=self.model()
        z=torch.randn((2,32))
        identity=net.transposed_latent(z,torch.zeros(2,dtype=torch.int64))
        raised=net.transposed_latent(z,torch.full((2,),12,dtype=torch.long))
        lower=net.transposed_latent(z,torch.full((2,),-12,dtype=torch.long))
        torch.testing.assert_close(identity,z)
        self.assertFalse(torch.equal(raised,lower))
        self.assertGreater(sum(p.numel() for p in net.transport.parameters()),0)
        with self.assertRaises(ValueError):
            net.transposed_latent(z,torch.full((2,),13,dtype=torch.long))

    def test_true_stem_supervision_losses_and_gradients(self):
        model=self.model()
        x=torch.randn((2,3,256))*0.01
        target=torch.randn((2,3,6,256))*0.01
        notes=torch.zeros((2,3,49));notes[:,:,0]=1
        strings=torch.zeros((2,3,6));strings[:,:,5]=1
        loss,part,_=reconstruction_loss(model,x,target,notes,strings)
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("waveform",part)
        loss.backward()
        self.assertGreater(float(model.stem_waveform[0].weight.grad.abs().sum()),0)
        self.assertGreater(float(model.view_reliability[0].weight.grad.abs().sum()),0)

    def test_real_recorded_shift_pair_model_has_trainable_gradient(self):
        model=self.model()
        src=torch.randn((4,1,256))*0.02
        tgt=torch.randn((4,1,256))*0.02
        truth=torch.zeros((4,49));truth[:,0]=1
        shifted=torch.zeros_like(truth);shifted[:,12]=1
        shift=torch.full((4,),12,dtype=torch.long)
        loss=real_shift_pair_loss(model,src,tgt,truth,shifted,shift)
        loss.backward()
        self.assertGreater(float(model.transport[0].weight.grad.abs().sum()),0)
        self.assertGreater(float(model.shifted_note.weight.grad.abs().sum()),0)

    def test_official_dataset_checksums_fail_closed_on_fixture(self):
        with tempfile.TemporaryDirectory() as td:
            from pathlib import Path
            a,b,c=(Path(td)/x for x in ("mic.zip","hex.zip","jams.zip"))
            for item in (a,b,c):item.write_bytes(b"not a real dataset")
            with self.assertRaises(ValueError):
                authenticate(a,b,c)
        self.assertEqual(len(OFFICIAL["mic_sha256"]),64)
        self.assertEqual(len(OFFICIAL["annotation_sha256"]),64)
        self.assertEqual(len(OFFICIAL["hex_clean_md5"]),32)

    def test_original_jams_gt_per_string_and_aligned_pcm_chunks(self):
        with tempfile.TemporaryDirectory() as td:
            from pathlib import Path
            path=Path(td)/"dummy_not_real.zip"
            example={"annotations":[
                dict(namespace="note_midi",data=[
                    dict(time=.01,duration=.2,value=40+i)])
                for i in range(6)]}
            with zipfile.ZipFile(path,"w") as zipf:
                zipf.writestr("00_test.jams",json.dumps(example))
                wav=io.BytesIO()
                wavfile.write(wav,44100,np.zeros(4096,np.int16))
                zipf.writestr("00_test_mic.wav",wav.getvalue())
            with zipfile.ZipFile(path) as zipf:
                notes=read_jams_note_midi(zipf,"00_test")
                audio=wav_in_zip(zipf,"00_test_mic.wav",1)
            self.assertEqual(len(notes),6)
            self.assertEqual(notes[0][2:],(40,0))
            target=np.zeros((len(audio),6),np.float32)
            mic,stems,gt,strings=real_blocks(audio,target,notes,
                sequences=2,seq_hops=4)
            self.assertEqual(mic.shape,(2,4,256))
            self.assertEqual(stems.shape,(2,4,6,256))
            self.assertEqual(gt.shape,(2,4,49))
            self.assertEqual(strings.shape,(2,4,6))

    def test_pair_pool_requires_separate_real_tracks(self):
        x=np.zeros((2,2,256),np.float32)
        gt=np.zeros((2,2,49),np.float32)
        gt[0,:,0]=1;gt[1,:,12]=1
        pool=pair_pool(dict(mic=x,note_gt=gt,track_id=np.array([0,0])))
        self.assertEqual(pool,[])
        pool=pair_pool(dict(mic=x.copy(),note_gt=gt,track_id=np.array([0,1])))
        self.assertEqual(pool,[]) # identical fake zeros never accepted as real pair
        with self.assertRaises(ValueError):
            pair_pool(dict(mic=x,note_gt=gt))

    def test_split_by_guitarist_and_no_official_midi(self):
        self.assertEqual(player_split("04_BN1-12-X_comp"),"train")
        self.assertEqual(player_split("05_BN1-12-X_comp"),"cal")
        self.assertEqual(player_split("00_BN1-12-X_comp"),"outer")
        self.assertEqual(len(set(TRAIN_PLAYERS+CAL_PLAYERS+OUTER_PLAYERS)),6)
        self.assertEqual(HOP,256)
        self.assertEqual(ABI,"NOTE_S843_CAUSAL_REAL_AUDIO_RECONSTRUCTION_TRANSPOSE_PCM256_V1")


if __name__=="__main__":
    unittest.main()
