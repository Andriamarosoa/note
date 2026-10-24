"""Onset neural transposer checks; dummy waveforms are NEVER scored as music."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from onset256.learned_latent_transposer256 import (
    HOP,MODEL_ABI,NeuralLatentTransposer256,CausalNeuralTransposerStream256,
    recorded_pair_training_loss,verify_recorded_pairs)
from scripts.train_onset_neural_transpose256 import (
    validate_partition,load_verified_pairs,train)


class NeuralTransposerPCM256Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(256)
    def model(self):
        return NeuralLatentTransposer256(latent_dim=32,shift_dim=8)

    def test_current_block_one_hypothesis_with_no_future_or_midi(self):
        stream=CausalNeuralTransposerStream256(self.model())
        out=stream.push(np.zeros(256,np.float32),start_sample=0)
        self.assertEqual(out.sample,256)
        self.assertEqual(out.note_probabilities.shape,(49,))
        self.assertEqual(out.plain_probabilities.shape,(49,))
        self.assertEqual(out.view_weights.shape,(5,))
        self.assertAlmostEqual(float(out.view_weights.sum()),1.,places=5)
        self.assertFalse(hasattr(out,"intents"))
        self.assertFalse(hasattr(out,"midi"))
        self.assertEqual(stream.push(np.ones(256,np.float32),start_sample=256).sample,512)

    def test_future_mutation_cannot_change_past(self):
        model=self.model()
        a=CausalNeuralTransposerStream256(model)
        b=CausalNeuralTransposerStream256(model)
        previous=np.random.default_rng(1).normal(size=256).astype(np.float32)
        x=a.push(previous,start_sample=0)
        y=b.push(previous,start_sample=0)
        a.push(np.ones(256,np.float32),start_sample=256)
        b.push(np.zeros(256,np.float32),start_sample=256)
        np.testing.assert_array_equal(x.note_probabilities,y.note_probabilities)

    def test_reject_128_512_1024_or_broken_pcm_clock(self):
        stream=CausalNeuralTransposerStream256(self.model())
        for size in (128,512,1024,255):
            with self.subTest(size=size),self.assertRaises(ValueError):
                stream.push(np.zeros(size,np.float32),start_sample=0)
        with self.assertRaises(ValueError):
            stream.push(np.zeros(256,np.float32),start_sample=256)
        with self.assertRaises(ValueError):
            stream.push(np.zeros(256,np.float64),start_sample=0)
        self.assertEqual(stream.received,0)

    def test_neural_transposed_embeddings_differ_with_shift(self):
        model=self.model()
        z=torch.randn(3,32)
        shift2=model.transport_latent(z,torch.full((3,),2))
        shift7=model.transport_latent(z,torch.full((3,),7))
        self.assertFalse(torch.equal(shift2,shift7))
        torch.testing.assert_close(model.transport_latent(z,torch.zeros(3,dtype=torch.long)),z)
        with self.assertRaises(ValueError):
            model.transport_latent(z,torch.full((3,),13))
        self.assertTrue(any(k.startswith("transport.") for k in dict(model.named_parameters())))

    def test_real_pair_truth_validation_and_learned_gradient(self):
        model=self.model()
        src=torch.randn(4,256)*.1
        dst=torch.randn(4,256)*.1
        truth=torch.zeros(4,49);target=torch.zeros(4,49)
        truth[:,10]=1.;target[:,17]=1.
        offsets=torch.full((4,),7,dtype=torch.long)
        verify_recorded_pairs(src,dst,truth,target,offsets)
        with self.assertRaises(ValueError):
            verify_recorded_pairs(src,dst,truth,truth,offsets)
        original=model.transport[0].weight.detach().clone()
        loss,detail=recorded_pair_training_loss(model,src,dst,truth,target,offsets)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(v) for v in detail.values()))
        opt=torch.optim.AdamW(model.parameters(),lr=.002)
        opt.zero_grad();loss.backward();opt.step()
        self.assertGreater(float((original-model.transport[0].weight).abs().sum()),0.)

    def test_original_pitch_neural_decoder_and_plain_ablation(self):
        model=self.model().eval()
        with torch.inference_mode():
            views,plain,weights=model(torch.zeros((2,256)),shifts=(2,-2))
        self.assertEqual(views.shape,(2,49))
        self.assertEqual(plain.shape,(2,49))
        self.assertEqual(weights.shape,(2,3))
        torch.testing.assert_close(weights.sum(dim=1),torch.ones(2))

    def fake_pairs_for_SMOKE_ONLY(self,folder):
        # Generated noise is not a guitar recording and never a real F1 score.
        n=9;rng=np.random.default_rng(26)
        wave=rng.normal(0,.1,(n,256)).astype(np.float32)
        target=rng.normal(0,.1,(n,256)).astype(np.float32)
        labels=np.zeros((n,49),np.float32);labels[:,10]=1.
        labels_t=np.zeros_like(labels);labels_t[:,17]=1.
        data=dict(source_pcm=wave,target_pcm=target,
                  source_note_mask=labels,target_note_mask=labels_t,
                  semitones=np.full(n,7,np.int64),
                  split=np.array(["train"]*3+["cal"]*3+["outer"]*3),
                  source_track=np.array(["S"+str(i) for i in range(n)]),
                  target_track=np.array(["T"+str(i) for i in range(n)]))
        path=folder/"pairs.npz";np.savez(path,**data)
        metadata=dict(schema="ONSET_REAL_RECORDED_NOTE_TRANSPOSE_PAIRS_256_V1",
            sample_rate_hz=44100,pcm_block_samples=256,
            audio_pair_source="independently_recorded_real_audio",
            dsp_pitch_shift_used=False,lookahead_samples=0,
            source_s708_teacher_used=False,
            paired_npz_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        manifest=folder/"manifest.json";manifest.write_text(json.dumps(metadata))
        return data,path,manifest

    def test_no_train_cal_outer_recording_identity_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            d,_,_=self.fake_pairs_for_SMOKE_ONLY(Path(tmp))
            d["target_track"][8]=d["source_track"][0]
            with self.assertRaises(ValueError):validate_partition(d)

    def test_data_checksum_and_no_dsp_declaration_are_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            _,file,manifest=self.fake_pairs_for_SMOKE_ONLY(Path(tmp))
            d,_=load_verified_pairs(file,manifest)
            self.assertEqual(d["source_pcm"].shape,(9,256))
            meta=json.loads(manifest.read_text());meta["dsp_pitch_shift_used"]=True
            manifest.write_text(json.dumps(meta))
            with self.assertRaises(ValueError):load_verified_pairs(file,manifest)
            meta["dsp_pitch_shift_used"]=False;meta["paired_npz_sha256"]="0"*64
            manifest.write_text(json.dumps(meta))
            with self.assertRaises(ValueError):load_verified_pairs(file,manifest)

    def test_training_smoke_does_not_claim_real_event_F1(self):
        with tempfile.TemporaryDirectory() as tmp:
            _,file,manifest=self.fake_pairs_for_SMOKE_ONLY(Path(tmp))
            out=Path(tmp)/"experiment"
            r=train(file,manifest,out,epochs=1,batch_size=3)
            self.assertEqual(r["train_input_samples"] if "train_input_samples" in r else r["training_pcm_samples"],256)
            self.assertTrue(r["onset_event_f1_80_wav_NOT_MEASURED"])
            self.assertFalse(r["promoted"])
            self.assertFalse(r["public_midi_changed"])
            self.assertTrue((out/"neural_latent_transposer256.pt").exists())
            with self.assertRaises(FileExistsError):
                train(file,manifest,out,epochs=1,batch_size=3)


if __name__=="__main__":
    unittest.main()
