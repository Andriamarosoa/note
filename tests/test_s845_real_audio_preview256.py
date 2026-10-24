"""S845 WAV preview software tests.

NO music is synthesized or used for training. Artificial zeros and white
noise are software TEST FIXTURES only, not evidence of a real sound.
"""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from scipy.io import wavfile

from onset256.real_audio_transpose_head256 import (
    ABI,NeuralAcousticTransposeHead256)
from scripts.s845_export_real_audio_estimates256 import run


class S845RealAudioPreviewSoftware(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(845)

    def test_raw_pcm256_streaming_and_eof_partial_without_generated_midi(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td)
            ckpt=base/"check.pt"
            net=NeuralAcousticTransposeHead256()
            # This checkpoint contains RANDOM software-only testing weights.
            torch.save(dict(abi=ABI,state_dict=net.state_dict(),
                            trained_hop_samples=256,promoted=False),ckpt)
            raw=base/"software_fixture.wav"
            source=np.random.default_rng(845).integers(
                -200,200,size=270,dtype=np.int16)
            wavfile.write(raw,44100,source)
            dest=base/"preview"
            report=run(raw,ckpt,dest)
            self.assertEqual(report["source_sample_count"],270)
            self.assertEqual(len(report["note_probabilities_per_block"]),2)
            self.assertEqual(report["note_probabilities_per_block"][-1]["sample_end"],270)
            self.assertTrue(report["estimated_six_strings_are_not_ground_truth"])
            names=list(dest.glob("*.wav"))
            self.assertEqual(len(names),19) # 6 orig, 6 minus12, 6 plus12, 1 mic
            for file in names:
                rate,payload=wavfile.read(file)
                self.assertEqual(rate,44100)
                self.assertEqual(payload.shape,(270,))
                self.assertEqual(payload.dtype,np.int16)
            self.assertFalse(list(dest.glob("*.mid")))
            with self.assertRaises(FileExistsError):
                run(raw,ckpt,dest)

    def test_reject_non_real_unsupported_mic_format(self):
        with tempfile.TemporaryDirectory() as td:
            td=Path(td)
            ckpt=td/"check.pt"
            net=NeuralAcousticTransposeHead256()
            torch.save(dict(abi=ABI,state_dict=net.state_dict(),
                            trained_hop_samples=256,promoted=False),ckpt)
            bad=td/"wrong_rate.wav"
            wavfile.write(bad,48000,np.zeros(256,np.int16))
            with self.assertRaises(ValueError):
                run(bad,ckpt,td/"bad_output")
            self.assertFalse((td/"bad_output").exists())

if __name__=="__main__":
    unittest.main()
