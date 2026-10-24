"""S851 A/B engineering tests. Fake arrays are not musical performance."""
import unittest
import numpy as np
import torch
from onset256.s850_causal_polyphonic_transposer256 import NeuralPolyphonicTranspose256
from scripts.s851_compare_neural_rank_vs_standard256 import play_shadow,result_for

class S851ActualInputContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);torch.manual_seed(851)

    def test_real_note_bank_playback_256_boundaries_no_synth(self):
        model=NeuralPolyphonicTranspose256(hidden=32).eval()
        audio=np.zeros(8*256,np.float32)
        clip=np.linspace(-.02,.02,256*16,dtype=np.float32)
        physical={52:clip}
        for reset in (0,4):
            output,notes,gate=play_shadow(model,audio,physical,gate_threshold=0,
                                          reset_hops=reset)
            self.assertEqual(len(output),len(audio))
            np.testing.assert_array_equal(output[:256],clip[:256])
            self.assertTrue(all(note==52 for note in notes))
            self.assertEqual(len(gate),8)

    def test_gate_can_reject_false_notes_without_future_audio(self):
        net=NeuralPolyphonicTranspose256(hidden=32).eval()
        audio=np.zeros(4*256,np.float32)
        physical={52:np.ones(256*8,np.float32)*.01}
        output,notes,gates=play_shadow(net,audio,physical,gate_threshold=1.0)
        self.assertTrue(np.all(output==0))
        self.assertTrue(all(v is None for v in notes))

    def test_real_comparison_fails_one_note_stuck_and_no_music_quality(self):
        net=NeuralPolyphonicTranspose256(hidden=32).eval()
        source=np.ones(256*50,np.float32)*.01
        standard=np.random.default_rng(1).normal(0,.01,256*50).astype(np.float32)
        bank={52:np.linspace(-.05,.05,256*16,dtype=np.float32)}
        predicted,report=result_for(net,source,standard,bank,0)
        self.assertEqual(len(predicted),len(source))
        self.assertFalse(report['musical_promotion_passed'])
        self.assertTrue(any('one pitch' in reason for reason in report['musical_rejection_reasons']))

if __name__=="__main__":
    unittest.main()
