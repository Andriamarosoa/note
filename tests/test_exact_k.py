"""Only Exact-K is runnable in this isolated branch."""
import unittest
import numpy as np
from pathlib import Path
from exact_k.acoustics import CausalPitchFeatures,HOP,SR,load_native_heads
from exact_k.temporal import TemporalHeads
from exact_k.runtime import ExactKOnly256


class ExactKOnlyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)
        cls.model=TemporalHeads(np.zeros(288,np.float32),np.ones(288,np.float32),width=8)
        cls.model.eval()
        cls.native=load_native_heads(
            Path(__file__).resolve().parents[1]/
            "model/live/strict_note/NOTE_strict_live.npz")

    def test_only_exposes_exact_k(self):
        from exact_k import runtime
        self.assertFalse(any(hasattr(runtime,a) for a in
            ("OnsetDecoder","OffsetDecoder","ActiveKDecoder",
             "FramePitchDecoder","FullNoteDecoder","SixIndependentAxes")))
        self.assertEqual(runtime.SCHEMA,"EXACT_K_ONLY_S834_TRUE_PCM256_V1")

    def test_native256_from_pcm_and_no_hold(self):
        stream=ExactKOnly256(self.native,self.model)
        rng=np.random.default_rng(123)
        samples=rng.normal(0,.1,5*HOP+17).astype(np.float32)
        output=stream.push(samples[:4*HOP],start_sample=0)
        self.assertEqual(len(output),4)
        self.assertEqual([r.sample for r in output],[HOP,2*HOP,3*HOP,4*HOP])
        self.assertTrue(all(0<=d.count<=6 for d in output))
        self.assertEqual(stream.diagnostics()["added_future_samples"],0)
        self.assertEqual(stream.diagnostics()["added_hold_samples"],0)
        output=stream.push(samples[4*HOP:],start_sample=4*HOP)
        self.assertEqual(len(output),1)
        self.assertEqual(output[0].sample,5*HOP)

    def test_audio_discontinuity_is_rejected(self):
        stream=ExactKOnly256(self.native,self.model)
        with self.assertRaises(ValueError):
            stream.push(np.zeros(HOP,np.float32),start_sample=1)
        stream.push(np.zeros(HOP,np.float32),start_sample=0)
        with self.assertRaises(ValueError):
            stream.push(np.zeros(HOP,np.float32),start_sample=0)

    def test_segmented_audio_identical_counts(self):
        rng=np.random.default_rng(234)
        x=rng.normal(0,.07,8*HOP).astype(np.float32)
        first=ExactKOnly256(self.native,self.model)
        second=ExactKOnly256(self.native,self.model)
        a=sum((list(first.push(x[j:j+HOP],start_sample=j))
            for j in range(0,len(x),HOP)),[])
        b=sum((list(second.push(x[j:j+2*HOP],start_sample=j))
            for j in range(0,len(x),2*HOP)),[])
        self.assertEqual([(v.sample,v.count) for v in a],
                         [(v.sample,v.count) for v in b])
        np.testing.assert_allclose(
            [v.confidence for v in a],[v.confidence for v in b],
            rtol=0,atol=1e-7)

    def test_frontend_causality_after_fft_history(self):
        rng=np.random.default_rng(99)
        x=rng.normal(0,.04,75*HOP).astype(np.float32)
        z=x.copy();z[69*HOP:]*=-1
        def compute(v):
            frontend=CausalPitchFeatures(hop=HOP)
            result=[]
            for j in range(0,len(v),HOP):
                result.extend(frontend.push(v[j:j+HOP],start_sample=j))
            return result
        before=compute(x);after=compute(z)
        self.assertEqual(len(before),75)
        for i in range(69):
            np.testing.assert_array_equal(before[i][1],after[i][1])


if __name__=="__main__":unittest.main()
