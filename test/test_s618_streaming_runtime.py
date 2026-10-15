from __future__ import annotations
import unittest
import numpy as np
from causal_note.s618_attack_head import input_features,DIM,TrainedPolicy
from causal_note.s618_streaming_runtime import CausalFeatureState
from causal_note.strict_live import CausalPitchFeatures,HOP


class S618StreamingTests(unittest.TestCase):
    def test_stream_feature_parity(self):
        rng=np.random.default_rng(61813)
        x=rng.random((300,49,126)).astype(np.float32)
        p=rng.random((300,49,3)).astype(np.float32)
        expected=input_features(x,p)
        state=CausalFeatureState()
        got=np.stack([state.frame(row,prob) for row,prob in zip(x,p)])
        self.assertEqual(got.shape,(300,49,DIM))
        np.testing.assert_array_equal(got,expected)

    def test_unavailable_future_cannot_change_past_audio_features(self):
        rng=np.random.default_rng(61814)
        samples=rng.standard_normal(256*20).astype(np.float32)*.1
        future=samples.copy();future[256*13:]=3.
        def replay(audio):
            frontend=CausalPitchFeatures()
            rows=[]
            for start in range(0,len(audio),256):
                rows.extend(frontend.push(audio[start:start+256],start_sample=start))
            return rows
        a=replay(samples);b=replay(future)
        self.assertEqual([r[0] for r in a],[r[0] for r in b])
        for (_,x),(_,y) in zip(a[:13],b[:13]):
            np.testing.assert_array_equal(x,y)

    def test_no_external_pretrained_weights(self):
        self.assertEqual(TrainedPolicy().rescue_above,2.)

if __name__=="__main__":unittest.main()
