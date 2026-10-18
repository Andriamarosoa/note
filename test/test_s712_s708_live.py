"""S712 safety tests for the exact S708 LIVE acoustic feature and event path."""
import unittest
from unittest.mock import patch

import numpy as np

from causal_note.strict_live import SR, HOP, feature_track
from causal_note.s704_joint_heads import acoustic_features
from causal_note.s706_sharp_spectra import sharp_track
from causal_note.s708_cardinality_selection import GuidedPolicy
from causal_note.s712_s708_live import S708StrictLiveStream, DECISION


class FrozenStubHeads:
    def predict(self, f):
        v=np.asarray(f, np.float32)
        p=np.zeros((49,3),np.float32)
        p[:,1]=.5
        p[:,2]=.1
        k=np.zeros(7,np.float32)
        k[0]=1.
        return p,k


class NativeTemporalStub:
    def eval(self):
        return self
    def __call__(self,x):
        import torch
        n=x.shape[1]
        logits=torch.full((1,n,49,3),-10.,dtype=torch.float32)
        logits[:,:,:,1]=5.
        k=torch.full((1,n,7),-10.,dtype=torch.float32)
        k[:,:,0]=10.
        return logits,k


class S712Tests(unittest.TestCase):
    def wave(self,n):
        rng=np.random.default_rng(712)
        t=np.arange(n)/SR
        a=.14*np.sin(2*np.pi*110*t)+.07*np.sin(2*np.pi*164.81*t)
        return (a+rng.normal(0,.005,n)).astype(np.float32)

    def make(self):
        return S708StrictLiveStream(FrozenStubHeads(),NativeTemporalStub(),
            GuidedPolicy(onset=.75,sustain_on=.5,sustain_off=.3,
                         release=.85,release_steps=1,count_confidence=1.1,
                         blend=.7,guide_confidence=.5,weak_onset=.15,
                         weak_sustain=.55,rise=.04,trim=False))

    def test_acoustic_stream_matches_offline_current_and_past(self):
        audio=self.wave(DECISION*5)
        ends,feat=feature_track(audio,block_size=4096)
        stub=FrozenStubHeads()
        pk=[stub.predict(f) for f in feat]
        p=np.asarray([v[0] for v in pk])
        k=np.asarray([v[1] for v in pk])
        chosen=np.arange(3,len(ends),4)
        x=acoustic_features(feat,p,k)[chosen]
        expected=np.concatenate((x,sharp_track(audio,ends[chosen])),-1)
        stream=self.make()
        observed=[]
        def capture(v):
            observed.append(v.copy())
            return np.zeros((49,3),np.float32),np.eye(7,dtype=np.float32)[0]
        stream.temporal.step=capture
        for j in range(0,len(audio),256):
            stream.push(audio[j:j+256],start_sample=j)
        self.assertEqual(len(observed),len(chosen))
        for a,b in zip(observed,expected):
            np.testing.assert_allclose(a,b,atol=3e-6,rtol=3e-6)

    def test_callback_sizes_are_event_and_feature_invariant(self):
        audio=self.wave(DECISION*3+137)
        outputs=[]
        for block in (128,256,333,2048,len(audio)):
            stream=self.make()
            events=[]
            offset=0
            while offset<len(audio):
                chunk=audio[offset:offset+block]
                events.extend(stream.push(chunk,start_sample=offset))
                offset+=len(chunk)
            events.extend(stream.finish())
            outputs.append([(e['type'],e['pitch'],e['decision_sample']) for e in events])
            self.assertEqual(stream.model_calls,3)
            self.assertEqual(stream.received,len(audio))
        for out in outputs[1:]:
            self.assertEqual(out,outputs[0])

    def test_future_audio_cannot_change_past_decisions(self):
        audio=self.wave(DECISION*4)
        b=audio.copy()
        b[DECISION*2:]=-b[DECISION*2:]
        events=[]
        for signal in (audio,b):
            stream=self.make()
            prefix=[]
            for j in range(0,len(signal),256):
                chunk=stream.push(signal[j:j+256],start_sample=j)
                if j<DECISION*2:
                    prefix.extend((e['type'],e['pitch'],e['decision_sample'])
                                  for e in chunk)
            events.append(prefix)
        self.assertEqual(events[0],events[1])

    def test_reject_gap_and_nonfinite(self):
        stream=self.make()
        with self.assertRaisesRegex(ValueError,"discontinuous"):
            stream.push(self.wave(256),start_sample=2)
        with self.assertRaisesRegex(ValueError,"finite"):
            stream.push(np.array([float('nan')],np.float32),start_sample=0)


if __name__=="__main__":
    unittest.main()
