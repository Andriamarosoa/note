"""S831 regression suite: inter-pitch context, silence confidence, event reattacks,
strict 256-hop causality after FFT ring saturation, lazy private model calls.
All signals synthetic; FIT80 GuitarSet and physical iRig are separate gates.
"""
import unittest
import numpy as np
from causal_note.s829_exactk_s809_native256 import HOP,fit_exact,summarize_one
from causal_note.s830_private_musical_evidence256 import (
    S830PCMAcoustics,S830FittedStream,pitch_features,
    positive_note_confidence,selected,train_evidence
)
from scripts.s831_train_audit_corrected256 import note_event_targets
from test.test_s830_private_musical_evidence256 import synthetic


class TestS831(unittest.TestCase):
    def test_harmonic_context_distinguishes_pitches(self):
        f=np.zeros((1,49,72),np.float32)
        f[0,0,[0,8,16]]=3.0  # low E2 fundamental
        f[0,12,[0,8,16]]=2.0 # E3 possible harmonic
        x=pitch_features(f)
        self.assertEqual(x.shape,(1,49,85))
        self.assertEqual(float(x[0,12,73]),3.0)  # E3 sees octave-lower E2
        self.assertEqual(float(x[0,0,73]),0.0) # E2 has no lower octave
        self.assertNotEqual(float(x[0,12,81]),float(x[0,0,81]))

    def test_silence_is_not_spuriously_confident(self):
        p=np.zeros((7,49,3),np.float32)
        p[:,:,1]=.01
        conf=positive_note_confidence(p)
        np.testing.assert_allclose(conf,np.full(7,.01),atol=1e-7)
        base=np.full(7,2,np.uint8)
        propose=np.zeros(7,np.uint8)
        np.testing.assert_array_equal(
            selected(base,propose,propose,conf,"router"),base)

    def test_repeat_note_without_silent_hop_produces_two_attacks(self):
        ends=np.arange(1,7)*HOP
        # E2 is plucked at frame 1 and restruck at frame 3,
        # with contiguous GT note intervals and no all-off frame.
        notes=[dict(pitch=40,time=0.,duration=2*HOP/44100),
               dict(pitch=40,time=2*HOP/44100,duration=2*HOP/44100)]
        onset,release=note_event_targets(notes,ends)
        self.assertEqual(np.flatnonzero(onset[:,0]).tolist(),[0,1])
        self.assertEqual(np.flatnonzero(release[:,0]).tolist(),[1,3])

    def test_fft_history_wrap_and_random_chunking_future(self):
        rng=np.random.default_rng(831)
        signal=rng.normal(0,.06,96*HOP+37).astype(np.float32)
        changed=signal.copy()
        changed[70*HOP:]*=-1
        def run(x,width):
            stream=S830PCMAcoustics();received=[];pos=0
            while pos<len(x):
                n=min(width,len(x)-pos)
                received.extend(stream.push(x[pos:pos+n],start_sample=pos))
                pos+=n
            return received
        base=run(signal,256)
        other=run(changed,128)
        self.assertEqual(len(base),96)
        self.assertEqual(len(other),96)
        np.testing.assert_array_equal([r[0] for r in base],
                                      np.arange(1,97)*HOP)
        for i in range(70):
            np.testing.assert_array_equal(base[i][1],other[i][1])

    def test_lazy_s829_mode_skips_musical_models_after_long_history(self):
        train=[synthetic("a","c1",51),synthetic("b","c2",52),
               synthetic("c","c3",53)]
        cal=[synthetic("d","cal",54)]
        count=fit_exact(train,cal,seed=831)
        pitches=train_evidence(train,seed=832)
        stream=S830FittedStream(count,pitches,("s829",.30,0))
        rng=np.random.default_rng(12)
        sound=rng.normal(0,.07,65*HOP).astype(np.float32)
        output=[]
        for i in range(65):
            output+=list(stream.push(sound[i*HOP:(i+1)*HOP],
                                     start_sample=i*HOP))
        self.assertEqual(len(output),65)
        self.assertEqual(stream.diagnostics()["musical_model_calls"],0)
        self.assertEqual(stream.diagnostics()["emitted_decisions"],65)
        self.assertIsNotNone(stream.diagnostics()["per_hop_model_cpu_p95_ms"])


if __name__=="__main__":
    unittest.main()
