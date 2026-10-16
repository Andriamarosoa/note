"""S634 tests: genuine factor-two waveform frequency, causal three view remap."""
from __future__ import annotations
import unittest
import numpy as np
from causal_note.three_view_pitch import (
    ThreeViewPitchProbe,remap_pitch_probabilities,warp_past_window,
    EXTENDED_HISTORY,
)
from causal_note.strict_live import SR,HOP,feature_track,load_heads
from scripts.live_strict_notes import DEFAULT_MODEL


class MockHeads:
    def __init__(self):
        self.calls=0
    def predict(self,x):
        self.calls+=1
        assert x.shape==(3,49,126),x.shape
        p=np.zeros((3,49,3),np.float32)
        p[1,64-40,:2]=[.9,.8]  # +12 view maps MIDI 64 to original MIDI 52
        p[2,64-40,:2]=[.95,.85] # -12 view maps MIDI 64 to original MIDI 76
        k=np.zeros((3,7),np.float32);k[:,1]=1
        return p,k


def tones(freq,n):
    return np.sin(2*np.pi*freq*np.arange(n)/SR).astype(np.float32)


def fundamental(audio):
    x=np.asarray(audio)
    win=np.hanning(len(x))
    spectrum=np.abs(np.fft.rfft(x*win))
    return np.fft.rfftfreq(len(x),1/SR)[np.argmax(spectrum[1:])+1]


class ThreeViewTests(unittest.TestCase):
    def test_frequency_double_and_half_are_real_audio_transformations(self):
        n=EXTENDED_HISTORY
        original=tones(110.,n)
        x2,half=warp_past_window(original)
        self.assertEqual(len(x2),8192)
        self.assertEqual(len(half),8192)
        self.assertLess(abs(fundamental(x2)-220),6)
        self.assertLess(abs(fundamental(half)-55),6)
        self.assertLess(abs(fundamental(original)-110),4)

    def test_two_exact_octave_inverse_pitch_mappings_and_boundary(self):
        a=np.zeros((49,3),np.float32)
        a[64-40,0]=.8;a[60-40,1]=.7
        up=remap_pitch_probabilities(a,12)
        down=remap_pitch_probabilities(a,-12)
        self.assertAlmostEqual(float(up[52-40,0]),.8,places=6)
        self.assertAlmostEqual(float(down[72-40,1]),.7,places=6)
        self.assertFalse(np.any(up[37:]))
        self.assertFalse(np.any(down[:12]))
        np.testing.assert_array_equal(remap_pitch_probabilities(a,0),a)
        with self.assertRaises(ValueError):
            remap_pitch_probabilities(a,2)

    def test_three_views_have_same_causal_clock_and_keep_native_K(self):
        heads=MockHeads()
        source=ThreeViewPitchProbe(heads)
        audio=tones(110.,HOP*75)
        records=[]
        start=0
        for sz in [256,512,1024,256,3072,4096,128,128,2048,4096,4096,4096]:
            if start>=len(audio):break
            frag=audio[start:min(start+sz,len(audio))]
            records.extend(source.push(frag,start_sample=start))
            start+=len(frag)
        if start<len(audio):
            records.extend(source.push(audio[start:],start_sample=start))
        self.assertEqual(len(records),len(audio)//HOP)
        self.assertEqual(heads.calls,len(records))
        self.assertTrue(all(r["decision_sample"]<=r["available_sample"] for r in records))
        self.assertEqual([r["decision_sample"] for r in records],list(range(HOP,len(audio)+1,HOP)))
        self.assertEqual(sum(r["valid_rescue"] for r in records),75-EXTENDED_HISTORY//HOP+1)
        for r in records:
            self.assertEqual(r["native_k"][1],1.)
        for r in records[:EXTENDED_HISTORY//HOP-1]:
            self.assertFalse(r["bass_rescue"].any())
        final=records[-1]
        self.assertAlmostEqual(float(final["bass_rescue"][52-40,0]),.9,places=6)
        self.assertAlmostEqual(float(final["high_rescue"][76-40,0]),.95,places=6)
        self.assertFalse(final["native"].any())

    def test_future_waveform_must_not_modify_prior_three_view_outputs(self):
        source=np.random.default_rng(4).normal(0,.03,100*HOP).astype(np.float32)
        alternate=source.copy()
        alternate[80*HOP:]*=-1
        p=ThreeViewPitchProbe(MockHeads())
        q=ThreeViewPitchProbe(MockHeads())
        a=p.push(source,start_sample=0)
        b=q.push(alternate,start_sample=0)
        for x,y in zip(a[:80],b[:80]):
            for key in ("native","native_k","bass_rescue","high_rescue"):
                np.testing.assert_array_equal(x[key],y[key])

    def test_real_wav_diagnostic_command_helpers(self):
        import tempfile
        import wave
        from pathlib import Path
        from scripts.s634_three_view_wav import load_audio,diagnostic
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/"fixture.wav"
            sound=tones(220.,round(.42*SR))
            with wave.open(str(path),"wb") as f:
                f.setnchannels(1);f.setsampwidth(2);f.setframerate(SR)
                f.writeframes((sound*32767).astype("<i2").tobytes())
            loaded=load_audio(path,seconds=1.)
            self.assertEqual(len(loaded),len(sound))
            result=diagnostic(loaded,MockHeads(),confidence=.6)
            self.assertEqual(result["frames_analyzed"],len(sound)//HOP)
            self.assertTrue(result["sound_realtime_notes_unchanged"])
            self.assertEqual(result["frequency_factors"],dict(bass_x2=2,original=1,high_half=.5))
            self.assertGreater(result["candidate_counts"].get("bass_x2_candidates",0),0)
            self.assertGreater(result["candidate_counts"].get("high_half_candidates",0),0)

    def test_real_frozen_head_original_branch_exact_parity(self):
        frozen=load_heads(DEFAULT_MODEL)
        data=tones(164.8,70*HOP)
        ends,features=feature_track(data)
        expected,_=frozen.predict(features)
        observed=ThreeViewPitchProbe(frozen).push(data,start_sample=0)
        self.assertEqual(len(ends),len(observed))
        for j in (0,15,63,69):
            np.testing.assert_allclose(observed[j]["native"],expected[j],
                                       rtol=5e-5,atol=5e-6)


if __name__=="__main__":
    unittest.main()
