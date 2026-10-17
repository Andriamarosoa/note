"""S652 checks that Tone proxy preserves pitch, is causal, and gates NoteOn safely."""
import unittest
import numpy as np
from scipy.signal import sosfilt,butter

from causal_note.strict_live import SR,HOP,Selection
from scripts.s652_tone_filter_multiview import (
    causal_lowpass,probabilities,POLICIES,FILTERS,
    aggregate,choose_on_calibration
)


class ToneMultiviewTests(unittest.TestCase):
    def test_tone_is_causal_at_original_sample_clock(self):
        rng=np.random.default_rng(652)
        src=rng.standard_normal(15000).astype(np.float32)*.01
        future=src.copy()
        future[10000:]=.9
        for fc in FILTERS.values():
            x=causal_lowpass(src,fc)
            y=causal_lowpass(future,fc)
            np.testing.assert_array_equal(x[:10000],y[:10000])
            self.assertEqual(len(x),len(src))
            self.assertTrue(np.isfinite(x).all())
            with self.assertRaises(ValueError):
                causal_lowpass(src,fc+1)

    def test_attenuates_upper_harmonics_not_fundamental_shift(self):
        duration=.60
        t=np.arange(round(duration*SR))/SR
        e2=440*2**((40-69)/12)
        x=np.asarray(.20*np.sin(2*np.pi*e2*t)+
                     .18*np.sin(2*np.pi*20*e2*t)+
                     .13*np.sin(2*np.pi*35*e2*t),np.float32)
        y=causal_lowpass(x,1100.)
        # Gain at e2 must remain strong, at 20*h progressively weaker.
        fft=np.abs(np.fft.rfft(y[SR//10:]))
        fft0=np.abs(np.fft.rfft(x[SR//10:]))
        for hz,cut in [(e2,.95),(20*e2,.75),(35*e2,.5)]:
            idx=round(hz*(len(y)-SR//10)/SR)
            ratio=float(fft[idx]/max(1e-10,fft0[idx]))
            self.assertLess(ratio,1.005)
            self.assertGreater(ratio,cut) if hz==e2 else self.assertLess(ratio,cut)
        self.assertEqual(HOP,256)

    def test_policies_all_valid_and_preserve_native_nothing_fabricated(self):
        cfg=Selection(onset=.50,active_on=.50)
        original=np.zeros((49,3),np.float32)
        mild=np.zeros_like(original)
        dark=np.zeros_like(original)
        idx=40-40
        original[idx,0:2]=[.49,.62]
        mild[idx,0:2]=[.88,.9]
        dark[idx,0:2]=[.91,.9]
        for policy in POLICIES:
            p=probabilities(policy,original,mild,dark,cfg)
            self.assertEqual(p.shape,(49,3),policy)
            self.assertTrue(np.isfinite(p).all())
            self.assertTrue(((p>=0)&(p<=1)).all())
        self.assertEqual(probabilities("S614_original",original,mild,dark,cfg)[idx,0],.49)
        self.assertEqual(probabilities("original_OR_both_confirmed",original,mild,dark,cfg)[idx,0],.91)
        self.assertEqual(probabilities("original_AND_mild",original,mild,dark,cfg)[idx,0],0)
        self.assertEqual(probabilities("vote_2of3",original,mild,dark,cfg)[idx,0],.91)
        with self.assertRaises(ValueError):
            probabilities("invalid",original,mild,dark,cfg)

    def test_calibration_guards_require_gain_across_holdout_compositions(self):
        def sample(composition,native=(5,5),rescue=(7,8)):
            measures={}
            for name in POLICIES:
                TP,FP=(native if name=="S614_original" else rescue)
                measures[name]=dict(
                    score=dict(TP=TP,FP=FP,FN=10-TP,latency_ms=[]),
                    onset_overlap_TP=TP,
                    onset_K2_TP=TP,onset_K3plus_TP=0,
                    pitch_E2_correct=0,pitch_E3_correct=0,
                    new_vs_S614=0,lost_vs_S614=0)
            return dict(composition=composition,results=measures)
        calibration=[sample("A"),sample("B"),sample("C"),sample("D"),sample("E")]
        held=[sample("X",rescue=(3,9)),sample("Y",rescue=(3,9)),
              sample("Z",rescue=(3,9)),sample("W",rescue=(3,9))]
        selected,info=choose_on_calibration(calibration,held)
        self.assertEqual(selected,"S614_original")
        self.assertEqual(info["held_positive_compositions"],0)


if __name__=="__main__":
    unittest.main()
