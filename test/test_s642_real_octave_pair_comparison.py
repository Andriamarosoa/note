"""S642 paired recording method and octave fundamental validity tests."""
import unittest
import numpy as np
from scripts.s642_real_octave_pair_comparison import (
    SR, BASE_T, DELAY, DURATION, LOW, HIGH,
    normalize_loudness,conditions,clip_has_other_true_note,source_clip,
)


class RealOctaveComparisonTests(unittest.TestCase):
    def test_known_E3_and_E4_are_not_the_same_annotated_pitch(self):
        self.assertEqual((LOW,HIGH),(52,64))
        self.assertEqual(HIGH-LOW,12)
        self.assertAlmostEqual(440*2**((LOW-69)/12),164.81,places=2)
        self.assertAlmostEqual(440*2**((HIGH-69)/12),329.63,places=2)

    def test_real_clip_protocol_uses_same_base_note_for_mixture(self):
        n=round(DURATION*SR)
        t=np.arange(n)/SR
        # Unit test input synthesized ONLY to test mixing, not measured experiment
        mask=(t>=BASE_T)
        a=np.asarray(mask*.04*np.sin(2*np.pi*164.81*t),np.float32)
        b=np.asarray(mask*.04*np.sin(2*np.pi*329.63*t),np.float32)
        sounds,truth=conditions(dict(wave=a),dict(wave=b))
        self.assertEqual(set(sounds),set(truth))
        self.assertEqual([e["pitch"] for e in truth["E3_alone"]],[LOW])
        self.assertEqual([e["pitch"] for e in truth["E4_alone"]],[HIGH])
        self.assertEqual([e["pitch"] for e in truth["E3_plus_E4_simultaneous"]],[LOW,HIGH])
        self.assertAlmostEqual(truth["E3_then_E4_120ms"][1]["time"]-
                               truth["E3_then_E4_120ms"][0]["time"],DELAY,places=5)
        for sound in sounds.values():
            self.assertEqual(sound.shape,(n,))
            self.assertTrue(np.isfinite(sound).all())
            self.assertLessEqual(np.max(np.abs(sound)),.9+1e-6)
        self.assertFalse(np.array_equal(sounds["E3_alone"],sounds["E3_plus_E4_simultaneous"]))
        self.assertTrue(np.all(sounds["E3_then_E4_120ms"][:round(DELAY*SR)]==
                               0)) # silence before .65s in the synthetic test

    def test_exclusion_prevents_another_true_note_in_onset(self):
        mono=[dict(pitch=LOW,time=2.,duration=.4)]
        duo=mono+[dict(pitch=HIGH,time=2.,duration=.4)]
        self.assertFalse(clip_has_other_true_note(mono,LOW,2.))
        self.assertTrue(clip_has_other_true_note(duo,LOW,2.))

    def test_clip_aligned_without_time_shifts(self):
        wave=np.arange(3*SR,dtype=np.float32)
        clip=source_clip(wave,1.)
        self.assertEqual(len(clip),round(DURATION*SR))
        self.assertEqual(clip[0],wave[round((1.-BASE_T)*SR)])

if __name__=="__main__":unittest.main()
