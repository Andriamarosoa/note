"""S643 tests: physical octave pair controls and composition leakage safeguards."""
import unittest
import numpy as np

from scripts.s643_octave_duet_head import (
    SR,DURATION,BASE_T,DELAY,extract_sources,candidate_pairs,
    select_composition_halves,make_conditions,
    spectral_signature,extracted_features,make_dataset,evaluate,
)


class OctaveDuetTests(unittest.TestCase):
    @staticmethod
    def made_note(frequency):
        n=round(DURATION*SR)
        t=np.arange(n)/SR
        envelope=np.where(t>=BASE_T,np.exp(-(t-BASE_T)*4),0)
        return np.asarray(.045*envelope*(
            np.sin(2*np.pi*frequency*t)+.45*np.sin(4*np.pi*frequency*t)),
            np.float32)

    def test_E2_and_its_octave_really_share_a_spectral_frequency(self):
        midi=40
        f0=440*2**((midi-69)/12)
        a=self.made_note(f0)
        b=self.made_note(2*f0)
        row_a=extracted_features(a,midi)
        row_ab=extracted_features(a+.75*b,midi)
        self.assertEqual(row_a.shape,row_ab.shape)
        self.assertTrue(np.isfinite(row_a).all())
        self.assertTrue(np.isfinite(row_ab).all())
        self.assertGreater(np.linalg.norm(row_ab-row_a),.1)

    def test_identical_original_note_used_in_single_and_duet(self):
        p=52
        f0=440*2**((p-69)/12)
        low=dict(wave=self.made_note(f0))
        high=dict(wave=self.made_note(2*f0))
        solo,duet=make_conditions(low,high)
        delayed_solo,delayed=make_conditions(low,high,offset=DELAY)
        np.testing.assert_array_equal(solo,delayed_solo)
        self.assertFalse(np.allclose(solo,duet))
        self.assertFalse(np.allclose(duet,delayed))
        self.assertEqual(len(solo),round(DURATION*SR))
        self.assertLessEqual(float(np.max(np.abs(duet))),.98)
        self.assertTrue(np.isfinite(delayed).all())

    def test_no_composition_appears_on_both_sides(self):
        clips=[]
        p_range=(44,49,52,56)
        for comp in "ABCDEFGHI":
            for pitch in [*p_range,*(p+12 for p in p_range)]:
                f0=440*2**((pitch-69)/12)
                clips.append(dict(composition=comp,name=comp+str(pitch),
                    pitch=pitch,onset=1.,wave=self.made_note(f0)))
        s=select_composition_halves(clips)
        self.assertTrue(set(s["a"]).isdisjoint(s["b"]))
        self.assertGreaterEqual(len(s["ab_pairs"]),12)
        self.assertGreaterEqual(len(s["ba_pairs"]),12)
        self.assertTrue(all(x["composition"] in s["a"]
                            for _,l,h in s["ab_pairs"] for x in (l,h)))
        self.assertTrue(all(x["composition"] in s["b"]
                            for _,l,h in s["ba_pairs"] for x in (l,h)))
        train=make_dataset(s["ab_pairs"][:4])
        test=make_dataset(s["ba_pairs"][:4])
        self.assertEqual(len(train[0]),8)
        self.assertEqual(len(test[0]),8)
        self.assertEqual(set(train[1]),{0,1})
        result=evaluate(train,test)
        self.assertEqual(result["n"],8)
        self.assertEqual(sum(result[k] for k in ("TP","FN")),4)
        self.assertEqual(sum(result[k] for k in ("FP","TN")),4)

if __name__=="__main__":
    unittest.main()
