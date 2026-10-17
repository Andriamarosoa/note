"""S644 natural (not digitally constructed) octave annotation scoring tests."""
import unittest

from scripts.s644_natural_octave_transfer import (
    truth_case,standard_metrics,ONSET_TOL,OBSERVE_AFTER
)


def n(pitch,t,duration=.4):
    return dict(pitch=pitch,time=t,duration=duration)


class NaturalOctaveTests(unittest.TestCase):
    def test_synchronous_note_plus_octave_is_true_separate_note(self):
        truth=[n(40,1.),n(52,1.02),n(59,1.01)]
        case=truth_case(truth,0)
        self.assertEqual(case["target"],1)
        self.assertEqual(case["low_pitch"],40)
        self.assertGreaterEqual(case["true_chord_K"],2)

    def test_lone_low_note_with_harmonic_is_negative(self):
        truth=[n(40,1.)]
        case=truth_case(truth,0)
        self.assertEqual(case["target"],0)
        self.assertEqual(case["upper_pitch"],52)
        self.assertEqual(case["true_chord_K"],1)

    def test_existing_upper_octave_and_delayed_onset_are_ambiguous(self):
        existing=[n(40,1.),n(52,.8,.4)]
        delayed=[n(40,1.),n(52,1.12,.25)]
        self.assertIsNone(truth_case(existing,0))
        self.assertIsNone(truth_case(delayed,0))
        self.assertIsNone(truth_case([n(40,.4)],0))
        self.assertIsNone(truth_case([n(40,1.,.1)],0))

    def test_metrics_handles_empty_and_single_class(self):
        self.assertEqual(standard_metrics([],[])["n"],0)
        s=standard_metrics([0,0],[.6,.3])
        self.assertIsNone(s["F1"])
        self.assertEqual((s["FP"],s["TN"]),(1,1))
        m=standard_metrics([0,1,1,0],[.1,.8,.3,.7])
        self.assertEqual((m["TP"],m["FP"],m["FN"],m["TN"]),(1,1,1,1))
        self.assertAlmostEqual(m["F1"],.5)

if __name__=="__main__":
    unittest.main()
