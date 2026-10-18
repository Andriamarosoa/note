"""S726 dataset counting must not count sustained harmonics as new attacks."""
import unittest
from scripts.s726_real_overlap_prevalence import annotate_overlap


class CorpusOverlapTests(unittest.TestCase):
    def test_E2_then_E3_distinct_start_and_concurrent_duration(self):
        notes=[dict(pitch=40,time=0.,duration=1.),
               dict(pitch=52,time=.23,duration=.30)]
        counts,cases=annotate_overlap(notes)
        self.assertEqual(counts["E2_then_E3"],1)
        self.assertEqual(counts["interval_12"],1)
        self.assertEqual(cases[0]["interval"],12)

    def test_same_time_and_after_release_are_not_independent_onsets(self):
        notes=[dict(pitch=40,time=0.,duration=.30),
               dict(pitch=52,time=0.,duration=.15),
               dict(pitch=52,time=.40,duration=.25)]
        counts,_=annotate_overlap(notes)
        self.assertEqual(counts.get("interval_12",0),0)

    def test_more_general_octaves_fifths_and_chord_intervals(self):
        notes=[dict(pitch=45,time=0.,duration=2.),
               dict(pitch=52,time=.18,duration=.25),
               dict(pitch=57,time=.28,duration=.35),
               dict(pitch=64,time=.38,duration=.25)]
        c,_=annotate_overlap(notes)
        self.assertGreaterEqual(c["interval_7"],1)
        self.assertGreaterEqual(c["interval_12"],1)
        self.assertGreaterEqual(c["interval_19"],1)
        self.assertEqual(c.get("E2_then_E3",0),0)


if __name__=="__main__":
    unittest.main()
