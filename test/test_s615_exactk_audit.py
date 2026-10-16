"""Deterministic S615 onset Exact-K audit tests; NOT historical S80 K."""
import unittest
from scripts.s615_exactk_audit import exact_k_onset_audit, merge_k_reports


class ExactKOnsetAuditTests(unittest.TestCase):
    def test_true_polyphony_is_not_suppressed_as_single_harmonic(self):
        truth=[dict(time=1.0,pitch=40),
               dict(time=1.004,pitch=52),
               dict(time=2.0,pitch=64)]
        emitted=[dict(type="note_on",time=1.015,pitch=40),
                 dict(type="note_on",time=2.030,pitch=64)]
        d=exact_k_onset_audit(emitted,truth)
        self.assertEqual(d["by_true_k"]["2"]["total"],1)
        self.assertEqual(d["by_true_k"]["2"]["undercount"],1)
        self.assertEqual(d["by_true_k"]["2"]["exact_k"],0)
        self.assertEqual(d["by_true_k"]["1"]["exact_pitch_set"],1)
        merged=merge_k_reports([d,d])
        self.assertEqual(merged["total_groups"],4)
        self.assertEqual(merged["exact_k_groups"],2)
        self.assertEqual(merged["onset_exact_k_rate"],.5)

    def test_false_octave_addition_is_overcount_even_if_onset_time_correct(self):
        truth=[dict(time=.5,pitch=40)]
        emitted=[dict(type="note_on",time=.52,pitch=40),
                 dict(type="note_on",time=.525,pitch=52)]
        d=exact_k_onset_audit(emitted,truth)
        self.assertEqual(d["by_true_k"]["1"]["overcount"],1)
        self.assertEqual(d["by_true_k"]["1"]["exact_pitch_set"],0)

    def test_unmatched_truth_is_missing_and_undercounted(self):
        d=exact_k_onset_audit([],[dict(time=.5,pitch=52)])
        self.assertEqual(d["by_true_k"]["1"]["missing"],1)
        self.assertEqual(d["by_true_k"]["1"]["undercount"],1)
        self.assertEqual(d["groups_matched"],0)


if __name__=="__main__":
    unittest.main()
