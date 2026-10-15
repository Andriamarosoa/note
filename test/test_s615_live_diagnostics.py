"""Synthetic checks for S615. No opened test is used for threshold fitting."""
from __future__ import annotations
import unittest
from scripts.s615_postfreeze_live_diagnostics import (
    classify_fp, classify_fn, intervals, unmatched_novelty,
)


class S615LiveDiagnosticsTests(unittest.TestCase):
    def test_duplicate_and_timing_are_disjoint(self):
        truth = [{"pitch": 60, "time": 1., "duration": .3}]
        match = [{"pitch": 60, "time": 1.02}]
        self.assertEqual(classify_fp({"pitch": 60, "time": 1.04}, truth, match),
                         "same_pitch_duplicate_near_matched_onset")
        self.assertEqual(classify_fp({"pitch": 60, "time": 1.15}, truth, match),
                         "same_pitch_onset_outside_80ms")
        self.assertEqual(classify_fp({"pitch": 72, "time": 1.2}, truth, match),
                         "harmonic_interval_coincidence_not_causation")
        self.assertEqual(classify_fp({"pitch": 81, "time": 3.}, truth, match),
                         "other_unmatched_onset")

    def test_missing_reattack_is_distinct_from_absent_note(self):
        held = [{"pitch": 60, "on": 1., "off": 2., "off_reason": "causal_release"}]
        self.assertEqual(classify_fn({"pitch": 60, "time": 1.55}, [], held),
                         "already_active_at_truth_rearticulation")
        self.assertEqual(classify_fn({"pitch": 61, "time": 1.55}, [], held),
                         "no_same_pitch_onset_within_180ms")
        predicted = [{"pitch": 60, "time": 1.7}]
        self.assertEqual(classify_fn({"pitch": 60, "time": 1.55}, predicted, held),
                         "same_pitch_emitted_outside_80ms")

    def test_novel_fp_pairing_uses_one_to_one_and_pitch(self):
        a = [{"pitch": 60, "time": 1.}, {"pitch": 61, "time": 2.},
             {"pitch": 60, "time": 1.002}]
        b = [{"pitch": 60, "time": 1.01}, {"pitch": 61, "time": 2.08}]
        self.assertEqual(unmatched_novelty(a, b),
                         {"common_fp_approx": 1, "new_fp_approx": 2,
                          "removed_fp_approx": 1})

    def test_midi_intervals_balance_and_panic(self):
        events = [{"type": "note_on", "pitch": 60, "time": .1},
                  {"type": "note_off", "pitch": 60, "time": .5, "reason": "stream_end"}]
        self.assertEqual(intervals(events)[0]["off_reason"], "stream_end")
        with self.assertRaisesRegex(AssertionError, "Unbalanced"):
            intervals(events + [events[-1]])
        with self.assertRaisesRegex(AssertionError, "Unclosed"):
            intervals(events[:1])


if __name__ == "__main__":
    unittest.main()
