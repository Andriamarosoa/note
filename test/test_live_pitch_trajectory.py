"""Unit tests for offline causal trajectory sampling (no model required)."""
import unittest

from scripts.audit_live_pitch_count import summarize_trajectory


class PitchTrajectoryAuditTests(unittest.TestCase):
    def frame(self, time_s, e2_onset, e2_h1):
        return {
            "time_s": time_s,
            "e2_onset": e2_onset,
            "e2_sustain": e2_onset / 2,
            "e2_h1_8192": e2_h1,
            "e2_harmonic_score_8192": 2 * e2_h1,
            "e3_harmonic_score_8192": .3,
            "original_note_on": [],
            "predicted_k": 3,
            "k_confidence": .4,
        }

    def test_missing_original_attack(self):
        frames = [self.frame(1.1, .1, .01)]
        summary = summarize_trajectory(frames, None)
        self.assertFalse(summary["available"])

    def test_future_trajectory_never_changes_original_attack_time(self):
        frames = [
            self.frame(1.0, .001, .01),
            self.frame(1.15, .00679, .02),
            self.frame(1.18, .07008, .089),
            self.frame(1.2, .4, .5),
            self.frame(1.35, .3, 1.0),
            self.frame(1.7, .5, .2),
        ]
        report = summarize_trajectory(frames, 1.15)
        self.assertTrue(report["available"])
        self.assertEqual(report["first_original_attack_s"], 1.15)
        self.assertAlmostEqual(report["maxima_during_next_500_ms"]["e2_h1_8192"]["value"], 1.0)
        self.assertAlmostEqual(report["maxima_during_next_500_ms"]["e2_onset"]["value"], .4)
        self.assertEqual(report["sampled_causal_frames"][0]["time_s"], 1.15)
        self.assertEqual(report["sampled_causal_frames"][1]["time_s"], 1.18)
        self.assertLessEqual(max(row["time_s"] for row in report["sampled_causal_frames"]), 1.7)


if __name__ == "__main__":
    unittest.main()
