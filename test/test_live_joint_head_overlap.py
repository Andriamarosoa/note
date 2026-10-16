"""Check onset/sustain overlap audit semantics on small synthetic tracks."""
import unittest
from scripts.audit_live_pitch_count import audit_joint_head_overlap


class JointHeadTests(unittest.TestCase):
    def test_separate_head_maxima_do_not_imply_a_valid_joint_threshold(self):
        frames = [
            {"time_s": 1.0, "e2_onset": .16, "e2_sustain": .2},
            {"time_s": 1.05, "e2_onset": .11, "e2_sustain": .64},
            {"time_s": 1.1, "e2_onset": .001, "e2_sustain": .99},
        ]
        result = audit_joint_head_overlap(frames, anchor_s=1.0)
        self.assertTrue(result["available"])
        maximum = {row["sustain_min"]: row for row in result["conditional_maxima"]}
        self.assertEqual(maximum[.5]["max_simultaneous_onset"], .11)
        self.assertEqual(maximum[.25]["max_simultaneous_onset"], .11)
        self.assertAlmostEqual(max(row["e2_onset"] for row in frames), .16)

    def test_late_head_peak_cannot_affect_first_half_second(self):
        frames = [
            {"time_s": 1.0, "e2_onset": .01, "e2_sustain": .6},
            {"time_s": 1.9, "e2_onset": .99, "e2_sustain": .99},
        ]
        result = audit_joint_head_overlap(frames, anchor_s=1.0)
        maximum = {row["sustain_min"]: row for row in result["conditional_maxima"]}
        self.assertEqual(maximum[.5]["max_simultaneous_onset"], .01)
        self.assertEqual(result["frames"], 1)

    def test_no_original_anchor(self):
        self.assertFalse(audit_joint_head_overlap([], anchor_s=None)["available"])


if __name__ == "__main__":
    unittest.main()
