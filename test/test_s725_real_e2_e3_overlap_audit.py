"""S725 one-to-one real overlap proof, no oracle at event decision."""
import unittest
from scripts.s725_real_e2_e3_overlap_audit import (
    actual_e3_with_e2_sounding,characterize_predicted_onset)


class E2E3GroundedAuditTests(unittest.TestCase):
    def test_real_E3_during_E2_only_if_E2_preexisted(self):
        notes=[
            dict(pitch=40,time=0.,duration=1.),
            dict(pitch=52,time=.20,duration=.3),
            dict(pitch=52,time=1.50,duration=.2),
            dict(pitch=52,time=.02,duration=.1)]
        self.assertEqual(actual_e3_with_e2_sounding(notes),[1])

    def test_matched_correct_E3_and_unmatched_false_E3_inside_held_E2(self):
        notes=[dict(pitch=40,time=0.,duration=1.),
               dict(pitch=52,time=.20,duration=.20)]
        events=[dict(type="note_on",pitch=40,time=0.),
                dict(type="note_on",pitch=52,time=.22),
                dict(type="note_on",pitch=52,time=.55)]
        score=characterize_predicted_onset(events,notes)
        self.assertEqual(score["E3_on_E2_total"],1)
        self.assertEqual(score["E3_on_E2_matched"],1)
        self.assertEqual(score["E3_on_E2_missed"],0)
        self.assertEqual(score["false_E3_while_true_E2_sustains"],1)

    def test_never_call_a_false_E3_true_due_to_active_E2(self):
        notes=[dict(pitch=40,time=0.,duration=1.)]
        events=[dict(type="note_on",pitch=52,time=.20)]
        score=characterize_predicted_onset(events,notes)
        self.assertEqual(score["E3_on_E2_total"],0)
        self.assertEqual(score["false_E3_while_true_E2_sustains"],1)


if __name__=="__main__":
    unittest.main()
