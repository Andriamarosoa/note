"""S652 exact event-level recovery and regression proof."""
from __future__ import annotations
import unittest
from causal_note.s652_event_round_audit import (
    audit_event_round,require_event_audit_before_next_round
)
from causal_note.strict_live import SR


def events(notes):
    output=[]
    for t,p in notes:
        for kind,when in (("note_on",t),("note_off",t+.025)):
            sample=round(when*SR)
            output.append(dict(type=kind,pitch=p,time=when,
                         decision_sample=sample,available_sample=sample,
                         reason="test_reference"))
    return sorted(output,key=lambda e:(e["available_sample"],
                      0 if e["type"]=="note_off" else 1))


def examples():
    truth=[dict(time=.10,pitch=52,duration=.10),
           dict(time=.25,pitch=55,duration=.10),
           dict(time=.25,pitch=59,duration=.10),
           dict(time=.45,pitch=40,duration=.10)]
    baseline=events([(.10,52),(.25,59),(.65,67)])
    candidate=events([(.10,52),(.25,55),(.45,40),(.25,67),(.70,80)])
    return truth,baseline,candidate


class TrueEventAuditTests(unittest.TestCase):
    def test_corrections_lost_reference_and_false_notes_are_separate(self):
        truth,before,after=examples()
        result=audit_event_round("clip-A","GuitarSet",truth,before,after)
        self.assertEqual(result["matched_reference_partition"]["corrected"],2)
        self.assertEqual(result["matched_reference_partition"]["lost"],1)
        self.assertEqual(result["matched_reference_partition"]["retained"],1)
        self.assertEqual(result["delta"]["TP"],1)
        self.assertEqual(result["delta"]["FP"],1)
        self.assertEqual(result["classifications"]["K2"]["newly_correct"],1)
        self.assertEqual(result["classifications"]["K2"]["regressed_true_note"],1)
        self.assertIn("possible_harmonic_confusion",
                      result["false_note_diagnostic_counts"])

    def test_same_aggregated_tp_can_hide_a_regression(self):
        truth,before,_=examples()
        after=events([(.10,52),(.25,55),(.65,67)])
        result=audit_event_round("clip-A","GuitarSet",truth,before,after)
        self.assertEqual(result["delta"]["TP"],0)
        self.assertEqual(result["matched_reference_partition"]["lost"],1)
        self.assertEqual(result["matched_reference_partition"]["corrected"],1)

    def test_guard_blocks_partial_audit_before_next_loop(self):
        truth,before,after=examples()
        verified=audit_event_round("clip-A","GuitarSet",truth,before,after)
        self.assertTrue(require_event_audit_before_next_round([verified],["clip-A"]))
        with self.assertRaises(AssertionError):
            require_event_audit_before_next_round([verified],["clip-A","clip-B"])

    def test_reject_anti_causal_or_missing_noteoff(self):
        truth,before,after=examples()
        after[0]["decision_sample"]=after[0]["available_sample"]+10
        with self.assertRaises(AssertionError):
            audit_event_round("clip-A","GuitarSet",truth,before,after)
        truth,before,after=examples()
        after.pop()
        with self.assertRaises(AssertionError):
            audit_event_round("clip-A","GuitarSet",truth,before,after)

if __name__=="__main__":unittest.main()
