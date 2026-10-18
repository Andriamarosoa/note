"""S713 hand-checkable guards for note event metrics and causal timestamps.

These fixtures are independent of any learned checkpoint. They validate the
metric definitions before a future independent agent improves Onset/Offset.
"""
import unittest

import numpy as np
from causal_note.strict_live import SR
from scripts.s704_joint_metrics import metric


def event(kind, sample, pitch=52, reason="causal_threshold"):
    return dict(type=kind,pitch=pitch,decision_sample=sample,
        available_sample=sample,time=sample/SR,
        reason=reason,velocity=96 if kind=="note_on" else 0)


class MetricContractTests(unittest.TestCase):
    def setUp(self):
        self.ends=np.arange(1,51,dtype=np.int64)*1024

    def at(self, seconds):
        # Decision sample anchored to one of the physical 1024-sample blocks.
        return int(round(seconds*SR/1024))*1024

    def test_one_pitch_perfect_note(self):
        truth=[dict(pitch=52,time=.2,duration=.4)]
        events=[event("note_on",self.at(.2)),
                event("note_off",self.at(.6))]
        result=metric(events,truth,self.ends)
        self.assertEqual((result["tp"],result["fp"],result["fn"]),(1,0,0))
        self.assertEqual(result["onset_f1"],1.0)
        self.assertEqual(result["offset_accuracy"],1.0)
        self.assertEqual(result["full_note_f1"],1.0)
        self.assertEqual(result["max_active"],1)
        self.assertEqual(result["long_offset_correct"],0)

    def test_eof_release_has_zero_offset_credit(self):
        truth=[dict(pitch=52,time=.2,duration=.4)]
        events=[event("note_on",self.at(.2)),
                event("note_off",int(self.ends[-1]),reason="stream_end")]
        result=metric(events,truth,self.ends)
        self.assertEqual(result["onset_f1"],1.0)
        self.assertEqual(result["offset_eligible"],1)
        self.assertEqual(result["offset_correct"],0)
        self.assertEqual(result["offset_accuracy"],0.0)
        self.assertEqual(result["full_note_f1"],0.0)

    def test_wrong_pitch_at_right_time_is_not_an_onset(self):
        truth=[dict(pitch=52,time=.2,duration=.4)]
        events=[event("note_on",self.at(.2),pitch=64),
                event("note_off",self.at(.6),pitch=64)]
        result=metric(events,truth,self.ends)
        self.assertEqual((result["tp"],result["fp"],result["fn"]),(0,1,1))
        self.assertEqual(result["onset_f1"],0.0)
        self.assertEqual(result["full_note_f1"],0.0)

    def test_two_same_pitch_notes_require_restrike_release(self):
        truth=[dict(pitch=52,time=.2,duration=.2),
               dict(pitch=52,time=.6,duration=.2)]
        events=[event("note_on",self.at(.2)),
                event("note_off",self.at(.4)),
                event("note_on",self.at(.6)),
                event("note_off",self.at(.8))]
        result=metric(events,truth,self.ends)
        self.assertEqual((result["tp"],result["fp"],result["fn"]),(2,0,0))
        self.assertEqual(result["offset_correct"],2)
        self.assertEqual(result["full_note_f1"],1.0)

    def test_duplicate_on_without_off_is_rejected(self):
        truth=[dict(pitch=52,time=.2,duration=.4)]
        events=[event("note_on",self.at(.2)),
                event("note_on",self.at(.25)),
                event("note_off",self.at(.6))]
        with self.assertRaisesRegex(AssertionError,"NoteOn"):
            metric(events,truth,self.ends)

    def test_backdating_is_rejected(self):
        truth=[dict(pitch=52,time=.2,duration=.4)]
        events=[event("note_on",self.at(.2)),
                event("note_off",self.at(.6))]
        events[0]["available_sample"]=events[0]["decision_sample"]-1
        with self.assertRaisesRegex(AssertionError,"Backdated"):
            metric(events,truth,self.ends)


if __name__=="__main__":
    unittest.main()
