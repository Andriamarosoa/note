"""S641 genuine causal scoring delay unit tests."""
from __future__ import annotations
import unittest
from causal_note.s635_shared_midi_metric import extract_reference,extract_events
from scripts.s641_frozen_output_delay_counterfactual import score_delay,HOP_MS,evaluate


class S641TimeShiftTests(unittest.TestCase):
    def test_early_detection_is_not_credit_until_emitted_after_truth(self):
        ref=extract_reference([dict(pitch=52,start_s=.10,end_s=.4)],1.)
        pred=extract_events([dict(type="note_on",pitch=52,time_s=.091)],1.)
        self.assertEqual(score_delay(ref,pred,0)["TP"],0)
        self.assertEqual(score_delay(ref,pred,2)["TP"],1)

    def test_late_correct_note_can_become_missed(self):
        ref=extract_reference([dict(pitch=52,start_s=.10,end_s=.4)],1.)
        pred=extract_events([dict(type="note_on",pitch=52,time_s=.178)],1.)
        self.assertEqual(score_delay(ref,pred,0)["TP"],1)
        self.assertEqual(score_delay(ref,pred,1)["TP"],0)

    def test_no_false_retrospective_credit(self):
        ref=extract_reference([dict(pitch=52,start_s=.10,end_s=.4)],1.)
        pred=extract_events([dict(type="note_on",pitch=52,time_s=.04)],1.)
        self.assertEqual(score_delay(ref,pred,6)["TP"],0)

    def test_positive_delay_preserves_note_count(self):
        ref=extract_reference([dict(pitch=52,start_s=.10,end_s=.4)],1.)
        pred=extract_events([dict(type="note_on",pitch=52,time_s=.09),
                             dict(type="note_on",pitch=58,time_s=.99)],1.)
        for hops in (0,2,14):
            x=score_delay(ref,pred,hops)
            self.assertEqual(x["TP"]+x["FP"],2)


if __name__=="__main__":unittest.main()
