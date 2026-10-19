"""S764 audit verifies distinct head metrics, source preservation, no oracle."""
import unittest

import numpy as np

from causal_note.s763_independent_decoders import DECISION_SAMPLES
from scripts.s764_six_decoder_shadow_cal32 import (
    replay_six, compare_track, merge_details, target_rows, intent_event)
from causal_note.s763_independent_decoders import Intent

SR=44100

def demo():
    n=6
    ends=np.arange(1,n+1,dtype=np.int64)*DECISION_SAMPLES
    post=np.zeros((n,49,3),np.float32)
    post[0:2,0,0]=.98;post[0:2,0,1]=.94
    post[2,0,2]=.98;post[2,0,1]=.05
    post[3:,0,2]=.02
    # A second note's attack is independent and disjoint.
    post[3:5,12,0]=.96;post[3:5,12,1]=.93
    post[5,12,2]=.96;post[5,12,1]=.02
    k=np.zeros((n,7),np.float32)
    k[:3,1]=1.;k[3:,1]=1.
    source=[
        dict(type="note_on",pitch=40,decision_sample=int(ends[0]),
             available_sample=int(ends[0]),time=float(ends[0]/SR),
             velocity=96,reason="signed_on"),
        dict(type="note_off",pitch=40,decision_sample=int(ends[2]),
             available_sample=int(ends[2]),time=float(ends[2]/SR),
             velocity=0,reason="signed_off"),
    ]
    truth=[
        dict(pitch=40,time=float(ends[0]/SR),duration=float(
            2*DECISION_SAMPLES/SR)),
        dict(pitch=52,time=float(ends[3]/SR),duration=float(
            2*DECISION_SAMPLES/SR)),
    ]
    native=np.arange(256,n*DECISION_SAMPLES+256,256,dtype=np.int64)
    return ends,post,k,source,truth,native

class ShadowAuditTests(unittest.TestCase):
    def test_six_decoders_are_evaluated_without_changing_source(self):
        ends,post,k,source,truth,native=demo()
        copy=[x.copy() for x in source]
        predicted=replay_six(ends,post,k,source,int(native[-1]))
        self.assertEqual(source,copy)
        self.assertEqual(predicted["signed"],source)
        self.assertEqual(len(predicted["counts"]["exact_k"]),6)
        self.assertEqual(len(predicted["pitches"]),6)
        self.assertTrue(predicted["onset"])
        self.assertTrue(predicted["offset"])
        self.assertTrue(predicted["full"])
        score=compare_track(predicted,truth,ends,native)
        self.assertEqual(set(score),{
            "exact_k","active_exact_k","onset","offset","frame_pitch","full_note"})
        agg=merge_details([score])
        self.assertEqual(set(agg),set(score))
        self.assertEqual(agg["onset"]["tp"],2)
        self.assertEqual(agg["offset"]["tp"],2)
        self.assertEqual(agg["full_note"]["tp"],2)
        self.assertEqual(agg["full_note"]["offset_correct"],2)
        self.assertGreaterEqual(agg["full_note"]["full_note_f1"],0)
        self.assertLessEqual(agg["full_note"]["full_note_f1"],1)

    def test_truth_never_enters_replay(self):
        ends,post,k,source,truth,native=demo()
        r1=replay_six(ends,post,k,source,int(native[-1]))
        changed_truth=[dict(x,pitch=75) for x in truth]
        r2=replay_six(ends,post,k,source,int(native[-1]))
        self.assertEqual(r1["onset"],r2["onset"])
        self.assertEqual(r1["offset"],r2["offset"])
        self.assertEqual(r1["full"],r2["full"])
        a=compare_track(r1,truth,ends,native)
        b=compare_track(r2,changed_truth,ends,native)
        self.assertNotEqual(a["onset"],b["onset"])

    def test_head_specific_metrics_are_distinct_from_full_midi(self):
        ends,post,k,source,truth,native=demo()
        predictions=replay_six(ends,post,k,source,int(native[-1]))
        a=compare_track(predictions,truth,ends,native)
        self.assertIn("actual_endings",a["offset"])
        self.assertNotIn("offset_accuracy",a["offset"])
        self.assertIn("six_score_diagnostic",a["full_note"])
        summary=merge_details([a])
        self.assertIn("unpaired release-boundary",summary["offset"]["definition"])
        self.assertEqual(summary["exact_k"]["total"],6)

    def test_k_and_pitch_ground_truth_not_resolved_until_after_proposals(self):
        ends,post,k,source,truth,native=demo()
        truek,truepitch=target_rows(ends,truth)
        self.assertEqual(truek.shape,(6,))
        self.assertEqual(truepitch.shape,(6,49))
        self.assertEqual(int(truek[0]),1)
        self.assertEqual(int(truek[3]),1)
        self.assertTrue(truepitch[0,0])
        self.assertTrue(truepitch[3,12])

    def test_no_offline_event_backdating(self):
        e=Intent("note_on",40,DECISION_SAMPLES,.95,"test")
        rec=intent_event(e)
        self.assertEqual(rec["available_sample"],DECISION_SAMPLES)
        self.assertEqual(rec["decision_sample"],DECISION_SAMPLES)
        self.assertEqual(rec["time"],DECISION_SAMPLES/SR)

if __name__=="__main__":
    unittest.main()
