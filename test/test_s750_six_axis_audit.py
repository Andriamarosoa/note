"""S750 post-hoc six-axis independent auditor synthetic regressions."""
import unittest
import numpy as np

from causal_note.strict_live import SR,HOP
from scripts.s748_six_axis_failure_audit import axis_audit,paired_audit
from scripts.s704_joint_metrics import metric


class SixAxisAuditTests(unittest.TestCase):
    def setUp(self):
        self.ends=np.arange(1,65,dtype=np.int64)*HOP
        self.on=int(self.ends[5])
        self.off=int(self.ends[25])
        self.events=[
            dict(type="note_on",pitch=40,available_sample=self.on,
                 decision_sample=self.on,time=self.on/SR,
                 reason="frozen_s708"),
            dict(type="note_off",pitch=40,available_sample=self.off,
                 decision_sample=self.off,time=self.off/SR,
                 reason="s707_causal_hazard")]
        self.truth=[dict(pitch=40,time=self.on/SR,
                         duration=(self.off-self.on)/SR)]

    def test_per_true_k_and_register_match_signed_s704(self):
        a=axis_audit(self.events,self.truth,self.ends)
        m=metric(self.events,self.truth,self.ends)
        for k in a["source_metrics"]:
            self.assertAlmostEqual(a["source_metrics"][k],m[k],places=12)
        self.assertEqual(a["per_true_k"]["1"]["under"],0)
        self.assertEqual(a["per_true_k"]["1"]["over"],0)
        self.assertGreater(a["per_true_k"]["1"]["exact"],0)
        self.assertEqual(a["register"]["low_E2_E3"]["onset_f1"],1.0)
        self.assertEqual(a["register"]["low_E2_E3"]["frame_pitch_f1"],1.0)
        self.assertEqual(a["offset_reasons"]["s707_causal_hazard"],1)

    def test_true_multiple_instances_same_pitch_keeps_s704_count(self):
        self.truth.append(dict(self.truth[0]))
        a=axis_audit(self.events,self.truth,self.ends)
        m=metric(self.events,self.truth,self.ends)
        self.assertGreater(a["per_true_k"]["2"]["true_frames"],0)
        self.assertEqual(a["source_metrics"]["exact_k"],m["exact_k"])
        self.assertTrue(np.asarray(a["confusion_k0_k6"]).shape==(7,7))

    def test_no_score_inflation_by_wrong_extra_attack(self):
        baseline=axis_audit(self.events,self.truth,self.ends)
        plus=[*self.events]
        plus.extend((
            dict(type="note_on",pitch=52,available_sample=self.on+4*HOP,
                 decision_sample=self.on+4*HOP,
                 time=(self.on+4*HOP)/SR,
                 reason="s743_multifundamental"),
            dict(type="note_off",pitch=52,available_sample=self.off,
                 decision_sample=self.off,time=self.off/SR,
                 reason="s707_causal_hazard")))
        plus.sort(key=lambda e:(e["available_sample"],0 if e["type"]=="note_off" else 1))
        audit=paired_audit(self.events,plus,self.truth,self.ends)
        self.assertFalse(audit["strict_six_nonregression"])
        self.assertLess(audit["trial"]["source_metrics"]["onset_f1"],
                        baseline["source_metrics"]["onset_f1"])

    def test_future_event_cannot_be_backdated(self):
        wrong=[dict(e) for e in self.events]
        wrong[0]["available_sample"]=self.on-HOP
        with self.assertRaises(AssertionError):
            axis_audit(wrong,self.truth,self.ends)


if __name__=="__main__":
    unittest.main()
