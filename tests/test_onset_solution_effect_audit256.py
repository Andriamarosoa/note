import unittest
import numpy as np

from onset256.learned_hgb_arbiter import GatePolicy, replay_precomputed
from scripts.audit_onset_solution_effect256 import (
    classify_truth, near_truth, reasons, summarize_rows, metric)


class OnsetSolutionEffectAuditTests(unittest.TestCase):
    def test_event_identity_recovered_lost_and_retained(self):
        self.assertEqual(classify_truth({0,1,3}, {0,2,3}, 5),
            ["retained_tp","lost_tp","recovered_tp","retained_tp","persistent_fn"])

    def test_audit_reasons_reproduce_causal_gate(self):
        baseline=np.zeros((20,49),bool)
        direct=np.zeros_like(baseline)
        baseline[0,0]=True
        baseline[1,0]=True
        baseline[12,0]=True
        baseline[3,1]=True
        direct[3,1]=True
        scores=np.asarray([.99,.99,.99,.99],np.float32)
        policy=GatePolicy(cut=.9,gap_hops=7)
        got=reasons(baseline,direct,scores,True,policy)
        self.assertEqual(got,[
            "accepted_hgb_addition","past_gate_duplicate_guard",
            "already_selected_by_acoustic","accepted_hgb_addition"])
        merged=replay_precomputed(direct,baseline,scores,policy)
        self.assertEqual(int(merged.sum()),3)
        self.assertTrue(merged[0,0]);self.assertTrue(merged[12,0]);self.assertTrue(merged[3,1])

    def test_threshold_and_previous_direct_are_explained(self):
        baseline=np.zeros((4,49),bool)
        direct=np.zeros_like(baseline)
        baseline[2,0]=True
        direct[1,0]=True
        policy=GatePolicy(cut=.6,gap_hops=7)
        self.assertEqual(reasons(baseline,direct,np.asarray([.9],np.float32),
                                 True,policy),["past_direct_duplicate_guard"])
        self.assertEqual(reasons(baseline,direct,np.asarray([.1],np.float32),
                                 True,policy),["gate_probability_below_cut"])
        self.assertEqual(reasons(baseline,direct,np.asarray([.9],np.float32),
                                 False,None),["disabled_by_calibration"])

    def test_harmonic_proximity_is_not_a_proven_cause(self):
        truth=[dict(pitch=52,time=.1,duration=.3),
               dict(pitch=55,time=.1,duration=.3)]
        o=near_truth(truth,40,.1)
        self.assertTrue(o["near_other_pitch_truth"])
        self.assertTrue(o["harmonic_nearby_truth"])
        self.assertFalse(o["near_same_pitch_truth"])

    def test_summaries_preserve_sign_and_tp_fp_accounting(self):
        rows=[dict(before=metric(3,2,2),after=metric(2,4,3),
                   recovered_tp=0,lost_tp=1,added_fp=2,added_events=2,
                   hgb_tp_rejected=1,direct_events_removed=0),
              dict(before=metric(1,1,1),after=metric(2,2,0),
                   recovered_tp=1,lost_tp=0,added_fp=1,added_events=1,
                   hgb_tp_rejected=0,direct_events_removed=0)]
        r=summarize_rows(rows)
        self.assertEqual(r["recovered_tp"],1)
        self.assertEqual(r["lost_tp"],1)
        self.assertEqual(r["added_fp"],3)
        self.assertEqual(r["direct_events_removed"],0)
        self.assertEqual(r["before"]["tp"],r["after"]["tp"])
        self.assertLess(r["delta_f1_points"],0)


if __name__=="__main__":
    unittest.main()
