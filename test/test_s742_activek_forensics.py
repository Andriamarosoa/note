"""S742 independent event-count audit tests; no synthetic 80-percent results."""
import unittest
import numpy as np
from causal_note.strict_live import SR, HOP
from causal_note.s708_cardinality_selection import GuidedPolicy
from scripts.s742_activek_forensics import (
    exact_active_breakdown, k_source_diagnostics, trace_weak_k_gate,
    aggregate_forensics)


def event(kind,pitch,hop):
    sample=int(hop*HOP)
    return dict(type=kind,pitch=pitch,decision_sample=sample,
                available_sample=sample,time=sample/SR,
                velocity=0,reason="test")


class S742Test(unittest.TestCase):
    def test_active_k_counts_real_emitted_instances_not_k_logits(self):
        ends=np.arange(1,17,dtype=np.int64)*HOP
        start=2*HOP/SR
        duration=8*HOP/SR
        truth=[dict(pitch=40,time=start,duration=duration),
               dict(pitch=47,time=start,duration=duration)]
        # Only one of two notes was emitted.
        events=[event("note_on",40,2),event("note_off",40,10)]
        audit=exact_active_breakdown(events,truth,ends)
        self.assertEqual(audit["active_frames"],8)
        self.assertEqual(audit["under_active"],8)
        self.assertEqual(audit["by_true_k"]["2"]["under"],8)
        self.assertEqual(audit["by_true_k"]["2"]["over"],0)

    def test_unison_truth_impossible_to_reproduce_with_one_pitch(self):
        ends=np.arange(1,17,dtype=np.int64)*HOP
        truth=[dict(pitch=40,time=2*HOP/SR,
                    duration=8*HOP/SR)]*2
        events=[event("note_on",40,2),event("note_off",40,10)]
        audit=exact_active_breakdown(events,truth,ends)
        self.assertEqual(audit["unison_or_same_pitch_impossible_frames"],8)
        self.assertEqual(audit["by_true_k"]["2"]["under"],8)

    def test_k0_is_excluded_from_active_but_in_other_count(self):
        ends=np.arange(1,17,dtype=np.int64)*HOP
        events=[event("note_on",40,3),event("note_off",40,6)]
        audit=exact_active_breakdown(events,[],ends)
        self.assertEqual(audit["active_frames"],0)
        self.assertEqual(audit["silent_over"],3)
        self.assertEqual(audit["by_true_k"]["0"]["over"],3)

    def test_five_distinct_head_rescue_coverage_by_true_k(self):
        k=np.zeros((3,7),np.float32)
        k[:,0]=1
        h=np.repeat(k[:,:,None],6,axis=-1)
        h[1,:,1]=0
        h[1,2,1]=1
        y=np.array([0,2,3],np.int64)
        out=k_source_diagnostics(k,h,y)
        self.assertEqual(out["baseline_wrong_active"],2)
        self.assertEqual(out["baseline_wrong_with_any_alternative_right"],1)
        self.assertEqual(out["baseline_wrong_without_any_alternative_right"],1)
        self.assertEqual(out["by_true_k"]["2"]["H0_wrong_with_alternative"],1)
        with self.assertRaises(ValueError):
            k_source_diagnostics(k,h,np.array([1,2,7]))

    def test_acoustic_ready_when_k_still_blocks_true_pitch(self):
        ends=np.array([4*HOP,8*HOP],np.int64)
        post=np.zeros((2,49,3),np.float32)
        post[0,0,0]=.3
        post[0,0,1]=.7
        post[1,0,0]=.2
        post[1,0,1]=.7
        k=np.zeros((2,7),np.float32)
        k[:,0]=.98
        k[:,1]=.02
        active=np.zeros((2,49),bool)
        active[:,0]=True
        policy=GuidedPolicy(guide_confidence=.7,weak_onset=.25,
                            weak_sustain=.65,rise=.04,
                            count_confidence=1.1)
        result=trace_weak_k_gate(post,k,ends,policy,active)
        flags=result["flags"]
        self.assertEqual(flags["weak_eligible_but_k_no_room"],1)
        self.assertEqual(flags["weak_eligible_and_K_support"],0)

    def test_aggregate_by_real_true_k_no_neural_accuracy_confusion(self):
        frame=dict(active_frames=4,active_correct=3,
            under_active=1,over_active=0,silent_over=0,
            missing_pitch_frames=1,spurious_pitch_frames=0,
            unison_or_same_pitch_impossible_frames=0,
            by_true_k={str(k):dict(frames=int(k==1)*4,
                correct=int(k==1)*3,under=int(k==1),
                over=0) for k in range(7)})
        row=dict(native_frames=frame,
            k_proposals=dict(baseline_wrong_with_any_alternative_right=1,
                baseline_wrong_without_any_alternative_right=0),
            source_gate=dict(flags=dict(weak_eligible_but_k_no_room=1)))
        out=aggregate_forensics([row])
        self.assertAlmostEqual(out["active_exact_k"],.75)
        self.assertEqual(out["by_true_k"]["1"]["frames"],4)
        self.assertEqual(out["weak_onset_gate"]["weak_eligible_but_k_no_room"],1)


if __name__=="__main__":
    unittest.main()
