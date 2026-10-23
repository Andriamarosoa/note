"""S837: fixed six-axis policy grid, score fidelity and exclusion security."""
from dataclasses import replace
import unittest
import numpy as np
from causal_note.s834_native256_decoders import AXES, Fragment, AxisDecision
from scripts.s837_six_axis_policy_loops256 import (
    Source, choices, new_decoder, score_axis, score, evaluate,
    select_across_compositions, audit_event_change, audit_count_change,
    original_for_axis, COMPS,
)


def source(comp):
    posts=[]
    for i in range(1,33):
        p=np.zeros((49,3),np.float32)
        p[0,0]=.9 if i==2 else (.05 if i<18 else .12)
        p[0,1]=.85 if 2<=i<18 else .04
        p[0,2]=.96 if i==18 else .02
        k=np.zeros(7,np.float32);k[1 if 2<=i<18 else 0]=1.
        posts.append(Fragment(i*256,p,k))
    src=Source(name=comp,composition=comp,snapshots=tuple(posts),
               truth=(dict(pitch=40,time=512/44100,duration=16*256/44100),),
               source_events=(),frame_ends=np.arange(256,33*256,256),
               baseline={},samples=32*256,sha256="synthetic")
    baseline={}
    for axis in AXES:
        d=new_decoder(axis,"reference",{})
        decisions=tuple(d.step(fr) for fr in src.snapshots)
        if axis=="full_note":
            ending=d.finish(src.samples)
            decisions=decisions+(AxisDecision("full_note",src.frame_ends[-1],
                                             intents=tuple(ending)),)
        baseline[axis]=score_axis(axis,decisions,src)
    return replace(src,baseline=baseline)


class IndependentLoopTest(unittest.TestCase):
    def test_all_axes_have_reference_and_at_least_one_distinct_candidate(self):
        for axis in AXES:
            group=choices(axis)
            self.assertGreater(len(group),1)
            self.assertEqual(group[0],("reference",{}))
            self.assertEqual(len({k for k,v in group}),len(group))

    def test_reference_decoder_replays_same_metric(self):
        src=source("BN1")
        for axis in AXES:
            self.assertEqual(evaluate(axis,"reference",{},src),src.baseline[axis],
                             axis)

    def test_leave_one_composition_out_selection_does_not_read_outer(self):
        arr=[source(c) for c in COMPS]
        axis="onset"
        results={(axis,label,s.name):evaluate(axis,label,params,s)
                 for label,params in choices(axis) for s in arr}
        before=select_across_compositions(axis,arr,results)
        # Change only BN1's candidate results and ensure BN1 selection
        # (calibrated from other four) remains exactly unchanged.
        scrambled=dict(results)
        for label,_ in choices(axis):
            scrambled[(axis,label,"BN1")]={"TP":0,"FP":100,"FN":100}
        after=select_across_compositions(axis,arr,scrambled)
        self.assertEqual(before["outer"]["BN1"]["selected_on_other_four"],
                         after["outer"]["BN1"]["selected_on_other_four"])

    def test_event_identity_postmortem_no_change(self):
        s=source("Rock3")
        for axis in ("onset","offset"):
            d=audit_event_change(axis,s,"reference","reference",{})
            self.assertEqual(d["recovered_gt"],0)
            self.assertEqual(d["lost_gt"],0)
            self.assertEqual(d["new_fp_events"],0)
            self.assertEqual(d["removed_fp_events"],0)

    def test_k_postmortem_explains_each_changed_frame_without_claiming_cause(self):
        s=source("BN1")
        for axis in ("exact_k","active_exact_k"):
            d=audit_count_change(axis,s,"reference",{})
            self.assertEqual(d["frames_changed"],0)
            self.assertEqual(d["recovered_correct"],0)
            self.assertEqual(d["lost_correct"],0)
            self.assertEqual(set(d["per_true_K"]),set(str(i) for i in range(7)))

    def test_onset_and_offset_count_keys_canonicalized_exactly(self):
        s=source("Rock3")
        for axis in ("onset","offset"):
            self.assertEqual(original_for_axis(axis,s),s.baseline[axis])

    def test_no_cross_axis_private_state_is_shared(self):
        first=source("SS1")
        copies=[]
        for axis in AXES:
            d=new_decoder(axis,"reference",{})
            copies.append(d)
            self.assertEqual(evaluate(axis,"reference",{},first),first.baseline[axis])
        self.assertEqual(len(set(map(id,copies))),6)


if __name__=="__main__":
    unittest.main()
