"""S714 independent tests for cardinality audit; cannot change S708 scoring."""
import unittest
import numpy as np
from causal_note.s708_cardinality_selection import GuidedPolicy, GuidedSelector
from scripts.s714_audit_s708_cardinality import (
    HOP, SR, events_to_frames, annotation_frames, one_item,
)


def e(typ, sample, pitch=52):
    return dict(type=typ,pitch=pitch,decision_sample=sample,
                available_sample=sample,time=sample/SR,reason="test",velocity=0)


class S714EventAuditTests(unittest.TestCase):
    def test_one_real_note_two_emissions_are_correct_k(self):
        ends=np.arange(1,9)*HOP
        truth=[dict(pitch=52,time=2*HOP/SR,duration=3*HOP/SR)]
        events=[e("note_on",2*HOP),e("note_off",5*HOP)]
        target,k=annotation_frames(truth,ends)
        pred=events_to_frames(events,ends)
        np.testing.assert_array_equal(target,pred)
        self.assertEqual(int(k.sum()),3)

    def test_annotation_same_pitch_unison_cannot_be_a_single_midi_pitch(self):
        ends=np.arange(1,9)*HOP
        truth=[dict(pitch=52,time=2*HOP/SR,duration=3*HOP/SR)]*2
        pitch,k=annotation_frames(truth,ends)
        self.assertEqual(int((k==2).sum()),3)
        self.assertEqual(int((pitch.sum(axis=1)==1).sum()),3)
        row=one_item(dict(name="fixture_comp",samples=8*HOP,truth=truth,
                         events={variant:[e("note_on",2*HOP),
                                  e("note_off",5*HOP)] for variant in
                                 ("candidate","native","native_s675_causal")}))
        self.assertEqual(row["simultaneous_same_pitch_frames"],3)
        self.assertEqual(row["variants"]["candidate"]["by_true_k"]["2"]["under"],3)
        self.assertEqual(row["variants"]["candidate"]["unique_pitch_k_correct"],8)

    def test_harmonic_coincidence_is_proxy_not_causation(self):
        notes=[dict(pitch=40,time=HOP/SR,duration=5*HOP/SR)]
        events=[e("note_on",HOP,40),e("note_on",HOP,52),
                e("note_off",6*HOP,40),e("note_off",6*HOP,52)]
        row=one_item(dict(name="fixture_comp",samples=8*HOP,truth=notes,
                         events={n:events for n in
                                 ("candidate","native","native_s675_causal")}))
        trial=row["variants"]["candidate"]
        self.assertEqual(trial["harmonic_coincidence_fp"],5)
        self.assertEqual(trial["false_note_frames"],5)
        self.assertEqual(trial["over"],5)

    def test_s708_frozen_policy_does_not_strictly_enforce_k_zero(self):
        # Forensic characterization, NOT a request to change frozen S708.
        # The historical count_confidence 1.1 explicitly disables the
        # JointSelector's strong K budget, while K still guides weak attacks.
        policy=GuidedPolicy(onset=.75,sustain_on=.50,sustain_off=.30,
            release=.85,count_confidence=1.1,blend=.70,
            guide_confidence=.50,weak_onset=.15,weak_sustain=.55,
            rise=.04,trim=False)
        selector=GuidedSelector(policy)
        post=np.zeros((49,3),dtype=np.float32)
        post[[1,8],0]=.9
        post[[1,8],1]=.9
        k=np.zeros(7,dtype=np.float32)
        k[0]=.999
        k[1]=.001
        emitted=selector.step(post,k,4*HOP)
        self.assertEqual([e['pitch'] for e in emitted if e['type']=='note_on'],[41,48])
        self.assertEqual(int(selector.active.sum()),2)
        self.assertEqual(int(k.argmax()),0)

    def test_reject_invalid_stream_states(self):
        ends=np.arange(1,9)*HOP
        with self.assertRaisesRegex(ValueError,"Duplicate"):
            events_to_frames([e("note_on",HOP),e("note_on",2*HOP)],ends)
        with self.assertRaisesRegex(ValueError,"without NoteOn"):
            events_to_frames([e("note_off",HOP)],ends)
        record=e("note_on",3*HOP)
        record["available_sample"]=2*HOP
        with self.assertRaisesRegex(ValueError,"Backdated"):
            events_to_frames([record],ends)


if __name__=="__main__":
    unittest.main()
