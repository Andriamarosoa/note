"""S639 root-cause categories and frozen K-guard semantics."""
from __future__ import annotations
import unittest
from dataclasses import replace
import numpy as np

from causal_note.strict_live import CausalSelection,Selection
from scripts.s639_bass_fp_forensic import classify_false_note,active_true_pitches
from scripts.s637_guitarset_three_view_groundtruth import corrected


def truth(pitch,start,length):
    return dict(pitch=pitch,time=start,duration=length)


def evt(pitch,t):
    return dict(pitch=pitch,time=t)


class ForensicTests(unittest.TestCase):
    def test_true_sustain_retrigger_and_octave_alias_are_distinct(self):
        annotations=[truth(52,.0,1.),truth(67,2.,.5)]
        same=classify_false_note(evt(52,.4),annotations,prior_predicted_gap_s=.08)
        octave=classify_false_note(evt(64,.4),annotations)
        third=classify_false_note(evt(71,.4),annotations)
        self.assertEqual(same["reason"],"sustain_retrigger_suspect")
        self.assertEqual(octave["reason"],"octave_alias_suspect")
        self.assertEqual(third["reason"],"higher_harmonic_alias_suspect")
        self.assertEqual(octave["active_truth_polyphony"],1)
        self.assertEqual(active_true_pitches(annotations,2.2),{67})

    def test_cross_pitch_onset_leak_and_background(self):
        notes=[truth(52,1.,.3)]
        self.assertEqual(classify_false_note(evt(54,1.),notes)["reason"],
                         "cross_pitch_onset_leakage")
        self.assertEqual(classify_false_note(evt(54,.3),notes)["reason"],
                         "outside_annotated_note_activity")

    def test_original_K_veto_explicitly_disabled_at_1_point_1(self):
        original=Selection(count_confidence=1.1)
        active=CausalSelection(original)
        k=np.zeros(7,np.float32);k[0]=1.0
        p=np.zeros((49,3),np.float32)
        p[12,0]=.75;p[12,1]=.85
        orig=active.step(p,k,256,256)
        self.assertEqual(len(orig),1)
        self.assertEqual(orig[0]["type"],"note_on")
        enforced=CausalSelection(replace(original,count_confidence=.35))
        blocked=enforced.step(p,k,256,256)
        self.assertEqual(len(blocked),0)

    def test_threshold_only_promotes_unarmed_native_attack(self):
        state=CausalSelection(Selection(onset=.6))
        native=np.zeros((49,3),np.float32)
        bass=np.zeros_like(native);bass[12,:2]=[.9,.9]
        k=np.zeros(7,np.float32);k[0]=1
        out,count=corrected(native,bass,np.zeros_like(bass),k,state,
                            ("bass_080",.8,None,False,0.),True)
        self.assertEqual(count,1)
        self.assertEqual(float(out[12,0]),.6)
        self.assertEqual(float(native[12,0]),0.)
        self.assertEqual(float(out[12,1]),.9)


if __name__=="__main__":
    unittest.main()
