"""Frozen selector counterexample fixtures for S718 onset error audit."""
import unittest
from unittest.mock import patch
import numpy as np

from causal_note.s708_cardinality_selection import GuidedPolicy
from causal_note.strict_live import SR
from scripts.s718_onset_false_negative_audit import categorize_missing, instrument

PITCH=52
IX=PITCH-40
END=4096


def state(on=.0,sustain=.0,release=.0,armed=True,cooldown=1000000,
          active_count=0,ktarget=2,kconf=1.,rising=True,source=False):
    onset=np.zeros(49,np.float32);onset[IX]=on
    sus=np.zeros(49,np.float32);sus[IX]=sustain
    rel=np.zeros(49,np.float32);rel[IX]=release
    armed_arr=np.ones(49,bool);armed_arr[IX]=armed
    active_arr=np.zeros(49,bool)
    rising_arr=np.zeros(49,bool);rising_arr[IX]=rising
    ages=np.full(49,cooldown,np.int64)
    return dict(end=END,on=onset,sustain=sus,release=rel,
       armed=armed_arr,active=active_arr,since_last=ages,
       rising=rising_arr,k_confidence=kconf,k_target=ktarget,
       active_count=active_count,
       source_on_pitches={PITCH} if source else set())


def cause(x,source_matched=False):
    note=dict(pitch=PITCH,time=END/SR,duration=.5)
    return categorize_missing(note,[x],source_matched=source_matched)[0]


class S718GatingTests(unittest.TestCase):
    def test_neural_absent_and_sustain_gate(self):
        self.assertEqual(cause(state(on=.05,sustain=.9)),
                         "network_onset_below_weak_threshold")
        self.assertEqual(cause(state(on=.95,sustain=.2)),
                         "high_onset_but_sustain_or_state_insufficient")

    def test_strong_attack_rearm_and_cooldown(self):
        self.assertEqual(cause(state(on=.95,sustain=.9,armed=False)),
                         "selector_rearming_blocked")
        self.assertEqual(cause(state(on=.95,sustain=.9,cooldown=100)),
                         "selector_refractory_65ms")
        self.assertEqual(cause(state(on=.95,sustain=.9,active_count=6)),
                         "full_six_note_capacity")
        self.assertEqual(cause(state(on=.95,sustain=.9,active_count=0)),
                         "eligible_strong_not_emitted")

    def test_weak_acoustic_K_veto_and_rising(self):
        self.assertEqual(cause(state(on=.4,sustain=.7,ktarget=0)),
                         "weak_evidence_blocked_by_K_gate")
        self.assertEqual(cause(state(on=.4,sustain=.7,rising=False)),
                         "weak_attack_without_rising_evidence")

    def test_source_only_hit_is_downstream_drop(self):
        self.assertEqual(cause(state(on=.95,sustain=.9),True),
                         "source_onset_removed_downstream")
        self.assertEqual(cause(state(on=.95,sustain=.9,source=True)),
                         "source_predicted_but_match_conflict")

    def test_frozen_S708_step_trace_matches_original_emitted_stream(self):
        policy=GuidedPolicy(onset=.75,sustain_on=.5,sustain_off=.3,
            release=.85,release_steps=1,count_confidence=1.1,
            blend=.7,guide_confidence=.5,weak_onset=.15,
            weak_sustain=.55,rise=.04,trim=False)
        rng=np.random.default_rng(718)
        ends=np.arange(1,18)*1024
        p=rng.random((len(ends),49,3),dtype=np.float32)
        k=np.zeros((len(ends),7),np.float32);k[:,1]=1.
        native=rng.random((len(ends),49,3),dtype=np.float32)
        item=dict(predictions={"s708_selected.pt":(p,k)},
              native_post=native,decision_ends=ends,samples=int(ends[-1]))
        from causal_note.s708_cardinality_selection import decode_guided
        fused=policy.blend*p+(1-policy.blend)*native
        expected=decode_guided(ends,fused,k,policy,item["samples"])
        with patch("scripts.s718_onset_false_negative_audit.evaluate_policy",
                    return_value=expected):
            observed,trace=instrument(item,policy)
        self.assertEqual(observed,expected)
        self.assertEqual(len(trace),len(ends))
        self.assertTrue(all(s["end"]==int(ends[i]) for i,s in enumerate(trace)))

if __name__=="__main__":
    unittest.main()
