"""S715 tests: frozen original baseline, shadow-only K tuning and six locks."""
import unittest
from unittest.mock import patch

from causal_note.s708_cardinality_selection import GuidedPolicy
from scripts.s704_joint_metrics import SCORES
from scripts.s715_shadow_k_calibration import variants, select_cal_only, emitted_for


class S715KCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.frozen=GuidedPolicy(onset=.75,sustain_on=.5,sustain_off=.3,
            release=.85,release_steps=1,count_confidence=1.1,blend=.7,
            guide_confidence=.5,weak_onset=.15,weak_sustain=.55,
            rise=.04,trim=False)
        self.options=variants(self.frozen)

    def test_no_frozen_policy_mutation_and_exact_grid_budget(self):
        self.assertIs(self.options["S708_frozen"],self.frozen)
        self.assertEqual(len(self.options),16)
        self.assertEqual(self.frozen.count_confidence,1.1)
        self.assertTrue(any(p.count_confidence<1 for p in self.options.values()))
        for policy in self.options.values():
            self.assertEqual(policy.trim,False)
            self.assertEqual(policy.onset,.75)
            self.assertEqual(policy.release,.85)

    def test_offset_loss_blocks_improved_k_and_onset(self):
        better="kconf0.55_guide0.50_sustain0.45"
        self.assertIn(better,self.options)
        base=dict.fromkeys(SCORES,.5)
        scores={"S708_frozen":dict(base),better:dict.fromkeys(SCORES,.8)}
        paired={"S708_frozen":dict(lost_long=0,lost_correct_offsets=0),
                better:dict(lost_long=0,lost_correct_offsets=1)}
        best,eligible=select_cal_only(scores,scores,paired)
        self.assertEqual(best,"S708_frozen")
        self.assertEqual(eligible,["S708_frozen"])

    def test_any_one_of_six_regressions_blocks_candidate(self):
        alt="kconf0.55_guide0.50_sustain0.45"
        base=dict.fromkeys(SCORES,.5)
        changed=dict.fromkeys(SCORES,.8)
        changed["active_exact_k"]=.499
        scores={"S708_frozen":base,alt:changed}
        paired={x:dict(lost_long=0,lost_correct_offsets=0) for x in scores}
        best,eligible=select_cal_only(scores,scores,paired)
        self.assertEqual(best,"S708_frozen")

    def test_true_six_metric_improvement_can_be_frozen_on_cal(self):
        alt="kconf0.55_guide0.50_sustain0.45"
        scores={"S708_frozen":dict.fromkeys(SCORES,.5),
                alt:dict.fromkeys(SCORES,.55)}
        paired={x:dict(lost_long=0,lost_correct_offsets=0) for x in scores}
        best,_=select_cal_only(scores,scores,paired)
        self.assertEqual(best,alt)

    def test_shadow_emits_one_actual_note_stream_with_frozen_release(self):
        item=dict(ends=[1024,2048],samples=2048)
        events=[{"type":"note_on","pitch":52,"decision_sample":1024}]
        released=[{"type":"note_on","pitch":52,"decision_sample":1024},
                  {"type":"note_off","pitch":52,"decision_sample":2048}]
        policy=self.frozen
        with patch("scripts.s715_shadow_k_calibration.evaluate_policy",return_value=events) as decode, \
             patch("scripts.s715_shadow_k_calibration.prepare_hazards",return_value={"notes":[]}) as prep, \
             patch("scripts.s715_shadow_k_calibration.render_release",return_value=(released,0)) as render:
            found,skipped=emitted_for(item,"s708_selected.pt",policy,object(),
                                      dict(gate=.8,persistence=1,grace=24))
        self.assertEqual(found,released)
        self.assertEqual(skipped,0)
        decode.assert_called_once()
        prep.assert_called_once()
        render.assert_called_once()


if __name__=="__main__":
    unittest.main()
