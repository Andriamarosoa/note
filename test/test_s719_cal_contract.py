"""S719 CAL-only selection no-regression unit contracts."""
import unittest
from scripts.s704_joint_metrics import SCORES
from scripts.s719_shadow_reattack_cal import variants,select_cal_only


class S719SelectionTests(unittest.TestCase):
    def test_baseline_is_always_protected_and_new_variants_are_shadow_only(self):
        opts=variants()
        self.assertEqual(len(opts),7)
        self.assertIsNone(opts["S708_frozen"])
        self.assertEqual({d["min_gap_ms"] for d in opts.values() if d},{90,150})
        self.assertEqual({d["rise"] for d in opts.values() if d},{.1,.2,.3})

    def test_no_single_metric_regression_even_for_larger_onset(self):
        b={k:.5 for k in SCORES}
        good={k:.6 for k in SCORES}
        good["offset_accuracy"]=.4999
        trial="reattack_rise0.20_gap150ms"
        s={"S708_frozen":b,trial:good}
        p={q:dict(lost_long=0,lost_correct_offsets=0) for q in s}
        name,qual=select_cal_only(s,p)
        self.assertEqual(name,"S708_frozen")
        self.assertEqual(qual,["S708_frozen"])

    def test_good_onset_with_lost_old_correct_offset_cannot_promote(self):
        trial="reattack_rise0.20_gap150ms"
        s={"S708_frozen":dict.fromkeys(SCORES,.5),
           trial:dict.fromkeys(SCORES,.7)}
        p={"S708_frozen":dict(lost_long=0,lost_correct_offsets=0),
           trial:dict(lost_long=0,lost_correct_offsets=1)}
        name,_=select_cal_only(s,p)
        self.assertEqual(name,"S708_frozen")

    def test_true_six_metric_win_can_be_frozen_on_cal(self):
        trial="reattack_rise0.20_gap150ms"
        s={"S708_frozen":dict.fromkeys(SCORES,.5),
           trial:dict.fromkeys(SCORES,.7)}
        p={q:dict(lost_long=0,lost_correct_offsets=0) for q in s}
        name,_=select_cal_only(s,p)
        self.assertEqual(name,trial)


if __name__=="__main__":
    unittest.main()
