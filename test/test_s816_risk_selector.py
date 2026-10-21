"""S816 pre-registered selection invariants; synthetic only, not science."""
import unittest
from causal_note.s815_activek_teacher import SELECTORS,SHIFTS
from scripts.s816_reselect_activek256 import choose_policy, PENALTY_OVERS,MAX_CAL_ACC_LOSS

def table():
    rows=[dict(selector=s,shift=shift,correct=60,frames=100,overcount=10)
          for s in SELECTORS for shift in SHIFTS]
    for r in rows:
        if (r["selector"],r["shift"])==("direct",0):
            r.update(correct=65,overcount=20)
        if (r["selector"],r["shift"])==("regression",0):
            r.update(correct=64,overcount=2)
        if (r["selector"],r["shift"])==("pitch_sum",0):
            r.update(correct=59,overcount=0)
    return rows

class S816RiskSelectorTest(unittest.TestCase):
    def test_penalty_trades_small_cal_accuracy_for_lower_overcounts(self):
        a=choose_policy(table())
        self.assertEqual((a["selector"],a["shift"]),("regression",0))
        self.assertEqual(a["calibration_best_accuracy"],.65)
        self.assertAlmostEqual(a["calibration_selected"]["accuracy"],.64)
        self.assertEqual(a["n_eligible"],2)
        self.assertEqual(a["penalty_overs"],PENALTY_OVERS)
        self.assertEqual(a["max_cal_accuracy_loss"],MAX_CAL_ACC_LOSS)

    def test_zero_penalty_selects_highest_calibration_accuracy(self):
        a=choose_policy(table(),penalty=0)
        self.assertEqual((a["selector"],a["shift"]),("direct",0))

    def test_no_outer_truth_or_predictions_accepted(self):
        with self.assertRaises(TypeError):
            choose_policy(table(),outer_labels=[1,2])
        with self.assertRaises(TypeError):
            choose_policy(table(),full_note_predictions=[0])

    def test_guard_refuses_to_sacrifice_many_correct(self):
        a=choose_policy(table(),max_loss=0)
        self.assertEqual((a["selector"],a["shift"]),("direct",0))

    def test_fail_closed_on_partial_wrong_or_duplicate_calibration(self):
        with self.assertRaisesRegex(ValueError,"full fifteen"):
            choose_policy(table()[:-1])
        x=table();x[0]=dict(x[1])
        with self.assertRaisesRegex(ValueError,"duplicate"):
            choose_policy(x)
        x=table();x[3]["frames"]=101
        with self.assertRaisesRegex(ValueError,"same inner"):
            choose_policy(x)
        x=table();x[1]["correct"]=80
        x[1]["overcount"]=30
        with self.assertRaisesRegex(ValueError,"invalid inner"):
            choose_policy(x)
        with self.assertRaisesRegex(ValueError,"invalid fixed"):
            choose_policy(table(),penalty=-1)

if __name__=="__main__":
    unittest.main()
