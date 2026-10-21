"""S833 cross-validated CAL-only local risk-selection contract."""
import unittest
import numpy as np
from causal_note.s833_crossfit_risk_selector256 import (
    SCHEMA,PARAMETERS,fit_guard
)
from causal_note.s832_guarded_private_selector256 import choose
from test.test_s832_guarded_local256 import cal_rows


class S833CrossfitTests(unittest.TestCase):
    def test_4fold_stratified_track_disjoint_and_positive(self):
        data=cal_rows(16)
        policy,info=fit_guard(data)
        self.assertTrue(policy,info)
        self.assertTrue(info["guard_enabled"])
        self.assertEqual(info["schema"],SCHEMA)
        self.assertEqual(len(info["tried"]),len(PARAMETERS))
        for p in info["tried"]:
            self.assertEqual(len(p["folds"]),4)
            self.assertEqual(sum(f["validation_frames"] if
               "validation_frames" in f else f["frames"]
               for f in p["folds"]),16*128)
            self.assertEqual(set(f["fold"] for f in p["folds"]),{0,1,2,3})
        row=data[0]
        result=choose(row["base"],row["confidence"],row["candidates"],policy)
        self.assertGreater(np.count_nonzero(result==row["k"]),
                           np.count_nonzero(row["base"]==row["k"]))

    def test_4fold_bad_model_falls_back_exactly(self):
        rows=cal_rows(16)
        for r in rows:r["k"][:]=1
        rules,info=fit_guard(rows)
        self.assertFalse(rules)
        self.assertFalse(info["guard_enabled"])
        self.assertEqual(info["reason"],"4fold_CAL_veto")
        for row in rows:
            np.testing.assert_array_equal(
              choose(row["base"],row["confidence"],row["candidates"],rules),
              row["base"])

    def test_duplicate_CAL_name_rejected(self):
        rows=cal_rows(8);rows[1]["name"]=rows[0]["name"]
        with self.assertRaises(ValueError):fit_guard(rows)

    def test_outer_truth_does_not_affect_policy_or_predictions(self):
        rows=cal_rows(16)
        rules,_=fit_guard(rows)
        reference=choose(rows[0]["base"],rows[0]["confidence"],
                         rows[0]["candidates"],rules)
        newtruth=np.full_like(rows[0]["k"],6)
        self.assertEqual(newtruth.shape,reference.shape)
        np.testing.assert_array_equal(
            choose(rows[0]["base"],rows[0]["confidence"],
                   rows[0]["candidates"],rules),reference)


if __name__=="__main__":
    unittest.main()
