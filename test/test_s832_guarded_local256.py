"""S832: conservative local gate, no CAL-label dependencies in inference."""
import unittest
import numpy as np
from causal_note.s832_guarded_private_selector256 import (
    Rule,SCHEMA,MODES,THRESHOLDS,make_candidates,
    learn_rules,fit_guard,choose,signed_margin
)


def cal_rows(n=8):
    rows=[]
    for i in range(n):
        base=np.ones(128,dtype=np.uint8)
        truth=base.copy()
        truth[:100]=2
        alt=base.copy();alt[:110]=2  # +100 corrections -10 regressions
        proposal={("rescue",.40):alt}
        confidence=np.full(128,.8,np.float32)
        rows.append(dict(name=f"cal_{i:02}",base=base,k=truth,
             confidence=confidence,candidates=proposal))
    return rows


class S832GateTests(unittest.TestCase):
    def test_wilson_lower_bound_improves_with_evidence(self):
        self.assertLess(signed_margin(1,0),signed_margin(80,0))
        self.assertLess(signed_margin(40,40),.50)

    def test_rule_learning_and_calibration_holdout(self):
        rows=cal_rows()
        rules,all_rules=learn_rules(rows,params=(64,1.35,.55,10))
        self.assertTrue(rules)
        best,proof=fit_guard(rows)
        self.assertTrue(best,proof)
        self.assertTrue(proof["guard_enabled"])
        self.assertIn("params",proof)
        b=rows[0]["base"]
        output=choose(b,rows[0]["confidence"],rows[0]["candidates"],best)
        self.assertTrue(np.array_equal(output[:100],np.full(100,2,np.uint8)))
        self.assertEqual(output[-1],1)

    def test_always_safe_fallback_on_bad_calibration(self):
        bad=cal_rows()
        for r in bad:r["k"][:]=1
        policy,proof=fit_guard(bad)
        self.assertFalse(policy)
        self.assertEqual(proof["reason"],"validation_veto")
        out=choose(bad[0]["base"],bad[0]["confidence"],
                   bad[0]["candidates"],policy)
        np.testing.assert_array_equal(out,bad[0]["base"])

    def test_inference_never_reads_gt(self):
        rows=cal_rows()
        policy,_=fit_guard(rows)
        row=rows[0]
        expected=choose(row["base"],row["confidence"],row["candidates"],policy)
        row["k"][:]=6
        actual=choose(row["base"],row["confidence"],row["candidates"],policy)
        np.testing.assert_array_equal(expected,actual)

    def test_all_20_private_candidates_are_causal_shape(self):
        rng=np.random.default_rng(832)
        post=rng.uniform(0,1,(38,49,3)).astype(np.float32)
        base=rng.integers(0,7,size=38,dtype=np.uint8)
        confidence,proposed=make_candidates(base,post)
        self.assertEqual(len(proposed),len(MODES)*len(THRESHOLDS))
        self.assertEqual(confidence.shape,(38,))
        for x in proposed.values():
            self.assertEqual(x.shape,base.shape)
            self.assertTrue(np.isin(x,range(7)).all())


if __name__=="__main__":
    unittest.main()
