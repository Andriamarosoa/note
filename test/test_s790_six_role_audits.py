"""S790 six independent report auditors reject false model improvements."""
import unittest
from scripts.s790_six_role_audits import score,check,ROLES

class S790AuditTests(unittest.TestCase):
    def test_six_roles_required(self):
        self.assertEqual(set(ROLES),
           {"data","acoustics","architecture","training","metrics","live"})

    def test_correct_frame_pitch_f1(self):
        r=dict(TP=8,FP=1,FN=3,
               precision=8/9,recall=8/11,f1=16/20)
        self.assertEqual(score(r,"synthetic")["TP"],8)
        r["f1"]=.99
        with self.assertRaises(AssertionError):score(r,"tampered")

    def test_unmeasured_and_invalid_counters_blocked(self):
        r=dict(TP=-1,FP=1,FN=3,
               precision=0,recall=0,f1=0)
        with self.assertRaises(AssertionError):score(r,"malformed")

    def test_missing_evidence_blocks(self):
        with self.assertRaises(AssertionError):check(False,"no report")

if __name__=="__main__":
    unittest.main()
