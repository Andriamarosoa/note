"""S687 nine-composition policy must avoid false-positive-dominated rescue."""
import unittest
from scripts.s687_composition_robust_novelty import choose_train_only,POLICIES

def metric(tp,fp,k):
    return dict(tp=tp,fp=fp,K_frames_correct=k,
                onset_F1=2*tp/(2*tp+fp+500),
                decoded_exact_k_accuracy=k/2000)

class S687Tests(unittest.TestCase):
    def test_reject_false_attack_explosion(self):
        base=metric(900,100,1200)
        variants={"native":base}
        group={}
        for p in POLICIES[1:]:
            variants[p]=metric(920,180,1210)
        for i in range(9):
            group[f"c{i}"]={p:variants[p] for p in POLICIES}
        selected,ledger,eligible=choose_train_only(variants,group)
        self.assertEqual(selected,"native")
        self.assertEqual(eligible,[])
    def test_requires_multi_composition_consistency(self):
        baseline=metric(900,200,1200)
        gain=metric(925,220,1220)
        stats={p:gain for p in POLICIES};stats["native"]=baseline
        groups={f"c{i}":{p:stats[p] for p in POLICIES} for i in range(9)}
        result,details,passing=choose_train_only(stats,groups)
        self.assertNotEqual(result,"native")
        self.assertTrue(passing)

if __name__=="__main__":unittest.main()
