"""S791 train-policy negative tests: independent domains, router, no masks bias."""
import unittest
import numpy as np
from scripts.s791_train_framepitch_complementary import (
    train_bank,head_predictions,totals,route,router_model)
from causal_note.s791_framepitch_expert_bank import ALL_MASKS

class S791TrainingTests(unittest.TestCase):
    def test_empty_domain_is_not_a_fake_trained_head(self):
        training=[]
        for comp in ("BN1","Jazz1"):
            x={name:np.zeros((0,40),np.float32)
               for name in ("general","bass","harmonic","transition")}
            y={name:np.empty(0,np.int8) for name in x}
            training.append(dict(composition=comp,sample_x=x,sample_y=y))
        bank=train_bank(training)
        self.assertEqual(len(bank),4)
        self.assertTrue(all(v is None for v in bank.values()))
        fake=np.zeros((3,49,40),np.float32)
        out=head_predictions(bank,fake)
        for prob in out.values():
            self.assertEqual(prob.shape,(3,49))
            self.assertTrue(np.all(prob==.5))

    def test_router_fallback_no_evidence(self):
        feat=np.zeros((6,19),np.float32)
        np.testing.assert_array_equal(route(None,feat,(0,1,2)),np.zeros(6))
        class Dummy:
            def predict(self,x):
                return np.array([0,1,2,3,4,5])[:len(x)]
        np.testing.assert_array_equal(
            route(Dummy(),feat,(0,2,4)),np.array([0,0,2,0,4,0]))

    def test_summary_produces_exact_counts(self):
        rows=[dict(stats=dict(tp=8,fp=1,fn=4)),
              dict(stats=dict(tp=2,fp=1,fn=1))]
        result=totals(rows,"stats")
        self.assertEqual((result["TP"],result["FP"],result["FN"]),(10,2,5))
        self.assertAlmostEqual(result["f1"],20/27)

if __name__=="__main__":
    unittest.main()
