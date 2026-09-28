import unittest
import numpy as np

from scripts.audit_v273_learning_bottleneck import metrics


class LearningBottleneckTest(unittest.TestCase):
    def test_oracle_only_changes_the_diagnostic_ranking(self):
        p = np.array([[.7,.05,.2,.02,.01,.01,.01],
                      [.6,.1,.05,.2,.02,.02,.01],
                      [.1,.1,.5,.2,.04,.03,.03]])
        k = np.array([0,3,3])
        out = metrics(k,p,np.ones(7))
        self.assertEqual(out['correct'],1)
        self.assertEqual(out['poly_correct'],0)
        self.assertEqual(out['poly_predicted_nonpoly'],1)
        self.assertEqual(out['poly_correct_if_true_poly_known'],1)
        self.assertEqual(out['poly_top2_correct'],2)
        self.assertEqual(out['by_true_k']['6']['rows'],0)
        self.assertIsNone(out['by_true_k']['6']['exact'])

    def test_loss_weighting_is_reported_separately_from_decisions(self):
        p=np.eye(7)*.4+.6/7
        k=np.arange(7)
        a=metrics(k,p,np.ones(7))
        b=metrics(k,p,np.arange(1,8))
        self.assertEqual(a['confusion_true_by_predicted'],b['confusion_true_by_predicted'])
        self.assertAlmostEqual(b['weighted_nll'],a['nll']*4)

    def test_invalid_probabilities_fail(self):
        with self.assertRaises(AssertionError):
            metrics(np.array([2]),np.ones((1,7)),np.ones(7))


if __name__=='__main__':
    unittest.main()
