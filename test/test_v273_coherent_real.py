import unittest
import numpy as np
from scripts.audit_v273_coherent_real import auc, measure, event_context
from scripts.probe_v273_coherent_decay import evidence, make_wave


class RealCoherentTest(unittest.TestCase):
    def test_auc_ties_and_composition_weights(self):
        score = np.array([0., 1., 1., 2., 3.])
        label = np.array([0, 1, 0, 1, 0], bool)
        weights = np.array([2, 2, 1, 1, 3])
        repeated = np.repeat(np.arange(5), weights)
        a = score[repeated][label[repeated]][:, None]
        b = score[repeated][~label[repeated]][None, :]
        expected = np.mean((a > b)+.5*(a == b))
        self.assertAlmostEqual(auc(score, label, weights), expected)
        self.assertEqual(auc([0, 1], [False, True]), 1.)
        self.assertIsNone(auc([0, 1], [False, False]))

    def test_audio_scores_match_frozen_synthetic_implementation(self):
        x = make_wave([110, 116.54], 1.2, noise_db=20, event='same_pitch', seed=7)
        old, _ = evidence(x); new = measure(x)
        for key, value in old.items():
            self.assertEqual(value, new[key])

    def test_ownership_and_forecast_horizon_are_distinct(self):
        groups = [np.array([1000, 2000]), np.array([4000])]
        events = np.array([[950, 1500], [1000, 2000], [2764, 3000], [3500, 4200]])
        a, b = event_context(groups, events)
        self.assertEqual((a['k'], a['owned_before'], a['owned_future'], a['owned_after']), (3, 1, 1, 1))
        self.assertEqual(a['future_births'], 1)
        self.assertEqual(b['k'], 1)


if __name__ == '__main__':
    unittest.main()
