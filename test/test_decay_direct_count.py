from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from scripts import test_decay_direct_count as trial


class DirectCountTests(unittest.TestCase):
    def test_learns_all_seven_counts_directly_from_acoustics(self):
        truth = np.tile(np.arange(7), 30)
        x = np.eye(7)[truth]
        probability, state = trial.fit_count(x, truth, np.eye(7))
        self.assertEqual(probability.shape, (7, 7))
        np.testing.assert_array_equal(probability.argmax(axis=1), np.arange(7))
        np.testing.assert_allclose(trial.predict_saved(state, np.eye(7)), probability)

    def test_missing_and_single_classes_have_valid_seven_way_outputs(self):
        for classes in ([0, 6], [2, 4, 6], [5]):
            truth = np.tile(classes, 15)
            x = np.eye(7)[truth]
            probability, state = trial.fit_count(x, truth, np.eye(7))
            np.testing.assert_allclose(probability.sum(axis=1), 1)
            np.testing.assert_allclose(trial.predict_saved(state, np.eye(7)), probability)
            self.assertTrue((probability[:, [k for k in range(7) if k not in classes]] == 0).all())

    def test_held_acoustics_do_not_change_fit_scaling_or_coefficients(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=(140, 9))
        truth = np.tile(np.arange(7), 20)
        _, first = trial.fit_count(x, truth, np.zeros((7, 9)))
        _, second = trial.fit_count(x, truth, np.full((7, 9), 1000.))
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])
        np.testing.assert_allclose(first['mean'], x.mean(axis=0))

    def test_pair_excludes_whole_compositions_and_zero_signal_matches_control(self):
        groups = np.repeat(trial.GROUPS, 21)
        truth = np.tile(np.arange(7), 12)
        x = np.eye(7)[truth]
        novelty = np.zeros((len(truth), 2))
        with TemporaryDirectory() as temp:
            output = Path(temp) / 'models'
            p, audit = trial.fit_pair(x, novelty, truth, groups, output)
            self.assertEqual(len(audit), 8)
            np.testing.assert_array_equal(p['control'], p['decay'])
            for number, group in enumerate(trial.GROUPS):
                with np.load(output / f'fold-{number}-decay.npz') as state:
                    fit, held = state['fit_rows'], state['held_rows']
                    self.assertNotIn(group, set(groups[fit]))
                    self.assertEqual(set(groups[held]), {group})
                    np.testing.assert_allclose(state['mean'], np.concatenate([x, novelty], axis=1)[fit].mean(axis=0))
                    np.testing.assert_allclose(trial.predict_saved(state, np.concatenate([x, novelty], axis=1)[held]), p['decay'][held])


if __name__ == '__main__':
    unittest.main()
