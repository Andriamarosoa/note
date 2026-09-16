from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from causal_note import decay_novelty as decay
from causal_note.v280_causal_cqt import CausalCQTConfig, causal_cqt, cluster_feature_map
from scripts import test_v273_decay_internal as trial


def exponential_log(batch=1, bins=238, rate=4.):
    t = np.arange(24)[None, :, None] * decay.HOP_SECONDS
    amplitude = np.broadcast_to(.1 * np.exp(-rate * t), (batch, 24, bins)).copy()
    return np.log1p(100 * amplitude)


class DecayFeatureTests(unittest.TestCase):
    def test_exact_exponential_has_no_novel_attack_and_recovers_decay_rate(self):
        novelty, audit = decay.estimate_decay(exponential_log(rate=4.))
        np.testing.assert_allclose(audit['slope'], -4., atol=1e-10)
        self.assertTrue(audit['reliable'].all())
        self.assertLess(float(novelty.max()), 1e-10)

    def test_weak_new_attack_detected_while_total_amplitude_keeps_falling(self):
        log = exponential_log(rate=20.)
        amplitude = np.expm1(log) / 100
        amplitude[:, 17:] += .001
        changed = np.log1p(100 * amplitude)
        self.assertLess(float(changed[0, 17, 0]), float(changed[0, 16, 0]))
        novelty, audit = decay.estimate_decay(changed)
        self.assertGreater(float(novelty[0, 0, 0]), .03)
        np.testing.assert_allclose(audit['slope'], -20., atol=1e-9)

    def test_later_frames_cannot_change_precluster_estimator(self):
        original = exponential_log()
        modified = original.copy()
        modified[:, 17:] = 8
        _, first = decay.estimate_decay(original)
        _, second = decay.estimate_decay(modified)
        for name in ('slope', 'intercept', 'log_rmse', 'reliable', 'predicted_log'):
            np.testing.assert_array_equal(first[name], second[name])

    def test_silence_short_history_and_growth_do_not_invent_decay(self):
        for log in (np.zeros((1, 24, 238)), exponential_log(rate=-4.)):
            novelty, audit = decay.estimate_decay(log)
            self.assertTrue(np.isfinite(novelty).all())
            self.assertFalse(audit['reliable'].any())
            self.assertEqual(float(novelty.max()), 0.)
        short = exponential_log()
        short[:, :10] = 0
        self.assertFalse(decay.estimate_decay(short)[1]['reliable'].any())
        with self.assertRaises(ValueError):
            decay.estimate_decay(np.full((1, 24, 238), np.nan))

    def test_audio_after_decision_cannot_change_decay_features(self):
        config = CausalCQTConfig()
        rng = np.random.default_rng(9)
        audio = rng.normal(0, .01, 8000).astype(np.float32)
        start = 4000
        decision = start + config.cluster_post_samples
        modified = audio.copy()
        modified[decision:] = rng.normal(0, .5, len(audio)-decision)
        outputs = []
        for signal in (audio, modified):
            cache = causal_cqt(signal, config)
            crop = cluster_feature_map(cache, start, config)
            outputs.append(decay.summarize_crop(crop[None]))
        for a, b in zip(outputs[0][:2], outputs[1][:2]):
            np.testing.assert_array_equal(a, b)


class PairedDiagnosticTests(unittest.TestCase):
    def test_adjacent_target_and_valid_decode_keep_on_ties(self):
        k = np.array([0, 2, 1, 4, 0])
        base = np.array([0, 1, 2, 1, 6])
        np.testing.assert_array_equal(trial.target_actions(k, base), [0, 2, 1, 0, 0])
        p = np.full((3, 3), 1/3)
        np.testing.assert_array_equal(trial.decode(p, np.array([0, 3, 6])), [0, 3, 6])
        p = np.array([[.1, .8, .1], [.1, .1, .8]])
        np.testing.assert_array_equal(trial.decode(p, np.array([0, 6])), [0, 6])

    def test_scaling_and_fit_are_independent_of_held_features_and_targets(self):
        rng = np.random.default_rng(44)
        x = rng.normal(size=(120, 5))
        y = np.arange(120) % 3
        p1, state1 = trial.fit_predict(x, y, np.zeros((9, 5)))
        _, state2 = trial.fit_predict(x, y, np.full((9, 5), 1000))
        for name in state1:
            np.testing.assert_array_equal(state1[name], state2[name])
        np.testing.assert_allclose(state1['mean'], x.mean(axis=0))
        self.assertEqual(p1.shape, (9, 3))
        np.testing.assert_allclose(p1, trial.predict_saved(state1, np.zeros((9, 5))), atol=1e-12)

    def test_zero_decay_signal_gives_exactly_equal_arms(self):
        rng = np.random.default_rng(10)
        table = {'ordinary': rng.normal(size=(60, 4)), 'decay': np.zeros((60, 2))}
        base = np.arange(60) % 7
        specialist = np.full((60, 5), .2)
        a = trial.design(table, base, specialist, 'control')
        b = trial.design(table, base, specialist, 'decay')
        np.testing.assert_array_equal(a, b)
        labels = np.arange(60) % 3
        np.testing.assert_array_equal(trial.fit_predict(a[:45], labels[:45], a[45:])[0],
                                      trial.fit_predict(b[:45], labels[:45], b[45:])[0])

    def test_bootstrap_uses_whole_compositions_and_zero_delta_is_zero(self):
        groups = np.repeat(trial.GROUPS, 3)
        k = np.tile([1, 2, 3], 4)
        base = np.tile([1, 1, 2], 4)
        result = trial.bootstrap(groups, k, base, base)
        self.assertEqual((result['lower_95_pp'], result['upper_95_pp']), (0, 0))
        improved = trial.bootstrap(groups, k, base, k)
        self.assertEqual((improved['lower_95_pp'], improved['upper_95_pp']), (100, 100))

    def test_any_failed_frozen_gate_keeps_v273_and_never_launches_more_training(self):
        base = {'rows': 1000, 'poly_rows': 100, 'correct': 800, 'poly_correct': 40, 'false_poly': 20}
        candidate = {**base, 'poly_correct': 45, 'correct': 805}
        stats = {'reference': base, 'control': base, 'decay': candidate}
        intervals = {arm: {'lower_95_pp': .1} for arm in ('reference', 'control')}
        by_group = {g: stats for g in trial.GROUPS}
        self.assertTrue(trial.decide(stats, intervals, by_group)['passed'])
        for change in ({'poly_correct': 41}, {'correct': 799}, {'false_poly': 21}):
            altered = {**stats, 'decay': {**candidate, **change}}
            decision = trial.decide(altered, intervals, by_group)
            self.assertFalse(decision['passed'])
            self.assertEqual(decision['official_reference'], 'V27.3')
            self.assertFalse(decision['automatic_next_training'])
        self.assertFalse(trial.decide(stats, {arm: {'lower_95_pp': 0} for arm in intervals}, by_group)['passed'])

    def test_end_to_end_group_exclusion_only_selected_crops_and_saved_replay(self):
        rng = np.random.default_rng(273)
        n = 96
        groups = np.repeat(trial.GROUPS, n//4)
        rows = np.arange(n) * 2 + 1
        base = np.tile(np.arange(6), n//6).astype(np.int16)
        truth = np.clip(base + np.tile([-1, 0, 1], n//3), 0, 6).astype(np.int16)
        members = np.array([f'00_{g}_comp.jams' for g in groups])
        ref = {'pred273_selective_transition': base, 'k': truth, 'member': members,
               'specialist_probability': np.full((n, 5), .2)}
        accessed = []
        raw = rng.uniform(.1, 1, (2*n, 24, 238, 3)).astype(np.float16)
        class Features:
            def __getitem__(self, selected):
                accessed.extend(selected.tolist())
                return raw[selected]
        with TemporaryDirectory() as temp:
            out = Path(temp)/'result'
            with patch.object(trial.frozen, 'load_data', return_value=(Features(), {}, {})), \
                 patch.object(trial, 'reference', return_value=(rows, ref, groups)):
                trial.run(SimpleNamespace(prepared_dir=Path(temp), reference_dir=Path(temp), output_dir=out))
            self.assertEqual(accessed, rows.tolist())
            report = json.loads((out/'comparison.json').read_text())
            self.assertEqual(report['status'], 'complete')
            self.assertEqual(report['metrics']['decay']['rows'], n)
            audit = json.loads((out/'audit.json').read_text())
            for group, fold in audit['folds'].items():
                self.assertNotIn(group, fold['fit_groups'])
                self.assertEqual(len(fold['fit_groups']), 3)
                self.assertEqual(fold['arms']['control']['features'], fold['arms']['decay']['features'])
            with np.load(out/'predictions.npz') as saved:
                for arm in ('control', 'decay'):
                    np.testing.assert_array_equal(saved['pred_'+arm], trial.decode(saved['action_probability_'+arm], base))
                    with np.load(out/'features.npz') as features:
                        x = trial.design(features, base, ref['specialist_probability'], arm)
                    for number, group in enumerate(trial.GROUPS):
                        held = groups == group
                        with np.load(out/'models'/f'fold-{number}-{arm}.npz') as state:
                            p = trial.predict_saved(state, x[held])
                        np.testing.assert_allclose(p, saved['action_probability_'+arm][held], atol=1e-12)
            self.assertEqual(len(list((out/'models').glob('*.npz'))), 8)


if __name__ == '__main__':
    unittest.main()
