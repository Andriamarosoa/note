import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from scripts import v273_native_protocol as protocol
from scripts import train_v273_native_paired as paired
from scripts import summarize_v273_native_paired as summary
from scripts import audit_v270_class_conditional_fusion as metrics
from scripts.rebuild_v273_sources import digest, write_json
from test import test_v273_decay_native_inputs as native_fixtures

CONFIG = Path(__file__).resolve().parents[1]/'analysis/v273-native-paired-config.json'


class PairedProtocolTests(unittest.TestCase):
    def test_compositions_stay_in_historical_folds_when_cluster_loads_change(self):
        cfg = protocol.load_config(CONFIG)
        members = list(cfg['member_folds'])
        tracks = [SimpleNamespace(annotation_member=m) for m in members]
        cache = {'members': np.array(members), 'track_members': members}
        first = protocol.frozen_group_folds(cache, tracks, CONFIG)
        cache['members'] = np.array(members + [members[0]]*500)
        second = protocol.frozen_group_folds(cache, tracks, CONFIG)
        self.assertEqual(first[:2], second[:2])
        self.assertNotEqual(first[2], second[2])
        self.assertEqual(len(first[0]), 24)
        with self.assertRaisesRegex(RuntimeError, 'outside frozen'):
            protocol.frozen_group_folds({**cache, 'members': ['05_invalid.jams']}, tracks, CONFIG)

    def test_manifest_rejects_a_composition_split_between_folds(self):
        cfg = protocol.load_config(CONFIG)
        first = next(iter(cfg['member_folds']))
        cfg['member_folds'][first] = (cfg['member_folds'][first]+1) % 5
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)/'config.json'
            write_json(p, cfg)
            with self.assertRaisesRegex(RuntimeError, 'composition split'):
                protocol.load_config(p)

    def test_bounded_feature_extraction_matches_direct_computation(self):
        crop = np.tile(native_fixtures.exponential_crop(), (5, 1, 1, 1)).astype(np.float16)
        crop[:, paired.native.PRE_FRAMES:, 3, 0] += .2
        batched, audit = paired.evidence_table(crop, batch_size=3)
        direct, _ = paired.native.native_decay_features(crop)
        np.testing.assert_array_equal(batched, direct)
        self.assertEqual(audit['nonzero_rows'], len(crop))

    def test_epoch_order_is_repeatable_changes_by_epoch_and_preserves_all_rows(self):
        idx = np.arange(100)*3
        np.testing.assert_array_equal(paired.epoch_order(idx, 71, 0), paired.epoch_order(idx, 71, 0))
        self.assertFalse(np.array_equal(paired.epoch_order(idx, 71, 0), paired.epoch_order(idx, 71, 1)))
        np.testing.assert_array_equal(np.sort(paired.epoch_order(idx, 71, 1)), idx)

    def test_summary_uses_paired_denominators_and_rejects_duplicate_coverage(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cfg = protocol.load_config(CONFIG)
            members = np.array(sorted(cfg['member_folds']))
            k = np.arange(len(members)) % 3
            cfg.update(expected_rows=len(members), expected_poly_rows=int(np.sum(k >= 2)))
            config = root/'config.json'
            write_json(config, cfg)
            for fold in range(5):
                idx = np.flatnonzero([cfg['member_folds'][str(m)] == fold for m in members])
                control = k[idx].copy()
                control[k[idx] == 2] = 1
                decay = k[idx].copy()
                report = {'experiment': 'v273_native_decay_paired_rebuilt_sources',
                          'fold': fold, 'config_sha256': digest(config),
                          'identical_pairing_verified': True, 'outer_labels_used_for_selection': False,
                          'arms': {}}
                for arm, pred in (('control', control), ('decay', decay)):
                    report['arms'][arm] = {'cardinality': metrics.cardinality_report(k[idx], pred),
                                          'event_50ms': {'true_positive': 10, 'false_positive': 1, 'false_negative': 2}}
                write_json(root/f'report-fold-{fold}.json', report)
                np.savez_compressed(root/f'predictions-fold-{fold}.npz', global_index=idx,
                                    member=members[idx], k=k[idx], control=control, decay=decay,
                                    outer_fold=np.full(len(idx), fold))
            with contextlib.redirect_stdout(io.StringIO()):
                result = summary.summarize(root, config, root/'result')
            self.assertEqual(result['paired_delta_poly_percentage_points'], 100)
            self.assertFalse(result['archived_reference_comparison_is_paired'])
            self.assertFalse(result['automatic_promotion'])
            p = root/'predictions-fold-0.npz'
            with np.load(p) as z:
                values = {key: z[key] for key in z.files}
            values['global_index'][0] = values['global_index'][1]
            np.savez_compressed(p, **values)
            with self.assertRaisesRegex(RuntimeError, 'exactly once'):
                summary.summarize(root, config, root/'bad')


@unittest.skipUnless(importlib.util.find_spec('tensorflow'), 'requires TensorFlow 2.15.1')
class PairedBatchTrainingTests(unittest.TestCase):
    def test_real_keras_sequence_keeps_zero_evidence_arms_identical_after_training(self):
        import tensorflow as tf
        self.assertEqual(tf.__version__, '2.15.1')
        hashes = []
        for arm in paired.native.ARMS:
            model, original = paired.native.build_count_model('v260_uniform', 713)
            inputs = native_fixtures.NativeNetworkParityTests.inputs(original)
            cache = {key: np.tile(inputs[name], (4, *([1]*(inputs[name].ndim-1))))
                     for key, name in [('sequence', 'candidate_set'), ('mask', 'candidate_mask'),
                                       ('stats', 'cluster_stats'), ('spectral', 'spectral_map')]}
            idx = np.arange(12)
            k = idx % 7
            seq = paired.make_sequence(cache, idx, np.zeros((12,128), np.float32), arm,
                                       k=k, table=np.ones(7, np.float32), seed=713,
                                       shuffle=True, batch_size=4)
            hashes.append(paired.weights_digest(model))
            history = model.fit(seq, epochs=2, shuffle=False, workers=0,
                                max_queue_size=1, verbose=0)
            self.assertEqual(len(history.history['loss']), 2)
            hashes.append(paired.weights_digest(model))
        self.assertEqual(hashes[0], hashes[2])
        self.assertEqual(hashes[1], hashes[3])


if __name__ == '__main__':
    unittest.main()
