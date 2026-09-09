from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np

from scripts import train_v280_internal_ablation as e
from causal_note.v280_causal_cqt import CausalCQTConfig, causal_cqt, cluster_feature_map, write_track_cache
from test.test_v280_real_smoke import _member, _metadata, _track


class InternalSplitTests(unittest.TestCase):
    def test_canonical_folds_exclude_compositions_and_ignore_target_values(self):
        members = tuple(_member(player, f"piece{group:02d}", arrangement)
                        for group in range(12) for player in range(2)
                        for arrangement in ("comp", "solo"))
        metadata = _metadata(members)
        tracks = tuple(_track(member) for member in metadata.track_members)
        fold, split = e.split_rows(metadata, tracks)
        for group in set(e.group_stem(m) for m in members):
            self.assertEqual(len(set(fold[[e.group_stem(m) == group for m in members]])), 1)
        metadata.exact[:] = 6
        after, after_split = e.split_rows(metadata, tracks)
        np.testing.assert_array_equal(after, fold)
        self.assertEqual(split, after_split)
        self.assertEqual(sum(split["row_counts_per_fold"]), len(members))

    def test_epoch_visits_every_row_once_and_anchors_polyphonic_examples(self):
        groups = np.asarray(["a"] * 11 + ["b"] * 6 + ["c"] * 12)
        k = np.asarray([2] * 8 + [0] * 21)
        order = e.epoch_order(groups, k, epoch=1, batch_size=4)
        np.testing.assert_array_equal(np.sort(order), np.arange(len(k)))
        np.testing.assert_array_equal(order, e.epoch_order(groups, k, epoch=1, batch_size=4))
        self.assertFalse(np.array_equal(order, e.epoch_order(groups, k, epoch=2, batch_size=4)))
        for start in range(0, len(order), 4):
            self.assertTrue(np.any(k[order[start:start + 4]] >= 2))

    def test_sampler_retains_all_rows_when_polyphonic_anchors_are_scarce(self):
        k = np.asarray([0, 0, 0, 2, 0, 0, 0, 0, 0])
        order = e.epoch_order(np.asarray(["a"] * 9), k, epoch=1, batch_size=4)
        np.testing.assert_array_equal(np.sort(order), np.arange(len(k)))


class AuxiliaryTargetTests(unittest.TestCase):
    def test_ambiguous_and_out_of_grid_pitch_keep_count_supervision(self):
        strings = np.zeros((3, 6), dtype=np.float32)
        strings[:, 0] = 1
        midi = np.zeros_like(strings)
        midi[:, 0] = [40.2, 40.5, 65.0]
        targets, weights, diag = e.training_targets(np.ones(3), strings, midi, strings)
        np.testing.assert_array_equal(targets["cardinality"], [1, 1, 1])
        np.testing.assert_array_equal(weights["cardinality"], [1, 1, 1])
        np.testing.assert_array_equal(weights["string_birth"], [1, 1, 1])
        np.testing.assert_array_equal(weights["pitch_onset"], [1, 0, 0])
        self.assertEqual(targets["string_fret_onset"][0, 0, 0], 1)
        self.assertEqual(diag["count_rows_discarded"], 0)
        self.assertEqual(diag["pitch_auxiliary_rows_masked"], 2)

    def test_inconsistent_string_count_masks_auxiliary_losses_only(self):
        strings = np.zeros((1, 6), dtype=np.float32)
        strings[0, 0] = 1
        midi = np.zeros_like(strings)
        midi[0, 0] = 40
        targets, weights, _ = e.training_targets(np.asarray([2]), strings, midi, strings)
        self.assertEqual(targets["cardinality"][0], 2)
        self.assertEqual(weights["cardinality"][0], 1)
        for key in ("string_birth", "poibin_cardinality", "string_fret_onset", "pitch_onset"):
            self.assertEqual(weights[key][0], 0)

    def test_nonfinite_active_note_stops_preparation(self):
        strings = np.ones((1, 6), dtype=np.float32)
        with self.assertRaisesRegex(e.AblationError, "non-finite"):
            e.training_targets(np.asarray([6]), strings, strings * np.nan, strings)


class EndOfStreamTests(unittest.TestCase):
    def test_tail_flush_uses_only_available_frames_and_keeps_normal_crops_identical(self):
        config = CausalCQTConfig(sample_rate=4000, hop_length=64, bins_per_semitone=3,
            input_min_midi=60.0, output_max_midi=72.0, max_harmonic_order=3,
            cluster_post_samples=192, cluster_frames=8, block_frames=7)
        track = causal_cqt(np.random.default_rng(36).normal(0, .1, 1024).astype(np.float32), config)
        with TemporaryDirectory() as temp:
            root = Path(temp)
            write_track_cache(root / "track.npz", track, config)
            # Compare against the actual float16 cache as used in training.
            cached = e.read_track_cache(root / "track.npz", config)
            manifest = {"cache": {"tracks": [{"annotation_member": "track", "cache_path": "track.npz"}]}}
            features, diagnostic = e.track_features(root, manifest, "track",
                [np.asarray([500, 520]), np.asarray([900])], config)
            np.testing.assert_array_equal(features[0], cluster_feature_map(cached, 500, config))
            np.testing.assert_array_equal(features[1], cluster_feature_map(cached, 900, config, decision_end=1024))
            self.assertEqual(diagnostic["end_of_stream_clipped_rows"], 1)
            self.assertEqual(diagnostic["end_of_stream_clipped_local_indices"], [1])
            self.assertEqual(diagnostic["missing_post_samples_max"], 68)
            self.assertEqual(diagnostic["decision_end_max"], 1024)
            with self.assertRaisesRegex(e.AblationError, "outside"):
                e.track_features(root, manifest, "track", [np.asarray([1025])], config)


class EvaluationTests(unittest.TestCase):
    def test_metrics_and_checkpoint_selection_prioritize_polyphonic_accuracy(self):
        k = np.asarray([0, 1, 2, 2, 3, 6])
        p = np.eye(7)[[0, 0, 1, 2, 4, 6]] * .94 + .06 / 7
        metrics = e.count_metrics(k, p)
        self.assertEqual(metrics["poly_correct"], 2)
        self.assertEqual(metrics["poly_rows"], 4)
        self.assertEqual(metrics["poly_exact_k"], .5)
        self.assertEqual(metrics["correct"], 3)
        self.assertEqual(metrics["by_true_k"]["2"]["undercount"], 1)
        self.assertEqual(metrics["by_true_k"]["3"]["overcount"], 1)
        self.assertGreater(e.checkpoint_key({"poly_correct": 3, "poly_nll": 5}),
                           e.checkpoint_key(metrics))

    def test_bad_probabilities_are_not_silently_normalized(self):
        with self.assertRaisesRegex(e.AblationError, "normalized"):
            e.count_metrics(np.asarray([2]), np.ones((1, 7)))

    def test_changed_prepared_features_are_rejected(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "features.npy").write_bytes(b"changed")
            e.smoke._atomic_json(root / "manifest.json", {
                "status": "complete", "contract": e.CONTRACT, "sources": e.SOURCES,
                "files": {"features.npy": {"sha256": "incorrect"},
                          "labels.npz": {}, "split.json": {}},
            })
            with self.assertRaisesRegex(e.AblationError, "digest mismatch"):
                e.load_prepared(root)

    def test_artifact_comparison_recomputes_paired_scores_and_retains_negative_result(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            k = np.asarray([0, 1, 2, 3, 4, 6])
            for arm in e.ARMS:
                output = root / f"v280-e-{arm}"
                output.mkdir()
                predictions = [0, 1, 2, 2, 3, 5] if arm == "harmonic" else k
                probability = np.eye(7)[predictions] * .93 + .01
                metrics = e.count_metrics(k, probability)
                np.savez_compressed(output / "validation-predictions.npz", global_index=np.arange(6),
                    member=np.asarray(["member"] * 6), k=k, probability=probability)
                (output / "best.weights.h5").write_bytes(b"test-weight-artifact")
                e.smoke._atomic_json(output / "report.json", {
                    "status": "complete", "arm": arm, "contract": e.CONTRACT, "sources": e.SOURCES,
                    "epochs_completed": e.EPOCHS, "trainable_parameters": e.expected_parameter_count(harmonic=arm == "harmonic"),
                    "training_rows": 32, "validation_rows": 6, "optimizer_updates": e.EPOCHS,
                    "history": [{"epoch": i, "validation": metrics, "training_order_sha256": str(i)}
                                for i in range(1, e.EPOCHS + 1)],
                    "best_epoch": 1, "selected_validation": metrics,
                    "weights_sha256": e.smoke._sha256_file(output / "best.weights.h5"),
                    "predictions_sha256": e.smoke._sha256_file(output / "validation-predictions.npz"),
                    "prepared_manifest_sha256": "paired-data", "shared_initial_weights_sha256": "paired-init",
                    "source_head_sha": "paired-commit", "source_run_id": "paired-run",
                })
            e.compare(SimpleNamespace(input_dir=root, output_dir=root / "comparison"))
            result = json.loads((root / "comparison" / "comparison.json").read_text())
            self.assertEqual(result["internal_preference"], "no_harmonic")
            self.assertEqual(result["poly_correct_delta_rows"], -3)
            self.assertFalse(result["outer_evaluation_launched"])


@unittest.skipUnless(importlib.util.find_spec("tensorflow"), "TensorFlow checked by GitHub Actions")
class TensorFlowPairTests(unittest.TestCase):
    def test_shared_initialization_and_masked_auxiliary_gradient_step(self):
        harmonic, first = e.initialize_arm("harmonic")
        control, second = e.initialize_arm("no_harmonic")
        self.assertEqual(first, second)
        strings = np.zeros((3, 6), dtype=np.float32)
        strings[:, 0] = 1
        midi = strings * 40.25
        midi[1, 0] = 40.5
        targets, weights, _ = e.training_targets(np.asarray([1, 1, 2]), strings, midi, strings)
        weights["string_fret_onset"] = weights["string_fret_onset"][:, None]
        x = np.random.default_rng(35).uniform(0, 1, (3, *e.FEATURE_SHAPE)).astype(np.float32)
        for model in (harmonic, control):
            result = model.train_on_batch(x, targets, sample_weight=weights, return_dict=True)
            self.assertTrue(np.isfinite(list(result.values())).all())


if __name__ == "__main__":
    unittest.main()
