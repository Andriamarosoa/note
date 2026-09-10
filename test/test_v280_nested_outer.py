from __future__ import annotations

import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from scripts import train_v280_nested_outer as f
from test.test_v280_real_smoke import _member


def synthetic_arrays():
    members = np.asarray([_member(0, f"piece{fold:02d}", "solo")
                          for fold in range(5) for _ in range(3)])
    k = np.tile([0, 2, 3], 5).astype(np.int32)
    strings = (np.arange(6)[None, :] < k[:, None]).astype(np.float32)
    midi = strings * np.asarray(f.e.STANDARD_TUNING_MIDI)[None, :]
    targets, weights, _ = f.e.training_targets(k, strings, midi, strings)
    return {"global_index": np.arange(len(k)), "member": members,
            "row_fold": np.repeat(np.arange(5), 3),
            **{"target_" + name: value for name, value in targets.items()},
            **{"weight_" + name: value for name, value in weights.items()}}


class NestedPartitionTests(unittest.TestCase):
    def test_inner_validation_uses_lowest_nonouter_and_refit_excludes_holdout(self):
        a = synthetic_arrays()
        for fold in range(5):
            roles = f.partitions(a["row_fold"], a["member"], fold)
            expected_val = 1 if fold == 0 else 0
            self.assertTrue(np.all(a["row_fold"][roles["validation"]] == expected_val))
            np.testing.assert_array_equal(np.sort(np.concatenate([roles["fit"], roles["validation"]])), roles["refit"])
            self.assertEqual(len(set(roles["outer"]) & set(roles["refit"])), 0)
            before = f.partition_record(a, fold)
            a["target_cardinality"][roles["outer"]] = 6
            self.assertEqual(before, f.partition_record(a, fold))

    def test_same_composition_cannot_cross_folds(self):
        a = synthetic_arrays()
        a["row_fold"][0] = 1
        with self.assertRaisesRegex(f.OuterError, "composition"):
            f.partitions(a["row_fold"], a["member"], 0)

    def test_selection_requires_all_epochs_and_uses_poly_score_nll_then_earlier(self):
        history = [{"epoch": epoch, "validation": {"poly_correct": 2, "poly_nll": 1.0}}
                   for epoch in range(1, 13)]
        self.assertEqual(f.choose_epoch(history)["epoch"], 1)
        history[7]["validation"]["poly_nll"] = .5
        self.assertEqual(f.choose_epoch(history)["epoch"], 8)
        history[3]["validation"]["poly_correct"] = 3
        self.assertEqual(f.choose_epoch(history)["epoch"], 4)
        with self.assertRaisesRegex(f.OuterError, "complete"):
            f.choose_epoch(history[:-1])


class AuditTests(unittest.TestCase):
    def test_oof_coverage_rejects_duplicates_changed_targets_and_missing_folds(self):
        a = synthetic_arrays()
        parts = []
        for fold in range(5):
            rows = f.partitions(a["row_fold"], a["member"], fold)["outer"]
            parts.append({"global_index": rows.copy(), "member": a["member"][rows].copy(),
                          "row_fold": a["row_fold"][rows].copy(), "k": a["target_cardinality"][rows].copy(),
                          "probability": np.eye(7)[a["target_cardinality"][rows]] * .93 + .01})
        merged = f.validate_oof_parts(parts, a)
        np.testing.assert_array_equal(merged["global_index"], a["global_index"])
        with self.assertRaisesRegex(f.OuterError, "five"):
            f.validate_oof_parts(parts[:-1], a)
        parts[4]["global_index"][0] = parts[3]["global_index"][0]
        with self.assertRaisesRegex(f.OuterError, "identities"):
            f.validate_oof_parts(parts, a)
        parts[4]["global_index"][0] = 12
        parts[4]["k"][1] = 6
        with self.assertRaisesRegex(f.OuterError, "identities"):
            f.validate_oof_parts(parts, a)

    def test_track_bootstrap_is_paired_and_retains_negative_result(self):
        a = synthetic_arrays()
        k = a["target_cardinality"]
        pred = k.copy()
        pred[k >= 2] = 0
        result = f.track_bootstrap(a["member"], k, k, pred, replicates=100)
        self.assertEqual(result["tracks"], 5)
        self.assertEqual(result["delta_pp"], -100.0)
        self.assertEqual(result["lower_95_pp"], -100.0)
        self.assertEqual(result["upper_95_pp"], -100.0)

    def test_frozen_ranking_preserves_unisons_and_event_false_negatives(self):
        arrays = {"member": np.asarray(["track", "track"])}
        evaluation = {"top_samples": np.asarray([[1000, 1000, -1], [3000, -1, -1]]),
                      "reference_member": np.asarray(["track"] * 3),
                      "reference_sample": np.asarray([1000, 1000, 3000])}
        result = f.event_metrics(arrays, evaluation, np.asarray([2, 0]))
        for tolerance in f.TOLERANCES_MS:
            self.assertEqual(result[str(tolerance)]["tp"], 2)
            self.assertEqual(result[str(tolerance)]["fp"], 0)
            self.assertEqual(result[str(tolerance)]["fn"], 1)

    def test_changed_checkpoint_file_is_rejected(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "optimizer.npz").write_bytes(b"original")
            records = f.file_records(root, ["optimizer.npz"])
            (root / "optimizer.npz").write_bytes(b"changed")
            with self.assertRaisesRegex(f.OuterError, "digest"):
                f.verify_files(root, records)


@unittest.skipUnless(importlib.util.find_spec("tensorflow"), "TensorFlow resume gates run in Actions before training")
class TensorFlowResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import tensorflow as tf
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(4)

    def test_restored_Adam_next_update_matches_uninterrupted_training(self):
        a = synthetic_arrays()
        x = np.random.default_rng(8).uniform(0, 1, (2, *f.e.FEATURE_SHAPE)).astype(np.float32)
        y = {name: a["target_" + name][1:3] for name in f.e.LOSS_WEIGHTS}
        w = {name: a["weight_" + name][1:3] for name in f.e.LOSS_WEIGHTS}
        w["string_fret_onset"] = w["string_fret_onset"][:, None]
        model, _ = f.e.initialize_arm("harmonic")
        model.optimizer.build(model.trainable_variables)
        model.train_on_batch(x, y, sample_weight=w)
        with TemporaryDirectory() as temp:
            root = Path(temp)
            specification = f.save_training_state(model, root)
            model.train_on_batch(x, y, sample_weight=w)
            expected = [v.numpy().copy() for v in model.weights + model.optimizer.variables()]
            restored, _ = f.e.initialize_arm("harmonic")
            f.restore_training_state(restored, root, specification)
            self.assertEqual(int(restored.optimizer.iterations.numpy()), 1)
            restored.train_on_batch(x, y, sample_weight=w)
            actual = [v.numpy() for v in restored.weights + restored.optimizer.variables()]
            self.assertEqual(len(actual), len(expected))
            for left, right in zip(actual, expected):
                np.testing.assert_allclose(left, right, rtol=1e-6, atol=1e-7)

    def test_two_probe_chunks_fresh_refit_and_single_outer_evaluation(self):
        a = synthetic_arrays()
        tensor = np.random.default_rng(9).uniform(0, 1, (15, *f.e.FEATURE_SHAPE)).astype(np.float16)
        accessed = []

        class TrackedFeatures:
            def __getitem__(self, rows):
                accessed.extend(np.asarray(rows).tolist())
                return tensor[rows]

        manifest = {"split": {"roles_by_fold": {str(i): f.partition_record(a, i) for i in range(5)}}}
        with TemporaryDirectory() as temp:
            root = Path(temp)
            prepared = root / "prepared"
            prepared.mkdir()
            f.e.smoke._atomic_json(prepared / "manifest.json", manifest)
            common = {"prepared_dir": prepared, "fold": 0}
            with patch.object(f, "load_prepared", return_value=(TrackedFeatures(), a, manifest)):
                for phase in ("probe", "refit"):
                    for chunk in (1, 2):
                        f.train_chunk(SimpleNamespace(**common, phase=phase, chunk=chunk,
                            output_dir=root / f"{phase}{chunk}",
                            previous_dir=root / f"{phase}1" if chunk == 2 else None,
                            probe_dir=root / "probe2" if phase == "refit" and chunk == 1 else None))
                self.assertTrue(set(accessed).isdisjoint({0, 1, 2}))
                probe = f.read_json(root / "probe2" / "state.json")
                refit = f.read_json(root / "refit2" / "state.json")
                self.assertEqual(probe["epochs_completed"], 12)
                self.assertEqual(refit["epochs_completed"], f.choose_epoch(probe["history"])["epoch"])
                self.assertEqual(refit["optimizer_updates"], refit["epochs_completed"])
                self.assertEqual(refit["history"][0]["updates"], 1)
                accessed.clear()
                f.evaluate(SimpleNamespace(**common, probe_dir=root / "probe2", refit_dir=root / "refit2",
                                           output_dir=root / "outer"))
                self.assertEqual(accessed, [0, 1, 2])
                report = f.read_json(root / "outer" / "report.json")
                self.assertEqual(report["outer_inference_passes"], 1)
                self.assertEqual(report["outer"]["rows"], 3)


if __name__ == "__main__":
    unittest.main()
