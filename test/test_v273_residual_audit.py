"""Regression checks for population accounting, leakage and portable score replay."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.audit_v273_internal_b_low_minimal_residual_failure import export_rows
from scripts.summarize_v273_internal_b_low_minimal_residual_failure import discover, summarize
from scripts.v273_residual_audit import (
    COUNT_KEYS, EXPERIMENT, FEATURES, FOLDS, REGISTERS, action_metrics, analyze,
    array_identity, check_partitions, extracted_matrix, fit_classifier,
    register_labels, sha256_file, stratify, verify_export, write_json,
)


def fixture(fold=0):
    rng = np.random.default_rng(42)
    arrays = {}
    for split, n, offset in (("fit", 90, 0), ("val", 30, 1000)):
        y = np.resize(np.array([2, 3, 2, 3, 4, 1]), n)
        pair = rng.uniform(.2, .8, n)
        X = np.column_stack([pair, pair - rng.uniform(.01, .06, n)])
        f0 = np.linspace(65, 500, n)
        valid = np.ones(n, bool)
        valid[-1] = False
        X[-1] = np.nan
        reason = np.full(n, "", dtype="U32")
        reason[-1] = "insufficient_f0_candidates"
        ids = np.arange(offset, offset + n)
        arrays.update({
            split + "_ids": ids, split + "_action_ids": ids.copy(),
            split + "_true_k": y, split + "_base_k": np.full(n, 3),
            split + "_b_low": np.ones(n, bool),
            split + "_fold": np.full(n, fold if split == "val" else next(f for f in FOLDS if f != fold)),
            split + "_recording": np.array([f"{split}_recording_{i // 10}" for i in range(n)]),
            split + "_start_sample": np.arange(n) * 512,
            split + "_X": X, split + "_f0": f0, split + "_valid": valid,
            split + "_excluded_reason": reason,
        })
    inputs = []
    for split in ("fit", "val"):
        v = arrays[split + "_valid"]
        inputs.extend([arrays[split + "_X"][v], arrays[split + "_true_k"][v], arrays[split + "_f0"][v]])
    report, state, predictions = analyze(*inputs, include_register_refit=True)
    arrays.update(predictions)
    report.update(schema_version=2, protocol={"experiment": EXPERIMENT, "validation_fold": fold,
                                              "outer_fold_3_used": False}, exclusions={})
    return arrays, report, state


def write_fixture(root, fold=0):
    root.mkdir()
    arrays, report, state = fixture(fold)
    write_json(root / "report.json", report)
    write_json(root / "model.json", state)
    (root / "report.md").write_text("Synthetic test fixture, not a project result.\n")
    export_rows(root / "rows.jsonl", arrays, 44100)
    np.savez_compressed(root / "replay.npz", **arrays)
    manifest = {"schema_version": 2, "commit": "fixture", "source_sha256": {},
                "config_sha256": "fixture", "runtime": {"test": True},
                "arrays": {k: array_identity(v) for k, v in arrays.items()},
                "files": {p: sha256_file(root / p) for p in
                          ("report.json", "report.md", "model.json", "rows.jsonl", "replay.npz")}}
    write_json(root / "manifest.json", manifest)
    write_json(root / "replay_check.json", verify_export(root))
    return arrays, report, state


class ResidualAuditTests(unittest.TestCase):
    def test_all_k_actions_and_threshold_equality(self):
        y = np.array([2, 3, 4, 1, 2, 3, 0])
        p = np.array([.5, .6, .9, .9, .49, .1, .7])
        result = action_metrics(y, p)
        self.assertEqual((result["applied"], result["corrections"], result["regressions"],
                          result["other_k_actions"], result["global_net"]), (5, 1, 1, 3, 0))
        self.assertEqual(result["val_k23_rows"], 4)
        labels = np.array(["low", "mid", "high", "unassigned", "low", "mid", "high"])
        parts = stratify(y, p, labels)
        for key in COUNT_KEYS:
            self.assertEqual(sum(r[key] for r in parts.values()), result[key])

    def test_register_boundaries_ties_and_missing_values(self):
        values = np.array([50, 100, 150, 200, np.nan, np.inf, -1])
        self.assertEqual(register_labels(values, [100, 200]).tolist(),
                         ["low", "mid", "mid", "high", "unassigned", "unassigned", "unassigned"])
        self.assertEqual(register_labels(np.array([100]), [100, 100]).tolist(), ["high"])
        empty = action_metrics(np.array([], int), np.array([]))
        self.assertIsNone(empty["joint_auc"])
        self.assertEqual(empty["applied"], 0)

    def test_global_predictions_do_not_depend_on_validation_register_or_labels(self):
        arrays, _, _ = fixture()
        data = []
        for split in ("fit", "val"):
            v = arrays[split + "_valid"]
            data.extend([arrays[split + "_X"][v], arrays[split + "_true_k"][v], arrays[split + "_f0"][v]])
        report, state, p = analyze(*data)
        altered = [x.copy() for x in data]
        altered[4][:] = 4  # no K2/K3 left in VAL; FIT and scores must remain unchanged.
        altered[5][:] = np.nan
        changed, changed_state, changed_p = analyze(*altered)
        self.assertEqual(state, changed_state)
        self.assertEqual(report["register_cuts_hz"], changed["register_cuts_hz"])
        np.testing.assert_array_equal(p["val_probability"], changed_p["val_probability"])
        self.assertIsNone(changed["joint"]["val_auc"])
        self.assertEqual(changed["global_stratified"]["unassigned"]["val_rows"], len(data[4]))

    def test_registers_use_global_scores_even_when_refits_differ(self):
        arrays, report, _ = fixture()
        y = arrays["val_true_k"][arrays["val_valid"]]
        for name in REGISTERS:
            mask = arrays["val_register"] == name
            expected = action_metrics(y[mask], arrays["val_probability"][mask])
            self.assertEqual({k: report["global_stratified"][name][k] for k in expected}, expected)
        self.assertTrue(report["register_refit"]["enabled"])
        self.assertTrue(any(r["status"] == "completed" for r in report["register_refit"]["registers"].values()))

    def test_shared_fit_canonicalizes_matrix_layout(self):
        X = np.arange(80, dtype=float).reshape(40, 2) / 100
        y = np.resize([0, 1, 1, 0], 40)
        c, f = fit_classifier(X, y), fit_classifier(np.asfortranarray(X), y)
        np.testing.assert_array_equal(c.decision_function(X), f.decision_function(X))

    def test_excluded_rows_keep_positions_and_reasons(self):
        rows = [{FEATURES[0]: .4, FEATURES[1]: .3, "median_triplet_f0": 100}, None,
                {FEATURES[0]: np.nan, FEATURES[1]: .3, "median_triplet_f0": 100}]
        X, _, valid, reasons = extracted_matrix(rows)
        self.assertEqual(X.shape, (3, 2))
        self.assertEqual(valid.tolist(), [True, False, False])
        self.assertEqual(reasons.tolist(), ["", "insufficient_f0_candidates", "nonfinite_residual"])

    def test_forbidden_fold_and_recording_overlap_are_rejected(self):
        arrays, _, _ = fixture()
        check_partitions(arrays, 0)
        with self.assertRaisesRegex(RuntimeError, "forbidden validation"):
            check_partitions(arrays, 3)
        arrays["fit_fold"][0] = 3
        with self.assertRaisesRegex(RuntimeError, "forbidden FIT"):
            check_partitions(arrays, 0)
        arrays["fit_fold"][0] = 1
        arrays["val_recording"][0] = arrays["fit_recording"][0]
        with self.assertRaisesRegex(RuntimeError, "recording overlap"):
            check_partitions(arrays, 0)

    def test_complete_export_replays_and_detects_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "fold"
            arrays, report, _ = write_fixture(root)
            self.assertEqual(verify_export(root)["global_net"], report["global_action"]["global_net"])
            rows = [json.loads(line) for line in (root / "rows.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 120)
            self.assertEqual(sum(not r["valid"] for r in rows), 2)
            val = [r for r in rows if r["split"] == "val" and r["valid"]]
            self.assertEqual([r["row_id"] for r in val], arrays["val_action_ids"][arrays["val_valid"]].tolist())
            self.assertEqual(sum(r["action_applied"] for r in val), report["global_action"]["applied"])
            self.assertTrue(all(r["global_score"] is None for r in rows if not r["valid"]))
            (root / "model.json").write_text("{}")
            with self.assertRaisesRegex(RuntimeError, "checksum changed: model.json"):
                verify_export(root)

    def test_summary_requires_four_verified_compatible_folds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for f in FOLDS:
                write_fixture(root / str(f), f)
            reports = discover(root)
            result = summarize(reports)
            self.assertEqual(result["global_totals"]["global_net"],
                             sum(r["global_action"]["global_net"] for r in reports.values()))
            p = root / "2" / "report.json"
            report = json.loads(p.read_text())
            report["schema_version"] = 1
            write_json(p, report)
            with self.assertRaisesRegex(RuntimeError, "legacy report"):
                discover(root)


if __name__ == "__main__":
    unittest.main()
