import unittest
import numpy as np

from scripts import audit_v273_internal_b_low_harmonic_strata as h
from scripts.audit_v273_attack_novelty import (
    harmonic_mask, component_metrics, case_measurement, aggregate, format_cell,
)


class AttackNoveltyTests(unittest.TestCase):
    def test_harmonic_mask_unions_overlapping_bands(self):
        freq = np.arange(50.0, 1001.0, 1.0)
        mask = harmonic_mask(freq, 100.0)
        self.assertTrue(mask[np.argmin(np.abs(freq - 100.0))])
        self.assertTrue(mask[np.argmin(np.abs(freq - 200.0))])
        self.assertTrue(mask[np.argmin(np.abs(freq - 300.0))])
        self.assertLessEqual(int(mask.sum()), len(freq))

    def test_component_metrics_detects_new_attack(self):
        freq = np.arange(50.0, 1001.0, 1.0)
        powers = np.ones((3, len(freq)), dtype=float)
        mask = harmonic_mask(freq, 100.0)
        powers[1, mask] = 10.0
        powers[2, mask] = 8.0
        m = component_metrics(freq, powers, 100.0)
        self.assertGreater(m["onset_contrast_raw"], 0.0)
        self.assertGreater(m["log_ratio_raw"], 0.0)
        self.assertGreater(m["positive_support"], 0.0)
        self.assertLess(m["late_retention_raw"], 1.0)

    def test_case_measurement_marks_unmatched_component_and_paired_delta(self):
        freq = np.arange(50.0, 1201.0, 1.0)
        powers = np.ones((3, len(freq)), dtype=float)
        for f0 in (100.0, 200.0, 300.0):
            powers[1, harmonic_mask(freq, f0)] += 8.0
        powers[0, harmonic_mask(freq, 500.0)] += 8.0
        powers[1, harmonic_mask(freq, 500.0)] += 8.0
        row = {
            "row_id": 1,
            "fold": 0,
            "group": "K3_regressed",
            "true_K": 3,
            "recording_id": "x",
            "start_sample": 1000,
            "owned_notes": [
                {"frequency_hz": 100.0},
                {"frequency_hz": 200.0},
                {"frequency_hz": 300.0},
            ],
            "decomposition": {"triplet_f0": [100.0, 200.0, 500.0]},
        }
        result = case_measurement(row, freq, powers, np.zeros(3, dtype=int))
        self.assertEqual(result["matched_selected_count"], 2)
        unmatched = [x for x in result["selected"] if x["role"] == "selected_unmatched"]
        self.assertEqual(len(unmatched), 1)
        self.assertGreater(
            result["summary"]["expected_minus_unmatched_onset_contrast_raw"], 0.0
        )

    def test_aggregate_keeps_component_and_paired_counts(self):
        freq = np.arange(50.0, 1201.0, 1.0)
        powers = np.ones((3, len(freq)), dtype=float)
        for f0 in (100.0, 200.0, 300.0):
            powers[1, harmonic_mask(freq, f0)] += 4.0
        row = {
            "row_id": 1, "fold": 0, "group": "K3_regressed", "true_K": 3,
            "recording_id": "x", "start_sample": 1000,
            "owned_notes": [{"frequency_hz": 100.0}, {"frequency_hz": 200.0}, {"frequency_hz": 300.0}],
            "decomposition": {"triplet_f0": [100.0, 200.0, 500.0]},
        }
        measured = case_measurement(row, freq, powers, np.zeros(3, dtype=int))
        out = aggregate([measured])
        self.assertEqual(out["rows"], 1)
        self.assertEqual(out["expected"]["components"], 3)
        self.assertEqual(out["selected_unmatched"]["components"], 1)
        self.assertEqual(out["expected_minus_unmatched"]["onset_contrast_raw"]["n"], 1)

    def test_report_formatter_accepts_labels_and_numbers(self):
        self.assertEqual(format_cell("K3_regressed"), "K3_regressed")
        self.assertEqual(format_cell(125), "125")
        self.assertEqual(format_cell(None), "n/a")
        self.assertEqual(format_cell(0.25), "0.250000")


if __name__ == "__main__":
    unittest.main()
