import unittest
import numpy as np

from scripts.audit_v273_exclusive_harmonic_support import (
    attack_spectrum, support_against_set, overlap_with_reference, measure_case, fmt,
)


class ExclusiveHarmonicSupportTests(unittest.TestCase):
    def test_attack_spectrum_is_positive_pre_post_difference(self):
        powers = np.array([[2., 5., 1.], [4., 3., 6.], [0., 0., 0.]])
        np.testing.assert_array_equal(attack_spectrum(powers), [2., 0., 5.])

    def test_unique_support_separates_shared_bins(self):
        freq = np.arange(50.0, 1001.0, 1.0)
        attack = np.ones(len(freq))
        out = support_against_set(freq, attack, 100.0, [200.0])
        self.assertGreaterEqual(out["attack_energy_total"], out["attack_energy_unique"])
        self.assertAlmostEqual(
            out["unique_attack_fraction"] + out["shared_attack_fraction"], 1.0, places=9
        )

    def test_overlap_with_expected_fraction_is_bounded(self):
        freq = np.arange(50.0, 1001.0, 1.0)
        attack = np.ones(len(freq))
        out = overlap_with_reference(freq, attack, 200.0, [100.0, 300.0])
        self.assertGreaterEqual(out["overlap_with_expected_fraction"], 0.0)
        self.assertLessEqual(out["overlap_with_expected_fraction"], 1.0)
        self.assertAlmostEqual(
            out["overlap_with_expected_fraction"] + out["outside_expected_fraction"],
            1.0,
            places=9,
        )

    def test_measure_case_labels_unmatched_selected_component(self):
        freq = np.arange(50.0, 1201.0, 1.0)
        powers = np.ones((3, len(freq)))
        powers[1] *= 2.0
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
        out = measure_case(row, freq, powers)
        self.assertEqual(out["matched_selected_count"], 2)
        unmatched = [x for x in out["selected"] if x["role"] == "selected_unmatched"]
        self.assertEqual(len(unmatched), 1)
        self.assertIn("overlap_with_expected_fraction", unmatched[0])

    def test_report_formatter_accepts_text_and_numbers(self):
        self.assertEqual(fmt("K3"), "K3")
        self.assertEqual(fmt(12), "12")
        self.assertEqual(fmt(None), "n/a")
        self.assertEqual(fmt(0.5), "0.500000")


if __name__ == "__main__":
    unittest.main()
