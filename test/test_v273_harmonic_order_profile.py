import unittest
import numpy as np

from scripts import audit_v273_internal_b_low_harmonic_strata as h
from scripts.audit_v273_harmonic_order_profile import (
    order_metrics, overlaps_expected, auc_expected_vs_unmatched
)


class HarmonicOrderProfileTests(unittest.TestCase):
    def test_order_metrics_returns_all_orders_and_normalized_profile(self):
        freq = np.linspace(h.MIN_HZ, h.MAX_ANALYSIS_HZ, 2000)
        powers = np.ones((3, len(freq)), dtype=float)
        powers[1] *= 2.0
        out = order_metrics(freq, powers, 100.0)
        self.assertEqual(len(out), h.MAX_HARMONICS)
        valid = [x for x in out if x["valid"]]
        self.assertTrue(valid)
        s = sum(x["profile_fraction"] for x in valid)
        self.assertAlmostEqual(s, 1.0, places=9)

    def test_overlap_detects_expected_harmonic(self):
        item = {"valid": True, "center_hz": 400.0}
        self.assertTrue(overlaps_expected(item, [100.0]))
        self.assertFalse(overlaps_expected({"valid": True, "center_hz": 455.0}, [100.0]))

    def test_auc_reports_direction(self):
        rows = [
            {"components": [
                {"role": "expected", "orders": [{"valid": True, "profile_fraction": .9}]},
                {"role": "selected_unmatched", "orders": [{"valid": True, "profile_fraction": .1}]},
            ]},
            {"components": [
                {"role": "expected", "orders": [{"valid": True, "profile_fraction": .8}]},
                {"role": "selected_unmatched", "orders": [{"valid": True, "profile_fraction": .2}]},
            ]},
        ]
        out = auc_expected_vs_unmatched(rows, "profile_fraction", 1)
        self.assertEqual(out["direction"], "expected_higher")
        self.assertAlmostEqual(out["oriented_auc"], 1.0)


if __name__ == "__main__":
    unittest.main()
