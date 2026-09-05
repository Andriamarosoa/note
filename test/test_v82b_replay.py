import json
from pathlib import Path
import tempfile
import unittest

from causal_note.v82b_replay import (
    ReplayError,
    ReplayPoint,
    arrangement_fraction,
    load_replay_points,
    select_replay_points,
)


class V82bReplayTests(unittest.TestCase):
    def test_load_deduplicates_and_preserves_real_fp_metadata(self):
        payload = {
            "scope": {"player_05_read": False},
            "false_positive_records": [
                {
                    "member": "00_a_solo.jams",
                    "sample": 100,
                    "arrangement": "solo",
                    "model_onset_score": 0.41,
                    "positive_flux_over_pre_energy": 0.6,
                    "fixed_positive_flux_fraction": 0.8,
                },
                {
                    "member": "00_a_solo.jams",
                    "sample": 100,
                    "arrangement": "solo",
                    "model_onset_score": 0.43,
                    "positive_flux_over_pre_energy": 0.1,
                    "fixed_positive_flux_fraction": 0.1,
                },
                {
                    "member": "01_b_comp.jams",
                    "sample": 200,
                    "arrangement": "comp",
                    "model_onset_score": 0.42,
                    "positive_flux_over_pre_energy": 0.7,
                    "fixed_positive_flux_fraction": 0.75,
                },
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            points = load_replay_points(path)
        self.assertEqual(2, len(points))
        solo = next(point for point in points if point.arrangement == "solo")
        self.assertAlmostEqual(0.43, solo.model_onset_score)
        self.assertFalse(solo.harmonic_proxy)
        comp = next(point for point in points if point.arrangement == "comp")
        self.assertTrue(comp.harmonic_proxy)

    def test_selection_is_unique_capped_and_preserves_pool_mix(self):
        points = []
        # 80% solo, 20% comp across enough tracks to satisfy a cap of 2.
        for track in range(8):
            arrangement = "solo" if track < 6 else "comp"
            per_track = 4 if arrangement == "solo" else 3
            for index in range(per_track):
                points.append(
                    ReplayPoint(
                        member=f"{track:02d}_{arrangement}.jams",
                        sample=track * 1000 + index,
                        arrangement=arrangement,
                        model_onset_score=0.41,
                    )
                )
        selected = select_replay_points(points, count=10, seed=7, max_per_track=2)
        self.assertEqual(10, len(selected))
        self.assertEqual(10, len({(point.member, point.sample) for point in selected}))
        counts = {}
        for point in selected:
            counts[point.member] = counts.get(point.member, 0) + 1
        self.assertLessEqual(max(counts.values()), 2)
        expected_solo = round(10 * arrangement_fraction(points, "solo"))
        self.assertEqual(expected_solo, sum(point.arrangement == "solo" for point in selected))

    def test_selection_fails_instead_of_repeating_when_pool_is_too_small(self):
        points = [ReplayPoint("00_a_solo.jams", index, "solo", 0.41) for index in range(3)]
        with self.assertRaises(ReplayError):
            select_replay_points(points, count=4, seed=1, max_per_track=10)


if __name__ == "__main__":
    unittest.main()
