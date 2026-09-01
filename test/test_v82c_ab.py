from __future__ import annotations

import unittest

import numpy as np

from scripts.train_v81 import BagExample
from scripts.train_v82c_ab import _apply_onset_replay_contract, _match_supervision_mass


class V82cABTests(unittest.TestCase):
    def test_onset_replay_contract_never_touches_offset_supervision(self):
        examples = (
            BagExample(0, 100, "positive_burst"),
            BagExample(0, 200, "true_fp_replay"),
            BagExample(0, 300, "background"),
        )
        targets = {
            "onset_bag_presence": np.array([[1.0], [1.0], [0.0]], dtype=np.float32),
            "onset_mass": np.array([[0.2], [0.3], [0.0]], dtype=np.float32),
            "onset_delay": np.array([[0.4], [0.5], [0.0]], dtype=np.float32),
            "onset_count": np.array([2, 1, 0], dtype=np.int32),
            "offset_bag_presence": np.array([[0.0], [1.0], [0.0]], dtype=np.float32),
            "offset_mass": np.array([[0.0], [0.1], [0.0]], dtype=np.float32),
            "offset_delay": np.array([[0.0], [0.2], [0.0]], dtype=np.float32),
            "offset_count": np.array([0, 2, 0], dtype=np.int32),
        }
        weights = {
            "onset_bag_presence": np.ones(3, dtype=np.float32),
            "onset_mass": np.ones(3, dtype=np.float32),
            "onset_delay": np.array([1.0, 1.0, 0.0], dtype=np.float32),
            "onset_count": np.array([1.0, 1.0, 0.0], dtype=np.float32),
            "offset_bag_presence": np.array([1.0, 1.0, 1.0], dtype=np.float32),
            "offset_mass": np.array([1.0, 1.0, 1.0], dtype=np.float32),
            "offset_delay": np.array([0.0, 1.0, 0.0], dtype=np.float32),
            "offset_count": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        }
        offset_targets = {
            name: value.copy() for name, value in targets.items() if name.startswith("offset_")
        }
        offset_weights = {
            name: value.copy() for name, value in weights.items() if name.startswith("offset_")
        }

        _apply_onset_replay_contract(targets, weights, examples)

        self.assertEqual(float(targets["onset_bag_presence"][1, 0]), 0.0)
        self.assertEqual(float(targets["onset_mass"][1, 0]), 0.0)
        self.assertEqual(int(targets["onset_count"][1]), 0)
        self.assertEqual(float(weights["onset_bag_presence"][1]), 1.0)
        self.assertEqual(float(weights["onset_delay"][1]), 0.0)
        self.assertEqual(float(targets["onset_bag_presence"][0, 0]), 1.0)

        for name, before in offset_targets.items():
            np.testing.assert_array_equal(before, targets[name])
        for name, before in offset_weights.items():
            np.testing.assert_array_equal(before, weights[name])

    def test_mass_matching_makes_replay_equal_control_for_every_head(self):
        control = {
            "onset_bag_presence": np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            "offset_bag_presence": np.array([1.0, 1.0, 0.0, 1.0], dtype=np.float32),
            "onset_count": np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32),
            "offset_count": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        }
        replay = {
            "onset_bag_presence": np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float32),
            "offset_bag_presence": np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            "onset_count": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            "offset_count": np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32),
        }

        report = _match_supervision_mass(control, replay)

        for name in control:
            self.assertAlmostEqual(float(control[name].sum()), float(replay[name].sum()), places=5)
            self.assertAlmostEqual(
                report[name]["control_mass"], report[name]["replay_matched_mass"], places=5
            )
        self.assertAlmostEqual(report["onset_bag_presence"]["replay_scale"], 4.0 / 3.0)
        self.assertAlmostEqual(report["offset_bag_presence"]["replay_scale"], 3.0 / 4.0)


if __name__ == "__main__":
    unittest.main()
