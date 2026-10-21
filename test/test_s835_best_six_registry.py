"""S835: do not silently overwrite the best known per-axis performance."""
import copy
import unittest

from causal_note.s835_best_six_registry import (
    AXES, load_registry, validate_registry, six_native256_ready,
    assert_single_axis_noninterference, assert_paired_no_regression
)


class BestSixAxisRegistryTests(unittest.TestCase):

    def setUp(self):
        self.manifest = load_registry()

    def test_every_axis_pins_real_repository_code_and_evidence(self):
        self.assertTrue(validate_registry(self.manifest))
        self.assertEqual(set(self.manifest["axes"]), set(AXES))
        native = [k for k, a in self.manifest["axes"].items()
                  if a["chosen"]["mode"] == "strict_256_shadow_candidate"]
        protected = [k for k, a in self.manifest["axes"].items()
                     if a["chosen"]["mode"] == "protected_legacy_keep"]
        self.assertEqual(set(native), {
            "exact_k", "active_exact_k", "frame_pitch", "full_note"})
        self.assertEqual(set(protected), {"onset", "offset"})

    def test_s834_and_s828_bests_are_not_pretended_deployed(self):
        self.assertFalse(six_native256_ready(self.manifest))
        self.assertFalse(self.manifest["policy"]["merged_midi_enabled"])
        self.assertFalse(self.manifest["policy"]["public_midi_modified"])
        self.assertFalse(any(
            x["chosen"]["ready_for_live"]
            for x in self.manifest["axes"].values()))
        self.assertEqual(self.manifest["policy"]["hop_samples"], 256)

    def test_quality_regression_not_accepted(self):
        corrupted = copy.deepcopy(self.manifest)
        corrupted["axes"]["exact_k"]["chosen"]["score"] = .55
        with self.assertRaises(ValueError):
            validate_registry(corrupted)

    def test_metric_swap_cannot_falsify_offset_progress(self):
        corrupted = copy.deepcopy(self.manifest)
        corrupted["axes"]["offset"]["chosen"]["mode"] = "strict_256_shadow_candidate"
        corrupted["axes"]["offset"]["chosen"]["series"] = "S834"
        corrupted["axes"]["offset"]["chosen"]["hop_samples"] = 256
        corrupted["axes"]["offset"]["chosen"]["score"] = .38677437
        corrupted["axes"]["offset"]["chosen"]["metric"] = "offset_event_f1"
        with self.assertRaises(ValueError):
            validate_registry(corrupted)

    def test_no_fake_production_promotion(self):
        invalid = copy.deepcopy(self.manifest)
        invalid["axes"]["frame_pitch"]["chosen"]["ready_for_live"] = True
        with self.assertRaises(ValueError):
            validate_registry(invalid)

    def test_independence_gate_checks_other_five_axes(self):
        unchanged = {k: (k, 1) for k in AXES}
        for axis in AXES:
            changed = dict(unchanged)
            changed[axis] = (axis, 2)
            self.assertTrue(assert_single_axis_noninterference(
                unchanged, changed, axis))
            foreign = next(k for k in AXES if k != axis)
            changed[foreign] = (foreign, 999)
            with self.assertRaises(AssertionError):
                assert_single_axis_noninterference(
                    unchanged, changed, axis)

    def test_paired_scores_need_six_identical_denominators(self):
        base = {x: dict(score=.7, metric=f"{x}_metric",
                        cohort="FIT80", frame_count=100) for x in AXES}
        after = copy.deepcopy(base)
        after["exact_k"]["score"] = .8
        self.assertTrue(assert_paired_no_regression(base, after))
        after["onset"]["score"] = .69
        with self.assertRaises(AssertionError):
            assert_paired_no_regression(base, after)
        after = copy.deepcopy(base)
        after["offset"]["metric"] = "different_offset_metric"
        with self.assertRaises(ValueError):
            assert_paired_no_regression(base, after)


if __name__ == "__main__":
    unittest.main()
