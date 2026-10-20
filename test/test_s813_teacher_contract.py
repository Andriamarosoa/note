"""Fail-closed teacher acceptance is distinct from unified MIDI and LIVE."""
from dataclasses import replace
import unittest

from causal_note.s813_teacher_contract import (
    AXES, REQUIRED_METRICS, RESOLUTION_MS, TeacherEvidence, assess_teacher,
    unification_gate, validate_observation_timing,
)


def qualified(axis, score=.850001):
    return TeacherEvidence(
        axis=axis, score=score, metric_name=REQUIRED_METRICS[axis],
        source_id="signed_source_fixture", evaluation_id="independent_evaluation_fixture",
        evaluation_role="independent_validation", generalization_verified=True,
        inference_uses_labels=False, private_parameters_verified=True,
        private_state_verified=True, causal_observations_verified=True,
        offset_anchors_fixed=True if axis == "offset" else None,
    )


class TeacherContractTests(unittest.TestCase):
    def test_exact_threshold_fails_for_each_axis(self):
        for axis in AXES:
            with self.subTest(axis=axis):
                self.assertEqual(assess_teacher(qualified(axis, .85))["status"], "FAIL")
                self.assertEqual(assess_teacher(qualified(axis))["status"], "PASS")

    def test_all_six_required_and_absence_is_unknown(self):
        self.assertEqual(unification_gate([])["status"], "UNKNOWN")
        for omit in AXES:
            gate = unification_gate(qualified(axis) for axis in AXES if axis != omit)
            self.assertFalse(gate["allow_unification"])
            self.assertEqual(gate["teachers"][omit]["status"], "UNKNOWN")
        self.assertTrue(unification_gate(map(qualified, AXES))["allow_unification"])

    def test_six_high_scores_on_different_cohorts_cannot_be_combined(self):
        teachers = [replace(qualified(axis, .9), source_id="cohort_" + axis,
                            evaluation_id="run_" + axis) for axis in AXES]
        gate = unification_gate(teachers)
        self.assertTrue(all(row["status"] == "PASS" for row in gate["teachers"].values()))
        self.assertFalse(gate["allow_unification"])
        self.assertEqual(gate["status"], "FAIL")
        self.assertIn("teacher_evaluation_cohort_or_split_mismatch", gate["failures"])
        # Separate training/evaluation runs are legitimate on the SAME cohort.
        same_cohort = [replace(row, source_id="cohort_fixed_held_split") for row in teachers]
        self.assertTrue(unification_gate(same_cohort)["allow_unification"])

    def test_missing_score_provenance_or_evidence_is_unknown(self):
        for field in ("score", "source_id", "evaluation_id", "metric_name",
                      "private_parameters_verified", "private_state_verified",
                      "generalization_verified", "causal_observations_verified",
                      "inference_uses_labels"):
            with self.subTest(field=field):
                self.assertEqual(assess_teacher(replace(qualified("onset"),
                                                       **{field: None}))["status"], "UNKNOWN")

    def test_invalid_scores_never_pass(self):
        for score in (float("nan"), float("inf"), -.1, 1.01, True, "0.99"):
            with self.subTest(score=score):
                self.assertEqual(assess_teacher(qualified("exact_k", score))["status"], "FAIL")

    def test_proxy_metric_and_unified_score_cannot_be_teacher_proofs(self):
        for changes in ({"metric_name": "four_state_accuracy"},
                        {"metric_name": "onset_f1"},
                        {"score_scope": "unified_midi"}):
            with self.subTest(changes=changes):
                self.assertEqual(assess_teacher(replace(qualified("full_note"),
                                                       **changes))["status"], "FAIL")

    def test_all_three_clocks_must_really_be_256(self):
        for field in ("feature_hop_samples", "decision_hop_samples", "metric_hop_samples"):
            with self.subTest(field=field):
                self.assertEqual(assess_teacher(replace(qualified("exact_k"),
                                                       **{field: 1024}))["status"], "FAIL")
        self.assertAlmostEqual(RESOLUTION_MS, 5.804988662131519)
        self.assertEqual(assess_teacher(replace(qualified("exact_k"),
                                               sample_rate_hz=48000))["status"], "FAIL")

    def test_fit_high_score_never_certifies_generalization(self):
        for changes in ({"generalization_verified": False},
                        {"evaluation_role": "fit_research"},
                        {"evaluation_role": "calibration"}):
            with self.subTest(changes=changes):
                self.assertEqual(assess_teacher(replace(qualified("frame_pitch", .99),
                                                       **changes))["status"], "FAIL")

    def test_label_access_or_shared_state_fails(self):
        for changes in ({"inference_uses_labels": True},
                        {"private_parameters_verified": False},
                        {"private_state_verified": False},
                        {"causal_observations_verified": False}):
            with self.subTest(changes=changes):
                self.assertEqual(assess_teacher(replace(qualified("onset"),
                                                       **changes))["status"], "FAIL")

    def test_offset_requires_fixed_onset_anchors(self):
        self.assertEqual(assess_teacher(replace(qualified("offset"),
                                               offset_anchors_fixed=False))["status"], "FAIL")
        self.assertEqual(assess_teacher(replace(qualified("offset"),
                                               offset_anchors_fixed=None))["status"], "UNKNOWN")

    def test_cpu_unknown_does_not_block_teacher_research_or_grant_live(self):
        for cpu in (None, {"p99_ms": 1.0, "max_ms": 2.0}):
            gate = unification_gate(map(qualified, AXES), cpu_evidence=cpu)
            self.assertTrue(gate["allow_unification"])
            self.assertFalse(gate["live_ready"])
            self.assertFalse(gate["unified_midi_evaluated"])

    def test_duplicate_or_foreign_teacher_rejected(self):
        with self.assertRaises(ValueError):
            unification_gate([qualified("onset"), qualified("onset")])
        with self.assertRaises(ValueError):
            assess_teacher(replace(qualified("onset"), axis="pitch_proxy"))

    def test_future_observation_cannot_leak_into_earlier_decision(self):
        self.assertTrue(validate_observation_timing(
            available_sample=256, decision_sample=256, latest_audio_sample=256))
        self.assertTrue(validate_observation_timing(
            available_sample=256, decision_sample=512, latest_audio_sample=512))
        for available, decision, audio in ((512, 256, 512), (256, 512, 256),
                                           (256, 300, 512), (-1, 256, 256),
                                           (True, 256, 256)):
            with self.subTest(available=available, decision=decision, audio=audio):
                with self.assertRaises(ValueError):
                    validate_observation_timing(available_sample=available,
                        decision_sample=decision, latest_audio_sample=audio)


if __name__ == "__main__":
    unittest.main()
