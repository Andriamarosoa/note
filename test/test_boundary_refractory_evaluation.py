import tempfile
import unittest
from pathlib import Path

from causal_note.detector import BoundaryScoreChunk, BoundaryType
from scripts.audit_boundary_candidate_relations import (
    IdentifiedReference,
    TracedCandidate,
)
from scripts.evaluate_boundary_refractory import (
    DEFAULT_SOURCE_REPORT,
    TREATMENT_REFRACTORY_SAMPLES,
    _decision,
    _distance_bin,
    _locked_configuration,
    _public_projection,
    _simultaneous_reference_support,
    classify_candidate_statuses,
    create_argument_parser,
    decode_refractory_stream,
    load_relation_source,
    summarize_delays,
    trace_fixed_refractory,
)


def traced(kind, sample, channel, episode_id):
    return TracedCandidate(
        kind=kind,
        sample=sample,
        channel=channel,
        entry_score=0.6,
        preceding_low_run_samples=16,
        previous_same_channel_gap_samples=None,
        episode_id=episode_id,
        survives_n16=True,
    )


def score_chunk(start, length, *, slots=2, onsets=(), offsets=()):
    onset_rows = [[0.0] * slots for _ in range(length)]
    offset_rows = [[0.0] * slots for _ in range(length)]
    for sample, slot in onsets:
        onset_rows[sample - start][slot] = 1.0
    for sample, slot in offsets:
        offset_rows[sample - start][slot] = 1.0
    return BoundaryScoreChunk(
        start,
        tuple(tuple(row) for row in onset_rows),
        tuple(tuple(row) for row in offset_rows),
    )


class ScriptedPredictor:
    slot_count = 2

    def __init__(self, chunks):
        self._chunks = dict(chunks)
        self.calls = 0

    def predict_chunk(self, samples, *, start_sample):
        self.calls += 1
        value = self._chunks[start_sample]
        if value.sample_count != len(samples):
            raise AssertionError("scripted score length differs")
        return value


class BoundaryRefractoryEvaluationTests(unittest.TestCase):
    def test_fixed_window_is_inclusive_and_suppression_does_not_extend_it(self):
        values = (
            traced(BoundaryType.ONSET, 100, 0, 0),
            traced(BoundaryType.ONSET, 2305, 0, 1),
            traced(BoundaryType.ONSET, 3000, 0, 2),
        )
        result = trace_fixed_refractory(values, refractory_samples=2205)

        self.assertEqual(tuple(item.sample for item in result.kept), (100, 3000))
        self.assertEqual(len(result.suppressed), 1)
        self.assertEqual(result.suppressed[0].distance_samples, 2205)
        self.assertEqual(result.suppressed[0].anchor.sample, 100)

        boundary = trace_fixed_refractory(
            (
                traced(BoundaryType.ONSET, 100, 0, 3),
                traced(BoundaryType.ONSET, 2306, 0, 4),
            ),
            refractory_samples=2205,
        )
        self.assertEqual(tuple(item.sample for item in boundary.kept), (100, 2306))

    def test_types_and_channels_are_independent(self):
        values = (
            traced(BoundaryType.ONSET, 100, 0, 0),
            traced(BoundaryType.ONSET, 100, 1, 1),
            traced(BoundaryType.OFFSET, 100, 0, 2),
            traced(BoundaryType.ONSET, 200, 0, 3),
            traced(BoundaryType.ONSET, 200, 1, 4),
            traced(BoundaryType.OFFSET, 200, 0, 5),
        )
        result = trace_fixed_refractory(values, refractory_samples=2205)

        self.assertEqual(
            tuple((item.kind, item.sample, item.channel) for item in result.kept),
            (
                (BoundaryType.ONSET, 100, 0),
                (BoundaryType.ONSET, 100, 1),
                (BoundaryType.OFFSET, 100, 0),
            ),
        )
        self.assertEqual(len(result.suppressed), 3)

    def test_zero_or_reverse_same_key_gap_fails_integrity(self):
        with self.assertRaises(AssertionError):
            trace_fixed_refractory(
                (
                    traced(BoundaryType.ONSET, 100, 0, 0),
                    traced(BoundaryType.ONSET, 100, 0, 1),
                )
            )
        with self.assertRaises(AssertionError):
            trace_fixed_refractory(
                (
                    traced(BoundaryType.ONSET, 100, 0, 0),
                    traced(BoundaryType.ONSET, 99, 0, 1),
                )
            )

    def test_decode_infers_once_and_matches_private_trace(self):
        first = score_chunk(
            0,
            4,
            onsets=((0, 0), (3, 0), (3, 1)),
            offsets=((3, 0),),
        )
        second = score_chunk(4, 2, onsets=((5, 0),))
        predictor = ScriptedPredictor(((0, first), (4, second)))

        result = decode_refractory_stream(
            predictor,
            ((0, (0.0,) * 4), (4, (0.0,) * 2)),
            rearm_low_samples=1,
            treatment_refractory_samples=4,
        )

        self.assertEqual(predictor.calls, 2)
        self.assertEqual(result.chunks, 2)
        self.assertEqual(_public_projection(result.traced_control), result.control_candidates)
        self.assertEqual(_public_projection(result.traced_treatment), result.treatment_candidates)
        self.assertEqual(
            tuple((item.kind.value, item.sample) for item in result.treatment_candidates),
            (("onset", 0), ("offset", 3), ("onset", 3), ("onset", 5)),
        )
        self.assertEqual(len(result.suppressions), 1)
        self.assertEqual(result.suppressions[0].candidate.sample, 3)

    def test_exp08_partition_and_simultaneous_support_are_identity_aware(self):
        references = (
            IdentifiedReference(BoundaryType.ONSET, 100, 0, 0),
            IdentifiedReference(BoundaryType.ONSET, 100, 1, 0),
        )
        candidates = (
            traced(BoundaryType.ONSET, 100, 0, 0),
            traced(BoundaryType.ONSET, 110, 0, 1),
            traced(BoundaryType.ONSET, 300, 0, 2),
        )
        statuses, partition, pairs = classify_candidate_statuses(
            references, candidates, tolerance_samples=50
        )

        self.assertEqual(len(pairs), 2)
        self.assertEqual(partition["isolated"], 1)
        self.assertEqual(sum(partition.values()), 1)
        self.assertEqual(statuses[2], "isolated")
        self.assertEqual(_simultaneous_reference_support(references, pairs), (2, 2))

    def test_distance_bins_are_non_overlapping_and_exhaustive(self):
        expected = {
            1: "1-44",
            44: "1-44",
            45: "45-220",
            220: "45-220",
            221: "221-441",
            441: "221-441",
            442: "442-882",
            882: "442-882",
            883: "883-2205",
            2205: "883-2205",
        }
        self.assertEqual({value: _distance_bin(value) for value in expected}, expected)
        with self.assertRaises(AssertionError):
            _distance_bin(0)
        with self.assertRaises(AssertionError):
            _distance_bin(2206)

    def test_parser_locks_the_preregistered_configuration(self):
        parser = create_argument_parser()
        arguments = parser.parse_args([])
        _locked_configuration(arguments)
        self.assertEqual(
            arguments.treatment_refractory_samples,
            TREATMENT_REFRACTORY_SAMPLES,
        )

        arguments.treatment_refractory_samples = 2204
        with self.assertRaises(Exception):
            _locked_configuration(arguments)

    def test_source_loader_accepts_only_completed_exp08(self):
        source = load_relation_source(DEFAULT_SOURCE_REPORT)
        self.assertEqual(source["kind"], "boundary_candidate_relation_audit")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            path.write_text('{"kind":"wrong"}', encoding="utf-8")
            with self.assertRaises(Exception):
                load_relation_source(path)

    def test_timing_summary_includes_signed_absolute_and_causal_metrics(self):
        summary = summarize_delays((-10, 0, 20))
        self.assertEqual(summary["matched_count"], 3)
        self.assertEqual(summary["early_match_count"], 1)
        self.assertEqual(summary["causal_match_count"], 2)
        self.assertEqual(summary["absolute_p50_samples"], 10.0)
        self.assertEqual(summary["absolute_max_samples"], 20)

    def test_decision_uses_the_amended_inclusive_thresholds(self):
        def head(false_positive, true_positive, prediction_count):
            return {
                "control": {"f1": 0.1},
                "treatment": {
                    "f1": 0.2,
                    "false_positive": false_positive,
                    "true_positive": true_positive,
                    "prediction_count": prediction_count,
                    "recall": 0.9,
                },
                "simultaneous_reference_support": {
                    "control_matched_instances": 10,
                    "treatment_matched_instances": 10,
                },
            }

        global_value = {
            "heads": {
                "onset": head(126553, 9055, 19082),
                "offset": head(213049, 8896, 19082),
            },
            "treatment_same_annotated_note_onset_offset_support": {"both": 7198},
        }
        regime = {
            "heads": {
                "onset": {"control": {"f1": 0.1}, "treatment": {"f1": 0.2}},
                "offset": {"control": {"f1": 0.1}, "treatment": {"f1": 0.2}},
            }
        }
        group = {
            "heads": {
                "onset": {"control": {"f1": 0.1}, "treatment": {"f1": 0.2}},
                "offset": {"control": {"f1": 0.1}, "treatment": {"f1": 0.2}},
            }
        }
        aggregates = {
            "global": global_value,
            "comp": regime,
            "solo": regime,
            "family_arrangement": [group for _ in range(12)],
        }

        decision = _decision(aggregates)
        self.assertTrue(decision["useful_filter_accepted"])
        self.assertTrue(
            decision["overprediction_resolved_on_adaptation_validation"]
        )

        global_value["heads"]["onset"]["treatment"]["true_positive"] = 9054
        self.assertFalse(_decision(aggregates)["useful_filter_accepted"])
        global_value["heads"]["onset"]["treatment"]["true_positive"] = 9055
        global_value["heads"]["offset"]["treatment"]["prediction_count"] = 19083
        self.assertFalse(
            _decision(aggregates)[
                "overprediction_resolved_on_adaptation_validation"
            ]
        )


if __name__ == "__main__":
    unittest.main()
