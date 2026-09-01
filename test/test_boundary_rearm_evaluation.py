import argparse
import json
from pathlib import Path
import tempfile
import unittest

from causal_note.detector import BoundaryCandidate, BoundaryScoreChunk, BoundaryType
from causal_note.guitarset import NoteBoundary
from scripts.evaluate_boundaries import EvaluationError
from scripts.evaluate_boundary_candidates import evaluate_boundary_lists
from scripts.evaluate_boundary_rearm import (
    CONTROL_REARM_LOW_SAMPLES,
    DEFAULT_SOURCE_REPORT,
    TREATMENT_REARM_LOW_SAMPLES,
    CandidateSourceExpectations,
    LowRunMorphology,
    RearmTrackOutcome,
    _decode_shared_score_stream,
    _locked_configuration,
    _metric_integer_counts,
    create_argument_parser,
    decode_rearm_stream,
    load_candidate_source,
    validate_control_reproduction,
)


class ScriptedPredictor:
    slot_count = 1

    def __init__(self, chunks):
        self._chunks = iter(chunks)
        self.calls = 0

    def predict_chunk(self, samples, *, start_sample):
        self.calls += 1
        chunk = next(self._chunks)
        if chunk.start_sample != start_sample:
            raise AssertionError("unexpected start sample")
        if chunk.sample_count != len(samples):
            raise AssertionError("unexpected sample count")
        return chunk


class RecordingDecoder:
    def __init__(self):
        self.score_ids = []

    def process_chunk(self, scores):
        self.score_ids.append(id(scores))
        return ()


def score_chunk(start, onset, offset=None):
    onset_rows = tuple((float(value),) for value in onset)
    if offset is None:
        offset = [0.0] * len(onset_rows)
    offset_rows = tuple((float(value),) for value in offset)
    return BoundaryScoreChunk(start, onset_rows, offset_rows)


class BoundaryRearmEvaluationTests(unittest.TestCase):
    def test_same_score_object_is_delivered_once_to_both_decoders(self):
        chunks = (score_chunk(0, [0.0, 0.0]), score_chunk(2, [0.0]))
        predictor = ScriptedPredictor(chunks)
        control = RecordingDecoder()
        treatment = RecordingDecoder()
        morphology = LowRunMorphology(slot_count=1, threshold=0.55)

        result = _decode_shared_score_stream(
            predictor,
            ((0, (0.0, 0.0)), (2, (0.0,))),
            control,
            treatment,
            morphology,
        )

        self.assertEqual(predictor.calls, 2)
        self.assertEqual(result.chunks, 2)
        self.assertEqual(control.score_ids, treatment.score_ids)
        self.assertEqual(len(set(control.score_ids)), 2)

    def test_n16_suppresses_only_short_bounded_low_run(self):
        # High, five lows, high, sixteen lows, high.  N=1 emits all three
        # fronts while N=16 suppresses only the middle front.
        onset = [0.6] + [0.0] * 5 + [0.6] + [0.0] * 16 + [0.6]
        chunk = score_chunk(0, onset)
        predictor = ScriptedPredictor((chunk,))

        result = decode_rearm_stream(
            predictor,
            ((0, tuple(0.0 for _ in onset)),),
        )

        self.assertEqual(
            result.control_candidates,
            (
                BoundaryCandidate(BoundaryType.ONSET, 0),
                BoundaryCandidate(BoundaryType.ONSET, 6),
                BoundaryCandidate(BoundaryType.ONSET, 23),
            ),
        )
        self.assertEqual(
            result.treatment_candidates,
            (
                BoundaryCandidate(BoundaryType.ONSET, 0),
                BoundaryCandidate(BoundaryType.ONSET, 23),
            ),
        )
        onset_morphology = result.morphology["onset"]
        self.assertEqual(onset_morphology["histogram"]["5"], 1)
        self.assertEqual(onset_morphology["histogram"]["16_plus"], 1)
        self.assertEqual(
            onset_morphology["shorter_than_treatment_run_count"], 1
        )

    def test_low_run_morphology_persists_across_chunks(self):
        morphology = LowRunMorphology(slot_count=1, threshold=0.55)
        morphology.process_chunk(score_chunk(0, [0.6, 0.0, 0.0]))
        morphology.process_chunk(score_chunk(3, [0.0, 0.6]))

        summary = morphology.summary()["onset"]
        self.assertEqual(summary["bounded_low_run_count"], 1)
        self.assertEqual(summary["bounded_low_run_samples"], 3)
        self.assertEqual(summary["histogram"]["3"], 1)

    def test_completed_exp06_source_is_loaded_per_track(self):
        source = load_candidate_source(DEFAULT_SOURCE_REPORT)

        self.assertEqual(source.kind, "boundary_candidate_evaluation")
        self.assertEqual(len(source.tracks), 60)
        self.assertEqual(
            source.global_metrics["onset"]["prediction_count"], 401104
        )
        self.assertEqual(
            source.global_metrics["offset"]["prediction_count"], 658417
        )

    def test_source_must_be_unassociated_exp06_shape(self):
        invalid = {
            "kind": "boundary_candidate_evaluation",
            "configuration": {
                "onset_threshold": 0.55,
                "offset_threshold": 0.55,
                "onset_release_threshold": 0.55,
                "offset_release_threshold": 0.55,
                "association": {"treatment": True},
                "interval_metrics": False,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            path.write_text(json.dumps(invalid), encoding="utf-8")
            with self.assertRaises(EvaluationError):
                load_candidate_source(path)

    def test_cli_and_experiment_values_are_locked(self):
        parser = create_argument_parser()
        arguments = parser.parse_args(
            ["--output", "out.json", "--source-report", "source.json"]
        )
        _locked_configuration(arguments)
        self.assertEqual(
            arguments.control_rearm_low_samples, CONTROL_REARM_LOW_SAMPLES
        )
        self.assertEqual(
            arguments.treatment_rearm_low_samples,
            TREATMENT_REARM_LOW_SAMPLES,
        )

        changed = argparse.Namespace(**vars(arguments))
        changed.treatment_rearm_low_samples = 15
        with self.assertRaises(EvaluationError):
            _locked_configuration(changed)
        with self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "--output",
                    "out.json",
                    "--source-report",
                    "source.json",
                    "--players",
                    "05",
                ]
            )

    def test_control_reproduction_checks_global_and_track_counts(self):
        references = (NoteBoundary(10, 20),)
        evaluation, onset_pairs, offset_pairs = evaluate_boundary_lists(
            references,
            (10,),
            (20,),
            onset_tolerance_samples=0,
            offset_tolerance_samples=0,
        )
        morphology = {
            "onset": {
                "bounded_low_run_count": 0,
                "bounded_low_run_samples": 0,
                "maximum_low_run_samples": 0,
                "shorter_than_treatment_run_count": 0,
                "histogram": {},
            },
            "offset": {
                "bounded_low_run_count": 0,
                "bounded_low_run_samples": 0,
                "maximum_low_run_samples": 0,
                "shorter_than_treatment_run_count": 0,
                "histogram": {},
            },
        }
        outcome = RearmTrackOutcome(
            annotation_member="00_test_comp.jams",
            audio_member="00_test_comp_mix.wav",
            family="test",
            arrangement="comp",
            audio_samples=100,
            control=evaluation,
            treatment=evaluation,
            control_open_events=0,
            control_onset_pairs=onset_pairs,
            control_offset_pairs=offset_pairs,
            treatment_onset_pairs=onset_pairs,
            treatment_offset_pairs=offset_pairs,
            morphology=morphology,
        )
        counts = {
            head: _metric_integer_counts(evaluation, head)
            for head in ("onset", "offset")
        }
        expectations = CandidateSourceExpectations(
            global_metrics=counts,
            tracks={outcome.annotation_member: counts},
            kind="boundary_candidate_evaluation",
        )

        answer = validate_control_reproduction(expectations, (outcome,))
        self.assertTrue(answer["matched"])
        self.assertEqual(answer["per_track_counts_checked"], 1)

        wrong = CandidateSourceExpectations(
            global_metrics={
                "onset": {**counts["onset"], "prediction_count": 2},
                "offset": counts["offset"],
            },
            tracks=expectations.tracks,
            kind=expectations.kind,
        )
        with self.assertRaises(EvaluationError):
            validate_control_reproduction(wrong, (outcome,))


if __name__ == "__main__":
    unittest.main()
