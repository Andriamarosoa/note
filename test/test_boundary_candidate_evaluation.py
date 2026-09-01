import io
from contextlib import redirect_stderr
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import wave
import zipfile

from causal_note.detector import BoundaryScoreChunk, BoundaryType
from causal_note.guitarset import NoteBoundary, SAMPLE_RATE, index_guitarset
from scripts.evaluate_boundaries import EvaluationError
from scripts.evaluate_boundary_candidates import (
    OFFICIAL_THRESHOLD,
    _decode_shared_score_stream,
    create_argument_parser,
    decode_candidate_stream,
    evaluate_boundary_lists,
    load_source_expectations,
    refuse_output_overwrite,
    run_candidate_evaluation,
    validate_control_reproduction,
    write_json_atomically,
)
from scripts.train_boundaries import split_tracks_by_group


class _ScriptedPredictor:
    slot_count = 1

    def __init__(self, onset, offset):
        self.onset = tuple(onset)
        self.offset = tuple(offset)
        self.calls = []

    def predict_chunk(self, samples, *, start_sample):
        self.calls.append((start_sample, len(samples)))
        end = start_sample + len(samples)
        return BoundaryScoreChunk(
            start_sample,
            tuple((self.onset[index],) for index in range(start_sample, end)),
            tuple((self.offset[index],) for index in range(start_sample, end)),
        )


class _ObjectRecordingDecoder:
    def __init__(self):
        self.scores = []

    def process_chunk(self, scores):
        # Retain each object so the assertion cannot be fooled by Python
        # reusing an id after an earlier chunk is released.
        self.scores.append(scores)
        return ()

    def active_events(self):
        return ()


class CandidateStreamingTests(unittest.TestCase):
    def test_same_score_object_is_delivered_to_both_decoders_once_per_chunk(self):
        predictor = _ScriptedPredictor((0.0,) * 6, (0.0,) * 6)
        control = _ObjectRecordingDecoder()
        treatment = _ObjectRecordingDecoder()

        result = _decode_shared_score_stream(
            predictor,
            ((0, (0.0, 0.0)), (2, (0.0, 0.0)), (4, (0.0, 0.0))),
            control,
            treatment,
        )

        self.assertEqual(predictor.calls, [(0, 2), (2, 2), (4, 2)])
        self.assertEqual(result.chunks, 3)
        for control_scores, treatment_scores in zip(
            control.scores, treatment.scores
        ):
            self.assertIs(control_scores, treatment_scores)
        self.assertIsNot(control.scores[0], control.scores[1])
        self.assertIsNot(control.scores[1], control.scores[2])

    def test_treatment_keeps_retrigger_while_control_waits_for_offset(self):
        predictor = _ScriptedPredictor(
            onset=(0.0, 0.6, 0.0, 0.6, 0.0, 0.0),
            offset=(0.0, 0.0, 0.0, 0.0, 0.6, 0.0),
        )

        result = decode_candidate_stream(
            predictor,
            ((0, (0.0, 0.0, 0.0)), (3, (0.0, 0.0, 0.0))),
        )

        self.assertEqual(
            tuple((event.kind.value, event.sample) for event in result.control_events),
            (("onset", 1), ("offset", 4)),
        )
        self.assertEqual(
            tuple((event.kind.value, event.sample) for event in result.candidates),
            (("onset", 1), ("onset", 3), ("offset", 4)),
        )
        self.assertEqual(result.control_open_events, 0)

    def test_treatment_preserves_two_simultaneous_slot_crossings(self):
        class _TwoSlotPredictor:
            slot_count = 2

            def predict_chunk(self, samples, *, start_sample):
                self.calls = getattr(self, "calls", 0) + 1
                return BoundaryScoreChunk(
                    start_sample,
                    ((0.0, 0.0), (0.6, 0.6), (0.0, 0.0)),
                    ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0)),
                )

        predictor = _TwoSlotPredictor()
        result = decode_candidate_stream(
            predictor,
            ((0, (0.0, 0.0, 0.0)),),
        )

        self.assertEqual(predictor.calls, 1)
        self.assertEqual(
            tuple((candidate.kind, candidate.sample) for candidate in result.candidates),
            (
                (BoundaryType.ONSET, 1),
                (BoundaryType.ONSET, 1),
            ),
        )

    def test_boundary_metrics_preserve_reference_and_prediction_multiplicity(self):
        references = (
            NoteBoundary(100, 700),
            NoteBoundary(100, 820),
            NoteBoundary(200, 900),
        )

        evaluation, onset_pairs, offset_pairs = evaluate_boundary_lists(
            references,
            predicted_onsets=(100, 100, 200),
            predicted_offsets=(700, 820, 900),
            onset_tolerance_samples=0,
            offset_tolerance_samples=0,
        )

        self.assertEqual(evaluation.onset.true_positive, 3)
        self.assertEqual(evaluation.offset.true_positive, 3)
        self.assertEqual(len(onset_pairs), 3)
        self.assertEqual(len(offset_pairs), 3)
        self.assertEqual(
            evaluation.onset_multiplicity.prediction.maximum_multiplicity,
            2,
        )
        self.assertFalse(hasattr(evaluation, "associated_interval"))


def _wav_bytes(sample_count):
    payload = io.BytesIO()
    with wave.open(payload, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(SAMPLE_RATE)
        writer.writeframes(struct.pack(f"<{sample_count}h", *((0,) * sample_count)))
    return payload.getvalue()


def _jams(onset=100, offset=700):
    return json.dumps(
        {
            "annotations": [
                {
                    "namespace": "note_midi",
                    "annotation_metadata": {"data_source": 0},
                    "data": [
                        {
                            "time": onset / SAMPLE_RATE,
                            "duration": (offset - onset) / SAMPLE_RATE,
                        }
                    ],
                }
            ]
        }
    )


def _write_zip(path, members):
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in members:
            archive.writestr(name, content)


class _FullTrackPredictor:
    slot_count = 6

    def __init__(self):
        self.calls = []
        self.reset_count = 0

    def warm_up(self, chunk_size):
        return None

    def reset(self):
        self.reset_count += 1

    def predict_chunk(self, samples, *, start_sample):
        self.calls.append((start_sample, len(samples)))
        onset = [[0.0] * self.slot_count for _ in samples]
        offset = [[0.0] * self.slot_count for _ in samples]
        for absolute, rows in ((100, onset), (700, offset)):
            if start_sample <= absolute < start_sample + len(samples):
                rows[absolute - start_sample][0] = 1.0
        return BoundaryScoreChunk(start_sample, tuple(onset), tuple(offset))


def _protocol(reference_count):
    return {
        "baseline": {
            "reference_onsets": reference_count,
            "reference_offsets": reference_count,
            "associated_decoder": {
                "predicted_onsets": reference_count,
                "predicted_offsets": reference_count,
                "incomplete_events": 0,
                "onset_true_positive": reference_count,
                "onset_false_positive": 0,
                "onset_false_negative": 0,
                "offset_true_positive": reference_count,
                "offset_false_positive": 0,
                "offset_false_negative": 0,
            },
        }
    }


def _per_track_source_report(validation_members):
    def head(count):
        return {
            "reference_count": count,
            "prediction_count": count,
            "true_positive": count,
            "false_positive": 0,
            "false_negative": 0,
        }

    count = len(validation_members)
    return {
        "kind": "boundary_evaluation",
        "aggregates": {
            "global": {
                "metrics": {"onset": head(count), "offset": head(count)},
                "counts": {"predicted_incomplete_events": 0},
            }
        },
        "tracks": [
            {
                "annotation_member": member,
                "metrics": {
                    "onset": head(1),
                    "offset": head(1),
                    "predicted_incomplete_events": 0,
                },
            }
            for member in validation_members
        ],
    }


class CandidateCliAndSourceTests(unittest.TestCase):
    def test_cli_is_locked_to_v7_thresholds_and_player_05(self):
        parser = create_argument_parser()
        arguments = parser.parse_args(
            (
                "--source-report",
                "source.json",
                "--output",
                "result.json",
            )
        )
        self.assertEqual(arguments.onset_threshold, OFFICIAL_THRESHOLD)
        self.assertEqual(arguments.offset_threshold, OFFICIAL_THRESHOLD)
        self.assertEqual(arguments.onset_release_threshold, OFFICIAL_THRESHOLD)
        self.assertEqual(arguments.offset_release_threshold, OFFICIAL_THRESHOLD)
        self.assertNotIn("05", arguments.players)

        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(
                (
                    "--source-report",
                    "source.json",
                    "--output",
                    "result.json",
                    "--players",
                    "05",
                )
            )

    def test_atomic_writer_refuses_an_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            write_json_atomically(output, {"complete": True})
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), {"complete": True})
            with self.assertRaises(FileExistsError):
                refuse_output_overwrite(output)
            with self.assertRaises(FileExistsError):
                write_json_atomically(output, {"complete": False})

    def test_protocol_counts_are_loaded_and_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "protocol.json"
            source.write_text(json.dumps(_protocol(1)), encoding="utf-8")
            expectations = load_source_expectations(source)

            self.assertEqual(
                expectations.global_metrics["onset"]["prediction_count"],
                1,
            )
            with self.assertRaisesRegex(EvaluationError, "differ from source"):
                validate_control_reproduction(expectations, ())

    def test_nonlocked_threshold_is_rejected_before_dataset_access(self):
        parser = create_argument_parser()
        with tempfile.TemporaryDirectory() as directory:
            arguments = parser.parse_args(
                (
                    "missing-dataset",
                    "--output",
                    str(Path(directory) / "result.json"),
                    "--source-report",
                    str(Path(directory) / "source.json"),
                    "--onset-threshold",
                    "0.5",
                )
            )
            with self.assertRaisesRegex(EvaluationError, "locked to 0.55"):
                run_candidate_evaluation(arguments)


class FullTrackCandidateEvaluationTests(unittest.TestCase):
    def test_one_pass_report_has_only_boundary_metrics_and_validates_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_zip(
                root / "annotation.zip",
                (
                    ("00_alpha_comp.jams", _jams()),
                    ("00_alpha_solo.jams", _jams()),
                    ("01_beta_comp.jams", _jams()),
                    ("01_beta_solo.jams", _jams()),
                    ("05_forbidden_comp.jams", b"not-json"),
                ),
            )
            _write_zip(
                root / "audio_mono-pickup_mix.zip",
                (
                    ("00_alpha_comp_mix.wav", _wav_bytes(900)),
                    ("00_alpha_solo_mix.wav", _wav_bytes(900)),
                    ("01_beta_comp_mix.wav", _wav_bytes(900)),
                    ("01_beta_solo_mix.wav", _wav_bytes(900)),
                    ("05_forbidden_comp_mix.wav", b"not-wav"),
                ),
            )
            _, validation = split_tracks_by_group(
                index_guitarset(root), validation_fraction=0.2, seed=1337
            )
            model = root / "model.keras"
            metadata = root / "model.metadata.json"
            metadata.write_text(
                json.dumps(
                    {
                        "seed": 1337,
                        "selected_players": ["00", "01", "02", "03", "04"],
                        "split": {
                            "validation_members": [
                                track.annotation_member for track in validation
                            ]
                        },
                        "model": {"receptive_field_samples": 1},
                    }
                ),
                encoding="utf-8",
            )
            source = root / "protocol.json"
            source.write_text(
                json.dumps(
                    _per_track_source_report(
                        tuple(track.annotation_member for track in validation)
                    )
                ),
                encoding="utf-8",
            )
            output = root / "result.json"
            predictor = _FullTrackPredictor()
            arguments = create_argument_parser().parse_args(
                (
                    str(root),
                    "--model",
                    str(model),
                    "--metadata",
                    str(metadata),
                    "--output",
                    str(output),
                    "--source-report",
                    str(source),
                    "--chunk-size",
                    "256",
                    "--onset-tolerance-ms",
                    "0",
                    "--offset-tolerance-ms",
                    "0",
                )
            )

            with patch(
                "scripts.evaluate_boundary_candidates.KerasBoundaryPredictor.from_path",
                return_value=predictor,
            ), redirect_stderr(io.StringIO()):
                result = run_candidate_evaluation(arguments)

            self.assertTrue(output.exists())
            self.assertEqual(result["split"]["validation_tracks"], 2)
            self.assertFalse(result["split"]["player_05_read"])
            self.assertTrue(result["source_validation"]["matched"])
            self.assertEqual(result["runtime"]["predictor_calls"], len(predictor.calls))
            self.assertEqual(result["runtime"]["control_score_chunk_deliveries"], len(predictor.calls))
            self.assertEqual(result["runtime"]["treatment_score_chunk_deliveries"], len(predictor.calls))
            self.assertEqual(
                result["aggregates"]["global"]["control"]["metrics"]["onset"]["f1"],
                1.0,
            )
            self.assertEqual(
                result["aggregates"]["global"]["treatment"]["metrics"]["offset"]["f1"],
                1.0,
            )
            self.assertEqual(len(result["aggregates"]["family_arrangement"]), 2)
            self.assertEqual(result["aggregates"]["comp"]["control"]["tracks"], 1)
            self.assertEqual(result["aggregates"]["solo"]["control"]["tracks"], 1)
            self.assertEqual(
                sum(
                    group["control"]["metrics"]["onset"]["prediction_count"]
                    for group in result["aggregates"]["family_arrangement"]
                ),
                result["aggregates"]["global"]["control"]["metrics"]["onset"]["prediction_count"],
            )
            self.assertTrue(
                all(track["control"]["metrics"]["onset"]["f1"] == 1.0 for track in result["tracks"])
            )
            self.assertNotIn("associated_interval", json.dumps(result))
            self.assertNotIn("event_id", json.dumps(result["aggregates"]))


if __name__ == "__main__":
    unittest.main()
