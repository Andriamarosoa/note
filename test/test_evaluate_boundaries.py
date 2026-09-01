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

from causal_note.detector import BoundaryEvent, BoundaryScoreChunk, BoundaryType
from causal_note.guitarset import NoteBoundary, SAMPLE_RATE, index_guitarset
from scripts.evaluate_boundaries import (
    aggregate_track_evaluations,
    compare_onset_multiplicity,
    create_argument_parser,
    evaluate_track_events,
    latency_metrics,
    match_boundaries,
    milliseconds_to_samples,
    onset_multiplicity_metrics,
    parse_predicted_events,
    run_evaluation,
)
from scripts.train_boundaries import split_tracks_by_group


def _onset(event_id, sample):
    return BoundaryEvent(BoundaryType.ONSET, event_id, sample)


def _offset(event_id, sample):
    return BoundaryEvent(BoundaryType.OFFSET, event_id, sample)


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


class _ScriptedFullTrackPredictor:
    slot_count = 6

    def warm_up(self, chunk_size):
        return None

    def reset(self):
        return None

    def predict_chunk(self, samples, *, start_sample):
        onset = [[0.0] * self.slot_count for _ in samples]
        offset = [[0.0] * self.slot_count for _ in samples]
        for absolute, rows in ((100, onset), (700, offset)):
            if start_sample <= absolute < start_sample + len(samples):
                rows[absolute - start_sample][0] = 1.0
        return BoundaryScoreChunk(start_sample, tuple(onset), tuple(offset))


class AnonymousBoundaryMetricsTests(unittest.TestCase):
    def test_overlapping_events_are_matched_without_exposing_slots(self):
        references = (
            NoteBoundary(100, 700),
            NoteBoundary(200, 820),
        )
        predictions = (
            _onset("opaque-A", 103),
            _onset("opaque-B", 205),
            _offset("opaque-A", 698),
            _offset("opaque-B", 830),
        )

        result = evaluate_track_events(
            references,
            predictions,
            onset_tolerance_samples=5,
            offset_tolerance_samples=10,
        )

        self.assertEqual(result.reference_complete_events, 2)
        self.assertEqual(result.predicted_complete_events, 2)
        self.assertEqual(result.predicted_incomplete_events, 0)
        self.assertEqual(result.onset.true_positive, 2)
        self.assertEqual(result.offset.true_positive, 2)
        self.assertEqual(result.associated_interval.true_positive, 2)
        self.assertEqual(result.associated_interval.f1, 1.0)
        self.assertEqual(result.onset_latency.causal_p50_samples, 4.0)

    def test_crossed_ids_fail_interval_association_despite_perfect_boundaries(self):
        references = (
            NoteBoundary(100, 700),
            NoteBoundary(200, 820),
        )
        predictions = (
            _onset("opaque-A", 100),
            _onset("opaque-B", 200),
            _offset("opaque-A", 820),
            _offset("opaque-B", 700),
        )

        result = evaluate_track_events(
            references,
            predictions,
            onset_tolerance_samples=0,
            offset_tolerance_samples=0,
        )

        self.assertEqual(result.onset.f1, 1.0)
        self.assertEqual(result.offset.f1, 1.0)
        self.assertEqual(result.associated_interval.true_positive, 0)
        self.assertEqual(result.associated_interval.f1, 0.0)

    def test_incomplete_and_malformed_event_ids_are_counted(self):
        predictions = (
            _onset("complete", 10),
            _offset("complete", 20),
            _onset("onset-only", 30),
            _offset("offset-only", 40),
            _onset("duplicate", 50),
            _onset("duplicate", 51),
            _offset("duplicate", 60),
        )

        parsed = parse_predicted_events(predictions)

        self.assertEqual(parsed.event_id_count, 4)
        self.assertEqual(len(parsed.intervals), 1)
        self.assertEqual(
            parsed.incomplete_event_ids,
            ("duplicate", "offset-only", "onset-only"),
        )
        self.assertEqual(
            parsed.onset_without_offset_event_ids,
            ("onset-only",),
        )
        self.assertEqual(
            parsed.offset_without_onset_event_ids,
            ("offset-only",),
        )
        self.assertEqual(parsed.malformed_event_ids, ("duplicate",))

    def test_boundary_matching_is_one_to_one(self):
        pairs = match_boundaries((100, 102), (101,), tolerance_samples=2)
        self.assertEqual(len(pairs), 1)


class LatencyMetricTests(unittest.TestCase):
    def test_reports_early_and_nonnegative_causal_delays_separately(self):
        metrics = latency_metrics(((100, 90), (200, 200), (300, 320)))

        self.assertEqual(metrics.matched_count, 3)
        self.assertEqual(metrics.early_match_count, 1)
        self.assertEqual(metrics.causal_match_count, 2)
        self.assertEqual(metrics.signed_p50_samples, 0.0)
        self.assertEqual(metrics.causal_p50_samples, 10.0)
        self.assertEqual(metrics.causal_max_samples, 20)

    def test_millisecond_conversion_uses_project_sample_rate(self):
        self.assertEqual(milliseconds_to_samples(50.0), 2205)

    def test_track_reports_offset_latency_with_the_same_rules(self):
        result = evaluate_track_events(
            (NoteBoundary(100, 700), NoteBoundary(200, 820)),
            (
                _onset("A", 100),
                _offset("A", 690),
                _onset("B", 200),
                _offset("B", 840),
            ),
            onset_tolerance_samples=0,
            offset_tolerance_samples=20,
        )

        self.assertEqual(result.offset_latency.matched_count, 2)
        self.assertEqual(result.offset_latency.early_match_count, 1)
        self.assertEqual(result.offset_latency.causal_match_count, 1)
        self.assertEqual(result.offset_latency.signed_p50_samples, 5.0)
        self.assertEqual(result.offset_latency.causal_p50_samples, 20.0)


class MultiplicityMetricTests(unittest.TestCase):
    def test_counts_only_exactly_equal_onset_positions(self):
        metrics = onset_multiplicity_metrics((100, 100, 101, 102, 102, 102))

        self.assertEqual(metrics.onset_count, 6)
        self.assertEqual(metrics.unique_position_count, 3)
        self.assertEqual(metrics.simultaneous_position_count, 2)
        self.assertEqual(metrics.simultaneous_onset_count, 5)
        self.assertEqual(metrics.extra_simultaneous_onset_count, 3)
        self.assertEqual(metrics.maximum_multiplicity, 3)
        self.assertEqual(metrics.position_histogram, {"1": 1, "2": 1, "3": 1})

    def test_compares_cardinality_at_matched_positions(self):
        comparison = compare_onset_multiplicity(
            (100, 100, 200, 300, 300, 300),
            (101, 200, 200, 400),
            tolerance_samples=2,
        )

        self.assertEqual(comparison.matched_position_count, 2)
        self.assertEqual(comparison.exact_multiplicity_match_count, 0)
        self.assertEqual(comparison.underpredicted_onset_count, 4)
        self.assertEqual(comparison.overpredicted_onset_count, 2)
        self.assertEqual(comparison.absolute_multiplicity_error, 6)

    def test_neighbouring_prediction_positions_are_not_clustered(self):
        comparison = compare_onset_multiplicity(
            (100, 100),
            (100, 101),
            tolerance_samples=2,
        )

        self.assertEqual(comparison.prediction.unique_position_count, 2)
        self.assertEqual(comparison.matched_position_count, 1)
        self.assertEqual(comparison.underpredicted_onset_count, 1)
        self.assertEqual(comparison.overpredicted_onset_count, 1)


class AggregateMetricTests(unittest.TestCase):
    def test_rates_count_false_and_orphan_event_ids_per_own_duration(self):
        evaluation = evaluate_track_events(
            (NoteBoundary(100, 200),),
            (
                _onset("true", 100),
                _offset("true", 200),
                _onset("false-complete", 300),
                _offset("false-complete", 400),
                _onset("onset-only", 500),
                _offset("offset-only", 600),
                _offset("malformed", 650),
                _onset("malformed", 700),
            ),
            onset_tolerance_samples=0,
            offset_tolerance_samples=0,
        )
        aggregate = aggregate_track_evaluations(
            (evaluation,),
            (SAMPLE_RATE * 3600,),
            (((100, 100),),),
            (((200, 200),),),
        )

        self.assertEqual(aggregate["duration"]["audio_hours"], 1.0)
        self.assertEqual(
            aggregate["counts"]["predicted_onset_without_offset_events"],
            1,
        )
        self.assertEqual(
            aggregate["counts"]["predicted_offset_without_onset_events"],
            1,
        )
        self.assertEqual(
            aggregate["counts"]["predicted_malformed_events"],
            1,
        )
        self.assertEqual(
            aggregate["rates_per_hour"],
            {
                "false_onsets": 3.0,
                "false_offsets": 3.0,
                "false_complete_intervals": 1.0,
                "orphan_ids": 3.0,
                "false_event_ids": 4.0,
            },
        )

    def test_latency_percentiles_are_rebuilt_from_raw_track_pairs(self):
        first = evaluate_track_events(
            (NoteBoundary(100, 500), NoteBoundary(200, 600)),
            (
                _onset("A", 100),
                _offset("A", 500),
                _onset("B", 300),
                _offset("B", 700),
            ),
            onset_tolerance_samples=1000,
            offset_tolerance_samples=1000,
        )
        second = evaluate_track_events(
            (NoteBoundary(100, 500),),
            (_onset("C", 1100), _offset("C", 1500)),
            onset_tolerance_samples=1000,
            offset_tolerance_samples=1000,
        )
        aggregate = aggregate_track_evaluations(
            (first, second),
            (SAMPLE_RATE, SAMPLE_RATE),
            (((100, 100), (200, 300)), ((100, 1100),)),
            (((500, 500), (600, 700)), ((500, 1500),)),
        )

        # Raw delays [0, 100, 1000] have p50=100; averaging per-track
        # percentiles would incorrectly produce (50 + 1000) / 2 = 525.
        self.assertEqual(
            aggregate["metrics"]["onset_latency"]["causal_p50_samples"],
            100.0,
        )
        self.assertEqual(
            aggregate["metrics"]["offset_latency"]["causal_p50_samples"],
            100.0,
        )


class EvaluationCliGuardTests(unittest.TestCase):
    def test_player_05_is_refused_by_argument_parser(self):
        parser = create_argument_parser()
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(("--players", "05"))

    def test_release_threshold_flags_default_to_none_and_accept_probabilities(self):
        parser = create_argument_parser()
        defaults = parser.parse_args(())
        explicit = parser.parse_args(
            (
                "--onset-release-threshold",
                "0.5",
                "--offset-release-threshold",
                "0.45",
            )
        )

        self.assertIsNone(defaults.onset_release_threshold)
        self.assertIsNone(defaults.offset_release_threshold)
        self.assertEqual(explicit.onset_release_threshold, 0.5)
        self.assertEqual(explicit.offset_release_threshold, 0.45)


class FullTrackEvaluationTests(unittest.TestCase):
    def test_rebuilds_seed_1337_split_and_never_opens_player_05(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            annotation_zip = root / "annotation.zip"
            audio_zip = root / "audio_mono-pickup_mix.zip"
            _write_zip(
                annotation_zip,
                (
                    ("00_alpha_comp.jams", _jams()),
                    ("00_alpha_solo.jams", _jams()),
                    ("01_beta_comp.jams", _jams()),
                    ("01_beta_solo.jams", _jams()),
                    # If evaluation opens this excluded member, JSON parsing fails.
                    ("05_forbidden_comp.jams", b"not-json"),
                ),
            )
            _write_zip(
                audio_zip,
                (
                    ("00_alpha_comp_mix.wav", _wav_bytes(900)),
                    ("00_alpha_solo_mix.wav", _wav_bytes(900)),
                    ("01_beta_comp_mix.wav", _wav_bytes(900)),
                    ("01_beta_solo_mix.wav", _wav_bytes(900)),
                    # If evaluation opens this excluded member, WAV parsing fails.
                    ("05_forbidden_comp_mix.wav", b"not-wav"),
                ),
            )
            _, validation = split_tracks_by_group(
                index_guitarset(root),
                validation_fraction=0.2,
                seed=1337,
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
            arguments = create_argument_parser().parse_args(
                (
                    str(root),
                    "--model",
                    str(model),
                    "--metadata",
                    str(metadata),
                    "--chunk-size",
                    "256",
                    "--onset-threshold",
                    "0.55",
                    "--offset-threshold",
                    "0.6",
                    "--onset-release-threshold",
                    "0.5",
                    "--offset-release-threshold",
                    "0.45",
                    "--onset-tolerance-ms",
                    "0",
                    "--offset-tolerance-ms",
                    "0",
                )
            )

            with patch(
                "scripts.evaluate_boundaries.KerasBoundaryPredictor.from_path",
                return_value=_ScriptedFullTrackPredictor(),
            ):
                result = run_evaluation(arguments)

            self.assertEqual(result["schema_version"], 2)
            self.assertEqual(result["split"]["validation_tracks"], 2)
            self.assertFalse(result["split"]["player_05_read"])
            self.assertEqual(result["counts"]["reference_complete_events"], 2)
            self.assertEqual(result["counts"]["predicted_complete_events"], 2)
            self.assertEqual(result["metrics"]["onset"]["f1"], 1.0)
            self.assertEqual(result["metrics"]["offset"]["f1"], 1.0)
            self.assertEqual(result["metrics"]["associated_interval"]["f1"], 1.0)
            self.assertEqual(result["configuration"]["onset_threshold"], 0.55)
            self.assertEqual(result["configuration"]["offset_threshold"], 0.6)
            self.assertEqual(
                result["configuration"]["onset_release_threshold"],
                0.5,
            )
            self.assertEqual(
                result["configuration"]["offset_release_threshold"],
                0.45,
            )
            self.assertEqual(result["metrics"]["offset_latency"]["matched_count"], 2)
            self.assertEqual(result["aggregates"]["global"]["counts"]["tracks"], 2)
            self.assertEqual(result["aggregates"]["comp"]["counts"]["tracks"], 1)
            self.assertEqual(result["aggregates"]["solo"]["counts"]["tracks"], 1)
            self.assertEqual(
                result["aggregates"]["comp"]["duration"]["audio_samples"],
                900,
            )
            self.assertEqual(
                result["aggregates"]["solo"]["duration"]["audio_samples"],
                900,
            )


if __name__ == "__main__":
    unittest.main()
