import tempfile
from pathlib import Path
import unittest

from causal_note.detector import BoundaryScoreChunk, LiveModelDetector
from scripts.evaluate_boundaries import EvaluationError
from scripts.sweep_boundary_hysteresis import (
    DEFAULT_ENTRY_THRESHOLD,
    DEFAULT_RELEASE_THRESHOLDS,
    create_argument_parser,
    decode_hysteresis_stream,
    refuse_output_overwrite,
    run_hysteresis_sweep,
    validate_hysteresis_thresholds,
)


class _ScriptedPredictor:
    slot_count = 1

    def __init__(self, onset, offset):
        self.onset = tuple(onset)
        self.offset = tuple(offset)
        self.calls = []

    def predict_chunk(self, samples, *, start_sample):
        self.calls.append((start_sample, len(samples)))
        end_sample = start_sample + len(samples)
        return BoundaryScoreChunk(
            start_sample,
            tuple(
                (self.onset[index],)
                for index in range(start_sample, end_sample)
            ),
            tuple(
                (self.offset[index],)
                for index in range(start_sample, end_sample)
            ),
        )


class HysteresisSweepStreamingTests(unittest.TestCase):
    def test_predictor_is_called_once_per_chunk_for_all_releases(self):
        predictor = _ScriptedPredictor(
            onset=(0.0,) * 6,
            offset=(0.0,) * 6,
        )
        chunks = (
            (0, (0.0, 0.0)),
            (2, (0.0, 0.0)),
            (4, (0.0, 0.0)),
        )

        result = decode_hysteresis_stream(
            predictor,
            chunks,
            entry_threshold=0.55,
            release_thresholds=(0.55, 0.50, 0.45),
        )

        self.assertEqual(predictor.calls, [(0, 2), (2, 2), (4, 2)])
        self.assertEqual(result.chunks, 3)
        self.assertEqual(tuple(result.events), (0.55, 0.50, 0.45))
        self.assertEqual(
            result.candidate_thresholds[0.50].onset_release_threshold,
            0.50,
        )
        self.assertEqual(
            result.candidate_thresholds[0.50].offset_release_threshold,
            0.50,
        )

    def test_release_states_are_independent_across_chunk_boundary(self):
        predictor = _ScriptedPredictor(
            onset=(0.60, 0.52, 0.60),
            offset=(0.00, 0.60, 0.52),
        )

        result = decode_hysteresis_stream(
            predictor,
            ((0, (0.0, 0.0)), (2, (0.0,))),
            entry_threshold=0.55,
            release_thresholds=(0.55, 0.50),
        )

        self.assertEqual(
            tuple((event.kind.value, event.sample) for event in result.events[0.55]),
            (("onset", 0), ("offset", 1), ("onset", 2)),
        )
        self.assertEqual(
            tuple((event.kind.value, event.sample) for event in result.events[0.50]),
            (("onset", 0), ("offset", 1)),
        )

    def test_fixed_onset_changes_only_offset_hysteresis_across_chunks(self):
        predictor = _ScriptedPredictor(
            # The release-band dip at sample 1 is bounded by onset highs on
            # opposite sides of the chunk boundary.  Fixed onset release at
            # entry must preserve the retrigger for both candidates.
            onset=(0.60, 0.52, 0.60, 0.00),
            # The offset dip at sample 2 crosses the chunk boundary before
            # the next high.  Only the 0.55 control rearms and closes at 3.
            offset=(0.00, 0.60, 0.52, 0.60),
        )

        result = decode_hysteresis_stream(
            predictor,
            ((0, (0.0, 0.0)), (2, (0.0, 0.0))),
            entry_threshold=0.55,
            release_thresholds=(0.55, 0.50),
            fixed_onset_release_threshold=0.55,
        )

        control = tuple(
            (event.kind.value, event.sample) for event in result.events[0.55]
        )
        treatment = tuple(
            (event.kind.value, event.sample) for event in result.events[0.50]
        )
        self.assertEqual(
            control,
            (("onset", 0), ("offset", 1), ("onset", 2), ("offset", 3)),
        )
        self.assertEqual(
            treatment,
            (("onset", 0), ("offset", 1), ("onset", 2)),
        )
        self.assertEqual(
            tuple(sample for kind, sample in control if kind == "onset"),
            tuple(sample for kind, sample in treatment if kind == "onset"),
        )
        self.assertEqual(predictor.calls, [(0, 2), (2, 2)])

        control_thresholds = result.candidate_thresholds[0.55]
        treatment_thresholds = result.candidate_thresholds[0.50]
        self.assertEqual(control_thresholds.onset_release_threshold, 0.55)
        self.assertEqual(control_thresholds.offset_release_threshold, 0.55)
        self.assertEqual(treatment_thresholds.onset_release_threshold, 0.55)
        self.assertEqual(treatment_thresholds.offset_release_threshold, 0.50)
        self.assertEqual(
            result.morphology[0.50]["onset"]["bridged_dip_count"],
            0,
        )
        self.assertEqual(
            result.morphology[0.50]["offset"]["bridged_dip_count"],
            1,
        )

    def test_release_equal_entry_matches_simple_live_decoding(self):
        onset = (0.0, 0.60, 0.52, 0.0, 0.80, 0.0)
        offset = (0.0, 0.0, 0.60, 0.52, 0.0, 0.90)
        chunks = ((0, (0.0, 0.0, 0.0)), (3, (0.0, 0.0, 0.0)))

        swept = decode_hysteresis_stream(
            _ScriptedPredictor(onset, offset),
            chunks,
            entry_threshold=0.55,
            release_thresholds=(0.55,),
        ).events[0.55]

        detector = LiveModelDetector(
            _ScriptedPredictor(onset, offset),
            onset_threshold=0.55,
            offset_threshold=0.55,
            onset_release_threshold=0.55,
            offset_release_threshold=0.55,
        )
        simple = tuple(
            event
            for start_sample, samples in chunks
            for event in detector.process_chunk(
                samples,
                start_sample=start_sample,
            )
        )

        self.assertEqual(swept, simple)

    def test_morphology_counts_only_release_band_dips_bounded_by_high_runs(self):
        predictor = _ScriptedPredictor(
            # One two-sample bridged dip across the chunk boundary, then a
            # below-release dip that must not be counted.
            onset=(0.60, 0.52, 0.51, 0.60, 0.49, 0.60),
            offset=(0.0,) * 6,
        )

        result = decode_hysteresis_stream(
            predictor,
            ((0, (0.0, 0.0)), (2, (0.0, 0.0)), (4, (0.0, 0.0))),
            entry_threshold=0.55,
            release_thresholds=(0.55, 0.50),
        )

        control = result.morphology[0.55]["onset"]
        treated = result.morphology[0.50]["onset"]
        self.assertEqual(control["bridged_dip_count"], 0)
        self.assertEqual(treated["bridged_dip_count"], 1)
        self.assertEqual(treated["bridged_dip_samples"], 2)
        self.assertEqual(treated["maximum_bridged_dip_samples"], 2)


class HysteresisSweepCliTests(unittest.TestCase):
    def test_defaults_seed_and_locked_player_choices(self):
        parser = create_argument_parser()
        arguments = parser.parse_args(["--output", "sweep.json"])
        self.assertEqual(arguments.entry_threshold, DEFAULT_ENTRY_THRESHOLD)
        self.assertEqual(
            tuple(arguments.release_thresholds),
            DEFAULT_RELEASE_THRESHOLDS,
        )
        self.assertEqual(arguments.seed, 1337)
        self.assertNotIn("05", arguments.players)
        self.assertIsNone(arguments.fixed_onset_release_threshold)

        with self.assertRaises(SystemExit):
            parser.parse_args(
                ["--output", "sweep.json", "--players", "05"]
            )

    def test_cli_requires_output_and_rejects_zero_probability(self):
        parser = create_argument_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(
                ["--output", "sweep.json", "--release-thresholds", "0"]
            )
        with self.assertRaises(SystemExit):
            parser.parse_args([])

    def test_cli_accepts_explicit_fixed_onset_release_threshold(self):
        arguments = create_argument_parser().parse_args(
            [
                "--output",
                "sweep.json",
                "--fixed-onset-release-threshold",
                "0.55",
                "--release-thresholds",
                "0.55",
                "0.50",
            ]
        )

        self.assertEqual(arguments.fixed_onset_release_threshold, 0.55)
        self.assertEqual(tuple(arguments.release_thresholds), (0.55, 0.50))

    def test_release_validation_rejects_duplicates_and_above_entry(self):
        with self.assertRaisesRegex(EvaluationError, "duplicates"):
            validate_hysteresis_thresholds(0.55, (0.50, 0.50))
        with self.assertRaisesRegex(EvaluationError, "less than or equal"):
            validate_hysteresis_thresholds(0.55, (0.60,))
        with self.assertRaises(EvaluationError):
            validate_hysteresis_thresholds(0.55, ())

    def test_existing_output_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "existing.json"
            output.write_text("existing", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                refuse_output_overwrite(output)

    def test_nonlocked_seed_is_rejected_before_dataset_access(self):
        parser = create_argument_parser()
        with tempfile.TemporaryDirectory() as directory:
            arguments = parser.parse_args(
                [
                    "missing-dataset",
                    "--output",
                    str(Path(directory) / "new.json"),
                    "--seed",
                    "42",
                ]
            )
            with self.assertRaisesRegex(
                EvaluationError,
                "locked to split seed 1337",
            ):
                run_hysteresis_sweep(arguments)


if __name__ == "__main__":
    unittest.main()
