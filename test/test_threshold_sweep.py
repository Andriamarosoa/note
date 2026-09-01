import tempfile
from pathlib import Path
import unittest

from causal_note.detector import (
    BoundaryScoreChunk,
    LiveModelDetector,
)
from scripts.evaluate_boundaries import EvaluationError
from scripts.sweep_boundary_thresholds import (
    DEFAULT_THRESHOLDS,
    create_argument_parser,
    decode_threshold_stream,
    refuse_output_overwrite,
    run_threshold_sweep,
    validate_thresholds,
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
            tuple((self.onset[index],) for index in range(start_sample, end_sample)),
            tuple((self.offset[index],) for index in range(start_sample, end_sample)),
        )


class ThresholdSweepStreamingTests(unittest.TestCase):
    def test_predictor_is_called_once_per_chunk_for_all_thresholds(self):
        predictor = _ScriptedPredictor(
            onset=(0.0,) * 6,
            offset=(0.0,) * 6,
        )
        chunks = (
            (0, (0.0, 0.0)),
            (2, (0.0, 0.0)),
            (4, (0.0, 0.0)),
        )

        result = decode_threshold_stream(
            predictor,
            chunks,
            thresholds=(0.5, 0.7, 0.9),
        )

        self.assertEqual(predictor.calls, [(0, 2), (2, 2), (4, 2)])
        self.assertEqual(result.chunks, 3)
        self.assertEqual(tuple(result.events), (0.5, 0.7, 0.9))

    def test_threshold_states_are_independent_across_chunk_plateau(self):
        predictor = _ScriptedPredictor(
            onset=(0.7, 0.7, 0.7, 0.9),
            offset=(0.0, 0.0, 0.0, 0.0),
        )

        result = decode_threshold_stream(
            predictor,
            ((0, (0.0, 0.0)), (2, (0.0, 0.0))),
            thresholds=(0.5, 0.8),
        )

        self.assertEqual(
            tuple(event.sample for event in result.events[0.5]),
            (0,),
        )
        self.assertEqual(
            tuple(event.sample for event in result.events[0.8]),
            (3,),
        )

    def test_baseline_and_selected_threshold_match_simple_live_decoding(self):
        onset = (0.0, 0.52, 0.52, 0.0, 0.8, 0.0)
        offset = (0.0, 0.0, 0.6, 0.0, 0.0, 0.9)
        chunks = ((0, (0.0, 0.0, 0.0)), (3, (0.0, 0.0, 0.0)))

        for threshold in (0.5, 0.55):
            with self.subTest(threshold=threshold):
                swept = decode_threshold_stream(
                    _ScriptedPredictor(onset, offset),
                    chunks,
                    thresholds=(threshold,),
                ).events[threshold]

                detector = LiveModelDetector(
                    _ScriptedPredictor(onset, offset),
                    onset_threshold=threshold,
                    offset_threshold=threshold,
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


class ThresholdSweepCliTests(unittest.TestCase):
    def test_default_grid_and_locked_player_choices(self):
        parser = create_argument_parser()
        arguments = parser.parse_args(["--output", "sweep.json"])
        self.assertEqual(tuple(arguments.thresholds), DEFAULT_THRESHOLDS)
        self.assertEqual(arguments.seed, 1337)
        self.assertNotIn("05", arguments.players)

        with self.assertRaises(SystemExit):
            parser.parse_args(
                ["--output", "sweep.json", "--players", "05"]
            )

    def test_cli_rejects_invalid_threshold_and_requires_output(self):
        parser = create_argument_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["--output", "sweep.json", "--thresholds", "0"])
        with self.assertRaises(SystemExit):
            parser.parse_args([])
        with self.assertRaises(EvaluationError):
            validate_thresholds((0.5, 0.5))

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
            with self.assertRaisesRegex(EvaluationError, "locked to split seed 1337"):
                run_threshold_sweep(arguments)


if __name__ == "__main__":
    unittest.main()
