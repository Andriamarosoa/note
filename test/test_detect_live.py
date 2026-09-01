from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import wave

from causal_note.detector import BoundaryScoreChunk, LiveModelDetector
from causal_note.pipeline import LiveChunkResult, LiveOnsetOffsetPipeline
from scripts.detect_live import emit, main, parse_args, stream_stdin, stream_wav


class _Predictor:
    slot_count = 2

    def predict_chunk(self, samples, *, start_sample):
        onset = [[0.0, 0.0] for _ in samples]
        offset = [[0.0, 0.0] for _ in samples]
        end = start_sample + len(samples)
        for sample, slot in ((100, 0), (200, 1)):
            if start_sample <= sample < end:
                onset[sample - start_sample][slot] = 1.0
        for sample, slot in ((700, 0), (820, 1)):
            if start_sample <= sample < end:
                offset[sample - start_sample][slot] = 1.0
        return BoundaryScoreChunk(start_sample, tuple(onset), tuple(offset))


class DetectLiveCliTests(unittest.TestCase):
    def test_wav_and_stdin_eof_finalize_exactly_once(self):
        empty_result = LiveChunkResult((), (), ())
        with tempfile.TemporaryDirectory() as temporary_dir:
            wav_path = Path(temporary_dir) / "three-samples.wav"
            with wave.open(str(wav_path), "wb") as audio:
                audio.setnchannels(1)
                audio.setsampwidth(2)
                audio.setframerate(44_100)
                audio.writeframes(b"\x00\x00" * 3)

            wav_pipeline = Mock()
            wav_pipeline.process_chunk.return_value = empty_result
            wav_pipeline.finalize_stream.return_value = empty_result
            stream_wav(wav_path, wav_pipeline, 2)

        self.assertEqual(wav_pipeline.process_chunk.call_count, 2)
        wav_pipeline.finalize_stream.assert_called_once_with()

        stdin_pipeline = Mock()
        stdin_pipeline.process_chunk.return_value = empty_result
        stdin_pipeline.finalize_stream.return_value = empty_result
        fake_stdin = Mock()
        fake_stdin.buffer = io.BytesIO(b"\x00\x00" * 3)
        with patch("scripts.detect_live.sys.stdin", fake_stdin):
            stream_stdin(stdin_pipeline, 2)

        self.assertEqual(stdin_pipeline.process_chunk.call_count, 2)
        stdin_pipeline.finalize_stream.assert_called_once_with()

    def test_release_threshold_flags_are_optional(self):
        defaults = parse_args(("--model", "model.keras", "--stdin-pcm16"))
        explicit = parse_args(
            (
                "--model",
                "model.keras",
                "--stdin-pcm16",
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

    def test_main_wires_release_thresholds_into_detector(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            model = root / "model.keras"
            metadata = root / "model.metadata.json"
            wav = root / "input.wav"
            metadata.write_text(
                json.dumps({"model": {"receptive_field_samples": 1}}),
                encoding="utf-8",
            )
            predictor = Mock()
            detector = Mock()
            pipeline = Mock()
            with patch(
                "scripts.detect_live.KerasBoundaryPredictor.from_path",
                return_value=predictor,
            ):
                with patch(
                    "scripts.detect_live.LiveModelDetector",
                    return_value=detector,
                ) as detector_class:
                    with patch(
                        "scripts.detect_live.LiveOnsetOffsetPipeline",
                        return_value=pipeline,
                    ):
                        with patch(
                            "scripts.detect_live.stream_wav"
                        ) as stream_wav:
                            main(
                                (
                                    "--model",
                                    str(model),
                                    "--metadata",
                                    str(metadata),
                                    "--wav",
                                    str(wav),
                                    "--onset-threshold",
                                    "0.55",
                                    "--offset-threshold",
                                    "0.6",
                                    "--onset-release-threshold",
                                    "0.5",
                                    "--offset-release-threshold",
                                    "0.45",
                                )
                            )

            detector_class.assert_called_once_with(
                predictor,
                onset_threshold=0.55,
                offset_threshold=0.6,
                onset_release_threshold=0.5,
                offset_release_threshold=0.45,
            )
            stream_wav.assert_called_once_with(wav, pipeline, 512)

    def test_cli_path_emits_only_boundaries_and_completes_restart_frames(self):
        pipeline = LiveOnsetOffsetPipeline(LiveModelDetector(_Predictor()))
        output = io.StringIO()
        with redirect_stdout(output):
            position = emit(pipeline, (0.0,) * 512, 0)
            position = emit(pipeline, (0.0,) * 512, position)
            position = emit(pipeline, (0.0,) * 188, position)
            position = emit(pipeline, (0.0,) * 120, position)

        self.assertEqual(
            output.getvalue().splitlines(),
            [
                "onset(event-000001, 100)",
                "onset(event-000002, 200)",
                "offset(event-000001, 700)",
                "offset(event-000002, 820)",
            ],
        )
        self.assertEqual(position, 1332)
        self.assertEqual(pipeline.scheduler.restart_frames(), ())
        self.assertEqual(
            pipeline.scheduler.audio_start_sample,
            pipeline.scheduler.audio_end_sample,
        )


if __name__ == "__main__":
    unittest.main()
