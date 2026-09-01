import io
from contextlib import redirect_stderr
import json
import importlib.util
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
import wave
import zipfile

from causal_note.guitarset import SAMPLE_RATE, index_guitarset, load_boundary_slots
from scripts.train_boundaries import (
    BalancedWindowBatcher,
    DEFAULT_POSITIVE_WEIGHTS,
    TrainingDataError,
    _prepare_tracks,
    create_argument_parser,
    decode_pcm16_mono_wav,
    group_stem,
    make_positive_sample_weights,
    make_window_targets,
    run_training,
    split_tracks_by_group,
)


NUMPY_AVAILABLE = importlib.util.find_spec("numpy") is not None


def _wav_bytes(samples, *, channels=1, sample_rate=SAMPLE_RATE):
    payload = io.BytesIO()
    with wave.open(payload, "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(struct.pack(f"<{len(samples)}h", *samples))
    return payload.getvalue()


def _write_zip(path, members):
    with zipfile.ZipFile(path, "w") as archive:
        for member, content in members:
            archive.writestr(member, content)


def _empty_jams():
    return json.dumps({"annotations": []})


def _boundary_jams(boundaries=((0, 100, 700), (1, 200, 820))):
    def annotation(slot, onset, offset):
        return {
            "namespace": "note_midi",
            "annotation_metadata": {"data_source": slot},
            "data": [
                {
                    "time": onset / SAMPLE_RATE,
                    "duration": (offset - onset) / SAMPLE_RATE,
                }
            ],
        }

    return json.dumps(
        {
            "annotations": [annotation(*boundary) for boundary in boundaries]
        }
    )


class TrainingModuleImportTests(unittest.TestCase):
    def test_import_is_lazy_for_numpy_and_tensorflow(self):
        project_root = Path(__file__).resolve().parents[1]
        code = (
            "import sys; "
            f"sys.path.insert(0, {str(project_root)!r}); "
            "import scripts.train_boundaries; "
            "raise SystemExit(int('numpy' in sys.modules or "
            "'tensorflow' in sys.modules))"
        )
        completed = subprocess.run(
            [sys.executable, "-c", code],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_player_05_is_rejected_by_the_cli(self):
        parser = create_argument_parser()
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(("--players", "05", "--smoke"))

    def test_target_width_cli_defaults_to_legacy_one(self):
        parser = create_argument_parser()
        legacy = parser.parse_args(())
        widened = parser.parse_args(
            (
                "--onset-target-width-samples",
                "512",
                "--offset-target-width-samples",
                "256",
            )
        )

        self.assertEqual(legacy.onset_target_width_samples, 1)
        self.assertEqual(legacy.offset_target_width_samples, 1)
        self.assertEqual(widened.onset_target_width_samples, 512)
        self.assertEqual(widened.offset_target_width_samples, 256)

        for option in (
            "--onset-target-width-samples",
            "--offset-target-width-samples",
        ):
            with self.subTest(option=option), redirect_stderr(
                io.StringIO()
            ), self.assertRaises(SystemExit):
                parser.parse_args((option, "0"))

    def test_player_05_is_rejected_by_programmatic_training(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            arguments = create_argument_parser().parse_args(
                (str(Path(temporary_dir) / "missing-data"), "--players", "00")
            )
            arguments.output = str(Path(temporary_dir) / "programmatic.keras")
            arguments.players = ("00", "05")
            with self.assertRaises(TrainingDataError):
                run_training(arguments)

    def test_existing_output_is_refused_before_training(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            output = Path(temporary_dir) / "existing.keras"
            output.write_bytes(b"do-not-replace")
            arguments = create_argument_parser().parse_args(
                (str(Path(temporary_dir) / "missing-data"), "--output", str(output))
            )
            with self.assertRaises(FileExistsError):
                run_training(arguments)
            self.assertEqual(output.read_bytes(), b"do-not-replace")

    def test_model_and_metadata_cannot_share_one_path(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            shared = Path(temporary_dir) / "artifact.keras"
            arguments = create_argument_parser().parse_args(
                (
                    str(Path(temporary_dir) / "missing-data"),
                    "--output",
                    str(shared),
                    "--metadata",
                    str(shared),
                )
            )
            with self.assertRaises(TrainingDataError):
                run_training(arguments)


class PcmZipTests(unittest.TestCase):
    def test_decodes_exact_mono_pcm16_samples_without_extraction(self):
        samples = (-32768, -1234, 0, 1234, 32767)
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            audio_zip = root / "audio.zip"
            _write_zip(
                audio_zip,
                (("nested/tiny.wav", _wav_bytes(samples)),),
            )

            decoded = decode_pcm16_mono_wav(audio_zip, "nested/tiny.wav")

            self.assertEqual(decoded.sample_rate, SAMPLE_RATE)
            self.assertEqual(tuple(decoded.samples), samples)
            self.assertFalse((root / "nested" / "tiny.wav").exists())

    def test_rejects_non_mono_pcm(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            audio_zip = Path(temporary_dir) / "audio.zip"
            _write_zip(audio_zip, (("stereo.wav", _wav_bytes((0, 0), channels=2)),))

            with self.assertRaises(TrainingDataError):
                decode_pcm16_mono_wav(audio_zip, "stereo.wav")


class ExactBoundaryTargetTests(unittest.TestCase):
    def test_exact_overlapping_a_and_b_labels(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            annotation_zip = root / "annotation.zip"
            audio_zip = root / "audio.zip"
            _write_zip(annotation_zip, (("00_overlap.jams", _boundary_jams()),))
            _write_zip(audio_zip, (("00_overlap_mix.wav", _wav_bytes((0,) * 900)),))

            slots = load_boundary_slots(annotation_zip, "00_overlap.jams")
            decoded = decode_pcm16_mono_wav(audio_zip, "00_overlap_mix.wav")
            targets = make_window_targets(
                slots,
                start_sample=0,
                window_samples=decoded.frame_count,
            )
            explicit_legacy_targets = make_window_targets(
                slots,
                start_sample=0,
                window_samples=decoded.frame_count,
                onset_target_width_samples=1,
                offset_target_width_samples=1,
            )

            self.assertEqual(targets, explicit_legacy_targets)
            self.assertEqual(targets["onset"][100][0], 1.0)
            self.assertEqual(targets["offset"][700][0], 1.0)
            self.assertEqual(targets["onset"][200][1], 1.0)
            self.assertEqual(targets["offset"][820][1], 1.0)
            self.assertEqual(set(targets), {"onset", "offset"})
            self.assertEqual(
                sum(sum(row) for row in targets["onset"]),
                2.0,
            )
            self.assertEqual(
                sum(sum(row) for row in targets["offset"]),
                2.0,
            )
            self.assertFalse((root / "00_overlap.jams").exists())
            self.assertFalse((root / "00_overlap_mix.wav").exists())

            weights = make_positive_sample_weights(targets)
            self.assertEqual(weights["onset"][100][0], 64.0)
            self.assertEqual(weights["offset"][820][1], 64.0)
            self.assertTrue(
                all(
                    value > 0.0
                    for output_weights in weights.values()
                    for row in output_weights
                    for value in row
                )
            )

    def test_causal_onset_window_is_absolute_clipped_and_offset_stays_exact(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            annotation_zip = Path(temporary_dir) / "annotation.zip"
            _write_zip(annotation_zip, (("00_overlap.jams", _boundary_jams()),))
            slots = load_boundary_slots(annotation_zip, "00_overlap.jams")

            targets = make_window_targets(
                slots,
                start_sample=0,
                window_samples=900,
                onset_target_width_samples=512,
            )

            self.assertTrue(
                all(
                    targets["onset"][sample][0] == 0.0
                    for sample in range(100)
                )
            )
            self.assertEqual(sum(row[0] for row in targets["onset"]), 512.0)
            self.assertEqual(sum(row[1] for row in targets["onset"]), 512.0)
            self.assertEqual(sum(sum(row) for row in targets["offset"]), 2.0)
            self.assertEqual(targets["offset"][700][0], 1.0)
            self.assertEqual(targets["offset"][820][1], 1.0)

            tail_crop = make_window_targets(
                slots,
                start_sample=400,
                window_samples=100,
                onset_target_width_samples=512,
            )
            self.assertTrue(
                all(row[0] == 1.0 and row[1] == 1.0 for row in tail_crop["onset"])
            )
            self.assertEqual(sum(sum(row) for row in tail_crop["offset"]), 0.0)

            end_clipped = make_window_targets(
                slots,
                start_sample=0,
                window_samples=300,
                onset_target_width_samples=512,
            )
            self.assertEqual(sum(row[0] for row in end_clipped["onset"]), 200.0)
            self.assertEqual(sum(row[1] for row in end_clipped["onset"]), 100.0)
            self.assertEqual(sum(sum(row) for row in end_clipped["offset"]), 0.0)

    def test_onset_target_width_must_be_positive(self):
        slots = ((), (), (), (), (), ())
        with self.assertRaises(TrainingDataError):
            make_window_targets(
                slots,
                start_sample=0,
                window_samples=1,
                onset_target_width_samples=0,
            )

    def test_causal_offset_window_is_absolute_clipped_and_onset_stays_exact(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            annotation_zip = Path(temporary_dir) / "annotation.zip"
            _write_zip(
                annotation_zip,
                (
                    ("00_overlap.jams", _boundary_jams()),
                    ("00_eof.jams", _boundary_jams(((0, 100, 1400),))),
                ),
            )
            slots = load_boundary_slots(annotation_zip, "00_overlap.jams")

            targets = make_window_targets(
                slots,
                start_sample=0,
                window_samples=1400,
                offset_target_width_samples=512,
            )

            self.assertEqual(sum(sum(row) for row in targets["onset"]), 2.0)
            self.assertTrue(
                all(
                    targets["offset"][sample][0] == 0.0
                    for sample in range(700)
                )
            )
            self.assertEqual(sum(row[0] for row in targets["offset"]), 512.0)
            self.assertEqual(sum(row[1] for row in targets["offset"]), 512.0)

            tail_crop = make_window_targets(
                slots,
                start_sample=900,
                window_samples=100,
                offset_target_width_samples=512,
            )
            self.assertTrue(
                all(row[0] == 1.0 and row[1] == 1.0 for row in tail_crop["offset"])
            )
            self.assertEqual(sum(sum(row) for row in tail_crop["onset"]), 0.0)

            end_clipped = make_window_targets(
                slots,
                start_sample=0,
                window_samples=900,
                offset_target_width_samples=512,
            )
            self.assertEqual(sum(row[0] for row in end_clipped["offset"]), 200.0)
            self.assertEqual(sum(row[1] for row in end_clipped["offset"]), 80.0)

            eof_slots = load_boundary_slots(annotation_zip, "00_eof.jams")
            eof_targets = make_window_targets(
                eof_slots,
                start_sample=0,
                window_samples=1400,
                offset_target_width_samples=512,
            )
            self.assertEqual(sum(sum(row) for row in eof_targets["offset"]), 0.0)

    def test_offset_target_width_must_be_positive(self):
        slots = ((), (), (), (), (), ())
        for invalid in (0, True):
            with self.subTest(invalid=invalid), self.assertRaises(TrainingDataError):
                make_window_targets(
                    slots,
                    start_sample=0,
                    window_samples=1,
                    offset_target_width_samples=invalid,
                )


@unittest.skipUnless(NUMPY_AVAILABLE, "NumPy is not installed")
class BalancedBatchTests(unittest.TestCase):
    def test_keras_weights_are_elementwise_and_audio_is_not_padded(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            _write_zip(
                root / "annotation.zip",
                (("00_overlap.jams", _boundary_jams()),),
            )
            _write_zip(
                root / "audio_mono-pickup_mix.zip",
                (("00_overlap_mix.wav", _wav_bytes((1,) * 1024)),),
            )
            prepared = _prepare_tracks(index_guitarset(root))
            batcher = BalancedWindowBatcher(
                prepared,
                window_samples=512,
                batch_size=2,
                seed=7,
                positive_weights=DEFAULT_POSITIVE_WEIGHTS,
                warmup_samples=100,
                numpy_module=np,
            )

            audio, targets, weights = next(batcher)

            self.assertEqual(audio.shape, (2, 512, 1))
            self.assertTrue(np.all(audio == np.float32(1 / 32768.0)))
            for name in ("onset", "offset"):
                self.assertEqual(targets[name].shape, (2, 512, 6))
                self.assertEqual(weights[name].shape, (2, 512, 6))
                expected = np.where(
                    targets[name] > 0.0,
                    DEFAULT_POSITIVE_WEIGHTS[name],
                    1.0,
                ).astype(np.float32)
                expected[1, :100, :] = 0.0
                np.testing.assert_array_equal(weights[name], expected)
                np.testing.assert_array_equal(
                    weights[name][1, :100, :],
                    np.zeros((100, 6), dtype=np.float32),
                )
                scored_positives = np.argwhere(
                    (targets[name] > 0.0) & (weights[name] > 0.0)
                )
                self.assertGreater(len(scored_positives), 0)
                for row, sample, slot in scored_positives:
                    self.assertEqual(weights[name][row, sample, slot], 64.0)
                    negative_slots = targets[name][row, sample] == 0.0
                    self.assertTrue(
                        np.all(weights[name][row, sample, negative_slots] == 1.0)
                    )

    def test_wide_onset_targets_weight_only_positive_slots(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            _write_zip(
                root / "annotation.zip",
                (("00_overlap.jams", _boundary_jams()),),
            )
            _write_zip(
                root / "audio_mono-pickup_mix.zip",
                (("00_overlap_mix.wav", _wav_bytes((1,) * 1024)),),
            )
            batcher = BalancedWindowBatcher(
                _prepare_tracks(index_guitarset(root)),
                window_samples=512,
                batch_size=2,
                seed=7,
                positive_weights=DEFAULT_POSITIVE_WEIGHTS,
                onset_target_width_samples=512,
                warmup_samples=0,
                numpy_module=np,
            )

            _, targets, weights = next(batcher)

            onset_positive = targets["onset"] > 0.0
            self.assertTrue(np.any(onset_positive))
            self.assertTrue(np.all(weights["onset"][onset_positive] == 64.0))
            self.assertTrue(np.all(weights["onset"][~onset_positive] == 1.0))
            for row, sample, slot in np.argwhere(onset_positive):
                negative_slots = targets["onset"][row, sample] == 0.0
                self.assertTrue(
                    np.all(weights["onset"][row, sample, negative_slots] == 1.0)
                )

    def test_wide_offset_targets_weight_only_positive_slots(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            _write_zip(
                root / "annotation.zip",
                (("00_overlap.jams", _boundary_jams()),),
            )
            _write_zip(
                root / "audio_mono-pickup_mix.zip",
                (("00_overlap_mix.wav", _wav_bytes((1,) * 1400)),),
            )
            batcher = BalancedWindowBatcher(
                _prepare_tracks(index_guitarset(root)),
                window_samples=512,
                batch_size=2,
                seed=7,
                positive_weights=DEFAULT_POSITIVE_WEIGHTS,
                offset_target_width_samples=512,
                warmup_samples=0,
                numpy_module=np,
            )

            _, targets, weights = next(batcher)

            offset_positive = targets["offset"] > 0.0
            self.assertTrue(np.any(offset_positive))
            self.assertTrue(np.all(weights["offset"][offset_positive] == 64.0))
            self.assertTrue(np.all(weights["offset"][~offset_positive] == 1.0))
            for row, sample, slot in np.argwhere(offset_positive):
                negative_slots = targets["offset"][row, sample] == 0.0
                self.assertTrue(
                    np.all(weights["offset"][row, sample, negative_slots] == 1.0)
                )


class LeakageSafeSplitTests(unittest.TestCase):
    def test_comp_and_solo_share_the_same_composition_group(self):
        self.assertEqual(
            group_stem("00_Funk1-114-Ab_comp.jams"),
            group_stem("04_Funk1-114-Ab_solo.jams"),
        )

    def test_player_guard_and_unprefixed_group_split(self):
        members = (
            "00_shared.jams",
            "01_shared.jams",
            "02_alpha.jams",
            "03_beta.jams",
            "04_gamma.jams",
            "05_forbidden.jams",
        )
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            _write_zip(
                root / "annotation.zip",
                tuple((member, _empty_jams()) for member in members),
            )
            _write_zip(
                root / "audio_mono-pickup_mix.zip",
                tuple(
                    (
                        f"{member[:-5]}_mix.wav",
                        _wav_bytes((0,) * 16),
                    )
                    for member in members
                ),
            )

            indexed = index_guitarset(root)
            train, validation = split_tracks_by_group(
                indexed,
                validation_fraction=0.5,
                seed=19,
            )

            self.assertEqual(len(indexed), 5)
            self.assertTrue(all(track.player_id in {"00", "01", "02", "03", "04"} for track in indexed))
            train_groups = {group_stem(track) for track in train}
            validation_groups = {group_stem(track) for track in validation}
            self.assertTrue(train_groups)
            self.assertTrue(validation_groups)
            self.assertFalse(train_groups & validation_groups)

            side_by_member = {
                track.annotation_member: "train" for track in train
            }
            side_by_member.update(
                {track.annotation_member: "validation" for track in validation}
            )
            self.assertEqual(
                side_by_member["00_shared.jams"],
                side_by_member["01_shared.jams"],
            )
            with self.assertRaises(TrainingDataError):
                group_stem("05_forbidden.jams")


if __name__ == "__main__":
    unittest.main()
