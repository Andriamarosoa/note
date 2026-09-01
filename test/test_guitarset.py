import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from causal_note.guitarset import (
    GuitarSetFormatError,
    GuitarSetTrack,
    NoteBoundary,
    index_guitarset,
    load_boundary_slots,
)


def _write_zip(path, members):
    with zipfile.ZipFile(path, "w") as archive:
        for member, content in members:
            archive.writestr(member, content)


def _jams(*annotations):
    return json.dumps({"annotations": list(annotations)})


def _note_annotation(slot, observations):
    return {
        "namespace": "note_midi",
        "annotation_metadata": {"data_source": slot},
        "data": observations,
    }


class GuitarSetIndexTests(unittest.TestCase):
    def test_maps_pickup_mix_and_excludes_player_05_deterministically(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            annotation_zip = root / "annotation.zip"
            audio_zip = root / "audio_mono-pickup_mix.zip"
            _write_zip(
                annotation_zip,
                (
                    ("05_forbidden.jams", _jams()),
                    ("nested/04_second.jams", _jams()),
                    ("00_first.jams", _jams()),
                    ("README.txt", "ignored"),
                ),
            )
            _write_zip(
                audio_zip,
                (
                    ("audio/04_second_mix.wav", b"wav-04"),
                    ("05_forbidden_mix.wav", b"wav-05"),
                    ("00_first_mix.wav", b"wav-00"),
                ),
            )

            tracks = index_guitarset(root)

            self.assertEqual(
                tuple(track.annotation_member for track in tracks),
                ("00_first.jams", "nested/04_second.jams"),
            )
            self.assertEqual(tuple(track.player_id for track in tracks), ("00", "04"))
            self.assertEqual(
                tuple(track.audio_member for track in tracks),
                ("00_first_mix.wav", "audio/04_second_mix.wav"),
            )
            self.assertTrue(all(track.annotation_zip == annotation_zip for track in tracks))
            self.assertTrue(all(track.audio_zip == audio_zip for track in tracks))

    def test_missing_mapped_audio_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            _write_zip(root / "annotation.zip", (("00_track.jams", _jams()),))
            _write_zip(root / "audio_mono-pickup_mix.zip", ())

            with self.assertRaises(GuitarSetFormatError):
                index_guitarset(root)

    def test_track_object_cannot_admit_player_05(self):
        with self.assertRaises(GuitarSetFormatError):
            GuitarSetTrack(
                player_id="05",
                annotation_zip=Path("annotation.zip"),
                annotation_member="05_track.jams",
                audio_zip=Path("audio_mono-pickup_mix.zip"),
                audio_member="05_track_mix.wav",
            )


class GuitarSetBoundaryTests(unittest.TestCase):
    def test_loads_six_slots_rounds_and_sorts_without_extracting(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            annotation_zip = root / "annotation.zip"
            document = _jams(
                {"namespace": "chord", "data": "ignored"},
                _note_annotation(
                    "5",
                    (
                        {"time": 700 / 44_100, "duration": 120 / 44_100},
                    ),
                ),
                _note_annotation(
                    0,
                    (
                        {"time": 0.02, "duration": 0.01},
                        {"time": 0.01, "duration": 0.02},
                    ),
                ),
            )
            _write_zip(annotation_zip, (("00_track.jams", document),))

            slots = load_boundary_slots(annotation_zip, "00_track.jams")

            self.assertEqual(len(slots), 6)
            self.assertEqual(
                slots[0],
                (NoteBoundary(441, 1323), NoteBoundary(882, 1323)),
            )
            self.assertEqual(slots[1:5], ((), (), (), ()))
            self.assertEqual(slots[5], (NoteBoundary(700, 820),))
            self.assertFalse((root / "00_track.jams").exists())

    def test_direct_loading_of_player_05_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            annotation_zip = Path(temporary_dir) / "annotation.zip"
            _write_zip(annotation_zip, (("05_track.jams", _jams()),))

            with self.assertRaises(GuitarSetFormatError):
                load_boundary_slots(annotation_zip, "05_track.jams")

    def test_invalid_slot_and_temporal_values_are_rejected(self):
        invalid_cases = (
            (_note_annotation(6, ()), "slot"),
            (_note_annotation(True, ()), "boolean slot"),
            (_note_annotation(0, ({"time": -0.1, "duration": 0.1},)), "negative time"),
            (_note_annotation(0, ({"time": 0.1, "duration": 0.0},)), "zero duration"),
            (_note_annotation(0, ({"time": "0.1", "duration": 0.1},)), "text time"),
            (_note_annotation(0, ({"time": 0.1, "duration": float("inf")},)), "infinite duration"),
            (_note_annotation(0, ({"time": 0.0, "duration": 1e-12},)), "same rounded boundary"),
        )
        for annotation, label in invalid_cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary_dir:
                annotation_zip = Path(temporary_dir) / "annotation.zip"
                _write_zip(
                    annotation_zip,
                    (("00_track.jams", _jams(annotation)),),
                )
                with self.assertRaises(GuitarSetFormatError):
                    load_boundary_slots(annotation_zip, "00_track.jams")


if __name__ == "__main__":
    unittest.main()
