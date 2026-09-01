import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from causal_note.guitarset import GuitarSetFormatError
from causal_note.guitarset_acoustics import load_rich_annotations


def _write_zip(path, member, document):
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(member, json.dumps(document))


class GuitarSetAcousticAnnotationTests(unittest.TestCase):
    def test_loads_note_pitch_and_pitch_contour_per_slot(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "annotation.zip"
            member = "00_track.jams"
            document = {
                "annotations": [
                    {
                        "namespace": "note_midi",
                        "annotation_metadata": {"data_source": "0"},
                        "data": [
                            {"time": 0.01, "duration": 0.02, "value": 40.0},
                        ],
                    },
                    {
                        "namespace": "pitch_contour",
                        "annotation_metadata": {"data_source": 0},
                        "data": [
                            {
                                "time": 0.0,
                                "duration": 0,
                                "value": {"voiced": False, "index": 0, "frequency": 0.0},
                            },
                            {
                                "time": 0.01,
                                "duration": 0,
                                "value": {"voiced": True, "index": 0, "frequency": 82.41},
                            },
                        ],
                    },
                ]
            }
            _write_zip(path, member, document)

            loaded = load_rich_annotations(path, member)

            self.assertEqual(len(loaded.notes_by_slot), 6)
            note = loaded.notes_by_slot[0][0]
            self.assertEqual((note.onset_sample, note.offset_sample), (441, 1323))
            self.assertAlmostEqual(note.midi, 40.0)
            self.assertAlmostEqual(note.frequency_hz, 82.4069, places=3)
            contour = loaded.contours_by_slot[0]
            self.assertEqual(tuple(point.sample for point in contour), (0, 441))
            self.assertFalse(contour[0].voiced)
            self.assertTrue(contour[1].voiced)
            self.assertAlmostEqual(contour[1].frequency_hz, 82.41)

    def test_loads_real_vectorized_pitch_contour_serialization(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "annotation.zip"
            member = "00_track.jams"
            _write_zip(
                path,
                member,
                {
                    "annotations": [
                        {
                            "namespace": "pitch_contour",
                            "annotation_metadata": {"data_source": "2"},
                            "data": {
                                "time": [0.0, 256 / 44_100],
                                "duration": [0.0, 0.0],
                                "value": [
                                    {"voiced": True, "index": 0, "frequency": 110.0},
                                    {"voiced": True, "index": 0, "frequency": 111.0},
                                ],
                                "confidence": [None, None],
                            },
                        }
                    ]
                },
            )

            loaded = load_rich_annotations(path, member)
            contour = loaded.contours_by_slot[2]

            self.assertEqual(tuple(point.sample for point in contour), (0, 256))
            self.assertEqual(tuple(point.frequency_hz for point in contour), (110.0, 111.0))
            self.assertTrue(all(point.voiced for point in contour))

    def test_player_05_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "annotation.zip"
            _write_zip(path, "05_track.jams", {"annotations": []})
            with self.assertRaises(GuitarSetFormatError):
                load_rich_annotations(path, "05_track.jams")

    def test_invalid_voiced_zero_frequency_is_normalized_unvoiced(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "annotation.zip"
            member = "00_track.jams"
            _write_zip(
                path,
                member,
                {
                    "annotations": [
                        {
                            "namespace": "pitch_contour",
                            "annotation_metadata": {"data_source": 1},
                            "data": [
                                {
                                    "time": 0.0,
                                    "duration": 0,
                                    "value": {"voiced": True, "index": 0, "frequency": 0.0},
                                }
                            ],
                        }
                    ]
                },
            )
            loaded = load_rich_annotations(path, member)
            self.assertFalse(loaded.contours_by_slot[1][0].voiced)


if __name__ == "__main__":
    unittest.main()
