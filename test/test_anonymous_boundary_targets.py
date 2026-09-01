from pathlib import Path
import subprocess
import sys
import unittest

from causal_note.guitarset import GuitarSetTrack, NoteBoundary
from scripts.audit_anonymous_boundary_targets import (
    AuditTrack,
    _block_histogram,
    _cross_type_pairs,
    _head_positions,
    _successive_note_relations,
    binary_plateau_count,
    causal_inverse_dense_counts,
    causal_inverse_roundtrip,
    constant_categorical_optimum,
    exact_count_histogram,
    naive_positive_delta_recovery,
    slot_binary_positive_elements,
    wide_count_histogram,
)


def _slots(*slot_notes):
    values = [tuple(notes) for notes in slot_notes]
    values.extend([tuple()] * (6 - len(values)))
    return tuple(values)


class AnonymousAuditImportTests(unittest.TestCase):
    def test_import_does_not_import_tensorflow(self):
        project_root = Path(__file__).resolve().parents[1]
        code = (
            "import sys; "
            f"sys.path.insert(0, {str(project_root)!r}); "
            "import scripts.audit_anonymous_boundary_targets; "
            "raise SystemExit(int('tensorflow' in sys.modules))"
        )
        completed = subprocess.run(
            [sys.executable, "-c", code],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


class AnonymousCountTests(unittest.TestCase):
    def test_exact_count_preserves_same_sample_multiplicity(self):
        histogram = exact_count_histogram((100, 100, 200), start=0, end=300)

        self.assertEqual(histogram, {0: 298, 1: 1, 2: 1})
        self.assertEqual(sum(key * value for key, value in histogram.items()), 3)

    def test_width_512_overlap_and_binary_plateau_are_distinct(self):
        overlap = wide_count_histogram((100, 611), start=0, end=1200)
        touching = wide_count_histogram((100, 612), start=0, end=1200)

        self.assertEqual(overlap[2], 1)
        self.assertEqual(touching.get(2, 0), 0)
        self.assertEqual(binary_plateau_count((100, 611)), 1)
        self.assertEqual(binary_plateau_count((100, 612)), 1)
        self.assertEqual(binary_plateau_count((100, 613)), 2)

    def test_causal_inverse_recovers_a_start_cancelled_by_an_expiry(self):
        positions = (100, 612)

        naive = naive_positive_delta_recovery(positions)

        self.assertEqual(naive["recovered_instances"], 1)
        self.assertEqual(naive["lost_instances"], 1)
        self.assertTrue(causal_inverse_roundtrip(positions, frame_count=1200))

    def test_dense_target_roundtrips_independently_of_sparse_formula(self):
        positions = (2, 2, 5, 7)
        width = 4
        dense_target = [0] * 12
        for event_position in positions:
            for target_position in range(
                event_position,
                min(len(dense_target), event_position + width),
            ):
                dense_target[target_position] += 1

        recovered = causal_inverse_dense_counts(dense_target, width=width)

        self.assertEqual(recovered, (0, 0, 2, 0, 0, 1, 0, 1, 0, 0, 0, 0))

    def test_slot_permutation_cannot_change_anonymous_counts(self):
        note_a = NoteBoundary(100, 700)
        note_b = NoteBoundary(100, 820)
        slots = _slots((note_a,), (note_b,))
        reversed_slots = tuple(reversed(slots))

        original = _head_positions(
            slots,
            "onset",
            frame_count=900,
            supervised_only=True,
        )
        permuted = _head_positions(
            reversed_slots,
            "onset",
            frame_count=900,
            supervised_only=True,
        )

        self.assertEqual(original, (100, 100))
        self.assertEqual(original, permuted)
        self.assertEqual(
            slot_binary_positive_elements(
                slots,
                "onset",
                start=0,
                end=900,
            ),
            1024,
        )

    def test_exclusive_end_offset_is_raw_but_not_sample_supervised(self):
        slots = _slots((NoteBoundary(100, 700),))

        raw = _head_positions(
            slots,
            "offset",
            frame_count=700,
            supervised_only=False,
        )
        supervised = _head_positions(
            slots,
            "offset",
            frame_count=700,
            supervised_only=True,
        )

        self.assertEqual(raw, (700,))
        self.assertEqual(supervised, ())

    def test_block_histogram_preserves_repeated_events(self):
        histogram = _block_histogram((100, 100, 511, 512), frame_count=1024)

        self.assertEqual(histogram, {3: 1, 1: 1})

    def test_weighted_constant_optimum_uses_all_count_classes(self):
        optimum = constant_categorical_optimum(
            {0: 90, 1: 8, 2: 2},
            nonzero_weight=2.0,
        )

        self.assertAlmostEqual(optimum["class_probabilities"]["0"], 90 / 110)
        self.assertAlmostEqual(optimum["class_probabilities"]["1"], 16 / 110)
        self.assertAlmostEqual(optimum["class_probabilities"]["2"], 4 / 110)


class TemporalRelationTests(unittest.TestCase):
    def _track(self):
        notes = (
            NoteBoundary(100, 700),
            NoteBoundary(600, 800),
            NoteBoundary(800, 900),
            NoteBoundary(1000, 1100),
            NoteBoundary(1700, 1800),
        )
        track = GuitarSetTrack(
            player_id="00",
            annotation_zip=Path("annotation.zip"),
            annotation_member="00_test.jams",
            audio_zip=Path("audio.zip"),
            audio_member="00_test_mix.wav",
        )
        return AuditTrack(track, _slots(notes), 2000)

    def test_successive_note_relation_categories(self):
        relations = _successive_note_relations((self._track(),))

        self.assertEqual(relations["retrigger_before_previous_offset"], 1)
        self.assertEqual(relations["same_sample"], 1)
        self.assertEqual(relations["positive_gap_below_512"], 1)
        self.assertEqual(relations["positive_gap_512_or_more"], 1)

    def test_cross_type_pair_distance_is_strictly_below_512(self):
        track = self._track()

        relations = _cross_type_pairs((track,))

        self.assertGreater(relations["total_pairs"], 0)
        self.assertEqual(
            relations["total_pairs"],
            relations.get("offset_before_onset", 0)
            + relations.get("onset_before_offset", 0)
            + relations.get("same_sample", 0),
        )


if __name__ == "__main__":
    unittest.main()
