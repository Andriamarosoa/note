from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from causal_note.guitarset import GuitarSetTrack
from causal_note.guitarset_acoustics import RichAnnotations, RichNote
from causal_note.v280_causal_cqt import CausalCQTConfig, causal_cqt, write_track_cache
from scripts import train_v280_real_smoke as smoke


def _member(player: int, composition: str, arrangement: str = "comp") -> str:
    return f"annotation/{player:02d}_{composition}_{arrangement}.jams"


def _track(member: str) -> GuitarSetTrack:
    name = Path(member).name[:-5]
    return GuitarSetTrack(
        player_id=Path(member).name[:2],
        annotation_zip=Path("annotation.zip"),
        annotation_member=member,
        audio_zip=Path("audio.zip"),
        audio_member=f"audio/{name}_mix.wav",
    )


def _metadata(members: tuple[str, ...]) -> smoke.ClusterMetadata:
    rows = len(members)
    sequence = np.zeros((rows, 2, 4), dtype=np.float16)
    sequence[:, 0, -2] = 0.25
    sequence[:, 1, -2] = 0.75
    mask = np.ones((rows, 2), dtype=np.uint8)
    exact = (np.arange(rows, dtype=np.int16) % 3) + 1
    slots = np.zeros((rows, 6), dtype=np.uint8)
    for row, count in enumerate(exact):
        slots[row, : int(count)] = 1
    return smoke.ClusterMetadata(
        sequence=sequence,
        mask=mask,
        exact=exact,
        members=np.asarray(members, dtype="U96"),
        top_samples=np.tile(np.asarray([100, 200], dtype=np.int32), (rows, 1)),
        slot_targets=slots,
        track_members=tuple(sorted(set(members))),
        shard_paths=("shard-00/v100-spectral-shard-00.npz",),
    )


def _write_v100_shard(path: Path, member: str, exact: int) -> None:
    sequence = np.zeros((1, 2, 4), dtype=np.float16)
    sequence[0, :, -2] = (0.25, 0.75)
    slot_targets = np.zeros((1, 6), dtype=np.uint8)
    slot_targets[0, :exact] = 1
    forbidden = np.asarray([{"must_not_deserialize": True}], dtype=object)
    np.savez(
        path,
        schema_version=np.asarray([1], dtype=np.int16),
        spectral=forbidden,
        sequence=sequence,
        mask=np.ones((1, 2), dtype=np.uint8),
        stats=forbidden,
        target=forbidden,
        exact=np.asarray([exact], dtype=np.int16),
        members=np.asarray([member], dtype="U96"),
        top_samples=np.asarray([[100, 200]], dtype=np.int32),
        slot_targets=slot_targets,
        track_members=np.asarray([member], dtype="U96"),
    )


class ClusterMetadataTests(unittest.TestCase):
    def test_full_loader_reads_required_rows_but_not_obsolete_spectral_arrays(self):
        members = (_member(0, "alpha"), _member(1, "beta", "solo"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_v100_shard(root / "v100-spectral-shard-00.npz", members[0], 1)
            _write_v100_shard(root / "v100-spectral-shard-01.npz", members[1], 2)
            result = smoke.load_v100_cluster_metadata(
                root,
                expected_shard_count=2,
                expected_track_count=2,
            )
        self.assertEqual(result.row_count, 2)
        self.assertEqual(result.track_members, tuple(sorted(members)))
        np.testing.assert_array_equal(result.exact, np.asarray([1, 2]))
        self.assertEqual(result.sequence.dtype, np.float16)

    def test_distilled_cache_round_trip_and_digest_guard(self):
        metadata = _metadata((_member(0, "alpha"), _member(1, "beta")))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "distilled"
            manifest = smoke.write_distilled_cluster_cache(output, metadata)
            loaded = smoke.load_distilled_cluster_cache(output)
            self.assertEqual(loaded.track_members, metadata.track_members)
            np.testing.assert_array_equal(loaded.slot_targets, metadata.slot_targets)
            self.assertTrue(manifest["protocol"]["outer_clean_supervision_arrays_present"])
            self.assertFalse(manifest["protocol"]["obsolete_v10_spectral_maps_copied"])

            with (output / "v280-cluster-metadata.npz").open("ab") as stream:
                stream.write(b"tamper")
            with self.assertRaisesRegex(smoke.V280RealSmokeError, "digest mismatch"):
                smoke.load_distilled_cluster_cache(output)

    def test_group_selection_uses_canonical_identities_only(self):
        alpha_a = _member(0, "alpha")
        alpha_b = _member(1, "alpha", "solo")
        beta = _member(0, "beta")
        gamma = _member(0, "gamma")
        metadata = _metadata((gamma, alpha_b, beta, alpha_a))
        groups, rows = smoke.select_development_groups(metadata, group_count=1)
        self.assertEqual(groups, ("alpha",))
        np.testing.assert_array_equal(rows, np.asarray([1, 3]))


class RealSupervisionTests(unittest.TestCase):
    def test_rich_midi_is_assigned_to_the_same_frozen_string_rows(self):
        member = _member(0, "alpha")
        annotations = RichAnnotations(
            notes_by_slot=(
                (RichNote(0, 102, 180, 40.0),),
                (RichNote(1, 998, 1100, 47.0),),
                (),
                (),
                (),
                (),
            ),
            contours_by_slot=((), (), (), (), (), ()),
        )
        expected = np.zeros((2, 6), dtype=np.uint8)
        expected[0, 0] = 1
        expected[1, 1] = 1
        with patch.object(smoke, "load_rich_annotations", return_value=annotations) as loader:
            midi, mask, diagnostics = smoke.derive_midi_supervision(
                {member: _track(member)},
                (member, member),
                (np.asarray([100, 130]), np.asarray([1000])),
                expected,
            )
        loader.assert_called_once()
        np.testing.assert_array_equal(mask, expected)
        self.assertEqual(midi[0, 0], 40.0)
        self.assertEqual(midi[1, 1], 47.0)
        self.assertEqual(diagnostics["unassigned_events"], 0)
        self.assertEqual(diagnostics["slot_mask_agreement"], 1.0)

    def test_targets_preserve_string_fret_unison_and_cardinality(self):
        slots = np.zeros((2, 6), dtype=np.float32)
        slots[0, 0] = 1
        slots[1, :2] = 1
        midi = np.zeros_like(slots)
        midi[0, 0] = 40
        midi[1, 0] = 45
        midi[1, 1] = 45
        targets, consistent, quantization = smoke.real_targets(
            np.asarray([1, 2]),
            slots,
            midi,
            slots,
        )
        np.testing.assert_array_equal(consistent, np.asarray([True, True]))
        self.assertEqual(targets["string_fret_onset"][1, 0, 5], 1.0)
        self.assertEqual(targets["string_fret_onset"][1, 1, 0], 1.0)
        self.assertEqual(np.sum(targets["pitch_onset"][1]), 1.0)
        np.testing.assert_array_equal(targets["poibin_cardinality"], np.asarray([1, 2]))
        self.assertEqual(quantization["fractional_label_count"], 0)
        self.assertEqual(quantization["absolute_cents_max"], 0.0)

    def test_fractional_note_label_is_quantized_and_reported(self):
        slots = np.zeros((1, 6), dtype=np.float32)
        slots[0, 0] = 1
        midi = np.zeros_like(slots)
        midi[0, 0] = 40.25
        targets, consistent, quantization = smoke.real_targets(
            np.asarray([1]), slots, midi, slots
        )
        np.testing.assert_array_equal(consistent, np.asarray([True]))
        self.assertEqual(targets["string_fret_onset"][0, 0, 0], 1.0)
        self.assertEqual(targets["pitch_onset"][0, 0], 1.0)
        self.assertEqual(quantization["fractional_label_count"], 1)
        self.assertEqual(quantization["absolute_cents_median"], 25.0)
        self.assertEqual(quantization["absolute_cents_p90"], 25.0)
        self.assertEqual(quantization["absolute_cents_max"], 25.0)

    def test_half_semitone_note_label_is_rejected_as_ambiguous(self):
        slots = np.zeros((1, 6), dtype=np.float32)
        slots[0, 0] = 1
        midi = np.zeros_like(slots)
        midi[0, 0] = 40.5
        with self.assertRaisesRegex(smoke.V280RealSmokeError, "ambiguous"):
            smoke.real_targets(np.asarray([1]), slots, midi, slots)

    def test_balanced_rows_are_deterministic_and_eligible_only(self):
        cardinality = np.asarray([1, 2, 1, 2, 1, 2, 1, 2])
        eligible = np.asarray([True, True, True, False, True, True, True, True])
        selected = smoke.select_balanced_rows(cardinality, eligible, rows=7)
        np.testing.assert_array_equal(selected, np.asarray([0, 1, 2, 5, 4, 7, 6]))
        self.assertTrue(np.all(eligible[selected]))

    def test_selected_feature_rows_are_read_from_the_matching_track_cache(self):
        config = CausalCQTConfig(
            sample_rate=4000,
            hop_length=64,
            bins_per_semitone=3,
            input_min_midi=60.0,
            output_max_midi=72.0,
            max_harmonic_order=3,
            cluster_post_samples=192,
            cluster_frames=8,
            block_frames=7,
        )
        member = _member(0, "alpha")
        rng = np.random.default_rng(28034)
        track = causal_cqt(rng.normal(0.0, 0.1, 1024).astype(np.float32), config)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "tracks" / "alpha.npz"
            write_track_cache(path, track, config)
            manifest = {
                "cache": {
                    "tracks": [
                        {"annotation_member": member, "cache_path": "tracks/alpha.npz"}
                    ]
                }
            }
            features, diagnostics = smoke.extract_selected_features(
                root,
                manifest,
                (member,),
                (np.asarray([128, 160], dtype=np.int32),),
                np.asarray([0], dtype=np.int64),
                config,
            )
        self.assertEqual(
            features.shape,
            (1, config.cluster_frames, len(config.center_frequencies_hz), 3),
        )
        self.assertTrue(np.isfinite(features).all())
        self.assertEqual(diagnostics["selected_track_count"], 1)
        self.assertEqual(diagnostics["cluster_start_min"], 128)
        self.assertEqual(diagnostics["decision_end_max"], 320)


if __name__ == "__main__":
    unittest.main()
