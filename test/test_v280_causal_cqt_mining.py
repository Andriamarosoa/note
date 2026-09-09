from __future__ import annotations

from array import array
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from causal_note.guitarset import GuitarSetTrack
from causal_note.v280_causal_cqt import CausalCQTConfig
from scripts import mine_v280_causal_cqt as mining
from scripts.train_boundaries import Pcm16Audio


def _small_config() -> CausalCQTConfig:
    return CausalCQTConfig(
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


def _track(player: int, group: int, arrangement: str) -> GuitarSetTrack:
    stem = f"{player:02d}_piece{group:02d}_{arrangement}"
    return GuitarSetTrack(
        player_id=f"{player:02d}",
        annotation_zip=Path("annotation.zip"),
        annotation_member=f"annotation/{stem}.jams",
        audio_zip=Path("audio.zip"),
        audio_member=f"audio/{stem}_mix.wav",
    )


def _canonical_fake_split():
    by_group = {
        group: tuple(
            _track(player, group, arrangement)
            for player in range(5)
            for arrangement in ("comp", "solo")
        )
        for group in range(30)
    }
    train = tuple(track for group in range(24) for track in by_group[group])
    validation = tuple(track for group in range(24, 30) for track in by_group[group])
    indexed = tuple(sorted(train + validation, key=lambda track: track.annotation_member))
    return indexed, train, validation


def _membership(train) -> mining.V100Membership:
    members = tuple(sorted(track.annotation_member for track in train))
    return mining.V100Membership(
        track_members=members,
        shard_paths=tuple(f"shard-{index}/v100-spectral-shard-{index:02d}.npz" for index in range(8)),
        sha256=mining._member_digest(members),
    )


def _write_fake_v100_shard(path: Path, members) -> None:
    # Object-valued supervision arrays would raise under allow_pickle=False if
    # the metadata-only loader accidentally attempted to deserialize them.
    forbidden = np.asarray([{"must_not_load": True}], dtype=object)
    np.savez(
        path,
        schema_version=np.asarray([1], dtype=np.int16),
        spectral=np.empty((0,), dtype=np.float16),
        sequence=np.empty((0,), dtype=np.float16),
        mask=np.empty((0,), dtype=np.uint8),
        stats=np.empty((0,), dtype=np.float16),
        target=forbidden,
        exact=forbidden,
        members=forbidden,
        top_samples=forbidden,
        slot_targets=forbidden,
        track_members=np.asarray(members, dtype="U96"),
    )


class V100MetadataOnlyTests(unittest.TestCase):
    def test_loader_never_deserializes_supervision_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_fake_v100_shard(
                root / "v100-spectral-shard-00.npz",
                ("annotation/00_piece00_comp.jams",),
            )
            _write_fake_v100_shard(
                root / "v100-spectral-shard-01.npz",
                ("annotation/01_piece00_comp.jams",),
            )
            result = mining.load_v100_track_members(root, expected_shard_count=2)
        self.assertEqual(len(result.track_members), 2)
        self.assertEqual(result.track_members, tuple(sorted(result.track_members)))
        self.assertEqual(result.sha256, mining._member_digest(result.track_members))

    def test_loader_rejects_duplicate_members_across_shards(self):
        member = "annotation/00_piece00_comp.jams"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_fake_v100_shard(root / "v100-spectral-shard-00.npz", (member,))
            _write_fake_v100_shard(root / "v100-spectral-shard-01.npz", (member,))
            with self.assertRaisesRegex(mining.V280CacheMiningError, "across V10 shards"):
                mining.load_v100_track_members(root, expected_shard_count=2)

    def test_loader_rejects_wrong_shard_count(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(mining.V280CacheMiningError, "expected 8"):
                mining.load_v100_track_members(Path(directory))


class OuterCleanSelectionTests(unittest.TestCase):
    def test_exact_v100_membership_and_group_isolation_are_required(self):
        indexed, train, validation = _canonical_fake_split()
        membership = _membership(train)
        result = mining.validate_outer_clean_selection(indexed, train, validation, membership)
        self.assertEqual(len(result.tracks), 240)
        self.assertEqual(result.validation_track_count, 60)
        self.assertEqual(len(result.train_groups), 24)
        self.assertEqual(len(result.validation_groups), 6)
        self.assertFalse(set(result.train_groups) & set(result.validation_groups))

    def test_historical_validation_member_is_rejected(self):
        indexed, train, validation = _canonical_fake_split()
        members = list(_membership(train).track_members)
        members[0] = validation[0].annotation_member
        bad = mining.V100Membership(
            tuple(sorted(members)),
            tuple(f"shard-{index}" for index in range(8)),
            mining._member_digest(members),
        )
        with self.assertRaisesRegex(mining.V280CacheMiningError, "differs from outer-clean"):
            mining.validate_outer_clean_selection(indexed, train, validation, bad)

    def test_track_cache_filename_is_deterministic_and_path_safe(self):
        member = "annotation/00_piece00_comp.jams"
        first = mining.track_cache_filename(member)
        self.assertEqual(first, mining.track_cache_filename(member))
        self.assertRegex(first, r"^track-[0-9a-f]{24}\.npz$")
        self.assertNotIn("/", first)
        self.assertNotEqual(first, mining.track_cache_filename(member.replace("comp", "solo")))


class CacheMiningEndToEndTests(unittest.TestCase):
    def test_bounded_mine_manifest_resume_and_integrity_verification(self):
        indexed, train, validation = _canonical_fake_split()
        membership = _membership(train)
        selection = mining.validate_outer_clean_selection(indexed, train, validation, membership)
        config = _small_config()
        samples = array("h", [int(12000 * np.sin(index / 11.0)) for index in range(768)])
        audio = Pcm16Audio(config.sample_rate, samples)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "cache"
            args = SimpleNamespace(
                dataset_dir=Path("unused-dataset"),
                v100_cache_dir=Path("unused-v100"),
                output_dir=output,
                limit_tracks=2,
                resume=False,
            )
            with patch.object(
                mining,
                "select_outer_clean_tracks",
                return_value=(selection, membership),
            ), patch.object(mining, "decode_pcm16_mono_wav", return_value=audio) as decode:
                report = mining.mine(args, config)
            self.assertEqual(decode.call_count, 2)
            self.assertFalse(report["formal"])
            self.assertEqual(report["data"]["outer_clean_track_count"], 240)
            self.assertEqual(report["data"]["historical_validation_track_count"], 60)
            self.assertEqual(report["data"]["mined_track_count"], 2)
            self.assertEqual(report["protocol"]["v100_arrays_deserialized"], ["schema_version", "track_members"])
            self.assertFalse(report["protocol"]["v100_supervision_arrays_deserialized"])
            self.assertFalse(report["protocol"]["jams_payloads_read"])
            self.assertFalse(report["protocol"]["outer_fold_trained_or_evaluated"])
            verified = mining.verify_cache(
                output,
                expected_track_count=2,
                expected_formal=False,
                config=config,
            )
            self.assertTrue(verified["verified"])
            self.assertGreater(verified["total_cache_bytes"], 0)

            args.resume = True
            with patch.object(
                mining,
                "select_outer_clean_tracks",
                return_value=(selection, membership),
            ), patch.object(mining, "decode_pcm16_mono_wav") as decode_again:
                resumed = mining.mine(args, config)
            decode_again.assert_not_called()
            self.assertEqual(resumed["cache"]["cache_set_sha256"], report["cache"]["cache_set_sha256"])

            first_path = output / report["cache"]["tracks"][0]["cache_path"]
            with first_path.open("ab") as stream:
                stream.write(b"tamper")
            with self.assertRaisesRegex(mining.V280CacheMiningError, "digest mismatch"):
                mining.verify_cache(
                    output,
                    expected_track_count=2,
                    expected_formal=False,
                    config=config,
                )


if __name__ == "__main__":
    unittest.main()
