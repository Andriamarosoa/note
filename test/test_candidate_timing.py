"""Regressions for ambiguous origins and supervision after top-48 truncation."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from scripts import candidate_timing as timing
from scripts import train_v90_structured_cluster_cardinality as v90
from scripts import train_v91_ordinal_cardinality as v91
from scripts import train_v92_string_factorized_cardinality as v92
from scripts import train_v100_spectral_string_slots as v100
from scripts import train_v102_source_time_assignment as v102


def fixture(groups, scores=None):
    records, clusters = [], []
    for samples in groups:
        ids = list(range(len(records), len(records) + len(samples)))
        records.extend({"sample": int(sample)} for sample in samples)
        clusters.append({"member": "00_test_comp.jams", "indices": ids})
    scores = np.asarray(scores if scores is not None else np.linspace(.9, .8, len(records)), np.float32)
    features = np.zeros((len(records), v90.FROZEN_CANDIDATE_DIM), np.float32)
    features[:, -1] = scores
    outputs = {"local_cardinality": np.tile([1., 0., 0., 0.], (len(records), 1)),
               "cluster_router": np.zeros(len(records))}
    seq, mask, stats, target, exact, truncated = v90._cluster_arrays(clusters, {}, records, features, outputs, scores)
    top = np.asarray([[records[max(c["indices"], key=lambda i: scores[i])]["sample"]] * 6
                      for c in clusters], np.int32)
    cache = dict(sequence=seq.astype(np.float16), mask=mask, stats=stats, target=target,
                 exact=exact, truncated=truncated, members=np.asarray([c["member"] for c in clusters]),
                 track_members=["00_test_comp.jams"], top_samples=top,
                 **timing.capture_timing(clusters, records, scores, v90.MAX_CANDIDATES))
    return cache, clusters, records


class CandidateTimingTests(unittest.TestCase):
    def test_ambiguous_origin_uses_original_integers(self):
        cache, _, _ = fixture([[1000, 1200]])
        self.assertEqual(v92._recover_cluster_start(cache["sequence"][0], cache["mask"][0], cache["top_samples"][0])[0], 800)
        retained, diag = v92._reconstruct_candidates(cache)
        np.testing.assert_array_equal(retained[0], [1000, 1200])
        self.assertEqual(diag["timing_source"], "stored_exact_retained_candidates")

    def test_dropped_endpoint_cannot_move_annotation_to_next_group(self):
        # The observed failure: onset 807819 is 1 sample from the dropped
        # endpoint, 319 from the last retained sample, 284 from the next group.
        first = np.rint(np.linspace(806264, 807500, 48)).astype(int).tolist()
        dropped = list(range(807700, 807708)) + [807820]
        cache, _, _ = fixture([first + dropped, [808103]], [.9] * 48 + [.1] * 9 + [.9])
        retained, _ = v92._reconstruct_candidates(cache)
        full, _ = v92._supervision_candidates(cache)
        track = SimpleNamespace(player_id="00", annotation_member="00_test_comp.jams",
                                annotation_zip=Path("annotation.zip"))
        slots = [[SimpleNamespace(onset_sample=807819)]] + [[] for _ in range(5)]
        with patch.object(v92, "load_boundary_slots", return_value=slots):
            old, _ = v92._slot_targets_for_member_clusters(track, [0, 1], retained)
            new, _ = v92._slot_targets_for_member_clusters(track, [0, 1], full)
        np.testing.assert_array_equal(old.sum(1), [0, 1])
        np.testing.assert_array_equal(new.sum(1), [1, 0])
        with patch.object(v102, "index_guitarset", return_value=[track]), \
             patch.object(v102.v101, "_pitch_events", return_value=[(0, 807819, 60)]):
            got = v102._derive_supervision(cache["members"], retained, Path("."),
                assignment_cache=cache, expected_slot_targets=new)
            oracle = v102._derive_supervision(cache["members"], full, Path("."), expected_slot_targets=new)
        for a, b in zip(got[:4], oracle[:4]):
            np.testing.assert_array_equal(a, b)
        self.assertEqual(len(retained[0]), 48)  # candidate IDs still index model rows
        self.assertEqual(len(full[0]), 57)

    def test_window_origin_survives_removal_of_first_candidate(self):
        cache, clusters, records = fixture([list(range(1000, 2000, 20))], [.01, .02] + [.9] * 48)
        self.assertEqual(v92._reconstruct_candidates(cache)[0][0][0], 1040)
        track = SimpleNamespace(player_id="00", annotation_member="00_test_comp.jams",
                                audio_zip=Path("audio.zip"), audio_member="audio.wav")
        audio = SimpleNamespace(samples=(np.sin(np.arange(6000) / 11) * 15000).astype(np.int16))
        with patch.object(v100, "index_guitarset", return_value=[track]), \
             patch.object(v100, "decode_pcm16_mono_wav", return_value=audio):
            cached, _ = v100._spectral_maps_for_cache(cache, Path("."))
            live = v100._spectral_maps_for_runtime([track], clusters, records)
        np.testing.assert_array_equal(cached, live.astype(np.float16))

    def test_same_string_collision_keeps_nearest_onset(self):
        track = SimpleNamespace(player_id="00", annotation_member="00_test_comp.jams")
        with patch.object(v102, "index_guitarset", return_value=[track]), \
             patch.object(v102.v101, "_pitch_events", return_value=[(0, 1030, 60), (0, 1100, 65)]):
            result = v102._derive_supervision([track.annotation_member], [np.asarray([1000, 1200])], Path("."))
        self.assertEqual(result[3][0, 0], 30)
        self.assertEqual(result[4]["same_slot_collisions"], 1)

    def test_pitch_targets_keep_global_rows_across_members(self):
        cache, _, _ = fixture([[1000], [5000]])
        names = ["00_test_comp.jams", "00_other_comp.jams"]
        cache["members"] = np.asarray(names)
        cache["slot_targets"] = np.asarray([[1, 0, 0, 0, 0, 0]] * 2)
        tracks = [SimpleNamespace(player_id="00", annotation_member=name) for name in names]
        def events(track):
            return [(0, 1030, 60)] if track.annotation_member == names[0] else [(0, 5030, 65)]
        with patch.object(v102.v101, "index_guitarset", return_value=tracks), \
             patch.object(v102.v101, "_pitch_events", side_effect=events):
            pitch, mask, _ = v102.v101._derive_pitch_targets(cache, Path("."))
        np.testing.assert_array_equal(mask, cache["slot_targets"])
        np.testing.assert_allclose(pitch[:, 0] * v102.v101.PITCH_SCALE, [60, 65])

    def test_v91_and_v100_multishard_round_trip(self):
        with TemporaryDirectory() as name:
            root = Path(name)
            originals = []
            for i, groups in enumerate(([[1000, 1200]], [[5000, 5300], [8000]])):
                c, _, _ = fixture(groups)
                originals.append(c)
                base = {k: c[k] for k in ("sequence", "mask", "stats", "target", "exact", "truncated", "members", "top_samples", "track_members")}
                v91._save_cache(root / f"v91-cache-shard-{i:02d}.npz", **base, timing=timing.timing_fields(c))
                spectral = np.zeros((len(groups), v100.TIME_FRAMES, v100.SPECTRAL_BANDS, v100.SPECTRAL_CHANNELS))
                v100._save_spectral_cache(root / f"v100-spectral-shard-{i:02d}.npz", c, spectral, np.zeros((len(groups), 6)))
            a, b = v91._load_caches(root), v100._load_spectral_caches(root)
            np.testing.assert_array_equal(a["full_candidate_offsets"], [0, 2, 4, 5])
            for key in timing.TIMING_KEYS:
                np.testing.assert_array_equal(a[key], b[key])
            self.assertEqual([x.tolist() for x in timing.full_samples(a)], [[1000, 1200], [5000, 5300], [8000]])

    def test_legacy_reads_but_cannot_be_mixed_or_written_as_exact(self):
        cache, _, _ = fixture([[1000, 1200]])
        legacy = {k: v for k, v in cache.items() if k not in timing.TIMING_KEYS}
        timing.check_schema(legacy, 1)
        with self.assertRaisesRegex(ValueError, "mix legacy"):
            timing.merge_timing([legacy, cache])
        with self.assertRaisesRegex(ValueError, "missing"):
            timing.check_schema(legacy, 2)
        with TemporaryDirectory() as name:
            path = Path(name) / "v91-cache-shard-00.npz"
            np.savez(path, schema_version=[1], **legacy)
            loaded = v91._load_caches(Path(name))
            self.assertEqual(v92._reconstruct_candidates(loaded)[1]["timing_source"], "legacy_ambiguous_reconstruction")
            with self.assertRaisesRegex(ValueError, "missing"):
                v100._save_spectral_cache(Path(name) / "new.npz", legacy, [], [])

    def test_corrupt_timing_is_rejected(self):
        c, _, _ = fixture([[1000, 1200]])
        partial = dict(c)
        partial.pop("candidate_samples")
        with self.assertRaisesRegex(ValueError, "incomplete"):
            timing.timing_fields(partial)
        for key, replacement in (("full_candidate_offsets", [0, 99]),
                                 ("cluster_start_samples", [800]),
                                 ("candidate_samples", c["candidate_samples"].astype(float))):
            with self.subTest(key=key), self.assertRaises(ValueError):
                timing.timing_fields({**c, key: np.asarray(replacement)})
        reordered = c["candidate_samples"].copy()
        reordered[0, :2] = [1200, 1000]
        with self.assertRaisesRegex(ValueError, "feature rows"):
            timing.timing_fields({**c, "candidate_samples": reordered})


if __name__ == "__main__":
    unittest.main()
