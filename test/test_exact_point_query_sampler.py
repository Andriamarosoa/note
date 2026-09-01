import copy
from pathlib import Path
import random
import subprocess
import sys
import unittest

from causal_note.guitarset import GuitarSetTrack, NoteBoundary
from scripts.audit_anonymous_boundary_targets import AuditTrack
from scripts.audit_exact_point_query_sampler import (
    LEFT_HISTORY_SAMPLES,
    _oracle_reconciliation,
    _sample_uniform_background,
    audit_fixed_sampler,
    background_near_boundary_count,
    build_population,
    full_stream_report,
    source_counts_for_queries,
)


def _slots(*slot_notes):
    values = [tuple(notes) for notes in slot_notes]
    values.extend([tuple()] * (6 - len(values)))
    return tuple(values)


def _audit_track(frame_count=100, player_id="00", stem="test"):
    track = GuitarSetTrack(
        player_id=player_id,
        annotation_zip=Path("annotation.zip"),
        annotation_member=f"annotation/{player_id}_{stem}_comp.jams",
        audio_zip=Path("audio.zip"),
        audio_member=f"audio/{player_id}_{stem}_comp_mix.wav",
    )
    slots = _slots(
        (NoteBoundary(2, 5),),
        (NoteBoundary(2, 7),),
        (NoteBoundary(5, frame_count),),
    )
    return AuditTrack(track, slots, frame_count)


class PointQueryAuditImportTests(unittest.TestCase):
    def test_import_does_not_import_tensorflow(self):
        project_root = Path(__file__).resolve().parents[1]
        code = (
            "import sys; "
            f"sys.path.insert(0, {str(project_root)!r}); "
            "import scripts.audit_exact_point_query_sampler; "
            "raise SystemExit(int('tensorflow' in sys.modules))"
        )
        completed = subprocess.run(
            [sys.executable, "-c", code],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


class PointPopulationTests(unittest.TestCase):
    def test_unique_position_pool_preserves_multiplicity_and_both_targets(self):
        population = build_population((_audit_track(frame_count=10),))

        self.assertEqual(
            population.stratum_sizes,
            {"onset_bearing": 2, "offset_only": 1, "background": 7},
        )
        onset_by_position = {
            target.position: target
            for target in population.positive_pools["onset_bearing"]
        }
        self.assertEqual(onset_by_position[2].joint, (2, 0))
        self.assertEqual(onset_by_position[5].joint, (1, 1))
        self.assertEqual(len(onset_by_position), 2)

        report = full_stream_report(population)
        self.assertEqual(report["heads"]["onset"]["event_instances"], 3)
        self.assertEqual(report["heads"]["offset"]["event_instances"], 2)
        self.assertEqual(report["positions_with_both_targets_positive"], 1)
        self.assertEqual(
            report["context_edges"][
                "exclusive_end_offset_instances_excluded_from_acoustic_target"
            ],
            1,
        )

    def test_near_boundary_count_excludes_boundaries_and_merges_overlap(self):
        count = background_near_boundary_count(
            (2, 5, 7),
            frame_count=10,
            radius=1,
        )

        self.assertEqual(count, 5)

    def test_source_cycle_is_exact(self):
        self.assertEqual(
            source_counts_for_queries(1600),
            {"onset_bearing": 534, "offset_only": 533, "background": 533},
        )
        self.assertEqual(
            source_counts_for_queries(400),
            {"onset_bearing": 134, "offset_only": 133, "background": 133},
        )

    def test_global_background_sampling_handles_unequal_track_lengths(self):
        population = build_population(
            (
                _audit_track(frame_count=100, player_id="00", stem="short"),
                _audit_track(frame_count=1000, player_id="01", stem="long"),
            )
        )
        cumulative_frames = (100, 1100)

        def draw(seed):
            rng = random.Random(seed)
            result = []
            for _ in range(5000):
                target, _ = _sample_uniform_background(
                    population,
                    rng=rng,
                    cumulative_frames=cumulative_frames,
                )
                self.assertNotIn(
                    target.position,
                    population.tracks[target.track_index].boundary_positions,
                )
                result.append((target.track_index, target.position))
            return result

        first = draw(91)
        second = draw(91)
        self.assertEqual(first, second)
        observed_long = sum(track_index == 1 for track_index, _ in first) / len(first)
        expected_long = 997 / (97 + 997)
        self.assertAlmostEqual(observed_long, expected_long, delta=0.03)

    def test_locked_oracle_reconciliation_is_not_self_referential(self):
        tracks = tuple(
            _audit_track(
                frame_count=100 + index,
                player_id=f"{index:02d}",
                stem=f"oracle{index}",
            )
            for index in range(5)
        )
        population = build_population(tracks)
        full = full_stream_report(population)

        def prior_split():
            heads = {}
            for head in ("onset", "offset"):
                actual = full["heads"][head]
                heads[head] = {
                    "exact_anonymous_count": {
                        "total_samples": full["frames"],
                        "histogram": actual["exact_count_histogram"],
                        "positive_positions": actual["positive_positions"],
                        "event_instances": actual["event_instances"],
                        "maximum_count": actual["maximum_exact_count"],
                    },
                    "raw_annotations": {
                        "exclusive_end_instances": (
                            full["context_edges"][
                                "exclusive_end_offset_instances_excluded_from_acoustic_target"
                            ]
                            if head == "offset"
                            else 0
                        ),
                        "event_instances": actual["event_instances"]
                        + (
                            full["context_edges"][
                                "exclusive_end_offset_instances_excluded_from_acoustic_target"
                            ]
                            if head == "offset"
                            else 0
                        ),
                    },
                }
            return {"heads": heads}

        prior = {
            "full_stream": {
                "train": prior_split(),
                "validation": prior_split(),
            }
        }
        exact = _oracle_reconciliation(
            {"train": population, "validation": population},
            {"train": full, "validation": full},
            prior,
        )
        self.assertTrue(exact["all_exact"])

        altered = copy.deepcopy(prior)
        altered["full_stream"]["train"]["heads"]["onset"][
            "exact_anonymous_count"
        ]["event_instances"] += 1
        mismatch = _oracle_reconciliation(
            {"train": population, "validation": population},
            {"train": full, "validation": full},
            altered,
        )
        self.assertFalse(mismatch["all_exact"])
        self.assertFalse(mismatch["checks"]["train_onset_event_instances"])


class FixedSamplerTests(unittest.TestCase):
    def test_importance_weights_recover_full_joint_prior_analytically(self):
        population = build_population((_audit_track(frame_count=100),))

        report = audit_fixed_sampler(population, query_count=60, seed=17)

        self.assertEqual(
            report["source_counts"],
            {"onset_bearing": 20, "offset_only": 20, "background": 20},
        )
        self.assertLessEqual(
            report["importance_correction"][
                "analytical_weighted_joint_max_absolute_error"
            ],
            1e-12,
        )
        self.assertAlmostEqual(
            report["importance_correction"][
                "analytical_sampled_mean_weight"
            ],
            1.0,
        )
        self.assertFalse(
            report["importance_correction"]["approved_for_training_loss"]
        )

    def test_causal_context_uses_only_left_initialization(self):
        population = build_population((_audit_track(frame_count=100),))

        report = audit_fixed_sampler(population, query_count=30, seed=3)

        self.assertGreater(
            report["selection"]["contexts_requiring_left_zero_initialization"],
            0,
        )
        self.assertEqual(report["selection"]["right_padding_samples"], 0)
        self.assertEqual(LEFT_HISTORY_SAMPLES, 4092)


if __name__ == "__main__":
    unittest.main()
