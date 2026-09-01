from pathlib import Path
from collections import Counter
import copy
import math
import random
import subprocess
import sys
import unittest

from causal_note.guitarset import GuitarSetTrack, NoteBoundary
from scripts.audit_anonymous_boundary_targets import AuditTrack
from scripts.audit_exact_point_query_sampler import (
    build_population,
    nearest_boundary_distance,
)
from scripts.audit_hard_negative_sampler import (
    BACKGROUND_STRATA,
    GRID_H,
    _distance_one_epoch_exposure,
    _rare_exposure_comparison,
    analytical_candidate_report,
    background_band_sizes,
    background_stratum_for_distance,
    candidate_source_counts,
    choose_candidate,
    sample_background_stratum,
)


def _slots(*slot_notes):
    values = [tuple(notes) for notes in slot_notes]
    values.extend([tuple()] * (6 - len(values)))
    return tuple(values)


def _audit_track(
    *,
    frame_count=200,
    notes=(),
    player_id="00",
    stem="hard-negative",
):
    track = GuitarSetTrack(
        player_id=player_id,
        annotation_zip=Path("annotation.zip"),
        annotation_member=f"annotation/{player_id}_{stem}_comp.jams",
        audio_zip=Path("audio.zip"),
        audio_member=f"audio/{player_id}_{stem}_comp_mix.wav",
    )
    return AuditTrack(track, _slots(tuple(notes)), frame_count)


def _sample_target(result):
    """Accept either the target itself or the audit's ``(target, attempts)``."""

    return result[0] if isinstance(result, tuple) else result


class HardNegativeAuditImportTests(unittest.TestCase):
    def test_import_does_not_import_tensorflow(self):
        project_root = Path(__file__).resolve().parents[1]
        code = (
            "import sys; "
            f"sys.path.insert(0, {str(project_root)!r}); "
            "import scripts.audit_hard_negative_sampler; "
            "raise SystemExit(int('tensorflow' in sys.modules))"
        )
        completed = subprocess.run(
            [sys.executable, "-c", code],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


class BackgroundBandTests(unittest.TestCase):
    def test_exact_distance_boundaries(self):
        near_one, near_fifteen, near_sixty_three, far = BACKGROUND_STRATA

        self.assertEqual(background_stratum_for_distance(1), near_one)
        self.assertEqual(background_stratum_for_distance(2), near_fifteen)
        self.assertEqual(background_stratum_for_distance(15), near_fifteen)
        self.assertEqual(background_stratum_for_distance(16), near_sixty_three)
        self.assertEqual(background_stratum_for_distance(63), near_sixty_three)
        self.assertEqual(background_stratum_for_distance(64), far)

    def test_bands_partition_background_exactly_and_exclude_boundary(self):
        # The sole acoustic boundary is 64.  The offset at the exclusive end
        # (130) is deliberately outside the acoustic target domain.
        population = build_population(
            (_audit_track(frame_count=130, notes=(NoteBoundary(64, 130),)),)
        )

        sizes = background_band_sizes(population)

        self.assertEqual(set(sizes), set(BACKGROUND_STRATA))
        self.assertEqual(tuple(sizes[key] for key in BACKGROUND_STRATA), (2, 28, 96, 3))
        self.assertEqual(sum(sizes.values()), population.stratum_sizes["background"])
        self.assertEqual(sum(sizes.values()), population.total_frames - 1)

    def test_track_without_boundary_is_entirely_far_background(self):
        population = build_population((_audit_track(frame_count=17),))

        sizes = background_band_sizes(population)

        self.assertEqual(sizes[BACKGROUND_STRATA[-1]], 17)
        self.assertEqual(
            sum(sizes[key] for key in BACKGROUND_STRATA[:-1]),
            0,
        )

    def test_sampling_is_deterministic_inside_every_nonempty_band(self):
        population = build_population(
            (_audit_track(frame_count=130, notes=(NoteBoundary(64, 130),)),)
        )
        cumulative_frames = (population.total_frames,)

        def draw(stratum):
            rng = random.Random(1729)
            result = []
            for _ in range(25):
                target = _sample_target(
                    sample_background_stratum(
                        population,
                        stratum,
                        rng,
                        cumulative_frames,
                    )
                )
                self.assertNotIn(
                    target.position,
                    population.tracks[target.track_index].boundary_positions,
                )
                distance = nearest_boundary_distance(
                    target.position,
                    tuple(
                        sorted(
                            population.tracks[
                                target.track_index
                            ].boundary_positions
                        )
                    ),
                )
                self.assertEqual(background_stratum_for_distance(distance), stratum)
                self.assertEqual(target.joint, (0, 0))
                result.append((target.track_index, target.position))
            return result

        for stratum in BACKGROUND_STRATA:
            with self.subTest(stratum=stratum):
                self.assertEqual(draw(stratum), draw(stratum))

    def test_distance_one_sampling_is_uniform_across_unequal_tracks(self):
        population = build_population(
            (
                _audit_track(
                    frame_count=10,
                    notes=(NoteBoundary(2, 10),),
                    player_id="00",
                    stem="uniform-short",
                ),
                _audit_track(
                    frame_count=100,
                    notes=(
                        NoteBoundary(40, 100),
                        NoteBoundary(60, 100),
                    ),
                    player_id="01",
                    stem="uniform-long",
                ),
            )
        )
        distance_one = BACKGROUND_STRATA[0]
        eligible = {(0, 1), (0, 3), (1, 39), (1, 41), (1, 59), (1, 61)}
        cumulative_frames = (10, 110)
        rng = random.Random(20260901)
        draws = 10_000
        observed = Counter()

        for _ in range(draws):
            target = _sample_target(
                sample_background_stratum(
                    population,
                    distance_one,
                    rng,
                    cumulative_frames,
                )
            )
            key = (target.track_index, target.position)
            self.assertIn(key, eligible)
            observed[key] += 1

        expected_position_frequency = 1.0 / len(eligible)
        for key in eligible:
            with self.subTest(position=key):
                self.assertAlmostEqual(
                    observed[key] / draws,
                    expected_position_frequency,
                    delta=0.025,
                )

        short_track_frequency = sum(
            observed[key] for key in eligible if key[0] == 0
        ) / draws
        long_track_frequency = sum(
            observed[key] for key in eligible if key[0] == 1
        ) / draws
        self.assertAlmostEqual(short_track_frequency, 2.0 / 6.0, delta=0.025)
        self.assertAlmostEqual(long_track_frequency, 4.0 / 6.0, delta=0.025)


class CandidateQuotaTests(unittest.TestCase):
    def test_every_grid_candidate_has_exact_split_totals(self):
        near_one, near_fifteen, near_sixty_three, far = BACKGROUND_STRATA
        expected_keys = {
            "onset_bearing",
            "offset_only",
            *BACKGROUND_STRATA,
        }

        for h in GRID_H:
            with self.subTest(h=h, split="train"):
                counts = candidate_source_counts(h, "train")
                self.assertEqual(set(counts), expected_keys)
                self.assertEqual(sum(counts.values()), 1600)
                self.assertEqual(counts["onset_bearing"], 534)
                self.assertEqual(counts["offset_only"], 533)
                self.assertEqual(counts[near_one], h)
                self.assertEqual(counts[near_fifteen], 4 * h)
                self.assertEqual(counts[near_sixty_three], 16 * h)
                self.assertEqual(counts[far], 533 - 21 * h)

            with self.subTest(h=h, split="validation"):
                counts = candidate_source_counts(h, "validation")
                self.assertEqual(set(counts), expected_keys)
                self.assertEqual(sum(counts.values()), 400)
                self.assertEqual(counts["onset_bearing"], 134)
                self.assertEqual(counts["offset_only"], 133)
                expected_near_one = math.ceil(h / 4)
                self.assertEqual(counts[near_one], expected_near_one)
                self.assertEqual(counts[near_fifteen], h)
                self.assertEqual(counts[near_sixty_three], 4 * h)
                self.assertEqual(
                    counts[far],
                    133 - expected_near_one - h - 4 * h,
                )


class AnalyticalCorrectionTests(unittest.TestCase):
    def test_six_strata_and_joint_prior_are_recovered_analytically(self):
        population = build_population(
            (_audit_track(notes=(NoteBoundary(64, 100),)),)
        )
        counts = candidate_source_counts(GRID_H[0], "train")

        report = analytical_candidate_report(population, counts)
        correction = report["importance_correction"]

        self.assertLessEqual(
            correction["analytical_sampled_mean_weight_absolute_error_from_one"],
            1e-12,
        )
        self.assertLessEqual(
            correction["analytical_weighted_stratum_max_absolute_error"],
            1e-12,
        )
        self.assertLessEqual(
            correction["analytical_weighted_joint_max_absolute_error"],
            1e-12,
        )
        self.assertGreater(correction["effective_sample_size_ratio"], 0.0)
        self.assertLessEqual(correction["effective_sample_size_ratio"], 1.0)

    def test_distance_one_exposure_distinguishes_fresh_and_cached_epochs(self):
        population = build_population(
            (_audit_track(frame_count=130, notes=(NoteBoundary(64, 130),)),)
        )
        analytical = {
            "source_counts": {"distance_1": 1},
            "live_stratum_sizes": {"distance_1": 1_000_000},
        }
        fixed = {
            "selection": {
                "by_stratum": {"distance_1": {"unique_positions": 1}}
            }
        }

        exposure = _distance_one_epoch_exposure(population, analytical, fixed)

        fresh = exposure["fresh_independent_epochs"]
        cached = exposure["same_cached_fixed_batch_repeated"]
        self.assertEqual(fresh["draws_per_epoch"], 1)
        self.assertEqual(fresh["draws_over_20_epochs"], 20)
        self.assertGreater(
            fresh["expected_unique_positions_over_20_epochs"],
            19.99,
        )
        self.assertLessEqual(
            fresh["expected_unique_positions_over_20_epochs"],
            20.0,
        )
        self.assertEqual(cached["draws_over_20_repetitions"], 20)
        self.assertEqual(cached["unique_positions_in_fixed_batch"], 1)
        self.assertEqual(cached["unique_positions_after_20_repetitions"], 1)

    def test_rare_exposure_comparison_detects_any_changed_value(self):
        fields = {
            "pool_positions": 3,
            "positions_by_stratum": {
                "onset_bearing": 3,
                "offset_only": 0,
            },
            "expected_draws_per_fixed_epoch": 2.5,
            "expected_draws_over_20_epochs": 50.0,
            "probability_seen_at_least_once_over_20_epochs": 0.875,
        }
        rare_exposure = {
            "by_head": {
                "onset": {"classes": {"1": copy.deepcopy(fields)}},
                "offset": {"classes": {"1": copy.deepcopy(fields)}},
            }
        }
        experiment_12 = {
            "splits": {
                "train": {
                    "fixed_point_query_sampler": {
                        "rare_count_class_exposure": {
                            "onset": {
                                "classes": {"1": copy.deepcopy(fields)}
                            },
                            "offset": {
                                "classes": {"1": copy.deepcopy(fields)}
                            },
                        }
                    }
                }
            }
        }

        unchanged = _rare_exposure_comparison(rare_exposure, experiment_12)
        self.assertTrue(unchanged["all_expected_exposures_exactly_unchanged"])

        altered = copy.deepcopy(rare_exposure)
        altered["by_head"]["offset"]["classes"]["1"][
            "expected_draws_over_20_epochs"
        ] += 1
        changed = _rare_exposure_comparison(altered, experiment_12)
        self.assertFalse(changed["all_expected_exposures_exactly_unchanged"])


class CandidateSelectionTests(unittest.TestCase):
    @staticmethod
    def _report(passes):
        return {"guards": {"passes_all_train_selection_gates": passes}}

    def test_smallest_passing_candidate_is_selected(self):
        reports = {
            h: self._report(h in {GRID_H[1], GRID_H[2]})
            for h in reversed(GRID_H)
        }

        self.assertEqual(choose_candidate(reports), GRID_H[1])

    def test_candidate_is_refused_when_no_guard_set_passes(self):
        reports = {h: self._report(False) for h in GRID_H}

        self.assertIsNone(choose_candidate(reports))


if __name__ == "__main__":
    unittest.main()
