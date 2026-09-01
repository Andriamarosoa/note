"""Audit distance-stratified background sampling for exact point queries.

Experiment 13 changes only the background draw distribution from Experiment
12.  It does not decode audio, import TensorFlow, select a loss or train a
model.  Candidate selection is analytical and train-only; validation is read
only after the train candidate has been locked by the preregistered rule.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path, PurePosixPath
import random
import sys
from typing import Dict, List, Mapping, Optional, Sequence, Tuple


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from causal_note.guitarset import ALLOWED_PLAYERS, index_guitarset  # noqa: E402
from scripts.audit_anonymous_boundary_targets import _prepare_tracks  # noqa: E402
from scripts.audit_exact_point_query_sampler import (  # noqa: E402
    HEADS,
    HISTORICAL_EPOCHS,
    LEFT_HISTORY_SAMPLES,
    PointQueryAuditError,
    PointTarget,
    SplitPopulation,
    _arrangement,
    _categorical_optimum,
    _head_histogram_from_joint,
    _histogram_json,
    _joint_json,
    _load_json,
    _oracle_reconciliation,
    _rare_class_exposure,
    _safe_ratio,
    background_near_boundary_count,
    build_population,
    full_stream_report,
    nearest_boundary_distance,
    sha256_file,
)
from scripts.train_boundaries import split_tracks_by_group  # noqa: E402


POSITIVE_STRATA = ("onset_bearing", "offset_only")
BACKGROUND_STRATA = (
    "distance_1",
    "distance_2_to_15",
    "distance_16_to_63",
    "distance_64_plus",
)
ALL_STRATA = POSITIVE_STRATA + BACKGROUND_STRATA
GRID_H = (1, 2, 4, 8)
SEED = 1337
MINIMUM_ESS_RATIO = 0.25
MINIMUM_FAR_BACKGROUND_FRACTION = 0.8
MINIMUM_DISTANCE_ONE_DRAWS_OVER_20_EPOCHS = 20
MAX_ANALYTICAL_ERROR = 1e-12

DEFAULT_PROTOCOL = (
    REPOSITORY_ROOT
    / "model"
    / "causal-boundaries-hard-negative-sampler-protocol.json"
)
DEFAULT_AMENDMENT = (
    REPOSITORY_ROOT
    / "model"
    / "causal-boundaries-hard-negative-sampler-protocol-amendment-01.json"
)
DEFAULT_REPORT_AMENDMENT = (
    REPOSITORY_ROOT
    / "model"
    / "causal-boundaries-hard-negative-sampler-protocol-amendment-02.json"
)
DEFAULT_REVIEW_AMENDMENT = (
    REPOSITORY_ROOT
    / "model"
    / "causal-boundaries-hard-negative-sampler-protocol-amendment-03.json"
)
DEFAULT_METADATA = (
    REPOSITORY_ROOT
    / "model"
    / "causal-boundaries-weight28-window512-v7-epoch08.recovery.metadata.json"
)
DEFAULT_PRIOR_AUDIT = (
    REPOSITORY_ROOT / "model" / "causal-boundaries-type-position-target-audit.json"
)
DEFAULT_EXPERIMENT_12 = (
    REPOSITORY_ROOT
    / "model"
    / "causal-boundaries-exact-point-query-sampler-audit.json"
)
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT
    / "model"
    / "causal-boundaries-hard-negative-sampler-audit.json"
)

EXPECTED_PROTOCOL_SHA256 = (
    "D3FF0F3F9A35C6B0AE8079B9781E961DE9FAE380A42A601922217D2530999CBB"
)
EXPECTED_AMENDMENT_SHA256 = (
    "46C739E90BD03469D3DB160DEF3A6215605C1C395E3FBF60865612C8EFD3F6EF"
)
EXPECTED_REPORT_AMENDMENT_SHA256 = (
    "1E130543D20C2B91AB620A26E9228FABE90FF3D5C52C5BBE338A1BC4DB38C055"
)
EXPECTED_REVIEW_AMENDMENT_SHA256 = (
    "69F12014A41766611F0ADE37080C68213E68C2ADD752A923FB84A2EE9DB21680"
)
EXPECTED_EXPERIMENT_12_SOURCE_SHA256 = (
    "4F8DDA1D84E41910E1DFEA17DAF4150BD278FDDEFD02415D20A6DA812D12B193"
)
EXPECTED_EXPERIMENT_12_RESULT_SHA256 = (
    "1056B0349E826FE9D2E8845AA8CE88D5E4F5E9C719E209CC73C41F9A19D1EF56"
)
EXPECTED_METADATA_SHA256 = (
    "8AB99DC6DD191B39043A303D41BB405A0C84C2CE607C608E330E4DEF33F52A25"
)
EXPECTED_PRIOR_AUDIT_SHA256 = (
    "D1B311666BAFE8191D6CE3B655EE6CFB8B140F053741B7DF9D98613D0E93CA14"
)
EXPECTED_ANONYMOUS_HELPER_SHA256 = (
    "7C414A863E941B3595F03F7A420A5FB80810A5F4D2D9C5350C8089491135B356"
)
EXPECTED_TRAINING_SOURCE_SHA256 = (
    "3006FD2A586CE87166A82021C5D461472D4A492E6B5EBFA18910D492D80F136A"
)
EXPECTED_GUITARSET_SOURCE_SHA256 = (
    "52E56D374294748074A34DB06B64B03D096008CA44B8ACA0C390C981C04F94E9"
)

_RNG_OFFSETS = {
    "onset_bearing": 11,
    "offset_only": 23,
    "distance_1": 101,
    "distance_2_to_15": 211,
    "distance_16_to_63": 307,
    "distance_64_plus": 401,
}


def background_stratum_for_distance(distance: int) -> str:
    """Return the locked background band for a positive integer distance."""

    if isinstance(distance, bool) or not isinstance(distance, int) or distance < 1:
        raise PointQueryAuditError("background distance must be an integer >= 1")
    if distance == 1:
        return "distance_1"
    if distance <= 15:
        return "distance_2_to_15"
    if distance <= 63:
        return "distance_16_to_63"
    return "distance_64_plus"


def _position_background_stratum(
    population: SplitPopulation,
    track_index: int,
    position: int,
    ordered_boundaries: Optional[Sequence[Sequence[int]]] = None,
) -> str:
    item = population.tracks[track_index]
    if position < 0 or position >= item.frame_count:
        raise PointQueryAuditError("sample position escaped its track")
    if position in item.boundary_positions:
        raise PointQueryAuditError("a true boundary cannot enter a background band")
    boundaries = (
        tuple(sorted(item.boundary_positions))
        if ordered_boundaries is None
        else ordered_boundaries[track_index]
    )
    if not boundaries:
        return "distance_64_plus"
    return background_stratum_for_distance(
        nearest_boundary_distance(position, boundaries)
    )


def background_band_sizes(population: SplitPopulation) -> Mapping[str, int]:
    """Compute exact disjoint background-band sizes without enumerating frames."""

    cumulative = {}
    for radius in (1, 15, 63):
        cumulative[radius] = sum(
            background_near_boundary_count(
                tuple(item.boundary_positions),
                frame_count=item.frame_count,
                radius=radius,
            )
            for item in population.tracks
        )
    sizes = {
        "distance_1": cumulative[1],
        "distance_2_to_15": cumulative[15] - cumulative[1],
        "distance_16_to_63": cumulative[63] - cumulative[15],
        "distance_64_plus": population.stratum_sizes["background"]
        - cumulative[63],
    }
    if any(value < 0 for value in sizes.values()):
        raise PointQueryAuditError("background distance bands are not monotonic")
    if sum(sizes.values()) != population.stratum_sizes["background"]:
        raise PointQueryAuditError("background distance bands do not partition background")
    return sizes


def candidate_source_counts(h: int, split: str) -> Mapping[str, int]:
    """Return the six locked source counts for one grid candidate."""

    if h not in GRID_H:
        raise PointQueryAuditError(f"h must be one of {GRID_H}")
    if split == "train":
        counts = {
            "onset_bearing": 534,
            "offset_only": 533,
            "distance_1": h,
            "distance_2_to_15": 4 * h,
            "distance_16_to_63": 16 * h,
        }
        counts["distance_64_plus"] = 533 - 21 * h
        expected_queries = 1600
        expected_background = 533
    elif split == "validation":
        counts = {
            "onset_bearing": 134,
            "offset_only": 133,
            "distance_1": math.ceil(h / 4),
            "distance_2_to_15": math.ceil((4 * h) / 4),
            "distance_16_to_63": math.ceil((16 * h) / 4),
        }
        counts["distance_64_plus"] = 133 - sum(
            counts[stratum] for stratum in BACKGROUND_STRATA[:-1]
        )
        expected_queries = 400
        expected_background = 133
    else:
        raise PointQueryAuditError("split must be 'train' or 'validation'")
    if tuple(counts) != ALL_STRATA:
        raise PointQueryAuditError("candidate source order changed")
    if any(counts[stratum] <= 0 for stratum in ALL_STRATA):
        raise PointQueryAuditError("every locked stratum must receive a draw")
    if sum(counts.values()) != expected_queries:
        raise PointQueryAuditError("candidate query total changed")
    if sum(counts[stratum] for stratum in BACKGROUND_STRATA) != expected_background:
        raise PointQueryAuditError("candidate background total changed")
    return counts


def _live_stratum_sizes(population: SplitPopulation) -> Mapping[str, int]:
    band_sizes = background_band_sizes(population)
    sizes = {
        "onset_bearing": population.stratum_sizes["onset_bearing"],
        "offset_only": population.stratum_sizes["offset_only"],
        **band_sizes,
    }
    if tuple(sizes) != ALL_STRATA or sum(sizes.values()) != population.total_frames:
        raise PointQueryAuditError("six live strata do not partition the stream")
    if any(value <= 0 for value in sizes.values()):
        raise PointQueryAuditError("a required live stratum is empty")
    return sizes


def _full_joint_probability(population: SplitPopulation) -> Counter:
    result: Counter = Counter()
    for histogram in population.joint_by_stratum.values():
        for joint, count in histogram.items():
            result[joint] += _safe_ratio(count, population.total_frames)
    return result


def analytical_candidate_report(
    population: SplitPopulation,
    counts: Mapping[str, int],
) -> Mapping[str, object]:
    """Audit priors, weights, ESS and train-selection guards analytically."""

    if tuple(counts) != ALL_STRATA:
        raise PointQueryAuditError("analytical counts must contain six ordered strata")
    if any(counts[stratum] <= 0 for stratum in ALL_STRATA):
        raise PointQueryAuditError("analytical strata must all be sampled")
    query_count = sum(counts.values())
    live_sizes = _live_stratum_sizes(population)
    live_probabilities = {
        stratum: _safe_ratio(live_sizes[stratum], population.total_frames)
        for stratum in ALL_STRATA
    }
    sampled_probabilities = {
        stratum: _safe_ratio(counts[stratum], query_count)
        for stratum in ALL_STRATA
    }
    weights = {
        stratum: live_probabilities[stratum] / sampled_probabilities[stratum]
        for stratum in ALL_STRATA
    }
    weighted_stratum_probabilities = {
        stratum: sampled_probabilities[stratum] * weights[stratum]
        for stratum in ALL_STRATA
    }
    stratum_max_error = max(
        abs(weighted_stratum_probabilities[stratum] - live_probabilities[stratum])
        for stratum in ALL_STRATA
    )
    mean_weight = sum(weighted_stratum_probabilities.values())

    sampled_joint_probability: Counter = Counter()
    weighted_joint_probability: Counter = Counter()
    for stratum in POSITIVE_STRATA:
        pool_size = population.stratum_sizes[stratum]
        for joint, pool_count in population.joint_by_stratum[stratum].items():
            conditional = _safe_ratio(pool_count, pool_size)
            sampled_joint_probability[joint] += (
                sampled_probabilities[stratum] * conditional
            )
            weighted_joint_probability[joint] += (
                sampled_probabilities[stratum] * weights[stratum] * conditional
            )
    for stratum in BACKGROUND_STRATA:
        sampled_joint_probability[(0, 0)] += sampled_probabilities[stratum]
        weighted_joint_probability[(0, 0)] += weighted_stratum_probabilities[stratum]

    full_joint_probability = _full_joint_probability(population)
    joint_keys = set(weighted_joint_probability) | set(full_joint_probability)
    joint_max_error = max(
        (
            abs(weighted_joint_probability[joint] - full_joint_probability[joint])
            for joint in joint_keys
        ),
        default=0.0,
    )
    denominator = sum(
        counts[stratum] * weights[stratum] * weights[stratum]
        for stratum in ALL_STRATA
    )
    weight_sum = sum(
        counts[stratum] * weights[stratum] for stratum in ALL_STRATA
    )
    ess = _safe_ratio(weight_sum * weight_sum, denominator)
    ess_ratio = _safe_ratio(ess, query_count)

    heads = {}
    for head in HEADS:
        sampled_histogram = _head_histogram_from_joint(
            sampled_joint_probability, head
        )
        weighted_histogram = _head_histogram_from_joint(
            weighted_joint_probability, head
        )
        full_histogram = _head_histogram_from_joint(full_joint_probability, head)
        sampled_positive = sum(
            mass for label, mass in sampled_histogram.items() if label > 0
        )
        weighted_positive = sum(
            mass for label, mass in weighted_histogram.items() if label > 0
        )
        full_positive = sum(
            mass for label, mass in full_histogram.items() if label > 0
        )
        heads[head] = {
            "unweighted_positive_probability": sampled_positive,
            "importance_weighted_positive_probability": weighted_positive,
            "full_stream_positive_probability": full_positive,
            "unweighted_over_full": _safe_ratio(sampled_positive, full_positive),
            "constant_categorical_optimum": {
                "unweighted": _categorical_optimum(sampled_histogram),
                "importance_weighted": _categorical_optimum(weighted_histogram),
                "full_stream": _categorical_optimum(full_histogram),
            },
            "diagnostic_mass_for_fixed_query_count": {
                "importance_weighted_positive": weighted_positive * query_count,
                "importance_weighted_negative": (1.0 - weighted_positive)
                * query_count,
            },
        }

    distance_one_exposure = counts["distance_1"] * HISTORICAL_EPOCHS
    background_draws = sum(counts[stratum] for stratum in BACKGROUND_STRATA)
    far_fraction = _safe_ratio(counts["distance_64_plus"], background_draws)
    mean_error = abs(mean_weight - 1.0)
    guards = {
        "distance_1_draws_over_20_fresh_epochs": distance_one_exposure,
        "distance_1_exposure_at_least_20": (
            distance_one_exposure
            >= MINIMUM_DISTANCE_ONE_DRAWS_OVER_20_EPOCHS
        ),
        "effective_sample_size_ratio_at_least_0_25": (
            ess_ratio >= MINIMUM_ESS_RATIO
        ),
        "distance_64_plus_fraction_of_background_draws": far_fraction,
        "distance_64_plus_fraction_at_least_0_8": (
            far_fraction >= MINIMUM_FAR_BACKGROUND_FRACTION
        ),
        "analytical_weighted_stratum_prior_within_1e_12": (
            stratum_max_error <= MAX_ANALYTICAL_ERROR
        ),
        "analytical_weighted_joint_prior_within_1e_12": (
            joint_max_error <= MAX_ANALYTICAL_ERROR
        ),
        "analytical_sampled_mean_weight_within_1e_12": (
            mean_error <= MAX_ANALYTICAL_ERROR
        ),
    }
    guards["passes_all_train_selection_gates"] = all(
        value
        for key, value in guards.items()
        if key
        not in {
            "distance_1_draws_over_20_fresh_epochs",
            "distance_64_plus_fraction_of_background_draws",
        }
    )

    return {
        "queries": query_count,
        "source_counts": dict(counts),
        "live_stratum_sizes": live_sizes,
        "heads": heads,
        "importance_correction": {
            "live_stratum_probabilities": live_probabilities,
            "sampled_stratum_probabilities": sampled_probabilities,
            "weights_p_live_over_p_sample": weights,
            "analytical_weighted_stratum_probabilities": weighted_stratum_probabilities,
            "analytical_weighted_stratum_max_absolute_error": stratum_max_error,
            "analytical_sampled_mean_weight": mean_weight,
            "analytical_sampled_mean_weight_absolute_error_from_one": mean_error,
            "analytical_weighted_joint_probabilities": _joint_json(
                weighted_joint_probability
            ),
            "full_stream_joint_probabilities": _joint_json(full_joint_probability),
            "analytical_weighted_joint_max_absolute_error": joint_max_error,
            "minimum_weight": min(weights.values()),
            "maximum_weight": max(weights.values()),
            "maximum_over_minimum": _safe_ratio(
                max(weights.values()), min(weights.values())
            ),
            "effective_sample_size": ess,
            "effective_sample_size_ratio": ess_ratio,
            "approved_for_training_loss": False,
        },
        "guards": guards,
    }


def choose_candidate(train_reports: Mapping[int, Mapping[str, object]]) -> Optional[int]:
    """Choose the first preregistered train candidate passing every gate."""

    for h in GRID_H:
        report = train_reports.get(h)
        if report is None:
            raise PointQueryAuditError(f"missing preregistered candidate h={h}")
        if report["guards"]["passes_all_train_selection_gates"]:
            return h
    return None


class _BackgroundSampler:
    def __init__(self, population: SplitPopulation) -> None:
        self.population = population
        running = 0
        cumulative = []
        for item in population.tracks:
            running += item.frame_count
            cumulative.append(running)
        self.cumulative_frames = tuple(cumulative)
        self.ordered_boundaries = tuple(
            tuple(sorted(item.boundary_positions)) for item in population.tracks
        )
        self.band_sizes = background_band_sizes(population)

    def sample(self, stratum: str, rng: random.Random) -> Tuple[PointTarget, int]:
        if stratum not in BACKGROUND_STRATA:
            raise PointQueryAuditError("unknown background distance stratum")
        if self.band_sizes[stratum] <= 0:
            raise PointQueryAuditError(f"cannot sample empty stratum {stratum}")
        attempts = 0
        while True:
            attempts += 1
            global_position = rng.randrange(self.population.total_frames)
            track_index = bisect_right(self.cumulative_frames, global_position)
            previous_end = (
                self.cumulative_frames[track_index - 1] if track_index else 0
            )
            position = global_position - previous_end
            item = self.population.tracks[track_index]
            if position in item.boundary_positions:
                continue
            actual = _position_background_stratum(
                self.population,
                track_index,
                position,
                self.ordered_boundaries,
            )
            if actual == stratum:
                return PointTarget(track_index, position, 0, 0, stratum), attempts


def sample_background_stratum(
    population: SplitPopulation,
    stratum: str,
    rng: random.Random,
    cumulative_frames: Optional[Sequence[int]] = None,
) -> Tuple[PointTarget, int]:
    """Public deterministic test helper for uniform band rejection sampling."""

    sampler = _BackgroundSampler(population)
    if cumulative_frames is not None and tuple(cumulative_frames) != sampler.cumulative_frames:
        raise PointQueryAuditError("provided cumulative frames do not match population")
    return sampler.sample(stratum, rng)


def _draw_fixed_candidate(
    population: SplitPopulation,
    *,
    h: int,
    split: str,
    base_seed: int,
) -> Tuple[Tuple[PointTarget, ...], Mapping[str, int]]:
    counts = candidate_source_counts(h, split)
    selected: List[PointTarget] = []
    attempts: Dict[str, int] = {stratum: 0 for stratum in BACKGROUND_STRATA}
    background_sampler = _BackgroundSampler(population)
    for stratum in ALL_STRATA:
        rng = random.Random(base_seed * 1000 + _RNG_OFFSETS[stratum])
        if stratum in POSITIVE_STRATA:
            pool = population.positive_pools[stratum]
            selected.extend(rng.choice(pool) for _ in range(counts[stratum]))
        else:
            for _ in range(counts[stratum]):
                target, used = background_sampler.sample(stratum, rng)
                selected.append(target)
                attempts[stratum] += used
    if len(selected) != sum(counts.values()):
        raise PointQueryAuditError("fixed candidate draw count changed")
    return tuple(selected), attempts


def _audit_fixed_selection(
    population: SplitPopulation,
    selected: Sequence[PointTarget],
    analytical: Mapping[str, object],
    attempts: Mapping[str, int],
) -> Mapping[str, object]:
    expected_counts = analytical["source_counts"]
    actual_counts = Counter(target.stratum for target in selected)
    if any(actual_counts[stratum] != expected_counts[stratum] for stratum in ALL_STRATA):
        raise PointQueryAuditError("fixed selection source counts changed")

    positive_lookup = {
        stratum: {
            (target.track_index, target.position): target.joint
            for target in population.positive_pools[stratum]
        }
        for stratum in POSITIVE_STRATA
    }
    invalid = []
    background_distances: Dict[str, List[int]] = {
        stratum: [] for stratum in BACKGROUND_STRATA
    }
    ordered_boundaries = tuple(
        tuple(sorted(item.boundary_positions)) for item in population.tracks
    )
    for target in selected:
        key = (target.track_index, target.position)
        if target.stratum in POSITIVE_STRATA:
            if positive_lookup[target.stratum].get(key) != target.joint:
                invalid.append((target.stratum, key, "positive_pool_mismatch"))
            continue
        item = population.tracks[target.track_index]
        if target.joint != (0, 0) or target.position in item.boundary_positions:
            invalid.append((target.stratum, key, "background_contains_boundary"))
            continue
        actual_stratum = _position_background_stratum(
            population,
            target.track_index,
            target.position,
            ordered_boundaries,
        )
        if actual_stratum != target.stratum:
            invalid.append((target.stratum, key, actual_stratum))
            continue
        boundaries = ordered_boundaries[target.track_index]
        distance = (
            nearest_boundary_distance(target.position, boundaries)
            if boundaries
            else population.tracks[target.track_index].frame_count + 64
        )
        background_distances[target.stratum].append(distance)
    if invalid:
        raise PointQueryAuditError(f"invalid selected positions: {invalid[:3]}")

    joint = Counter(target.joint for target in selected)
    selected_keys = {
        stratum: Counter(
            (target.track_index, target.position)
            for target in selected
            if target.stratum == stratum
        )
        for stratum in ALL_STRATA
    }
    by_player = Counter(
        population.tracks[target.track_index].track.player_id for target in selected
    )
    by_arrangement = Counter(
        _arrangement(population.tracks[target.track_index].member)
        for target in selected
    )
    by_track = Counter(
        population.tracks[target.track_index].member for target in selected
    )
    all_track_members = tuple(sorted(item.member for item in population.tracks))
    missing_tracks = [member for member in all_track_members if not by_track[member]]
    all_track_counts = sorted(by_track[member] for member in all_track_members)
    represented_counts = sorted(
        by_track[member] for member in all_track_members if by_track[member]
    )
    left_initialized = [
        target for target in selected if target.position < LEFT_HISTORY_SAMPLES
    ]
    weights = analytical["importance_correction"]["weights_p_live_over_p_sample"]
    weighted_joint: Counter = Counter()
    weight_sum = 0.0
    for target in selected:
        weight = weights[target.stratum]
        weight_sum += weight
        weighted_joint[target.joint] += weight

    head_reports = {}
    for head in HEADS:
        histogram = _head_histogram_from_joint(joint, head)
        weighted_histogram = _head_histogram_from_joint(weighted_joint, head)
        head_reports[head] = {
            "unweighted_exact_count_histogram": _histogram_json(histogram),
            "importance_weighted_exact_count_mass": _histogram_json(
                weighted_histogram
            ),
            "unweighted_positive_queries": sum(
                count for label, count in histogram.items() if label > 0
            ),
        }

    return {
        "source_counts": dict(actual_counts),
        "all_selected_positions_match_declared_strata": True,
        "joint_exact_count_histogram": _joint_json(joint),
        "heads": head_reports,
        "selection": {
            "by_stratum": {
                stratum: {
                    "draws": actual_counts[stratum],
                    "unique_positions": len(selected_keys[stratum]),
                    "repeated_draws": actual_counts[stratum]
                    - len(selected_keys[stratum]),
                }
                for stratum in ALL_STRATA
            },
            "by_player": dict(sorted(by_player.items())),
            "by_arrangement": dict(sorted(by_arrangement.items())),
            "by_track": dict(sorted(by_track.items())),
            "track_exposure": {
                "represented_tracks": len(represented_counts),
                "total_tracks": len(all_track_members),
                "missing_tracks": missing_tracks,
                "missing_track_count": len(missing_tracks),
                "all_tracks_query_count_summary": {
                    "minimum": min(all_track_counts),
                    "median_lower": all_track_counts[(len(all_track_counts) - 1) // 2],
                    "maximum": max(all_track_counts),
                },
                "represented_tracks_query_count_summary": {
                    "minimum": min(represented_counts),
                    "median_lower": represented_counts[
                        (len(represented_counts) - 1) // 2
                    ],
                    "maximum": max(represented_counts),
                },
                "selected_queries_reconciled": sum(by_track.values()),
                "total_query_accounting_exact": sum(by_track.values())
                == len(selected),
            },
            "contexts_requiring_left_zero_initialization": len(left_initialized),
            "left_zero_padding_samples_total": sum(
                LEFT_HISTORY_SAMPLES - target.position
                for target in left_initialized
            ),
            "right_padding_samples": 0,
            "background_rejection_attempts": dict(attempts),
            "background_distance_summary": {
                stratum: {
                    "minimum": min(values) if values else None,
                    "maximum": max(values) if values else None,
                }
                for stratum, values in background_distances.items()
            },
        },
        "empirical_importance_weighted_joint_mass": _joint_json(weighted_joint),
        "empirical_weight_sum": weight_sum,
    }


def _validate_locked_inputs(
    protocol_path: Path,
    amendment_path: Path,
    report_amendment_path: Path,
    review_amendment_path: Path,
    metadata_path: Path,
    prior_audit_path: Path,
    experiment_12_path: Path,
) -> Tuple[Mapping[str, object], Mapping[str, object], Mapping[str, object], Mapping[str, object]]:
    locked = {
        protocol_path: EXPECTED_PROTOCOL_SHA256,
        amendment_path: EXPECTED_AMENDMENT_SHA256,
        report_amendment_path: EXPECTED_REPORT_AMENDMENT_SHA256,
        review_amendment_path: EXPECTED_REVIEW_AMENDMENT_SHA256,
        metadata_path: EXPECTED_METADATA_SHA256,
        prior_audit_path: EXPECTED_PRIOR_AUDIT_SHA256,
        experiment_12_path: EXPECTED_EXPERIMENT_12_RESULT_SHA256,
        REPOSITORY_ROOT / "scripts" / "audit_exact_point_query_sampler.py": (
            EXPECTED_EXPERIMENT_12_SOURCE_SHA256
        ),
        REPOSITORY_ROOT / "scripts" / "audit_anonymous_boundary_targets.py": (
            EXPECTED_ANONYMOUS_HELPER_SHA256
        ),
        REPOSITORY_ROOT / "scripts" / "train_boundaries.py": (
            EXPECTED_TRAINING_SOURCE_SHA256
        ),
        REPOSITORY_ROOT / "src" / "causal_note" / "guitarset.py": (
            EXPECTED_GUITARSET_SOURCE_SHA256
        ),
    }
    for path, expected in locked.items():
        if sha256_file(path) != expected:
            raise PointQueryAuditError(f"locked input SHA-256 changed: {path}")
    protocol = _load_json(protocol_path)
    amendment = _load_json(amendment_path)
    report_amendment = _load_json(report_amendment_path)
    review_amendment = _load_json(review_amendment_path)
    metadata = _load_json(metadata_path)
    prior_audit = _load_json(prior_audit_path)
    experiment_12 = _load_json(experiment_12_path)
    if protocol.get("status") != "preregistered_before_implementation":
        raise PointQueryAuditError("Exp13 protocol status changed")
    if amendment.get("status") != "preregistered_clarification_before_implementation":
        raise PointQueryAuditError("Exp13 amendment status changed")
    if report_amendment.get("status") != "report_correction_registered_before_final_audit":
        raise PointQueryAuditError("Exp13 report amendment status changed")
    if review_amendment.get("status") != "independent_review_corrections_registered_before_final_audit":
        raise PointQueryAuditError("Exp13 review amendment status changed")
    if experiment_12.get("status") != "completed_audit_only":
        raise PointQueryAuditError("Exp12 audit status changed")
    return protocol, metadata, prior_audit, experiment_12


def _rare_exposure_summary(
    population: SplitPopulation,
    counts: Mapping[str, int],
) -> Mapping[str, object]:
    exposure = _rare_class_exposure(
        population,
        counts,
        epochs=HISTORICAL_EPOCHS,
    )
    values = []
    for head in HEADS:
        values.extend(
            item["expected_draws_over_20_epochs"]
            for item in exposure[head]["classes"].values()
        )
    return {
        "by_head": exposure,
        "minimum_expected_draws_over_20_epochs": min(values) if values else 0.0,
        "required_minimum": 20.0,
        "passes": bool(values) and min(values) >= 20.0,
    }


def _comparison_to_experiment_12(
    selected_h: int,
    train_analytical: Mapping[str, object],
    validation_analytical: Mapping[str, object],
    experiment_12: Mapping[str, object],
) -> Mapping[str, object]:
    result = {}
    for split, candidate in (
        ("train", train_analytical),
        ("validation", validation_analytical),
    ):
        baseline = experiment_12["splits"][split]["fixed_point_query_sampler"]
        candidate_ess = candidate["importance_correction"][
            "effective_sample_size_ratio"
        ]
        baseline_ess = baseline["importance_correction"][
            "effective_sample_size_ratio"
        ]
        candidate_d1 = candidate["source_counts"]["distance_1"]
        baseline_d1_expected = baseline["background_near_boundary"]["1"][
            "expected_fixed_draws"
        ]
        positive_source_counts = {
            stratum: {
                "experiment_12": baseline["source_counts"][stratum],
                "experiment_13": candidate["source_counts"][stratum],
                "exact_match": (
                    baseline["source_counts"][stratum]
                    == candidate["source_counts"][stratum]
                ),
            }
            for stratum in POSITIVE_STRATA
        }
        result[split] = {
            "selected_h": selected_h,
            "distance_1_draws_per_fixed_epoch": {
                "experiment_12_expected_uniform": baseline_d1_expected,
                "experiment_13_exact": candidate_d1,
                "experiment_13_over_experiment_12": _safe_ratio(
                    candidate_d1, baseline_d1_expected
                ),
            },
            "distance_1_draws_over_20_epochs": {
                "experiment_12_expected_uniform": baseline[
                    "background_near_boundary"
                ]["1"]["expected_draws_over_20_repeated_fixed_epochs"],
                "experiment_13_exact": candidate_d1 * HISTORICAL_EPOCHS,
            },
            "effective_sample_size_ratio": {
                "experiment_12": baseline_ess,
                "experiment_13": candidate_ess,
                "absolute_change": candidate_ess - baseline_ess,
            },
            "positive_source_draw_counts": positive_source_counts,
            "unweighted_head_positive_probability": {
                head: {
                    "experiment_12_empirical_fixed_draw": baseline["heads"][head][
                        "unweighted_positive_probability"
                    ],
                    "experiment_13_analytical_expectation": candidate["heads"][
                        head
                    ]["unweighted_positive_probability"],
                    "signed_analytical_minus_empirical": candidate["heads"][head][
                        "unweighted_positive_probability"
                    ]
                    - baseline["heads"][head]["unweighted_positive_probability"],
                    "interpretation": (
                        "distinct quantities; simultaneous onset+offset positions make "
                        "the offset empirical draw vary while source quotas stay fixed"
                    ),
                }
                for head in HEADS
            },
        }
    return result


def _train_grid_comparison_to_experiment_12(
    train_grid: Mapping[int, Mapping[str, object]],
    experiment_12: Mapping[str, object],
) -> Mapping[str, object]:
    """Compare every preregistered train candidate with the locked Exp12 draw."""

    baseline = experiment_12["splits"]["train"]["fixed_point_query_sampler"]
    baseline_ess = baseline["importance_correction"][
        "effective_sample_size_ratio"
    ]
    baseline_d1 = baseline["background_near_boundary"]["1"][
        "expected_fixed_draws"
    ]
    result = {}
    for h in GRID_H:
        candidate = train_grid[h]
        candidate_ess = candidate["importance_correction"][
            "effective_sample_size_ratio"
        ]
        result[str(h)] = {
            "source_counts": candidate["source_counts"],
            "positive_source_draw_counts_exactly_match": all(
                candidate["source_counts"][stratum]
                == baseline["source_counts"][stratum]
                for stratum in POSITIVE_STRATA
            ),
            "distance_1_draws_per_epoch": candidate["source_counts"][
                "distance_1"
            ],
            "experiment_12_expected_distance_1_draws_per_epoch": baseline_d1,
            "distance_1_over_experiment_12": _safe_ratio(
                candidate["source_counts"]["distance_1"], baseline_d1
            ),
            "distance_1_draws_over_20_epochs": candidate["guards"][
                "distance_1_draws_over_20_fresh_epochs"
            ],
            "distance_64_plus_fraction_of_background_draws": candidate[
                "guards"
            ]["distance_64_plus_fraction_of_background_draws"],
            "effective_sample_size_ratio": candidate_ess,
            "experiment_12_effective_sample_size_ratio": baseline_ess,
            "effective_sample_size_ratio_absolute_change": candidate_ess
            - baseline_ess,
            "passes_all_train_selection_gates": candidate["guards"][
                "passes_all_train_selection_gates"
            ],
        }
    return result


def _rare_exposure_comparison(
    rare_exposure: Mapping[str, object],
    experiment_12: Mapping[str, object],
) -> Mapping[str, object]:
    baseline = experiment_12["splits"]["train"]["fixed_point_query_sampler"][
        "rare_count_class_exposure"
    ]
    checks = {}
    details = {}
    fields = (
        "pool_positions",
        "positions_by_stratum",
        "expected_draws_per_fixed_epoch",
        "expected_draws_over_20_epochs",
        "probability_seen_at_least_once_over_20_epochs",
    )
    for head in HEADS:
        current_classes = rare_exposure["by_head"][head]["classes"]
        baseline_classes = baseline[head]["classes"]
        class_checks = {}
        for count_class, current in current_classes.items():
            expected = baseline_classes[count_class]
            field_checks = {field: current[field] == expected[field] for field in fields}
            class_checks[count_class] = all(field_checks.values())
            details[f"{head}_{count_class}"] = field_checks
        checks[head] = all(class_checks.values()) and set(current_classes) == set(
            baseline_classes
        )
    return {
        "all_expected_exposures_exactly_unchanged": all(checks.values()),
        "checks_by_head": checks,
        "details": details,
    }


def _distance_one_epoch_exposure(
    population: SplitPopulation,
    analytical: Mapping[str, object],
    fixed: Mapping[str, object],
) -> Mapping[str, object]:
    draws_per_epoch = analytical["source_counts"]["distance_1"]
    pool_positions = analytical["live_stratum_sizes"]["distance_1"]
    total_draws = draws_per_epoch * HISTORICAL_EPOCHS
    expected_unique_fresh = pool_positions * (
        1.0 - (1.0 - 1.0 / pool_positions) ** total_draws
    )
    fixed_unique = fixed["selection"]["by_stratum"]["distance_1"][
        "unique_positions"
    ]
    return {
        "pool_positions": pool_positions,
        "fresh_independent_epochs": {
            "draws_per_epoch": draws_per_epoch,
            "draws_over_20_epochs": total_draws,
            "expected_unique_positions_over_20_epochs": expected_unique_fresh,
            "assumption": "fresh independent draws with replacement every epoch",
        },
        "same_cached_fixed_batch_repeated": {
            "draws_over_20_repetitions": total_draws,
            "unique_positions_in_fixed_batch": fixed_unique,
            "unique_positions_after_20_repetitions": fixed_unique,
        },
        "interpretation": (
            "the draw-count guard passes; it does not assert 20 unique positions "
            "when one cached batch is repeated"
        ),
    }


def run_audit(
    dataset_dir: Path,
    *,
    output_path: Path,
    protocol_path: Path = DEFAULT_PROTOCOL,
    amendment_path: Path = DEFAULT_AMENDMENT,
    report_amendment_path: Path = DEFAULT_REPORT_AMENDMENT,
    review_amendment_path: Path = DEFAULT_REVIEW_AMENDMENT,
    metadata_path: Path = DEFAULT_METADATA,
    prior_audit_path: Path = DEFAULT_PRIOR_AUDIT,
    experiment_12_path: Path = DEFAULT_EXPERIMENT_12,
) -> Mapping[str, object]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing audit: {output_path}")
    protocol, metadata, prior_audit, experiment_12 = _validate_locked_inputs(
        protocol_path,
        amendment_path,
        report_amendment_path,
        review_amendment_path,
        metadata_path,
        prior_audit_path,
        experiment_12_path,
    )

    tracks = index_guitarset(dataset_dir)
    if not tracks or any(track.player_id not in ALLOWED_PLAYERS for track in tracks):
        raise PointQueryAuditError("index escaped locked players 00 through 04")
    train_tracks, validation_tracks = split_tracks_by_group(
        tracks,
        validation_fraction=0.2,
        seed=SEED,
    )
    metadata_split = metadata.get("split")
    if not isinstance(metadata_split, dict) or not isinstance(
        metadata_split.get("validation_members"), list
    ):
        raise PointQueryAuditError("V7 metadata has no validation member list")
    expected_validation = set(metadata_split["validation_members"])
    actual_validation = {
        PurePosixPath(track.annotation_member).name for track in validation_tracks
    }
    if actual_validation != expected_validation:
        raise PointQueryAuditError("generated validation split differs from V7")

    # Selection is deliberately completed from train before validation
    # annotations or WAV headers are opened.
    train_population = build_population(_prepare_tracks(train_tracks))
    train_full_report = full_stream_report(train_population)
    if train_full_report != experiment_12["splits"]["train"]["full_stream"]:
        raise PointQueryAuditError("train full-stream population differs from Exp12")
    train_grid = {
        h: analytical_candidate_report(
            train_population, candidate_source_counts(h, "train")
        )
        for h in GRID_H
    }
    selected_h = choose_candidate(train_grid)
    if selected_h is None:
        raise PointQueryAuditError("no preregistered train candidate passed")
    train_analytical = train_grid[selected_h]

    validation_population = build_population(_prepare_tracks(validation_tracks))
    populations = {
        "train": train_population,
        "validation": validation_population,
    }
    full_reports = {
        "train": train_full_report,
        "validation": full_stream_report(validation_population),
    }
    full_matches_exp12 = {
        split: full_reports[split]
        == experiment_12["splits"][split]["full_stream"]
        for split in ("train", "validation")
    }
    if not all(full_matches_exp12.values()):
        raise PointQueryAuditError("full-stream population differs from Exp12")

    band_sizes = {
        split: background_band_sizes(population)
        for split, population in populations.items()
    }
    band_reconciliation = {}
    for split in ("train", "validation"):
        baseline_near = experiment_12["splits"][split][
            "fixed_point_query_sampler"
        ]["background_near_boundary"]
        sizes = band_sizes[split]
        checks = {
            "distance_1_matches_exp12": sizes["distance_1"]
            == baseline_near["1"]["full_stream_background_positions"],
            "through_15_matches_exp12": (
                sizes["distance_1"] + sizes["distance_2_to_15"]
                == baseline_near["15"]["full_stream_background_positions"]
            ),
            "through_63_matches_exp12": (
                sizes["distance_1"]
                + sizes["distance_2_to_15"]
                + sizes["distance_16_to_63"]
                == baseline_near["63"]["full_stream_background_positions"]
            ),
            "all_background_reconciles": sum(sizes.values())
            == populations[split].stratum_sizes["background"],
        }
        band_reconciliation[split] = {
            "checks": checks,
            "all_exact": all(checks.values()),
        }

    validation_analytical = analytical_candidate_report(
        populations["validation"],
        candidate_source_counts(selected_h, "validation"),
    )
    train_selected, train_attempts = _draw_fixed_candidate(
        populations["train"], h=selected_h, split="train", base_seed=SEED
    )
    validation_selected, validation_attempts = _draw_fixed_candidate(
        populations["validation"],
        h=selected_h,
        split="validation",
        base_seed=SEED + 1,
    )
    fixed_reports = {
        "train": _audit_fixed_selection(
            populations["train"],
            train_selected,
            train_analytical,
            train_attempts,
        ),
        "validation": _audit_fixed_selection(
            populations["validation"],
            validation_selected,
            validation_analytical,
            validation_attempts,
        ),
    }
    rare_exposure = _rare_exposure_summary(
        populations["train"], train_analytical["source_counts"]
    )
    rare_comparison = _rare_exposure_comparison(rare_exposure, experiment_12)
    oracle = _oracle_reconciliation(populations, full_reports, prior_audit)
    actual_players = sorted({track.player_id for track in tracks})

    structural_checks = {
        "full_stream_exactly_matches_experiment_12": all(
            full_matches_exp12.values()
        ),
        "background_bands_reconcile_exactly": all(
            item["all_exact"] for item in band_reconciliation.values()
        ),
        "selected_train_positions_valid": fixed_reports["train"][
            "all_selected_positions_match_declared_strata"
        ],
        "selected_validation_positions_valid": fixed_reports["validation"][
            "all_selected_positions_match_declared_strata"
        ],
        "selected_train_source_counts_exact": fixed_reports["train"][
            "source_counts"
        ]
        == train_analytical["source_counts"],
        "selected_validation_source_counts_exact": fixed_reports["validation"][
            "source_counts"
        ]
        == validation_analytical["source_counts"],
        "selected_train_track_exposure_reconciles": fixed_reports["train"][
            "selection"
        ]["track_exposure"]["total_query_accounting_exact"],
        "selected_validation_track_exposure_reconciles": fixed_reports[
            "validation"
        ]["selection"]["track_exposure"]["total_query_accounting_exact"],
        "positive_source_draw_counts_exactly_match_experiment_12": all(
            candidate_source_counts(selected_h, split)[stratum]
            == experiment_12["splits"][split]["fixed_point_query_sampler"][
                "source_counts"
            ][stratum]
            for split in ("train", "validation")
            for stratum in POSITIVE_STRATA
        ),
        "rare_count_expected_exposure_exactly_unchanged": rare_comparison[
            "all_expected_exposures_exactly_unchanged"
        ],
        "locked_oracle_reconciliation_exact": oracle["all_exact"],
        "validation_ess_at_least_0_25": validation_analytical[
            "importance_correction"
        ]["effective_sample_size_ratio"]
        >= MINIMUM_ESS_RATIO,
        "validation_weighted_stratum_prior_within_1e_12": validation_analytical[
            "importance_correction"
        ]["analytical_weighted_stratum_max_absolute_error"]
        <= MAX_ANALYTICAL_ERROR,
        "validation_weighted_joint_prior_within_1e_12": validation_analytical[
            "importance_correction"
        ]["analytical_weighted_joint_max_absolute_error"]
        <= MAX_ANALYTICAL_ERROR,
        "actual_players_are_exactly_00_through_04": actual_players
        == sorted(ALLOWED_PLAYERS),
        "player_05_absent": "05" not in actual_players,
    }
    structural_pass = all(structural_checks.values())
    background_constraint_repaired = train_analytical["guards"][
        "distance_1_exposure_at_least_20"
    ]
    category = (
        "background_constraint_repaired_but_sampler_not_training_ready"
        if structural_pass and background_constraint_repaired
        else "hard_negative_sampler_structurally_rejected"
    )

    report: Mapping[str, object] = {
        "schema_version": 1,
        "status": "completed_audit_only",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "single_change": "uniform background subdivided by exact nearest-boundary distance",
            "targets": ["onset_count[t]", "offset_count[t]"],
            "positive_pools_changed": False,
            "positive_draw_counts_changed": False,
            "total_query_counts_changed": False,
            "loss_selected_or_changed": False,
            "model_selected_or_changed": False,
            "audio_crop_implemented": False,
            "decoder_changed": False,
            "live_changed": False,
            "training_started": False,
            "player_05_content_read": False,
        },
        "locked_inputs": {
            "protocol": str(protocol_path.resolve()),
            "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
            "protocol_amendment": str(amendment_path.resolve()),
            "protocol_amendment_sha256": EXPECTED_AMENDMENT_SHA256,
            "protocol_report_amendment": str(report_amendment_path.resolve()),
            "protocol_report_amendment_sha256": EXPECTED_REPORT_AMENDMENT_SHA256,
            "protocol_review_amendment": str(review_amendment_path.resolve()),
            "protocol_review_amendment_sha256": EXPECTED_REVIEW_AMENDMENT_SHA256,
            "experiment_12": str(experiment_12_path.resolve()),
            "experiment_12_sha256": EXPECTED_EXPERIMENT_12_RESULT_SHA256,
            "experiment_12_source_sha256": EXPECTED_EXPERIMENT_12_SOURCE_SHA256,
            "anonymous_target_helper_sha256": EXPECTED_ANONYMOUS_HELPER_SHA256,
            "training_source_sha256": EXPECTED_TRAINING_SOURCE_SHA256,
            "guitarset_source_sha256": EXPECTED_GUITARSET_SOURCE_SHA256,
            "metadata_sha256": EXPECTED_METADATA_SHA256,
            "prior_audit_sha256": EXPECTED_PRIOR_AUDIT_SHA256,
            "audit_source_sha256": sha256_file(Path(__file__)),
            "protocol_status": protocol["status"],
        },
        "data_guard": {
            "actual_players": actual_players,
            "locked_test_player": "05",
            "locked_test_used": False,
            "train_tracks": len(train_tracks),
            "validation_tracks": len(validation_tracks),
            "validation_members_exactly_match_v7": True,
        },
        "background_partition": {
            split: {
                "positions": band_sizes[split],
                "reconciliation": band_reconciliation[split],
            }
            for split in ("train", "validation")
        },
        "train_grid": {str(h): train_grid[h] for h in GRID_H},
        "train_grid_comparison_to_experiment_12": (
            _train_grid_comparison_to_experiment_12(train_grid, experiment_12)
        ),
        "selected_candidate": {
            "h": selected_h,
            "selection_used_validation": False,
            "train": {
                "analytical": train_analytical,
                "fixed_draw": fixed_reports["train"],
            },
            "validation": {
                "analytical": validation_analytical,
                "fixed_draw": fixed_reports["validation"],
            },
            "rare_count_exposure": rare_exposure,
            "rare_count_comparison_to_experiment_12": rare_comparison,
            "distance_1_epoch_exposure": {
                "train": _distance_one_epoch_exposure(
                    populations["train"], train_analytical, fixed_reports["train"]
                ),
                "validation": _distance_one_epoch_exposure(
                    populations["validation"],
                    validation_analytical,
                    fixed_reports["validation"],
                ),
            },
        },
        "comparison_to_experiment_12": _comparison_to_experiment_12(
            selected_h,
            train_analytical,
            validation_analytical,
            experiment_12,
        ),
        "decision": {
            "category": category,
            "selected_h": selected_h,
            "structural_checks": structural_checks,
            "structural_pass": structural_pass,
            "background_distance_1_constraint_repaired": background_constraint_repaired,
            "background_distance_1_guard_passes_at_exact_minimum_without_margin": (
                train_analytical["guards"][
                    "distance_1_draws_over_20_fresh_epochs"
                ]
                == MINIMUM_DISTANCE_ONE_DRAWS_OVER_20_EPOCHS
            ),
            "rare_count_exposure_pass": rare_exposure["passes"],
            "importance_weights_approved_for_training_loss": False,
            "loss_selected": False,
            "model_selected": False,
            "audio_crop_implemented": False,
            "crop_live_equivalence_verified": False,
            "training_ready": False,
            "training_started": False,
            "live_changed": False,
            "stop_reasons": [
                "rare_exact_count_classes_remain_underexposed",
                "selected_h_meets_the_distance_1_draw_guard_without_margin",
                "unweighted_positive_prior_remains_far_above_live",
                "importance_weights_are_not_approved_as_training_loss",
                "point_query_loss_and_model_are_not_selected",
                "audio_crop_and_crop_live_equivalence_are_not_verified",
            ],
            "next_action_requires_user_approval": True,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    try:
        temporary.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return report


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset_dir",
        nargs="?",
        default=str(REPOSITORY_ROOT / "data" / "GuitarSet"),
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--protocol-amendment", default=str(DEFAULT_AMENDMENT))
    parser.add_argument(
        "--protocol-report-amendment", default=str(DEFAULT_REPORT_AMENDMENT)
    )
    parser.add_argument(
        "--protocol-review-amendment", default=str(DEFAULT_REVIEW_AMENDMENT)
    )
    parser.add_argument("--metadata", default=str(DEFAULT_METADATA))
    parser.add_argument("--prior-audit", default=str(DEFAULT_PRIOR_AUDIT))
    parser.add_argument("--experiment-12", default=str(DEFAULT_EXPERIMENT_12))
    return parser


def main(argv: Sequence[str] = ()) -> int:
    arguments = create_argument_parser().parse_args(argv or None)
    report = run_audit(
        Path(arguments.dataset_dir),
        output_path=Path(arguments.output),
        protocol_path=Path(arguments.protocol),
        amendment_path=Path(arguments.protocol_amendment),
        report_amendment_path=Path(arguments.protocol_report_amendment),
        review_amendment_path=Path(arguments.protocol_review_amendment),
        metadata_path=Path(arguments.metadata),
        prior_audit_path=Path(arguments.prior_audit),
        experiment_12_path=Path(arguments.experiment_12),
    )
    print(json.dumps(report["decision"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
