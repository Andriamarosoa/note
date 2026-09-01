"""Audit an exact causal point-query sampler without training.

Each query is one unique ``(track, sample t)`` position.  The two targets are
the exact anonymous counts ``onset_count[t]`` and ``offset_count[t]``.  This
script reads GuitarSet players 00 through 04, reproduces the locked V7 split,
and never decodes audio samples or imports TensorFlow.
"""

from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import random
import sys
from typing import Dict, List, Mapping, MutableMapping, Sequence, Tuple


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from causal_note.guitarset import (  # noqa: E402 - local bootstrap above
    ALLOWED_PLAYERS,
    SAMPLE_RATE,
    GuitarSetTrack,
    index_guitarset,
)
from scripts.audit_anonymous_boundary_targets import (  # noqa: E402
    AuditTrack,
    _head_positions,
    _prepare_tracks,
)
from scripts.train_boundaries import group_stem, split_tracks_by_group  # noqa: E402


HEADS = ("onset", "offset")
STRATA = ("onset_bearing", "offset_only", "background")
NEAR_BOUNDARY_RADII = (1, 15, 63)
CONTEXT_SAMPLES = 4093
LEFT_HISTORY_SAMPLES = CONTEXT_SAMPLES - 1
HISTORICAL_THRESHOLD = 0.55
HISTORICAL_EPOCHS = 20
SEED = 1337
FIXED_QUERIES = {"train": 1600, "validation": 400}

DEFAULT_PROTOCOL = (
    REPOSITORY_ROOT
    / "model"
    / "causal-boundaries-exact-point-query-sampler-protocol.json"
)
DEFAULT_AMENDMENT = (
    REPOSITORY_ROOT
    / "model"
    / "causal-boundaries-exact-point-query-sampler-protocol-amendment-01.json"
)
DEFAULT_REVIEW_AMENDMENT = (
    REPOSITORY_ROOT
    / "model"
    / "causal-boundaries-exact-point-query-sampler-protocol-amendment-02.json"
)
DEFAULT_METADATA = (
    REPOSITORY_ROOT
    / "model"
    / "causal-boundaries-weight28-window512-v7-epoch08.recovery.metadata.json"
)
DEFAULT_PRIOR_AUDIT = (
    REPOSITORY_ROOT / "model" / "causal-boundaries-type-position-target-audit.json"
)
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT
    / "model"
    / "causal-boundaries-exact-point-query-sampler-audit.json"
)

EXPECTED_PROTOCOL_SHA256 = (
    "A57062D99AC907E35E081BF8A73CE12CBE9A7E9477D8CB915D11D88D1D25804F"
)
EXPECTED_AMENDMENT_SHA256 = (
    "DB3D5937B0A4F5DF7C004E7D7742ACC1FD5A5268E4AC876C1B4E4CCA0D701930"
)
EXPECTED_REVIEW_AMENDMENT_SHA256 = (
    "A54B4CE82DA79A67AAAB1D46F572F0B2CB237B3EA1C485DD4A891A1D25FCDDA6"
)
EXPECTED_HELPER_SHA256 = (
    "7C414A863E941B3595F03F7A420A5FB80810A5F4D2D9C5350C8089491135B356"
)


class PointQueryAuditError(ValueError):
    """Raised when a locked input or exact accounting rule is violated."""


@dataclass(frozen=True)
class TrackTargets:
    audit_track: AuditTrack
    onset_counts: Mapping[int, int]
    offset_counts: Mapping[int, int]
    boundary_positions: frozenset
    exclusive_end_offsets: int

    @property
    def track(self) -> GuitarSetTrack:
        return self.audit_track.track

    @property
    def frame_count(self) -> int:
        return self.audit_track.frame_count

    @property
    def member(self) -> str:
        return self.audit_track.member


@dataclass(frozen=True)
class PointTarget:
    track_index: int
    position: int
    onset_count: int
    offset_count: int
    stratum: str

    @property
    def joint(self) -> Tuple[int, int]:
        return (self.onset_count, self.offset_count)


@dataclass(frozen=True)
class SplitPopulation:
    tracks: Tuple[TrackTargets, ...]
    positive_pools: Mapping[str, Tuple[PointTarget, ...]]
    joint_by_stratum: Mapping[str, Mapping[Tuple[int, int], int]]
    total_frames: int
    stratum_sizes: Mapping[str, int]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _load_json(path: Path) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PointQueryAuditError(f"cannot read JSON input: {path}") from exc
    if not isinstance(value, dict):
        raise PointQueryAuditError(f"JSON root must be an object: {path}")
    return value


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _joint_key(joint: Tuple[int, int]) -> str:
    return f"{joint[0]}|{joint[1]}"


def _joint_json(histogram: Mapping[Tuple[int, int], float]) -> Dict[str, object]:
    result: Dict[str, object] = {}
    for joint in sorted(histogram):
        value = histogram[joint]
        if value:
            result[_joint_key(joint)] = value
    return result


def _histogram_json(histogram: Mapping[int, float]) -> Dict[str, object]:
    return {
        str(key): histogram[key]
        for key in sorted(histogram)
        if histogram[key]
    }


def _arrangement(member: str) -> str:
    stem = PurePosixPath(member).stem
    if stem.endswith("_comp"):
        return "comp"
    if stem.endswith("_solo"):
        return "solo"
    return "unknown"


def _track_targets(item: AuditTrack) -> TrackTargets:
    onsets = Counter(
        _head_positions(
            item.slots,
            "onset",
            frame_count=item.frame_count,
            supervised_only=True,
        )
    )
    raw_offsets = _head_positions(
        item.slots,
        "offset",
        frame_count=item.frame_count,
        supervised_only=False,
    )
    offsets = Counter(position for position in raw_offsets if position < item.frame_count)
    if any(position >= item.frame_count for position in onsets):
        raise PointQueryAuditError("onset escaped the acoustic sample domain")
    exclusive_end = sum(
        multiplicity
        for position, multiplicity in Counter(raw_offsets).items()
        if position == item.frame_count
    )
    if len(raw_offsets) != sum(offsets.values()) + exclusive_end:
        raise PointQueryAuditError("offset accounting did not reconcile")
    return TrackTargets(
        audit_track=item,
        onset_counts=dict(onsets),
        offset_counts=dict(offsets),
        boundary_positions=frozenset(onsets) | frozenset(offsets),
        exclusive_end_offsets=exclusive_end,
    )


def build_population(tracks: Sequence[AuditTrack]) -> SplitPopulation:
    prepared = tuple(_track_targets(item) for item in tracks)
    if not prepared:
        raise PointQueryAuditError("split contains no tracks")

    pools: Dict[str, List[PointTarget]] = {
        "onset_bearing": [],
        "offset_only": [],
    }
    joint_by_stratum: Dict[str, Counter] = {stratum: Counter() for stratum in STRATA}
    total_frames = 0

    for track_index, item in enumerate(prepared):
        total_frames += item.frame_count
        for position in sorted(item.boundary_positions):
            onset_count = item.onset_counts.get(position, 0)
            offset_count = item.offset_counts.get(position, 0)
            if onset_count > 0:
                stratum = "onset_bearing"
            elif offset_count > 0:
                stratum = "offset_only"
            else:
                raise PointQueryAuditError("boundary union contains an empty target")
            target = PointTarget(
                track_index=track_index,
                position=position,
                onset_count=onset_count,
                offset_count=offset_count,
                stratum=stratum,
            )
            pools[stratum].append(target)
            joint_by_stratum[stratum][target.joint] += 1

    positive_positions = sum(len(pool) for pool in pools.values())
    background = total_frames - positive_positions
    if background <= 0:
        raise PointQueryAuditError("split has no background positions")
    joint_by_stratum["background"][(0, 0)] = background
    stratum_sizes = {
        "onset_bearing": len(pools["onset_bearing"]),
        "offset_only": len(pools["offset_only"]),
        "background": background,
    }
    if sum(stratum_sizes.values()) != total_frames:
        raise PointQueryAuditError("strata do not partition the full stream")
    for stratum, pool in pools.items():
        keys = {(target.track_index, target.position) for target in pool}
        if len(keys) != len(pool):
            raise PointQueryAuditError(f"{stratum} pool contains duplicate positions")

    return SplitPopulation(
        tracks=prepared,
        positive_pools={key: tuple(value) for key, value in pools.items()},
        joint_by_stratum={
            key: dict(value) for key, value in joint_by_stratum.items()
        },
        total_frames=total_frames,
        stratum_sizes=stratum_sizes,
    )


def _merged_interval_length(intervals: Sequence[Tuple[int, int]]) -> int:
    if not intervals:
        return 0
    ordered = sorted(intervals)
    start, end = ordered[0]
    covered = 0
    for next_start, next_end in ordered[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            covered += end - start
            start, end = next_start, next_end
    return covered + end - start


def background_near_boundary_count(
    boundary_positions: Sequence[int],
    *,
    frame_count: int,
    radius: int,
) -> int:
    """Count non-boundary samples no farther than ``radius`` from a boundary."""

    unique = sorted(set(boundary_positions))
    if radius < 0 or frame_count < 0:
        raise PointQueryAuditError("invalid near-boundary interval")
    if any(position < 0 or position >= frame_count for position in unique):
        raise PointQueryAuditError("boundary escaped the acoustic sample domain")
    intervals = [
        (max(0, position - radius), min(frame_count, position + radius + 1))
        for position in unique
    ]
    return _merged_interval_length(intervals) - len(unique)


def nearest_boundary_distance(position: int, boundary_positions: Sequence[int]) -> int:
    ordered = boundary_positions
    if not ordered:
        raise PointQueryAuditError("cannot measure distance without a boundary")
    index = bisect_left(ordered, position)
    distances = []
    if index:
        distances.append(position - ordered[index - 1])
    if index < len(ordered):
        distances.append(ordered[index] - position)
    return min(distances)


def _head_histogram_from_joint(
    joint_histogram: Mapping[Tuple[int, int], float],
    head: str,
) -> Counter:
    index = HEADS.index(head)
    result: Counter = Counter()
    for joint, mass in joint_histogram.items():
        result[joint[index]] += mass
    return result


def _categorical_optimum(histogram: Mapping[int, float]) -> Mapping[str, object]:
    total = float(sum(histogram.values()))
    probabilities = {
        str(label): _safe_ratio(mass, total)
        for label, mass in sorted(histogram.items())
        if mass
    }
    nonzero_probability = sum(
        probability
        for label, probability in probabilities.items()
        if int(label) > 0
    )
    return {
        "class_probabilities": probabilities,
        "nonzero_probability": nonzero_probability,
        "historical_threshold": HISTORICAL_THRESHOLD,
        "constant_threshold_decision": (
            "always_nonzero"
            if nonzero_probability >= HISTORICAL_THRESHOLD
            else "always_zero"
        ),
    }


def _quantile_summary(values: Sequence[int]) -> Mapping[str, object]:
    ordered = sorted(values)
    if not ordered:
        return {"minimum": 0, "median": 0, "p95": 0, "maximum": 0}
    median_index = (len(ordered) - 1) // 2
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "minimum": ordered[0],
        "median_lower": ordered[median_index],
        "p95_nearest_rank": ordered[p95_index],
        "maximum": ordered[-1],
    }


def full_stream_report(population: SplitPopulation) -> Mapping[str, object]:
    joint = Counter()
    for histogram in population.joint_by_stratum.values():
        joint.update(histogram)
    if sum(joint.values()) != population.total_frames:
        raise PointQueryAuditError("joint histogram does not reconcile")

    by_player_frames: Counter = Counter()
    by_arrangement_frames: Counter = Counter()
    by_track: Dict[str, object] = {}
    stratum_by_player = {stratum: Counter() for stratum in STRATA}
    stratum_by_arrangement = {stratum: Counter() for stratum in STRATA}
    per_track_strata = {stratum: [] for stratum in STRATA}
    early_onsets = 0
    early_offsets = 0
    late_internal_offsets = 0
    exclusive_end_offsets = 0

    for item in population.tracks:
        player = item.track.player_id
        arrangement = _arrangement(item.member)
        by_player_frames[player] += item.frame_count
        by_arrangement_frames[arrangement] += item.frame_count
        onset_positions = set(item.onset_counts)
        offset_positions = set(item.offset_counts)
        counts = {
            "onset_bearing": len(onset_positions),
            "offset_only": len(offset_positions - onset_positions),
            "background": item.frame_count - len(onset_positions | offset_positions),
        }
        for stratum, count in counts.items():
            stratum_by_player[stratum][player] += count
            stratum_by_arrangement[stratum][arrangement] += count
            per_track_strata[stratum].append(count)
        by_track[item.member] = {
            "player": player,
            "arrangement": arrangement,
            "composition_group": group_stem(item.track),
            "frames": item.frame_count,
            "strata": counts,
        }
        early_onsets += sum(position < LEFT_HISTORY_SAMPLES for position in onset_positions)
        early_offsets += sum(position < LEFT_HISTORY_SAMPLES for position in offset_positions)
        late_internal_offsets += sum(
            position >= max(0, item.frame_count - LEFT_HISTORY_SAMPLES)
            for position in offset_positions
        )
        exclusive_end_offsets += item.exclusive_end_offsets

    head_reports: Dict[str, object] = {}
    for head in HEADS:
        histogram = _head_histogram_from_joint(joint, head)
        positive_positions = sum(
            mass for count, mass in histogram.items() if count > 0
        )
        event_mass = sum(count * mass for count, mass in histogram.items())
        head_reports[head] = {
            "exact_count_histogram": _histogram_json(histogram),
            "positive_positions": positive_positions,
            "positive_position_density": _safe_ratio(
                positive_positions, population.total_frames
            ),
            "event_instances": event_mass,
            "maximum_exact_count": max(histogram),
            "constant_categorical_optimum": _categorical_optimum(histogram),
        }

    near_background: Dict[str, object] = {}
    for radius in NEAR_BOUNDARY_RADII:
        count = sum(
            background_near_boundary_count(
                tuple(item.boundary_positions),
                frame_count=item.frame_count,
                radius=radius,
            )
            for item in population.tracks
        )
        near_background[str(radius)] = {
            "positions": count,
            "fraction_of_background": _safe_ratio(
                count, population.stratum_sizes["background"]
            ),
        }

    return {
        "tracks": len(population.tracks),
        "frames": population.total_frames,
        "duration_seconds": population.total_frames / SAMPLE_RATE,
        "joint_exact_count_histogram": _joint_json(joint),
        "strata": {
            stratum: {
                "positions": population.stratum_sizes[stratum],
                "fraction_of_full_stream": _safe_ratio(
                    population.stratum_sizes[stratum], population.total_frames
                ),
                "by_player": dict(sorted(stratum_by_player[stratum].items())),
                "by_arrangement": dict(
                    sorted(stratum_by_arrangement[stratum].items())
                ),
                "per_track_summary": _quantile_summary(per_track_strata[stratum]),
            }
            for stratum in STRATA
        },
        "positions_with_both_targets_positive": sum(
            mass
            for (onset_count, offset_count), mass in joint.items()
            if onset_count > 0 and offset_count > 0
        ),
        "heads": head_reports,
        "background_near_boundary": near_background,
        "context_edges": {
            "onset_positions_requiring_left_zero_initialization": early_onsets,
            "offset_positions_requiring_left_zero_initialization": early_offsets,
            "internal_offset_positions_in_final_4092_samples": late_internal_offsets,
            "exclusive_end_offset_instances_excluded_from_acoustic_target": exclusive_end_offsets,
            "right_padding_samples": 0,
        },
        "frame_distribution": {
            "by_player": dict(sorted(by_player_frames.items())),
            "by_arrangement": dict(sorted(by_arrangement_frames.items())),
            "by_track": dict(sorted(by_track.items())),
        },
    }


def source_counts_for_queries(query_count: int) -> Mapping[str, int]:
    if query_count <= 0:
        raise PointQueryAuditError("query count must be positive")
    counts = Counter(STRATA[index % len(STRATA)] for index in range(query_count))
    return {stratum: counts[stratum] for stratum in STRATA}


def _sample_uniform_background(
    population: SplitPopulation,
    *,
    rng: random.Random,
    cumulative_frames: Sequence[int],
) -> Tuple[PointTarget, int]:
    attempts = 0
    while True:
        attempts += 1
        global_position = rng.randrange(population.total_frames)
        track_index = bisect_right(cumulative_frames, global_position)
        previous_end = cumulative_frames[track_index - 1] if track_index else 0
        position = global_position - previous_end
        item = population.tracks[track_index]
        if position not in item.boundary_positions:
            return (
                PointTarget(track_index, position, 0, 0, "background"),
                attempts,
            )


def _rare_class_exposure(
    population: SplitPopulation,
    source_counts: Mapping[str, int],
    *,
    epochs: int,
) -> Mapping[str, object]:
    result: Dict[str, object] = {}
    for head_index, head in enumerate(HEADS):
        class_by_stratum: Dict[int, Counter] = {}
        observed_classes = set()
        for stratum in ("onset_bearing", "offset_only"):
            counts = Counter(
                target.joint[head_index]
                for target in population.positive_pools[stratum]
                if target.joint[head_index] > 0
            )
            class_by_stratum[stratum] = counts
            observed_classes.update(counts)
        classes: Dict[str, object] = {}
        for count_class in sorted(observed_classes):
            expected_per_epoch = 0.0
            probability_absent = 1.0
            pool_positions = 0
            positions_by_stratum: Dict[str, int] = {}
            for stratum in ("onset_bearing", "offset_only"):
                count = class_by_stratum[stratum][count_class]
                pool_size = population.stratum_sizes[stratum]
                draws = source_counts[stratum]
                fraction = _safe_ratio(count, pool_size)
                positions_by_stratum[stratum] = count
                pool_positions += count
                expected_per_epoch += draws * fraction
                probability_absent *= (1.0 - fraction) ** (draws * epochs)
            classes[str(count_class)] = {
                "pool_positions": pool_positions,
                "positions_by_stratum": positions_by_stratum,
                "expected_draws_per_fixed_epoch": expected_per_epoch,
                "expected_draws_over_20_epochs": expected_per_epoch * epochs,
                "probability_seen_at_least_once_over_20_epochs": 1.0
                - probability_absent,
            }
        result[head] = {
            "classes": classes,
            "fresh_epoch_probability_assumption": (
                "fresh independent draws with replacement in every epoch"
            ),
        }
    return result


def audit_fixed_sampler(
    population: SplitPopulation,
    *,
    query_count: int,
    seed: int,
) -> Mapping[str, object]:
    expected_sources = source_counts_for_queries(query_count)
    rng = random.Random(seed)
    cumulative_frames = []
    running_frames = 0
    for item in population.tracks:
        running_frames += item.frame_count
        cumulative_frames.append(running_frames)

    selected: List[PointTarget] = []
    rejection_attempts = 0
    for query_index in range(query_count):
        stratum = STRATA[query_index % len(STRATA)]
        if stratum == "background":
            target, attempts = _sample_uniform_background(
                population,
                rng=rng,
                cumulative_frames=cumulative_frames,
            )
            rejection_attempts += attempts
        else:
            target = rng.choice(population.positive_pools[stratum])
        selected.append(target)

    source_counts = Counter(target.stratum for target in selected)
    if any(source_counts[stratum] != expected_sources[stratum] for stratum in STRATA):
        raise PointQueryAuditError("fixed source cycle changed")

    sample_joint = Counter(target.joint for target in selected)
    selected_keys_by_stratum = {
        stratum: Counter(
            (target.track_index, target.position)
            for target in selected
            if target.stratum == stratum
        )
        for stratum in STRATA
    }
    selected_by_player = Counter(
        population.tracks[target.track_index].track.player_id for target in selected
    )
    selected_by_arrangement = Counter(
        _arrangement(population.tracks[target.track_index].member)
        for target in selected
    )
    left_initialized = [
        target for target in selected if target.position < LEFT_HISTORY_SAMPLES
    ]

    live_probabilities = {
        stratum: _safe_ratio(
            population.stratum_sizes[stratum], population.total_frames
        )
        for stratum in STRATA
    }
    sampled_probabilities = {
        stratum: _safe_ratio(source_counts[stratum], query_count)
        for stratum in STRATA
    }
    importance_weights = {
        stratum: live_probabilities[stratum] / sampled_probabilities[stratum]
        for stratum in STRATA
    }
    analytical_mean_weight = sum(
        sampled_probabilities[stratum] * importance_weights[stratum]
        for stratum in STRATA
    )

    analytical_joint_probability: Counter = Counter()
    for stratum in STRATA:
        stratum_size = population.stratum_sizes[stratum]
        for joint, count in population.joint_by_stratum[stratum].items():
            analytical_joint_probability[joint] += (
                sampled_probabilities[stratum]
                * importance_weights[stratum]
                * _safe_ratio(count, stratum_size)
            )
    full_joint_probability: Counter = Counter()
    for histogram in population.joint_by_stratum.values():
        for joint, count in histogram.items():
            full_joint_probability[joint] += _safe_ratio(
                count, population.total_frames
            )
    joint_keys = set(analytical_joint_probability) | set(full_joint_probability)
    analytical_max_error = max(
        (
            abs(
                analytical_joint_probability[joint]
                - full_joint_probability[joint]
            )
            for joint in joint_keys
        ),
        default=0.0,
    )

    empirical_weighted_joint_mass: Counter = Counter()
    weights = []
    for target in selected:
        weight = importance_weights[target.stratum]
        weights.append(weight)
        empirical_weighted_joint_mass[target.joint] += weight
    weight_sum = sum(weights)
    empirical_weighted_joint_probability = Counter(
        {
            joint: _safe_ratio(mass, weight_sum)
            for joint, mass in empirical_weighted_joint_mass.items()
        }
    )
    effective_sample_size = _safe_ratio(
        weight_sum * weight_sum, sum(weight * weight for weight in weights)
    )

    head_reports: Dict[str, object] = {}
    for head in HEADS:
        sampled_histogram = _head_histogram_from_joint(sample_joint, head)
        weighted_histogram = _head_histogram_from_joint(
            empirical_weighted_joint_mass, head
        )
        full_histogram = _head_histogram_from_joint(
            {
                joint: probability
                for joint, probability in full_joint_probability.items()
            },
            head,
        )
        sample_positive = sum(
            mass for label, mass in sampled_histogram.items() if label > 0
        )
        weighted_positive_mass = sum(
            mass for label, mass in weighted_histogram.items() if label > 0
        )
        full_positive_probability = sum(
            mass for label, mass in full_histogram.items() if label > 0
        )
        weighted_positive_probability = _safe_ratio(
            weighted_positive_mass, weight_sum
        )
        head_reports[head] = {
            "unweighted_exact_count_histogram": _histogram_json(sampled_histogram),
            "unweighted_positive_queries": sample_positive,
            "unweighted_positive_probability": _safe_ratio(
                sample_positive, query_count
            ),
            "importance_weighted_exact_count_mass": _histogram_json(
                weighted_histogram
            ),
            "importance_weighted_positive_mass": weighted_positive_mass,
            "importance_weighted_negative_mass": weight_sum
            - weighted_positive_mass,
            "importance_weighted_positive_probability": weighted_positive_probability,
            "full_stream_positive_probability": full_positive_probability,
            "weighted_empirical_relative_error_to_full": _safe_ratio(
                abs(weighted_positive_probability - full_positive_probability),
                full_positive_probability,
            ),
            "constant_categorical_optimum": {
                "unweighted_fixed_queries": _categorical_optimum(sampled_histogram),
                "importance_weighted_fixed_queries": _categorical_optimum(
                    weighted_histogram
                ),
                "full_stream": _categorical_optimum(full_histogram),
            },
        }

    background_selected = [
        target for target in selected if target.stratum == "background"
    ]
    observed_near = Counter()
    for target in background_selected:
        boundaries = tuple(
            sorted(population.tracks[target.track_index].boundary_positions)
        )
        distance = nearest_boundary_distance(target.position, boundaries)
        for radius in NEAR_BOUNDARY_RADII:
            if distance <= radius:
                observed_near[radius] += 1

    near_report: Dict[str, object] = {}
    for radius in NEAR_BOUNDARY_RADII:
        full_count = sum(
            background_near_boundary_count(
                tuple(item.boundary_positions),
                frame_count=item.frame_count,
                radius=radius,
            )
            for item in population.tracks
        )
        fraction = _safe_ratio(
            full_count, population.stratum_sizes["background"]
        )
        expected = source_counts["background"] * fraction
        near_report[str(radius)] = {
            "full_stream_background_positions": full_count,
            "fraction_of_background": fraction,
            "expected_fixed_draws": expected,
            "observed_fixed_draws": observed_near[radius],
            "expected_draws_over_20_repeated_fixed_epochs": expected
            * HISTORICAL_EPOCHS,
        }

    rare_exposure = _rare_class_exposure(
        population,
        source_counts,
        epochs=HISTORICAL_EPOCHS,
    )
    for head in HEADS:
        observed = Counter(
            target.joint[HEADS.index(head)]
            for target in selected
            if target.joint[HEADS.index(head)] > 0
        )
        rare_exposure[head]["observed_fixed_draws"] = _histogram_json(observed)
        for count_class, class_report in rare_exposure[head]["classes"].items():
            observed_count = observed[int(count_class)]
            class_report["observed_in_single_cached_fixed_batch"] = observed_count
            class_report["seen_if_that_same_fixed_batch_is_repeated"] = (
                observed_count > 0
            )
            class_report["draws_if_that_same_fixed_batch_is_repeated_20_times"] = (
                observed_count * HISTORICAL_EPOCHS
            )

    per_stratum_selection = {}
    for stratum in STRATA:
        counter = selected_keys_by_stratum[stratum]
        draws = source_counts[stratum]
        per_stratum_selection[stratum] = {
            "draws": draws,
            "unique_positions": len(counter),
            "repeated_draws": draws - len(counter),
            "positions_selected_more_than_once": sum(
                count > 1 for count in counter.values()
            ),
        }

    return {
        "seed": seed,
        "queries": query_count,
        "source_counts": dict(source_counts),
        "source_probabilities": sampled_probabilities,
        "selection": {
            "by_stratum": per_stratum_selection,
            "by_player": dict(sorted(selected_by_player.items())),
            "by_arrangement": dict(sorted(selected_by_arrangement.items())),
            "background_global_frame_attempts": rejection_attempts,
            "background_rejections_on_boundary_positions": rejection_attempts
            - source_counts["background"],
            "contexts_requiring_left_zero_initialization": len(left_initialized),
            "left_zero_padding_samples_total": sum(
                LEFT_HISTORY_SAMPLES - target.position
                for target in left_initialized
            ),
            "right_padding_samples": 0,
        },
        "joint_exact_count_histogram": _joint_json(sample_joint),
        "heads": head_reports,
        "importance_correction": {
            "live_stratum_probabilities": live_probabilities,
            "sampled_stratum_probabilities": sampled_probabilities,
            "weights_p_live_over_p_sample": importance_weights,
            "minimum_weight": min(importance_weights.values()),
            "maximum_weight": max(importance_weights.values()),
            "maximum_over_minimum": _safe_ratio(
                max(importance_weights.values()), min(importance_weights.values())
            ),
            "analytical_sampled_mean_weight": analytical_mean_weight,
            "analytical_sampled_mean_weight_absolute_error_from_one": abs(
                analytical_mean_weight - 1.0
            ),
            "analytical_weighted_joint_probabilities": _joint_json(
                analytical_joint_probability
            ),
            "full_stream_joint_probabilities": _joint_json(full_joint_probability),
            "analytical_weighted_joint_max_absolute_error": analytical_max_error,
            "empirical_weighted_joint_probabilities": _joint_json(
                empirical_weighted_joint_probability
            ),
            "effective_sample_size": effective_sample_size,
            "effective_sample_size_ratio": _safe_ratio(
                effective_sample_size, query_count
            ),
            "approved_for_training_loss": False,
        },
        "background_near_boundary": near_report,
        "rare_count_class_exposure": rare_exposure,
    }


def _validate_locked_inputs(
    protocol_path: Path,
    amendment_path: Path,
    review_amendment_path: Path,
    metadata_path: Path,
    prior_audit_path: Path,
) -> Tuple[Mapping[str, object], Mapping[str, object], Mapping[str, object]]:
    if sha256_file(protocol_path) != EXPECTED_PROTOCOL_SHA256:
        raise PointQueryAuditError("point-query protocol SHA-256 changed")
    if sha256_file(amendment_path) != EXPECTED_AMENDMENT_SHA256:
        raise PointQueryAuditError("point-query amendment SHA-256 changed")
    if sha256_file(review_amendment_path) != EXPECTED_REVIEW_AMENDMENT_SHA256:
        raise PointQueryAuditError("point-query review amendment SHA-256 changed")
    helper_path = REPOSITORY_ROOT / "scripts" / "audit_anonymous_boundary_targets.py"
    if sha256_file(helper_path) != EXPECTED_HELPER_SHA256:
        raise PointQueryAuditError("anonymous-target helper SHA-256 changed")
    protocol = _load_json(protocol_path)
    amendment = _load_json(amendment_path)
    metadata = _load_json(metadata_path)
    prior_audit = _load_json(prior_audit_path)
    baseline = protocol.get("baseline")
    if not isinstance(baseline, dict):
        raise PointQueryAuditError("protocol baseline is missing")
    for path_key, hash_key in (
        ("training_source", "training_source_sha256"),
        ("guitarset_source", "guitarset_source_sha256"),
        ("anonymous_target_audit", "anonymous_target_audit_sha256"),
        ("end_of_stream_audit", "end_of_stream_audit_sha256"),
        ("v7_metadata", "v7_metadata_sha256"),
    ):
        path_value = baseline.get(path_key)
        expected_hash = baseline.get(hash_key)
        if not isinstance(path_value, str) or not isinstance(expected_hash, str):
            raise PointQueryAuditError(f"invalid locked baseline field: {path_key}")
        actual_path = REPOSITORY_ROOT / path_value
        if sha256_file(actual_path) != expected_hash:
            raise PointQueryAuditError(f"locked input changed: {path_value}")
    if Path(metadata_path).resolve() != (
        REPOSITORY_ROOT / str(baseline["v7_metadata"])
    ).resolve():
        raise PointQueryAuditError("metadata path differs from protocol")
    if Path(prior_audit_path).resolve() != (
        REPOSITORY_ROOT / str(baseline["anonymous_target_audit"])
    ).resolve():
        raise PointQueryAuditError("prior audit path differs from protocol")
    return protocol, metadata, prior_audit


def _comparison_to_v7(
    full_reports: Mapping[str, Mapping[str, object]],
    sampler_reports: Mapping[str, Mapping[str, object]],
    prior_audit: Mapping[str, object],
) -> Mapping[str, object]:
    result: Dict[str, object] = {}
    for split_name in ("train", "validation"):
        result[split_name] = {}
        for head in HEADS:
            full_density = full_reports[split_name]["heads"][head][
                "positive_position_density"
            ]
            candidate_density = sampler_reports[split_name]["heads"][head][
                "unweighted_positive_probability"
            ]
            v7_density = prior_audit["current_v7_sampler"][split_name]["heads"][
                head
            ]["exact_anonymous_count"]["positive_position_density"]
            result[split_name][head] = {
                "full_stream_exact_density": full_density,
                "audited_v7_sampled_exact_density": v7_density,
                "v7_over_full": _safe_ratio(v7_density, full_density),
                "candidate_unweighted_point_density": candidate_density,
                "candidate_over_full": _safe_ratio(
                    candidate_density, full_density
                ),
            }
    return result


def _oracle_reconciliation(
    populations: Mapping[str, SplitPopulation],
    full_reports: Mapping[str, Mapping[str, object]],
    prior_audit: Mapping[str, object],
) -> Mapping[str, object]:
    """Reconcile the new representation against the locked independent audit."""

    checks: Dict[str, bool] = {}
    details: Dict[str, object] = {}
    prior_splits = prior_audit["full_stream"]
    for split_name in ("train", "validation"):
        population = populations[split_name]
        actual_full = full_reports[split_name]
        prior_full = prior_splits[split_name]
        split_details: Dict[str, object] = {}
        expected_frames = prior_full["heads"]["onset"]["exact_anonymous_count"][
            "total_samples"
        ]
        checks[f"{split_name}_frames"] = population.total_frames == expected_frames
        for head in HEADS:
            actual_head = actual_full["heads"][head]
            expected_head = prior_full["heads"][head]["exact_anonymous_count"]
            prefix = f"{split_name}_{head}"
            head_checks = {
                "histogram": actual_head["exact_count_histogram"]
                == expected_head["histogram"],
                "positive_positions": actual_head["positive_positions"]
                == expected_head["positive_positions"],
                "event_instances": actual_head["event_instances"]
                == expected_head["event_instances"],
                "maximum_exact_count": actual_head["maximum_exact_count"]
                == expected_head["maximum_count"],
            }
            for name, passed in head_checks.items():
                checks[f"{prefix}_{name}"] = passed
            split_details[head] = head_checks

        expected_eof = prior_full["heads"]["offset"]["raw_annotations"][
            "exclusive_end_instances"
        ]
        actual_eof = actual_full["context_edges"][
            "exclusive_end_offset_instances_excluded_from_acoustic_target"
        ]
        all_supervised_offsets_internal = all(
            all(position < item.frame_count for position in item.offset_counts)
            for item in population.tracks
        )
        checks[f"{split_name}_exclusive_end_count"] = actual_eof == expected_eof
        checks[f"{split_name}_all_supervised_offsets_internal"] = (
            all_supervised_offsets_internal
        )
        expected_raw_offsets = prior_full["heads"]["offset"]["raw_annotations"][
            "event_instances"
        ]
        checks[f"{split_name}_raw_offset_reconciliation"] = (
            actual_full["heads"]["offset"]["event_instances"] + actual_eof
            == expected_raw_offsets
        )
        split_details["exclusive_end"] = {
            "expected_instances": expected_eof,
            "actual_excluded_instances": actual_eof,
            "all_supervised_offset_positions_strictly_before_frame_count": (
                all_supervised_offsets_internal
            ),
        }
        details[split_name] = split_details

    actual_players = sorted(
        {
            item.track.player_id
            for population in populations.values()
            for item in population.tracks
        }
    )
    checks["actual_players_are_exactly_00_through_04"] = actual_players == sorted(
        ALLOWED_PLAYERS
    )
    checks["player_05_absent_from_indexed_tracks"] = "05" not in actual_players
    return {
        "all_exact": all(checks.values()),
        "checks": checks,
        "details": details,
        "actual_players": actual_players,
    }


def _derive_decision(
    populations: Mapping[str, SplitPopulation],
    full_reports: Mapping[str, Mapping[str, object]],
    sampler_reports: Mapping[str, Mapping[str, object]],
    prior_audit: Mapping[str, object],
) -> Mapping[str, object]:
    train_sampler = sampler_reports["train"]
    oracle = _oracle_reconciliation(populations, full_reports, prior_audit)
    structural_checks = {
        "strata_partition_all_full_stream_samples_exactly": all(
            sum(population.stratum_sizes.values()) == population.total_frames
            for population in populations.values()
        ),
        "positive_pools_use_unique_positions": all(
            len(pool)
            == len({(target.track_index, target.position) for target in pool})
            for population in populations.values()
            for pool in population.positive_pools.values()
        ),
        "locked_oracle_reconciliation_exact": oracle["all_exact"],
        "exclusive_end_offsets_excluded_and_reconciled": all(
            oracle["checks"][f"{split_name}_exclusive_end_count"]
            and oracle["checks"][
                f"{split_name}_all_supervised_offsets_internal"
            ]
            and oracle["checks"][f"{split_name}_raw_offset_reconciliation"]
            for split_name in ("train", "validation")
        ),
        "source_cycle_matches_fixed_counts": all(
            sampler_reports[split_name]["source_counts"]
            == source_counts_for_queries(FIXED_QUERIES[split_name])
            for split_name in ("train", "validation")
        ),
        "analytical_importance_weighted_joint_prior_within_1e_12": all(
            sampler_reports[split_name]["importance_correction"][
                "analytical_weighted_joint_max_absolute_error"
            ]
            <= 1e-12
            for split_name in ("train", "validation")
        ),
        "analytical_sampled_mean_importance_weight_within_1e_12": all(
            sampler_reports[split_name]["importance_correction"][
                "analytical_sampled_mean_weight_absolute_error_from_one"
            ]
            <= 1e-12
            for split_name in ("train", "validation")
        ),
        "effective_sample_size_ratio_at_least_0_25": all(
            sampler_reports[split_name]["importance_correction"][
                "effective_sample_size_ratio"
            ]
            >= 0.25
            for split_name in ("train", "validation")
        ),
        "actual_players_are_exactly_00_through_04": oracle["checks"][
            "actual_players_are_exactly_00_through_04"
        ],
        "player_05_absent_from_indexed_tracks": oracle["checks"][
            "player_05_absent_from_indexed_tracks"
        ],
    }
    hard_pass = all(structural_checks.values())

    rare_values = []
    for head in HEADS:
        rare_values.extend(
            item["expected_draws_over_20_epochs"]
            for item in train_sampler["rare_count_class_exposure"][head][
                "classes"
            ].values()
        )
    minimum_rare_exposure = min(rare_values) if rare_values else 0.0
    near_one_exposure = train_sampler["background_near_boundary"]["1"][
        "expected_draws_over_20_repeated_fixed_epochs"
    ]
    training_guards = {
        "minimum_expected_count_class_draws_over_20_train_epochs": minimum_rare_exposure,
        "required_minimum_count_class_draws": 20.0,
        "count_class_exposure_pass": minimum_rare_exposure >= 20.0,
        "expected_background_draws_within_one_sample_over_20_train_epochs": near_one_exposure,
        "required_minimum_near_boundary_draws": 20.0,
        "near_boundary_exposure_pass": near_one_exposure >= 20.0,
        "unweighted_sampler_live_calibrated": all(
            math.isclose(
                sampler_reports[split_name]["heads"][head][
                    "unweighted_positive_probability"
                ],
                full_reports[split_name]["heads"][head][
                    "positive_position_density"
                ],
                rel_tol=1e-6,
                abs_tol=1e-12,
            )
            for split_name in ("train", "validation")
            for head in HEADS
        ),
        "point_crop_equals_continuous_model_output_verified": False,
        "candidate_context_interval_contract_ends_at_t_plus_1": True,
        "audio_crop_implementation_verified": False,
        "point_crop_equivalence_deferred_reason": "no compatible point-query causal model exists in this sampler-only audit",
    }
    training_ready = (
        hard_pass
        and training_guards["count_class_exposure_pass"]
        and training_guards["near_boundary_exposure_pass"]
        and training_guards["audio_crop_implementation_verified"]
        and training_guards["point_crop_equals_continuous_model_output_verified"]
    )
    if not hard_pass:
        category = "structurally_rejected"
    elif training_ready:
        category = "structurally_admissible_with_prior_correction_and_training_ready"
    else:
        category = "structurally_admissible_with_prior_correction_but_not_training_ready"

    stop_reasons = []
    if not training_guards["unweighted_sampler_live_calibrated"]:
        stop_reasons.append("unweighted_fixed_cycle_prior_is_not_the_live_prior")
    if not training_guards["near_boundary_exposure_pass"]:
        stop_reasons.append("uniform_background_underexposes_boundary_adjacent_negatives")
    if not training_guards["count_class_exposure_pass"]:
        stop_reasons.append("fixed_cycle_underexposes_rare_exact_count_classes")
    if not training_guards["audio_crop_implementation_verified"]:
        stop_reasons.append("candidate_audio_crop_is_not_implemented_or_verified")
    if not training_guards["point_crop_equals_continuous_model_output_verified"]:
        stop_reasons.append("crop_live_equivalence_requires_a_future_compatible_model")

    return {
        "category": category,
        "structural_admissibility_scope": (
            "target partition, draw distribution and analytical prior correction only"
        ),
        "structurally_admissible_with_prior_correction": hard_pass,
        "training_ready": training_ready,
        "structural_checks": structural_checks,
        "training_readiness_guards": training_guards,
        "importance_weights_approved_for_training_loss": False,
        "model_selected": False,
        "loss_selected": False,
        "training_started": False,
        "live_changed": False,
        "stop_reasons": stop_reasons,
        "locked_oracle_reconciliation": oracle,
        "next_action_requires_user_approval": True,
    }


def run_audit(
    dataset_dir: Path,
    *,
    output_path: Path,
    protocol_path: Path = DEFAULT_PROTOCOL,
    amendment_path: Path = DEFAULT_AMENDMENT,
    review_amendment_path: Path = DEFAULT_REVIEW_AMENDMENT,
    metadata_path: Path = DEFAULT_METADATA,
    prior_audit_path: Path = DEFAULT_PRIOR_AUDIT,
) -> Mapping[str, object]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing audit: {output_path}")
    protocol, metadata, prior_audit = _validate_locked_inputs(
        protocol_path,
        amendment_path,
        review_amendment_path,
        metadata_path,
        prior_audit_path,
    )

    tracks = index_guitarset(dataset_dir)
    if not tracks or any(track.player_id not in ALLOWED_PLAYERS for track in tracks):
        raise PointQueryAuditError("GuitarSet index escaped players 00 through 04")
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

    prepared = {
        "train": _prepare_tracks(train_tracks),
        "validation": _prepare_tracks(validation_tracks),
    }
    populations = {
        split_name: build_population(split_tracks)
        for split_name, split_tracks in prepared.items()
    }
    full_reports = {
        split_name: full_stream_report(population)
        for split_name, population in populations.items()
    }
    sampler_reports = {
        "train": audit_fixed_sampler(
            populations["train"],
            query_count=FIXED_QUERIES["train"],
            seed=SEED,
        ),
        "validation": audit_fixed_sampler(
            populations["validation"],
            query_count=FIXED_QUERIES["validation"],
            seed=SEED + 1,
        ),
    }
    report: Mapping[str, object] = {
        "schema_version": 1,
        "status": "completed_audit_only",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "candidate_only": "exact causal point-query sampler",
            "targets": ["onset_count[t]", "offset_count[t]"],
            "context_samples": CONTEXT_SAMPLES,
            "future_audio_samples": 0,
            "model_changed": False,
            "loss_changed": False,
            "training_run": False,
            "decoder_changed": False,
            "live_changed": False,
            "player_05_content_read": False,
        },
        "locked_inputs": {
            "protocol": str(protocol_path.resolve()),
            "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
            "protocol_amendment": str(amendment_path.resolve()),
            "protocol_amendment_sha256": EXPECTED_AMENDMENT_SHA256,
            "protocol_review_amendment": str(review_amendment_path.resolve()),
            "protocol_review_amendment_sha256": EXPECTED_REVIEW_AMENDMENT_SHA256,
            "metadata": str(metadata_path.resolve()),
            "metadata_sha256": sha256_file(metadata_path),
            "prior_target_audit": str(prior_audit_path.resolve()),
            "prior_target_audit_sha256": sha256_file(prior_audit_path),
            "audit_helper_sha256": EXPECTED_HELPER_SHA256,
            "audit_source_sha256": sha256_file(Path(__file__)),
            "protocol_status": protocol["status"],
        },
        "data_guard": {
            "selected_players": sorted(ALLOWED_PLAYERS),
            "locked_test_player": "05",
            "locked_test_used": False,
            "train_tracks": len(train_tracks),
            "validation_tracks": len(validation_tracks),
            "validation_members_exactly_match_v7": True,
        },
        "splits": {
            split_name: {
                "full_stream": full_reports[split_name],
                "fixed_point_query_sampler": sampler_reports[split_name],
            }
            for split_name in ("train", "validation")
        },
        "comparison_to_v7": _comparison_to_v7(
            full_reports, sampler_reports, prior_audit
        ),
        "decision": _derive_decision(
            populations, full_reports, sampler_reports, prior_audit
        ),
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
        "--protocol-review-amendment", default=str(DEFAULT_REVIEW_AMENDMENT)
    )
    parser.add_argument("--metadata", default=str(DEFAULT_METADATA))
    parser.add_argument("--prior-audit", default=str(DEFAULT_PRIOR_AUDIT))
    return parser


def main(argv: Sequence[str] = ()) -> int:
    arguments = create_argument_parser().parse_args(argv or None)
    report = run_audit(
        Path(arguments.dataset_dir),
        output_path=Path(arguments.output),
        protocol_path=Path(arguments.protocol),
        amendment_path=Path(arguments.protocol_amendment),
        review_amendment_path=Path(arguments.protocol_review_amendment),
        metadata_path=Path(arguments.metadata),
        prior_audit_path=Path(arguments.prior_audit),
    )
    print(json.dumps(report["decision"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
