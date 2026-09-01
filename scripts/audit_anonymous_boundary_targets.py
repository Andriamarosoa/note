"""Audit anonymous ``type, position`` boundary targets without training.

The audit compares exact per-sample multiplicity, anonymous count windows of
512 samples, non-overlapping 512-sample multisets, and the historical V7
six-slot binary targets.  It reads only GuitarSet players 00 through 04 and
never decodes audio samples or imports TensorFlow.
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
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from causal_note.guitarset import (  # noqa: E402 - local bootstrap above
    ALLOWED_PLAYERS,
    BoundarySlots,
    GuitarSetTrack,
    NoteBoundary,
    SAMPLE_RATE,
    SLOT_COUNT,
    index_guitarset,
    load_boundary_slots,
)
from scripts.train_boundaries import (  # noqa: E402
    inspect_pcm16_mono_wav,
    split_tracks_by_group,
)


HEADS = ("onset", "offset")
WIDTH_SAMPLES = 512
BLOCK_SAMPLES = 512
WINDOW_SAMPLES = 8192
BATCH_SIZE = 8
TRAIN_STEPS = 200
VALIDATION_STEPS = 50
WARMUP_SAMPLES = 4092
SEED = 1337
HISTORICAL_POSITIVE_WEIGHT = 28.0
HISTORICAL_THRESHOLD = 0.55
CAPACITIES = (1, 2, 4, 6, 8, 12, 16)

DEFAULT_PROTOCOL = (
    REPOSITORY_ROOT
    / "model"
    / "causal-boundaries-type-position-target-audit-protocol.json"
)
DEFAULT_AMENDMENT = (
    REPOSITORY_ROOT
    / "model"
    / "causal-boundaries-type-position-target-audit-protocol-amendment-01.json"
)
DEFAULT_METADATA = (
    REPOSITORY_ROOT
    / "model"
    / "causal-boundaries-weight28-window512-v7-epoch08.recovery.metadata.json"
)
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT
    / "model"
    / "causal-boundaries-type-position-target-audit.json"
)
EXPECTED_PROTOCOL_SHA256 = (
    "BDAFD38D09C230841CE4AE42EC341129FFB7263F662683C3BA7F10F6F194D1A9"
)
EXPECTED_AMENDMENT_SHA256 = (
    "8DADC12687037C9AC5DF6408F5C3F88A9AEF06145DBB214B4B53351E0B310BAF"
)


class AnonymousTargetAuditError(ValueError):
    """Raised when the locked audit inputs or accounting are inconsistent."""


@dataclass(frozen=True)
class AuditTrack:
    track: GuitarSetTrack
    slots: BoundarySlots
    frame_count: int

    @property
    def member(self) -> str:
        return PurePosixPath(self.track.annotation_member).name


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
        raise AnonymousTargetAuditError(f"cannot read JSON input: {path}") from exc
    if not isinstance(value, dict):
        raise AnonymousTargetAuditError(f"JSON root must be an object: {path}")
    return value


def _histogram_json(histogram: Mapping[int, int]) -> Dict[str, int]:
    return {
        str(key): int(histogram[key])
        for key in sorted(histogram)
        if histogram[key]
    }


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _head_positions(
    slots: BoundarySlots,
    head: str,
    *,
    frame_count: int,
    supervised_only: bool,
) -> Tuple[int, ...]:
    if head not in HEADS:
        raise AnonymousTargetAuditError(f"unknown head: {head!r}")
    positions: List[int] = []
    for notes in slots:
        for note in notes:
            position = note.onset_sample if head == "onset" else note.offset_sample
            if position > frame_count:
                raise AnonymousTargetAuditError(
                    f"{head} position {position} exceeds {frame_count} frames"
                )
            if supervised_only and position >= frame_count:
                continue
            positions.append(position)
    return tuple(sorted(positions))


def exact_count_histogram(
    positions: Sequence[int],
    *,
    start: int,
    end: int,
) -> Counter:
    """Return the per-sample multiplicity histogram on ``[start, end)``."""

    if not 0 <= start <= end:
        raise AnonymousTargetAuditError("invalid exact-count interval")
    counts = Counter(position for position in positions if start <= position < end)
    histogram = Counter(counts.values())
    histogram[0] += end - start - len(counts)
    if sum(histogram.values()) != end - start:
        raise AnonymousTargetAuditError("exact-count histogram does not reconcile")
    return histogram


def wide_count_histogram(
    positions: Sequence[int],
    *,
    start: int,
    end: int,
    width: int = WIDTH_SAMPLES,
) -> Counter:
    """Return the anonymous rectangular-count histogram on ``[start, end)``."""

    if not 0 <= start <= end or width <= 0:
        raise AnonymousTargetAuditError("invalid wide-count interval")
    deltas: Counter = Counter()
    for position, multiplicity in Counter(positions).items():
        interval_start = max(start, position)
        interval_end = min(end, position + width)
        if interval_start < interval_end:
            deltas[interval_start] += multiplicity
            deltas[interval_end] -= multiplicity

    histogram: Counter = Counter()
    level = 0
    cursor = start
    for position in sorted(deltas):
        if position < start or position > end:
            raise AnonymousTargetAuditError("wide-count delta escaped its interval")
        histogram[level] += position - cursor
        level += deltas[position]
        if level < 0:
            raise AnonymousTargetAuditError("wide-count level became negative")
        cursor = position
    histogram[level] += end - cursor
    if level != 0:
        raise AnonymousTargetAuditError("wide-count sweep did not return to zero")
    if sum(histogram.values()) != end - start:
        raise AnonymousTargetAuditError("wide-count histogram does not reconcile")
    return Counter({key: value for key, value in histogram.items() if value})


def _union_length(
    positions: Sequence[int],
    *,
    start: int,
    end: int,
    width: int,
) -> int:
    intervals = []
    for position in sorted(set(positions)):
        interval_start = max(start, position)
        interval_end = min(end, position + width)
        if interval_start < interval_end:
            intervals.append((interval_start, interval_end))
    if not intervals:
        return 0
    covered = 0
    current_start, current_end = intervals[0]
    for interval_start, interval_end in intervals[1:]:
        if interval_start <= current_end:
            current_end = max(current_end, interval_end)
        else:
            covered += current_end - current_start
            current_start, current_end = interval_start, interval_end
    return covered + current_end - current_start


def slot_binary_positive_elements(
    slots: BoundarySlots,
    head: str,
    *,
    start: int,
    end: int,
    width: int = WIDTH_SAMPLES,
) -> int:
    positive = 0
    for notes in slots:
        positions = [
            note.onset_sample if head == "onset" else note.offset_sample
            for note in notes
        ]
        positive += _union_length(
            positions,
            start=start,
            end=end,
            width=width,
        )
    return positive


def binary_plateau_count(
    positions: Sequence[int],
    *,
    width: int = WIDTH_SAMPLES,
) -> int:
    """Count rising edges of the binary union of discrete causal windows."""

    plateau_count = 0
    current_end = -1
    for position in sorted(set(positions)):
        if position > current_end:
            plateau_count += 1
        current_end = max(current_end, position + width)
    return plateau_count


def naive_positive_delta_recovery(
    positions: Sequence[int],
    *,
    width: int = WIDTH_SAMPLES,
) -> Mapping[str, object]:
    counts = Counter(positions)
    recovered_instances = 0
    fully_visible_positions = 0
    partially_visible_positions = 0
    hidden_positions = 0
    for position, multiplicity in counts.items():
        recovered = max(0, multiplicity - counts.get(position - width, 0))
        recovered_instances += recovered
        if recovered == multiplicity:
            fully_visible_positions += 1
        elif recovered:
            partially_visible_positions += 1
        else:
            hidden_positions += 1
    return {
        "event_instances": sum(counts.values()),
        "recovered_instances": recovered_instances,
        "lost_instances": sum(counts.values()) - recovered_instances,
        "recovered_fraction": _safe_ratio(recovered_instances, sum(counts.values())),
        "fully_visible_positions": fully_visible_positions,
        "partially_visible_positions": partially_visible_positions,
        "hidden_positions": hidden_positions,
    }


def causal_inverse_roundtrip(
    positions: Sequence[int],
    *,
    frame_count: int,
    width: int = WIDTH_SAMPLES,
) -> bool:
    """Verify ``x[t] = (y[t]-y[t-1]) + x[t-width]`` sparsely."""

    expected = Counter(position for position in positions if 0 <= position < frame_count)
    deltas: Counter = Counter()
    for position, multiplicity in expected.items():
        deltas[position] += multiplicity
        expiry = position + width
        if expiry < frame_count:
            deltas[expiry] -= multiplicity

    recovered: Dict[int, int] = {}
    for position in sorted(deltas):
        value = deltas[position] + recovered.get(position - width, 0)
        if value < 0:
            return False
        if value:
            recovered[position] = value
    return Counter(recovered) == expected


def causal_inverse_dense_counts(
    wide_counts: Sequence[int],
    *,
    width: int = WIDTH_SAMPLES,
) -> Tuple[int, ...]:
    """Invert an explicit dense rectangular-count target causally."""

    if width <= 0:
        raise AnonymousTargetAuditError("inverse width must be positive")
    recovered: List[int] = []
    previous_level = 0
    for position, raw_level in enumerate(wide_counts):
        level = int(raw_level)
        if level != raw_level or level < 0:
            raise AnonymousTargetAuditError(
                "dense wide counts must be nonnegative integers"
            )
        expired_start = recovered[position - width] if position >= width else 0
        value = level - previous_level + expired_start
        if value < 0:
            raise AnonymousTargetAuditError(
                "dense wide counts do not describe nonnegative event counts"
            )
        recovered.append(value)
        previous_level = level
    return tuple(recovered)


def _quantile(sorted_values: Sequence[int], fraction: float) -> int:
    if not sorted_values:
        return 0
    index = int(round((len(sorted_values) - 1) * fraction))
    return int(sorted_values[index])


def _gap_report(gaps: Sequence[int]) -> Mapping[str, object]:
    ordered = sorted(gaps)
    bins = Counter()
    for gap in ordered:
        if gap <= 15:
            bins["1-15"] += 1
        elif gap <= 63:
            bins["16-63"] += 1
        elif gap <= 255:
            bins["64-255"] += 1
        elif gap <= 511:
            bins["256-511"] += 1
        elif gap == 512:
            bins["512"] += 1
        elif gap <= 2205:
            bins["513-2205"] += 1
        else:
            bins["2206_plus"] += 1
    return {
        "count": len(ordered),
        "minimum_samples": ordered[0] if ordered else None,
        "median_samples": _quantile(ordered, 0.5) if ordered else None,
        "p95_samples": _quantile(ordered, 0.95) if ordered else None,
        "bins": dict(bins),
    }


def _block_histogram(
    positions: Sequence[int],
    *,
    frame_count: int,
    block_samples: int = BLOCK_SAMPLES,
) -> Counter:
    block_count = (frame_count + block_samples - 1) // block_samples
    counts = Counter(
        position // block_samples
        for position in positions
        if 0 <= position < frame_count
    )
    histogram = Counter(counts.values())
    empty_blocks = block_count - len(counts)
    if empty_blocks:
        histogram[0] = empty_blocks
    if sum(histogram.values()) != block_count:
        raise AnonymousTargetAuditError("block histogram does not reconcile")
    return histogram


def _block_report(histogram: Mapping[int, int]) -> Mapping[str, object]:
    total_blocks = sum(histogram.values())
    total_instances = sum(count * blocks for count, blocks in histogram.items())
    nonempty_blocks = total_blocks - histogram.get(0, 0)
    capacity_curve: Dict[str, object] = {}
    for capacity in CAPACITIES:
        fitting_blocks = sum(
            blocks for count, blocks in histogram.items() if count <= capacity
        )
        fitting_nonempty_blocks = sum(
            blocks
            for count, blocks in histogram.items()
            if 0 < count <= capacity
        )
        retained_instances = sum(
            min(count, capacity) * blocks for count, blocks in histogram.items()
        )
        capacity_curve[str(capacity)] = {
            "all_blocks_fit_fraction": _safe_ratio(fitting_blocks, total_blocks),
            "nonempty_blocks_fit_fraction": _safe_ratio(
                fitting_nonempty_blocks, nonempty_blocks
            ),
            "instances_retained_fraction_if_clipped": _safe_ratio(
                retained_instances, total_instances
            ),
        }
    return {
        "histogram": _histogram_json(histogram),
        "total_blocks": total_blocks,
        "empty_blocks": histogram.get(0, 0),
        "empty_block_fraction": _safe_ratio(histogram.get(0, 0), total_blocks),
        "nonempty_blocks": nonempty_blocks,
        "event_instances": total_instances,
        "maximum_events_in_one_block": max(histogram) if histogram else 0,
        "smallest_capacity_covering_every_block": max(histogram) if histogram else 0,
        "capacity_curve": capacity_curve,
    }


def constant_categorical_optimum(
    histogram: Mapping[int, int],
    *,
    nonzero_weight: float,
) -> Mapping[str, object]:
    if nonzero_weight <= 0.0 or not math.isfinite(nonzero_weight):
        raise AnonymousTargetAuditError("nonzero weight must be finite and positive")
    counts = {int(key): int(value) for key, value in histogram.items() if value}
    total_examples = sum(counts.values())
    weighted_counts = {
        key: value * (nonzero_weight if key else 1.0)
        for key, value in counts.items()
    }
    weighted_total = sum(weighted_counts.values())
    probabilities = {
        key: _safe_ratio(value, weighted_total)
        for key, value in weighted_counts.items()
    }
    normalized_loss = -sum(
        probability * math.log(probability)
        for probability in probabilities.values()
        if probability > 0.0
    )
    per_example_weighted_loss = -sum(
        weighted_counts[key] * math.log(probabilities[key])
        for key in probabilities
        if probabilities[key] > 0.0
    ) / total_examples
    nonzero_probability = sum(
        probability for key, probability in probabilities.items() if key
    )
    return {
        "nonzero_weight": nonzero_weight,
        "class_probabilities": {
            str(key): probabilities[key] for key in sorted(probabilities)
        },
        "nonzero_probability": nonzero_probability,
        "zero_probability": probabilities.get(0, 0.0),
        "normalized_weighted_cross_entropy": normalized_loss,
        "per_example_weighted_cross_entropy": per_example_weighted_loss,
        "weighted_zero_mass": weighted_counts.get(0, 0.0),
        "weighted_nonzero_mass": sum(
            value for key, value in weighted_counts.items() if key
        ),
        "historical_threshold": HISTORICAL_THRESHOLD,
        "constant_threshold_decision": (
            "always_nonzero"
            if nonzero_probability >= HISTORICAL_THRESHOLD
            else "always_zero"
        ),
    }


def _loss_report(histogram: Mapping[int, int]) -> Mapping[str, object]:
    return {
        "unweighted": constant_categorical_optimum(
            histogram,
            nonzero_weight=1.0,
        ),
        "historical_weight_28_diagnostic": constant_categorical_optimum(
            histogram,
            nonzero_weight=HISTORICAL_POSITIVE_WEIGHT,
        ),
    }


def _basic_histogram_report(histogram: Mapping[int, int]) -> Mapping[str, object]:
    total_samples = sum(histogram.values())
    nonzero_samples = total_samples - histogram.get(0, 0)
    event_mass = sum(count * samples for count, samples in histogram.items())
    return {
        "histogram": _histogram_json(histogram),
        "total_samples": total_samples,
        "positive_positions": nonzero_samples,
        "positive_position_density": _safe_ratio(nonzero_samples, total_samples),
        "event_mass": event_mass,
        "mean_event_count_per_sample": _safe_ratio(event_mass, total_samples),
        "maximum_count": max(histogram) if histogram else 0,
        "loss": _loss_report(histogram),
    }


def _prepare_tracks(tracks: Sequence[GuitarSetTrack]) -> Tuple[AuditTrack, ...]:
    prepared: List[AuditTrack] = []
    for track in tracks:
        if track.player_id not in ALLOWED_PLAYERS or track.player_id == "05":
            raise AnonymousTargetAuditError("player 05 content must remain locked")
        slots = load_boundary_slots(track.annotation_zip, track.annotation_member)
        info = inspect_pcm16_mono_wav(track.audio_zip, track.audio_member)
        for notes in slots:
            for note in notes:
                if note.offset_sample > info.frame_count:
                    raise AnonymousTargetAuditError(
                        f"boundary exceeds WAV frames in {track.annotation_member!r}"
                    )
        prepared.append(AuditTrack(track, slots, info.frame_count))
    return tuple(prepared)


def _cross_type_pairs(tracks: Sequence[AuditTrack]) -> Mapping[str, int]:
    result = Counter()
    for item in tracks:
        onsets = _head_positions(
            item.slots,
            "onset",
            frame_count=item.frame_count,
            supervised_only=True,
        )
        offsets = _head_positions(
            item.slots,
            "offset",
            frame_count=item.frame_count,
            supervised_only=True,
        )
        for onset in onsets:
            first = bisect_left(offsets, onset - (WIDTH_SAMPLES - 1))
            last = bisect_right(offsets, onset + (WIDTH_SAMPLES - 1))
            for offset in offsets[first:last]:
                if offset < onset:
                    result["offset_before_onset"] += 1
                elif onset < offset:
                    result["onset_before_offset"] += 1
                else:
                    result["same_sample"] += 1
    result["total_pairs"] = sum(
        result[key]
        for key in ("offset_before_onset", "onset_before_offset", "same_sample")
    )
    return dict(result)


def _successive_note_relations(tracks: Sequence[AuditTrack]) -> Mapping[str, int]:
    result = Counter()
    for item in tracks:
        for notes in item.slots:
            for previous, current in zip(notes, notes[1:]):
                delta = current.onset_sample - previous.offset_sample
                if delta < 0:
                    result["retrigger_before_previous_offset"] += 1
                elif delta == 0:
                    result["same_sample"] += 1
                elif delta < WIDTH_SAMPLES:
                    result["positive_gap_below_512"] += 1
                else:
                    result["positive_gap_512_or_more"] += 1
    result["total_successive_pairs"] = sum(
        result[key]
        for key in (
            "retrigger_before_previous_offset",
            "same_sample",
            "positive_gap_below_512",
            "positive_gap_512_or_more",
        )
    )
    return dict(result)


def audit_full_split(tracks: Sequence[AuditTrack]) -> Mapping[str, object]:
    total_frames = sum(item.frame_count for item in tracks)
    if total_frames <= 0:
        raise AnonymousTargetAuditError("split contains no audio frames")

    head_accumulators: Dict[str, MutableMapping[str, object]] = {}
    for head in HEADS:
        head_accumulators[head] = {
            "raw_instances": 0,
            "raw_unique_positions": 0,
            "raw_multiplicity_histogram": Counter(),
            "raw_maximum_multiplicity": 0,
            "supervised_instances": 0,
            "exclusive_end_instances": 0,
            "exact_histogram": Counter(),
            "wide_histogram": Counter(),
            "plateaus": 0,
            "gaps": [],
            "naive_event_instances": 0,
            "naive_recovered_instances": 0,
            "naive_fully_visible_positions": 0,
            "naive_partially_visible_positions": 0,
            "naive_hidden_positions": 0,
            "inverse_passed_tracks": 0,
            "slot_positive_elements": 0,
            "block_histogram": Counter(),
            "slot_permutation_invariant": True,
        }
    combined_block_histogram: Counter = Counter()

    for item in tracks:
        supervised_by_head: Dict[str, Tuple[int, ...]] = {}
        for head in HEADS:
            accumulator = head_accumulators[head]
            raw = _head_positions(
                item.slots,
                head,
                frame_count=item.frame_count,
                supervised_only=False,
            )
            supervised = _head_positions(
                item.slots,
                head,
                frame_count=item.frame_count,
                supervised_only=True,
            )
            supervised_by_head[head] = supervised
            raw_counts = Counter(raw)
            supervised_counts = Counter(supervised)
            accumulator["raw_instances"] += len(raw)
            accumulator["raw_unique_positions"] += len(raw_counts)
            accumulator["raw_multiplicity_histogram"].update(raw_counts.values())
            accumulator["raw_maximum_multiplicity"] = max(
                accumulator["raw_maximum_multiplicity"],
                max(raw_counts.values(), default=0),
            )
            accumulator["supervised_instances"] += len(supervised)
            accumulator["exclusive_end_instances"] += len(raw) - len(supervised)
            accumulator["exact_histogram"].update(
                exact_count_histogram(supervised, start=0, end=item.frame_count)
            )
            accumulator["wide_histogram"].update(
                wide_count_histogram(supervised, start=0, end=item.frame_count)
            )
            accumulator["plateaus"] += binary_plateau_count(supervised)
            unique_positions = sorted(supervised_counts)
            accumulator["gaps"].extend(
                right - left
                for left, right in zip(unique_positions, unique_positions[1:])
            )
            naive = naive_positive_delta_recovery(supervised)
            accumulator["naive_event_instances"] += naive["event_instances"]
            accumulator["naive_recovered_instances"] += naive["recovered_instances"]
            accumulator["naive_fully_visible_positions"] += naive[
                "fully_visible_positions"
            ]
            accumulator["naive_partially_visible_positions"] += naive[
                "partially_visible_positions"
            ]
            accumulator["naive_hidden_positions"] += naive["hidden_positions"]
            if causal_inverse_roundtrip(
                supervised,
                frame_count=item.frame_count,
            ):
                accumulator["inverse_passed_tracks"] += 1
            accumulator["slot_positive_elements"] += slot_binary_positive_elements(
                item.slots,
                head,
                start=0,
                end=item.frame_count,
            )
            accumulator["block_histogram"].update(
                _block_histogram(supervised, frame_count=item.frame_count)
            )
            reversed_slots = tuple(reversed(item.slots))
            reversed_positions = _head_positions(
                reversed_slots,
                head,
                frame_count=item.frame_count,
                supervised_only=True,
            )
            if Counter(reversed_positions) != supervised_counts:
                accumulator["slot_permutation_invariant"] = False

        combined_block_histogram.update(
            _block_histogram(
                supervised_by_head["onset"] + supervised_by_head["offset"],
                frame_count=item.frame_count,
            )
        )

    head_reports: Dict[str, object] = {}
    for head in HEADS:
        accumulator = head_accumulators[head]
        exact = _basic_histogram_report(accumulator["exact_histogram"])
        wide = _basic_histogram_report(accumulator["wide_histogram"])
        supervised_instances = int(accumulator["supervised_instances"])
        plateau_count = int(accumulator["plateaus"])
        wide_event_mass = int(wide["event_mass"])
        wide_positive_samples = int(wide["positive_positions"])
        naive_recovered = int(accumulator["naive_recovered_instances"])
        slot_positive = int(accumulator["slot_positive_elements"])
        slot_total = total_frames * SLOT_COUNT
        slot_histogram = Counter({0: slot_total - slot_positive, 1: slot_positive})
        head_reports[head] = {
            "raw_annotations": {
                "event_instances": int(accumulator["raw_instances"]),
                "unique_positions": int(accumulator["raw_unique_positions"]),
                "simultaneous_extra_instances": int(accumulator["raw_instances"])
                - int(accumulator["raw_unique_positions"]),
                "multiplicity_position_histogram": _histogram_json(
                    accumulator["raw_multiplicity_histogram"]
                ),
                "maximum_multiplicity": int(
                    accumulator["raw_maximum_multiplicity"]
                ),
                "exclusive_end_instances": int(
                    accumulator["exclusive_end_instances"]
                ),
            },
            "exact_anonymous_count": {
                **exact,
                "event_instances": supervised_instances,
                "simultaneous_extra_instances": supervised_instances
                - int(exact["positive_positions"]),
                "successive_unique_position_gaps": _gap_report(
                    accumulator["gaps"]
                ),
                "slot_permutation_invariant": bool(
                    accumulator["slot_permutation_invariant"]
                ),
                "instance_sum_reconciles": int(exact["event_mass"])
                == supervised_instances,
            },
            "anonymous_count_window_512": {
                **wide,
                "binary_union": {
                    "positive_samples": wide_positive_samples,
                    "overlap_mass_removed_by_union": wide_event_mass
                    - wide_positive_samples,
                    "overlap_mass_removed_fraction": _safe_ratio(
                        wide_event_mass - wide_positive_samples,
                        wide_event_mass,
                    ),
                    "positive_plateaus_or_rising_edges": plateau_count,
                    "event_instances_not_separable_by_rising_edges":
                    supervised_instances - plateau_count,
                    "event_instance_loss_fraction": _safe_ratio(
                        supervised_instances - plateau_count,
                        supervised_instances,
                    ),
                },
                "naive_positive_delta_decoder": {
                    "event_instances": int(accumulator["naive_event_instances"]),
                    "recovered_instances": naive_recovered,
                    "lost_instances": supervised_instances - naive_recovered,
                    "recovered_fraction": _safe_ratio(
                        naive_recovered, supervised_instances
                    ),
                    "fully_visible_positions": int(
                        accumulator["naive_fully_visible_positions"]
                    ),
                    "partially_visible_positions": int(
                        accumulator["naive_partially_visible_positions"]
                    ),
                    "hidden_positions": int(accumulator["naive_hidden_positions"]),
                },
                "causal_inverse_roundtrip": {
                    "passed_tracks": int(accumulator["inverse_passed_tracks"]),
                    "total_tracks": len(tracks),
                    "exact": int(accumulator["inverse_passed_tracks"])
                    == len(tracks),
                    "dense_target_crosscheck_is_unit_tested": True,
                },
            },
            "anonymous_multiset_per_nonoverlapping_block_512": _block_report(
                accumulator["block_histogram"]
            ),
            "historical_v7_slot_binary_window_512": {
                "positive_elements": slot_positive,
                "negative_elements": slot_total - slot_positive,
                "total_elements": slot_total,
                "positive_element_density": _safe_ratio(slot_positive, slot_total),
                "loss": _loss_report(slot_histogram),
            },
        }

    return {
        "tracks": len(tracks),
        "players": dict(Counter(item.track.player_id for item in tracks)),
        "frames": total_frames,
        "duration_seconds": total_frames / SAMPLE_RATE,
        "heads": head_reports,
        "combined_anonymous_multiset_per_nonoverlapping_block_512": _block_report(
            combined_block_histogram
        ),
        "temporal_relations": {
            "cross_type_pairs_strictly_below_512_samples": _cross_type_pairs(tracks),
            "successive_notes_within_original_slot": _successive_note_relations(
                tracks
            ),
        },
    }


def _anchor_pool(
    tracks: Sequence[AuditTrack],
    head: str,
) -> List[Tuple[int, int]]:
    anchors: List[Tuple[int, int]] = []
    for track_index, item in enumerate(tracks):
        for notes in item.slots:
            for note in notes:
                position = note.onset_sample if head == "onset" else note.offset_sample
                if position < item.frame_count:
                    anchors.append((track_index, position))
    return anchors


def audit_sampler(
    tracks: Sequence[AuditTrack],
    *,
    steps: int,
    seed: int,
) -> Mapping[str, object]:
    if not tracks:
        raise AnonymousTargetAuditError("cannot audit an empty sampler split")
    anchors = {head: _anchor_pool(tracks, head) for head in HEADS}
    if not anchors["onset"] or not anchors["offset"]:
        raise AnonymousTargetAuditError("sampler split has no anchors")
    for item in tracks:
        if item.frame_count < WINDOW_SAMPLES:
            raise AnonymousTargetAuditError("track is shorter than one V7 window")

    rng = random.Random(seed)
    next_kind = 0
    source_counts = Counter()
    scored_samples = 0
    windows_starting_at_zero = 0
    exact_histograms = {head: Counter() for head in HEADS}
    wide_histograms = {head: Counter() for head in HEADS}
    slot_positives = Counter()
    selected_anchor_positions = {head: Counter() for head in HEADS}
    selected_simultaneous_anchor_rows = Counter()

    for _ in range(steps * BATCH_SIZE):
        kind = ("onset", "offset", "random")[next_kind]
        next_kind = (next_kind + 1) % 3
        source_counts[kind] += 1
        if kind == "random":
            track_index = rng.randrange(len(tracks))
            item = tracks[track_index]
            start = rng.randint(0, item.frame_count - WINDOW_SAMPLES)
        else:
            track_index, boundary_sample = rng.choice(anchors[kind])
            item = tracks[track_index]
            local_boundary = rng.randint(
                max(WINDOW_SAMPLES // 2, WARMUP_SAMPLES),
                WINDOW_SAMPLES - 1,
            )
            desired = max(0, boundary_sample - local_boundary)
            start = min(desired, item.frame_count - WINDOW_SAMPLES)
            key = f"{item.member}:{boundary_sample}"
            selected_anchor_positions[kind][key] += 1
            pool_multiplicity = Counter(anchors[kind])[(track_index, boundary_sample)]
            if pool_multiplicity > 1:
                selected_simultaneous_anchor_rows[kind] += 1

        if start == 0:
            valid_start = 0
            windows_starting_at_zero += 1
        else:
            valid_start = WARMUP_SAMPLES
        scored_start = start + valid_start
        scored_end = start + WINDOW_SAMPLES
        scored_samples += scored_end - scored_start

        for head in HEADS:
            positions = _head_positions(
                item.slots,
                head,
                frame_count=item.frame_count,
                supervised_only=True,
            )
            exact_histograms[head].update(
                exact_count_histogram(
                    positions,
                    start=scored_start,
                    end=scored_end,
                )
            )
            wide_histograms[head].update(
                wide_count_histogram(
                    positions,
                    start=scored_start,
                    end=scored_end,
                )
            )
            slot_positives[head] += slot_binary_positive_elements(
                item.slots,
                head,
                start=scored_start,
                end=scored_end,
            )

    head_reports: Dict[str, object] = {}
    for head in HEADS:
        slot_total = scored_samples * SLOT_COUNT
        slot_histogram = Counter(
            {0: slot_total - slot_positives[head], 1: slot_positives[head]}
        )
        pool_counts = Counter(anchors[head])
        selected_count = sum(selected_anchor_positions[head].values())
        head_reports[head] = {
            "anchor_pool": {
                "instances": len(anchors[head]),
                "unique_anonymous_track_positions": len(pool_counts),
                "duplicate_instances": len(anchors[head]) - len(pool_counts),
                "maximum_multiplicity": max(pool_counts.values(), default=0),
            },
            "selected_anchors": {
                "rows": selected_count,
                "unique_anonymous_track_positions": len(
                    selected_anchor_positions[head]
                ),
                "repeated_selection_instances": selected_count
                - len(selected_anchor_positions[head]),
                "rows_anchored_on_a_simultaneous_position": int(
                    selected_simultaneous_anchor_rows[head]
                ),
            },
            "exact_anonymous_count": _basic_histogram_report(
                exact_histograms[head]
            ),
            "anonymous_count_window_512": _basic_histogram_report(
                wide_histograms[head]
            ),
            "historical_v7_slot_binary_window_512": {
                "positive_elements": int(slot_positives[head]),
                "negative_elements": slot_total - int(slot_positives[head]),
                "total_elements": slot_total,
                "positive_element_density": _safe_ratio(
                    slot_positives[head], slot_total
                ),
                "loss": _loss_report(slot_histogram),
            },
        }

    return {
        "steps": steps,
        "batch_size": BATCH_SIZE,
        "windows": steps * BATCH_SIZE,
        "seed": seed,
        "source_counts": {
            "onset_anchor": source_counts["onset"],
            "offset_anchor": source_counts["offset"],
            "random": source_counts["random"],
        },
        "windows_starting_at_zero": windows_starting_at_zero,
        "scored_samples": scored_samples,
        "heads": head_reports,
    }


def _add_density_comparisons(
    full_report: Mapping[str, object],
    sampler_report: MutableMapping[str, object],
) -> None:
    for head in HEADS:
        full_head = full_report["heads"][head]
        sampled_head = sampler_report["heads"][head]
        comparisons = {}
        for representation, density_key in (
            ("exact_anonymous_count", "positive_position_density"),
            ("anonymous_count_window_512", "positive_position_density"),
            ("historical_v7_slot_binary_window_512", "positive_element_density"),
        ):
            full_density = full_head[representation][density_key]
            sampled_density = sampled_head[representation][density_key]
            ratio = _safe_ratio(sampled_density, full_density)
            comparisons[representation] = {
                "full_density": full_density,
                "sampled_density": sampled_density,
                "sampled_over_full_ratio": ratio,
                "outside_preregistered_0_5_to_2_0_range": ratio < 0.5
                or ratio > 2.0,
            }
        sampled_head["density_comparison_to_full_stream"] = comparisons


def _validate_locked_inputs(
    protocol_path: Path,
    amendment_path: Path,
    metadata_path: Path,
) -> Tuple[Mapping[str, object], Mapping[str, object], Mapping[str, object]]:
    protocol_hash = sha256_file(protocol_path)
    amendment_hash = sha256_file(amendment_path)
    if protocol_hash != EXPECTED_PROTOCOL_SHA256:
        raise AnonymousTargetAuditError(
            f"protocol SHA-256 changed: {protocol_hash}"
        )
    if amendment_hash != EXPECTED_AMENDMENT_SHA256:
        raise AnonymousTargetAuditError(
            f"protocol amendment SHA-256 changed: {amendment_hash}"
        )
    protocol = _load_json(protocol_path)
    amendment = _load_json(amendment_path)
    metadata = _load_json(metadata_path)
    baseline = protocol.get("baseline")
    if not isinstance(baseline, dict):
        raise AnonymousTargetAuditError("protocol baseline is missing")
    for path_key, hash_key in (
        ("training_source", "training_source_sha256"),
        ("guitarset_source", "guitarset_source_sha256"),
        ("v7_metadata", "v7_metadata_sha256"),
    ):
        relative_path = baseline.get(path_key)
        expected_hash = baseline.get(hash_key)
        if not isinstance(relative_path, str) or not isinstance(expected_hash, str):
            raise AnonymousTargetAuditError(f"invalid protocol baseline key {path_key}")
        actual_hash = sha256_file(REPOSITORY_ROOT / relative_path)
        if actual_hash != expected_hash:
            raise AnonymousTargetAuditError(
                f"locked input changed: {relative_path} -> {actual_hash}"
            )
    if sha256_file(metadata_path) != baseline["v7_metadata_sha256"]:
        raise AnonymousTargetAuditError("selected metadata differs from protocol")
    return protocol, amendment, metadata


def _derive_decision(
    full_splits: Mapping[str, Mapping[str, object]],
    samplers: Mapping[str, Mapping[str, object]],
) -> Mapping[str, object]:
    exact_integrity = True
    inverse_integrity = True
    binary_hidden = 0
    exclusive_end_offsets = 0
    distribution_shift_flags: List[str] = []
    constant_pressure: Dict[str, object] = {}
    for split_name in ("train", "validation"):
        constant_pressure[split_name] = {}
        for head in HEADS:
            full_head = full_splits[split_name]["heads"][head]
            exact_integrity = exact_integrity and bool(
                full_head["exact_anonymous_count"]["instance_sum_reconciles"]
            ) and bool(full_head["exact_anonymous_count"]["slot_permutation_invariant"])
            inverse_integrity = inverse_integrity and bool(
                full_head["anonymous_count_window_512"][
                    "causal_inverse_roundtrip"
                ]["exact"]
            )
            binary_hidden += full_head["anonymous_count_window_512"][
                "binary_union"
            ]["event_instances_not_separable_by_rising_edges"]
            exclusive_end_offsets += (
                full_head["raw_annotations"]["exclusive_end_instances"]
                if head == "offset"
                else 0
            )
            for representation, comparison in samplers[split_name]["heads"][head][
                "density_comparison_to_full_stream"
            ].items():
                if comparison["outside_preregistered_0_5_to_2_0_range"]:
                    distribution_shift_flags.append(
                        f"{split_name}.{head}.{representation}"
                    )
            constant_pressure[split_name][head] = {
                representation: samplers[split_name]["heads"][head][representation][
                    "loss"
                ]["historical_weight_28_diagnostic"][
                    "constant_threshold_decision"
                ]
                for representation in (
                    "exact_anonymous_count",
                    "anonymous_count_window_512",
                    "historical_v7_slot_binary_window_512",
                )
            }

    combined_maximum = max(
        full_splits[split_name][
            "combined_anonymous_multiset_per_nonoverlapping_block_512"
        ]["smallest_capacity_covering_every_block"]
        for split_name in ("train", "validation")
    )
    stop_reasons = []
    if exclusive_end_offsets:
        stop_reasons.append("exclusive_end_offsets_require_an_explicit_policy")
    if distribution_shift_flags:
        stop_reasons.append("current_sampler_changes_target_density_by_more_than_2x")
    pressure_rows = [
        constant_pressure[split_name][head]
        for split_name in ("train", "validation")
        for head in HEADS
    ]
    if all(row["exact_anonymous_count"] == "always_zero" for row in pressure_rows):
        stop_reasons.append(
            "exact_count_with_historical_weight_favors_constant_no_event"
        )
    if all(
        row["anonymous_count_window_512"] == "always_nonzero"
        for row in pressure_rows
    ):
        stop_reasons.append(
            "anonymous_window_with_historical_weight_favors_constant_nonzero"
        )
    if any(
        len(
            {
                row["exact_anonymous_count"],
                row["anonymous_count_window_512"],
                row["historical_v7_slot_binary_window_512"],
            }
        )
        > 1
        for row in pressure_rows
    ):
        stop_reasons.append(
            "historical_weight_loss_and_threshold_are_not_transferable"
        )
    return {
        "exact_anonymous_count_structurally_admissible_on_sample_grid":
        exact_integrity,
        "exact_anonymous_count_preserves_supervised_multiplicity": exact_integrity,
        "full_annotation_stream_admissible_before_end_policy": exact_integrity
        and exclusive_end_offsets == 0,
        "exclusive_end_offsets": exclusive_end_offsets,
        "exclusive_end_policy_required_before_training": exclusive_end_offsets > 0,
        "anonymous_window_causal_inverse_structurally_exact": inverse_integrity,
        "binary_union_rejected": binary_hidden > 0,
        "binary_union_unseparable_event_instances": binary_hidden,
        "smallest_observed_combined_block_query_capacity": combined_maximum,
        "block_query_capacity_selected": False,
        "sampler_distribution_shift_flags": distribution_shift_flags,
        "historical_weight_28_constant_pressure_on_sampled_data": constant_pressure,
        "model_selected": False,
        "training_ready": False,
        "training_started": False,
        "live_changed": False,
        "stop_reasons": stop_reasons,
        "next_action_requires_user_approval": True,
    }


def run_audit(
    dataset_dir: Path,
    *,
    output_path: Path,
    protocol_path: Path = DEFAULT_PROTOCOL,
    amendment_path: Path = DEFAULT_AMENDMENT,
    metadata_path: Path = DEFAULT_METADATA,
) -> Mapping[str, object]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing audit: {output_path}")
    protocol, amendment, metadata = _validate_locked_inputs(
        protocol_path,
        amendment_path,
        metadata_path,
    )
    tracks = index_guitarset(dataset_dir)
    if not tracks or any(track.player_id not in ALLOWED_PLAYERS for track in tracks):
        raise AnonymousTargetAuditError("GuitarSet index escaped players 00 through 04")
    train_tracks, validation_tracks = split_tracks_by_group(
        tracks,
        validation_fraction=0.2,
        seed=SEED,
    )
    metadata_split = metadata.get("split")
    if not isinstance(metadata_split, dict) or not isinstance(
        metadata_split.get("validation_members"), list
    ):
        raise AnonymousTargetAuditError("V7 metadata has no validation member list")
    expected_validation = set(metadata_split["validation_members"])
    actual_validation = {
        PurePosixPath(track.annotation_member).name for track in validation_tracks
    }
    if actual_validation != expected_validation:
        raise AnonymousTargetAuditError("generated validation split differs from V7")

    prepared_train = _prepare_tracks(train_tracks)
    prepared_validation = _prepare_tracks(validation_tracks)
    full_splits = {
        "train": audit_full_split(prepared_train),
        "validation": audit_full_split(prepared_validation),
    }
    samplers: Dict[str, MutableMapping[str, object]] = {
        "train": dict(
            audit_sampler(prepared_train, steps=TRAIN_STEPS, seed=SEED)
        ),
        "validation": dict(
            audit_sampler(
                prepared_validation,
                steps=VALIDATION_STEPS,
                seed=SEED + 1,
            )
        ),
    }
    _add_density_comparisons(full_splits["train"], samplers["train"])
    _add_density_comparisons(
        full_splits["validation"], samplers["validation"]
    )

    expected_validation_positives = amendment["known_before_implementation"][
        "v7_fixed_validation_sampled_slot_positive_elements"
    ]
    validation_reproduction = {
        head: {
            "expected": expected_validation_positives[head],
            "actual": samplers["validation"]["heads"][head][
                "historical_v7_slot_binary_window_512"
            ]["positive_elements"],
            "exact": samplers["validation"]["heads"][head][
                "historical_v7_slot_binary_window_512"
            ]["positive_elements"]
            == expected_validation_positives[head],
        }
        for head in HEADS
    }
    source_cycle_reproduction = samplers["validation"]["source_counts"] == {
        "onset_anchor": 134,
        "offset_anchor": 133,
        "random": 133,
    }
    if not all(item["exact"] for item in validation_reproduction.values()):
        raise AnonymousTargetAuditError(
            "fixed V7 validation positive elements were not reproduced"
        )
    if not source_cycle_reproduction:
        raise AnonymousTargetAuditError("fixed V7 validation source cycle changed")

    report: Mapping[str, object] = {
        "schema_version": 1,
        "status": "completed_audit_only",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "public_event_fields": ["type", "position"],
            "multiplicity_encoding": "repeat the identical type-position event k times",
            "model_changed": False,
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
            "metadata": str(metadata_path.resolve()),
            "metadata_sha256": sha256_file(metadata_path),
            "audit_source_sha256": sha256_file(Path(__file__)),
        },
        "data_guard": {
            "selected_players": sorted(ALLOWED_PLAYERS),
            "locked_test_player": "05",
            "locked_test_used": False,
            "validation_members_exactly_match_v7": True,
            "validation_tracks": len(validation_tracks),
            "train_tracks": len(train_tracks),
        },
        "full_stream": full_splits,
        "current_v7_sampler": samplers,
        "integrity": {
            "fixed_validation_source_cycle_exact": source_cycle_reproduction,
            "fixed_validation_slot_positive_elements": validation_reproduction,
            "exact_instance_sums": all(
                full_splits[split_name]["heads"][head]["exact_anonymous_count"][
                    "instance_sum_reconciles"
                ]
                for split_name in ("train", "validation")
                for head in HEADS
            ),
            "slot_permutation_invariance": all(
                full_splits[split_name]["heads"][head]["exact_anonymous_count"][
                    "slot_permutation_invariant"
                ]
                for split_name in ("train", "validation")
                for head in HEADS
            ),
            "causal_inverse_roundtrip": all(
                full_splits[split_name]["heads"][head][
                    "anonymous_count_window_512"
                ]["causal_inverse_roundtrip"]["exact"]
                for split_name in ("train", "validation")
                for head in HEADS
            ),
        },
        "decision": _derive_decision(full_splits, samplers),
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
    parser.add_argument("--metadata", default=str(DEFAULT_METADATA))
    return parser


def main(argv: Sequence[str] = ()) -> int:
    arguments = create_argument_parser().parse_args(argv or None)
    report = run_audit(
        Path(arguments.dataset_dir),
        output_path=Path(arguments.output),
        protocol_path=Path(arguments.protocol),
        amendment_path=Path(arguments.protocol_amendment),
        metadata_path=Path(arguments.metadata),
    )
    print(json.dumps(report["decision"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
