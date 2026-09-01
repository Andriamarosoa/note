"""Deterministic replay selection for V8.2b true V8.1 false positives.

The replay source is the train-only frozen-V8.1 audit.  This module deliberately
contains no acoustic proxy mining: every candidate must be an actual unmatched
onset prediction emitted by V8.1 on the train split.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
from typing import Iterable, Sequence, Tuple


class ReplayError(ValueError):
    pass


@dataclass(frozen=True, order=True)
class ReplayPoint:
    member: str
    sample: int
    arrangement: str
    model_onset_score: float
    harmonic_proxy: bool = False

    def __post_init__(self) -> None:
        if not self.member:
            raise ReplayError("replay member must be non-empty")
        if isinstance(self.sample, bool) or not isinstance(self.sample, int) or self.sample < 0:
            raise ReplayError("replay sample must be an integer >= 0")
        if self.arrangement not in ("comp", "solo"):
            raise ReplayError("replay arrangement must be 'comp' or 'solo'")
        score = float(self.model_onset_score)
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ReplayError("model_onset_score must be a finite probability")
        object.__setattr__(self, "model_onset_score", score)
        object.__setattr__(self, "harmonic_proxy", bool(self.harmonic_proxy))


def load_replay_points(path: Path) -> Tuple[ReplayPoint, ...]:
    """Load and deduplicate actual train false-positive onset positions."""
    with Path(path).open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    scope = payload.get("scope", {})
    if scope.get("player_05_read") is not False:
        raise ReplayError("replay audit must explicitly confirm player_05_read=false")
    records = payload.get("false_positive_records")
    if not isinstance(records, list) or not records:
        raise ReplayError("replay audit contains no false_positive_records")

    by_key = {}
    for record in records:
        if not isinstance(record, dict):
            raise ReplayError("false-positive records must be objects")
        member = str(record.get("member", ""))
        sample = record.get("sample")
        arrangement = str(record.get("arrangement", ""))
        score = record.get("model_onset_score")
        if isinstance(sample, bool) or not isinstance(sample, int):
            raise ReplayError("false-positive sample must be an integer")
        proxy = (
            float(record.get("positive_flux_over_pre_energy", 0.0)) >= 0.50
            and float(record.get("fixed_positive_flux_fraction", 0.0)) >= 0.70
        )
        point = ReplayPoint(member, sample, arrangement, float(score), proxy)
        key = (point.member, point.sample)
        previous = by_key.get(key)
        if previous is None or point.model_onset_score > previous.model_onset_score:
            by_key[key] = point
    return tuple(sorted(by_key.values()))


def arrangement_fraction(points: Sequence[ReplayPoint], arrangement: str = "solo") -> float:
    if not points:
        raise ReplayError("cannot compute arrangement fraction of an empty pool")
    return sum(point.arrangement == arrangement for point in points) / float(len(points))


def select_replay_points(
    points: Sequence[ReplayPoint],
    *,
    count: int,
    seed: int,
    max_per_track: int,
) -> Tuple[ReplayPoint, ...]:
    """Select a unique replay batch while preserving the pool's comp/solo mix.

    Selection is without replacement within an epoch.  The requested arrangement
    counts are derived from the actual replay pool rather than a hand-tuned ratio.
    A per-track cap prevents one performance from dominating the replay batch.
    """
    if count <= 0:
        raise ReplayError("replay count must be positive")
    if max_per_track <= 0:
        raise ReplayError("max_per_track must be positive")
    unique = {(point.member, point.sample): point for point in points}
    pool = list(unique.values())
    if len(pool) < count:
        raise ReplayError(f"replay pool has only {len(pool)} unique points for requested count={count}")

    solo_fraction = arrangement_fraction(pool, "solo")
    targets = {
        "solo": int(round(count * solo_fraction)),
    }
    targets["comp"] = count - targets["solo"]

    rng = random.Random(seed)
    by_arrangement = {
        arrangement: [point for point in pool if point.arrangement == arrangement]
        for arrangement in ("solo", "comp")
    }
    for values in by_arrangement.values():
        rng.shuffle(values)

    selected = []
    per_track = Counter()
    for arrangement in ("solo", "comp"):
        needed = targets[arrangement]
        for point in by_arrangement[arrangement]:
            if len([item for item in selected if item.arrangement == arrangement]) >= needed:
                break
            if per_track[point.member] >= max_per_track:
                continue
            selected.append(point)
            per_track[point.member] += 1
        obtained = sum(item.arrangement == arrangement for item in selected)
        if obtained < needed:
            raise ReplayError(
                f"cannot satisfy {arrangement} replay target {needed} with max_per_track={max_per_track}; obtained {obtained}"
            )

    if len(selected) != count:
        raise ReplayError(f"selected {len(selected)} replay points, expected {count}")
    if len({(point.member, point.sample) for point in selected}) != len(selected):
        raise ReplayError("replay selection unexpectedly contains duplicate positions")
    rng.shuffle(selected)
    return tuple(selected)


def summarize_replay(points: Iterable[ReplayPoint]) -> dict:
    frozen = tuple(points)
    tracks = Counter(point.member for point in frozen)
    arrangements = Counter(point.arrangement for point in frozen)
    return {
        "positions": len(frozen),
        "tracks": len(tracks),
        "arrangement": dict(sorted(arrangements.items())),
        "harmonic_proxy": sum(point.harmonic_proxy for point in frozen),
        "max_per_track": max(tracks.values()) if tracks else 0,
        "top_tracks": tracks.most_common(10),
    }


__all__ = [
    "ReplayError",
    "ReplayPoint",
    "arrangement_fraction",
    "load_replay_points",
    "select_replay_points",
    "summarize_replay",
]
