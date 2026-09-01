"""Audit the exact V8.3 control/replay training distributions without training.

The V8.3 experiment matched total supervision mass per head, but replay still
collapsed offset precision.  This audit reconstructs the exact three epoch
samples and asks whether the *composition* of that mass changed: positive vs
negative offset mass, arrangement, active-note state, recent/future boundaries,
and the acoustic profile of selected frozen-V8.1 false positives.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import sys
from typing import Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from causal_note.guitarset import ALLOWED_PLAYERS, SAMPLE_RATE, index_guitarset
from causal_note.v81_targets import (
    DEFAULT_OFFSET_HORIZON_SAMPLES,
    DEFAULT_ONSET_HORIZON_SAMPLES,
)
from causal_note.v8_model import calculate_receptive_field
from causal_note.v82b_replay import load_replay_points
from scripts.audit_anonymous_boundary_targets import _prepare_tracks
from scripts.train_boundaries import TrainingDataError, split_tracks_by_group
from scripts.train_v81 import (
    BagExample,
    _burst_maps,
    _draw_examples,
    _head_supervision,
    _prepare_burst_tracks,
)
from scripts.train_v82b import _clean_replay_pool
from scripts.train_v82c_ab import (
    DEFAULT_MAX_REPLAY_PER_TRACK,
    DEFAULT_NEGATIVE_MARGIN,
    DEFAULT_REPLAY_FRACTION,
    DEFAULT_TRAIN_EXAMPLES,
    _draw_replay_examples,
)

RECEPTIVE_FIELD = calculate_receptive_field()
MAXIMUM_HORIZON = max(DEFAULT_ONSET_HORIZON_SAMPLES, DEFAULT_OFFSET_HORIZON_SAMPLES)


def _ms(samples: int | None) -> float | None:
    return None if samples is None else float(samples) * 1000.0 / SAMPLE_RATE


def _arrangement(member: str) -> str:
    name = Path(member).name
    return "solo" if "_solo" in name else "comp"


def _previous_distance(positions: Sequence[int], sample: int) -> int | None:
    index = bisect_left(positions, sample)
    if index <= 0:
        return None
    return sample - int(positions[index - 1])


def _next_distance(positions: Sequence[int], sample: int) -> int | None:
    index = bisect_left(positions, sample)
    if index >= len(positions):
        return None
    return int(positions[index]) - sample


def _active_note_count(item, sample: int) -> int:
    return sum(
        1
        for slot in item.audit_track.slots
        for note in slot
        if note.onset_sample <= sample < note.offset_sample
    )


def _bin_age_ms(value: float | None) -> str:
    if value is None:
        return "none"
    if value <= 5.0:
        return "0_5"
    if value <= 20.0:
        return "5_20"
    if value <= 50.0:
        return "20_50"
    if value <= 100.0:
        return "50_100"
    return "gt_100"


def _quantiles(values: Iterable[float]) -> Mapping[str, float | int | None]:
    finite = sorted(float(v) for v in values if v is not None and math.isfinite(float(v)))
    if not finite:
        return {"n": 0, "p10": None, "p50": None, "p90": None, "mean": None}

    def q(fraction: float) -> float:
        if len(finite) == 1:
            return finite[0]
        position = fraction * (len(finite) - 1)
        low = int(math.floor(position))
        high = int(math.ceil(position))
        if low == high:
            return finite[low]
        weight = position - low
        return finite[low] * (1.0 - weight) + finite[high] * weight

    return {
        "n": len(finite),
        "p10": q(0.10),
        "p50": q(0.50),
        "p90": q(0.90),
        "mean": sum(finite) / len(finite),
    }


def _row_record(tracks, example: BagExample, *, replay_contract: bool) -> dict:
    item = tracks[example.track_index]
    maps = _burst_maps(item)
    onset = _head_supervision(
        item,
        maps,
        kind="onset",
        position=example.position,
        negative_margin=DEFAULT_NEGATIVE_MARGIN,
    )
    offset = _head_supervision(
        item,
        maps,
        kind="offset",
        position=example.position,
        negative_margin=DEFAULT_NEGATIVE_MARGIN,
    )

    is_replay = example.stratum == "true_fp_replay"
    if replay_contract and is_replay:
        onset = dict(onset)
        onset.update(
            presence=0.0,
            mass=0.0,
            delay=0.0,
            count=0,
            presence_weight=1.0,
            mass_weight=1.0,
            delay_weight=0.0,
            count_weight=0.0,
        )

    previous_onset = _previous_distance(item.onset_positions, example.position)
    next_onset = _next_distance(item.onset_positions, example.position)
    previous_offset = _previous_distance(item.offset_positions, example.position)
    next_offset = _next_distance(item.offset_positions, example.position)

    return {
        "member": str(item.audit_track.track.annotation_member),
        "sample": example.position,
        "stratum": example.stratum,
        "arrangement": _arrangement(str(item.audit_track.track.annotation_member)),
        "active_note_count": _active_note_count(item, example.position),
        "previous_onset_ms": _ms(previous_onset),
        "next_onset_ms": _ms(next_onset),
        "previous_offset_ms": _ms(previous_offset),
        "next_offset_ms": _ms(next_offset),
        "previous_onset_within_rf": previous_onset is not None and previous_onset <= RECEPTIVE_FIELD - 1,
        "previous_offset_within_rf": previous_offset is not None and previous_offset <= RECEPTIVE_FIELD - 1,
        "onset": onset,
        "offset": offset,
        "is_replay": is_replay,
    }


def _presence_state(supervision: Mapping[str, float]) -> str:
    if not supervision["presence_weight"]:
        return "ambiguous"
    return "positive" if supervision["presence"] >= 0.5 else "negative"


def _summarize_rows(rows: Sequence[dict]) -> dict:
    strata = Counter(row["stratum"] for row in rows)
    arrangements = Counter(row["arrangement"] for row in rows)
    active = Counter(str(row["active_note_count"]) for row in rows)
    result = {
        "rows": len(rows),
        "unique_anchors": len({(row["member"], row["sample"]) for row in rows}),
        "strata": dict(sorted(strata.items())),
        "arrangement": dict(sorted(arrangements.items())),
        "active_note_count": dict(sorted(active.items(), key=lambda item: int(item[0]))),
        "recent_boundary_context": {
            "previous_onset_within_rf": sum(row["previous_onset_within_rf"] for row in rows),
            "previous_offset_within_rf": sum(row["previous_offset_within_rf"] for row in rows),
            "previous_onset_age_ms": _quantiles(row["previous_onset_ms"] for row in rows),
            "next_onset_distance_ms": _quantiles(row["next_onset_ms"] for row in rows),
            "previous_offset_age_ms": _quantiles(row["previous_offset_ms"] for row in rows),
            "next_offset_distance_ms": _quantiles(row["next_offset_ms"] for row in rows),
        },
        "presence_state": {},
    }
    for kind in ("onset", "offset"):
        result["presence_state"][kind] = dict(
            sorted(Counter(_presence_state(row[kind]) for row in rows).items())
        )
    return result


def _presence_mass(rows: Sequence[dict], kind: str) -> dict:
    raw = {"positive": 0.0, "negative": 0.0, "ambiguous": 0.0}
    for row in rows:
        sup = row[kind]
        weight = float(sup["presence_weight"])
        if not weight:
            raw["ambiguous"] += 1.0
        elif sup["presence"] >= 0.5:
            raw["positive"] += weight
        else:
            raw["negative"] += weight
    raw["total_supervised"] = raw["positive"] + raw["negative"]
    raw["positive_fraction"] = (
        raw["positive"] / raw["total_supervised"] if raw["total_supervised"] else 0.0
    )
    return raw


def _mass_comparison(control_rows: Sequence[dict], replay_rows: Sequence[dict], kind: str) -> dict:
    control = _presence_mass(control_rows, kind)
    replay = _presence_mass(replay_rows, kind)
    scale = (
        control["total_supervised"] / replay["total_supervised"]
        if replay["total_supervised"]
        else 1.0
    )
    matched_positive = replay["positive"] * scale
    matched_negative = replay["negative"] * scale
    return {
        "control": control,
        "replay_raw": replay,
        "replay_mass_scale": scale,
        "replay_matched": {
            "positive": matched_positive,
            "negative": matched_negative,
            "total_supervised": matched_positive + matched_negative,
            "positive_fraction": (
                matched_positive / (matched_positive + matched_negative)
                if matched_positive + matched_negative
                else 0.0
            ),
        },
        "matched_delta": {
            "positive": matched_positive - control["positive"],
            "negative": matched_negative - control["negative"],
            "positive_fraction": (
                replay["positive_fraction"] - control["positive_fraction"]
            ),
        },
    }


def _replay_focus(rows: Sequence[dict]) -> dict:
    selected = [row for row in rows if row["is_replay"]]
    prior_onset_bins = Counter(_bin_age_ms(row["previous_onset_ms"]) for row in selected)
    next_offset_bins = Counter(_bin_age_ms(row["next_offset_ms"]) for row in selected)
    return {
        **_summarize_rows(selected),
        "previous_onset_age_bins_ms": dict(sorted(prior_onset_bins.items())),
        "next_offset_distance_bins_ms": dict(sorted(next_offset_bins.items())),
        "recent_onset_tail": {
            "within_20ms": sum(
                row["previous_onset_ms"] is not None and row["previous_onset_ms"] <= 20.0
                for row in selected
            ),
            "within_50ms": sum(
                row["previous_onset_ms"] is not None and row["previous_onset_ms"] <= 50.0
                for row in selected
            ),
            "within_100ms": sum(
                row["previous_onset_ms"] is not None and row["previous_onset_ms"] <= 100.0
                for row in selected
            ),
        },
        "forward_onset_contract_conflicts": sum(
            _presence_state(row["onset"]) != "negative" for row in selected
        ),
        "offset_state_on_replay": dict(
            sorted(Counter(_presence_state(row["offset"]) for row in selected).items())
        ),
    }


def _acoustic_join(audit_path: Path, selected_keys: set[tuple[str, int]]) -> dict:
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    false_records = payload.get("false_positive_records", [])
    controls = payload.get("control_records", [])
    by_key = {
        (Path(str(row["member"])).name, int(row["sample"])): row
        for row in false_records
    }
    selected = [by_key[key] for key in selected_keys if key in by_key]
    selected_fp_keys = {(Path(str(row["member"])).name, int(row["sample"])) for row in selected}
    matched_controls = [
        row
        for row in controls
        if (Path(str(row["member"])).name, int(row.get("matched_fp_sample", -1))) in selected_fp_keys
    ]
    fields = (
        "model_onset_score",
        "active_note_count",
        "fixed_positive_flux_fraction",
        "fixed_positive_flux_enrichment",
        "contour_positive_flux_fraction",
        "contour_positive_flux_enrichment",
        "positive_flux_over_pre_energy",
        "rms_delta_db",
    )
    return {
        "selected_unique_records": len(selected),
        "matched_control_records": len(matched_controls),
        "selected": {field: _quantiles(row.get(field) for row in selected) for field in fields},
        "matched_controls": {
            field: _quantiles(row.get(field) for row in matched_controls) for field in fields
        },
    }


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit exact V8.3 control/replay training data.")
    parser.add_argument("dataset_dir", nargs="?", type=Path, default=ROOT / "data" / "GuitarSet")
    parser.add_argument("--replay-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "model" / "v83-training-data-audit.json")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train-examples", type=int, default=DEFAULT_TRAIN_EXAMPLES)
    parser.add_argument("--replay-fraction", type=float, default=DEFAULT_REPLAY_FRACTION)
    parser.add_argument("--max-replay-per-track", type=int, default=DEFAULT_MAX_REPLAY_PER_TRACK)
    return parser


def main(argv=None) -> int:
    args = create_argument_parser().parse_args(argv)
    if not args.replay_audit.is_file():
        raise TrainingDataError("replay audit does not exist")

    indexed = tuple(
        track for track in index_guitarset(args.dataset_dir)
        if track.player_id in ALLOWED_PLAYERS
    )
    train_tracks, validation_tracks = split_tracks_by_group(
        indexed, validation_fraction=0.2, seed=args.seed
    )
    train = _prepare_burst_tracks(_prepare_tracks(train_tracks))
    raw_replay = load_replay_points(args.replay_audit)
    replay_pool = _clean_replay_pool(
        train,
        raw_replay,
        maximum_horizon=MAXIMUM_HORIZON,
        negative_margin=DEFAULT_NEGATIVE_MARGIN,
    )

    epochs = []
    all_selected_keys: set[tuple[str, int]] = set()
    aggregate_control = []
    aggregate_replay = []
    for epoch in range(1, args.epochs + 1):
        epoch_seed = args.seed + epoch * 10007
        control_examples = _draw_examples(
            train,
            count=args.train_examples,
            seed=epoch_seed,
            maximum_horizon=MAXIMUM_HORIZON,
            negative_margin=DEFAULT_NEGATIVE_MARGIN,
        )
        replay_examples, selected_replay = _draw_replay_examples(
            train,
            replay_pool,
            count=args.train_examples,
            seed=epoch_seed,
            replay_fraction=args.replay_fraction,
            maximum_horizon=MAXIMUM_HORIZON,
            negative_margin=DEFAULT_NEGATIVE_MARGIN,
            max_replay_per_track=args.max_replay_per_track,
        )
        control_rows = [_row_record(train, example, replay_contract=False) for example in control_examples]
        replay_rows = [_row_record(train, example, replay_contract=True) for example in replay_examples]
        aggregate_control.extend(control_rows)
        aggregate_replay.extend(replay_rows)
        for point in selected_replay:
            all_selected_keys.add((Path(point.member).name, int(point.sample)))

        epochs.append(
            {
                "epoch": epoch,
                "seed": epoch_seed,
                "control": _summarize_rows(control_rows),
                "replay": _summarize_rows(replay_rows),
                "presence_mass": {
                    kind: _mass_comparison(control_rows, replay_rows, kind)
                    for kind in ("onset", "offset")
                },
                "replay_focus": _replay_focus(replay_rows),
            }
        )

    report = {
        "schema_version": 1,
        "experiment": "v83_training_data_distribution_audit",
        "configuration": {
            "seed": args.seed,
            "epochs": args.epochs,
            "train_examples": args.train_examples,
            "replay_fraction": args.replay_fraction,
            "max_replay_per_track": args.max_replay_per_track,
            "negative_margin_samples": DEFAULT_NEGATIVE_MARGIN,
            "onset_horizon_samples": DEFAULT_ONSET_HORIZON_SAMPLES,
            "offset_horizon_samples": DEFAULT_OFFSET_HORIZON_SAMPLES,
            "receptive_field_samples": RECEPTIVE_FIELD,
            "receptive_field_ms": _ms(RECEPTIVE_FIELD - 1),
            "train_tracks": len(train),
            "validation_tracks": len(validation_tracks),
        },
        "raw_replay_positions": len(raw_replay),
        "clean_replay_positions": len(replay_pool),
        "selected_unique_replay_positions": len(all_selected_keys),
        "aggregate": {
            "control": _summarize_rows(aggregate_control),
            "replay": _summarize_rows(aggregate_replay),
            "presence_mass": {
                kind: _mass_comparison(aggregate_control, aggregate_replay, kind)
                for kind in ("onset", "offset")
            },
            "replay_focus": _replay_focus(aggregate_replay),
        },
        "acoustic_selected_replay_vs_matched_controls": _acoustic_join(
            args.replay_audit, all_selected_keys
        ),
        "epochs": epochs,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    agg = report["aggregate"]
    print("V8.3 DATA AUDIT")
    print("clean replay:", report["clean_replay_positions"], "selected unique:", report["selected_unique_replay_positions"])
    for kind in ("onset", "offset"):
        mass = agg["presence_mass"][kind]
        print(kind, "control positive fraction", mass["control"]["positive_fraction"],
              "replay positive fraction", mass["replay_matched"]["positive_fraction"],
              "matched positive delta", mass["matched_delta"]["positive"],
              "matched negative delta", mass["matched_delta"]["negative"])
    focus = agg["replay_focus"]
    print("replay arrangement:", focus["arrangement"])
    print("replay previous onset bins:", focus["previous_onset_age_bins_ms"])
    print("replay recent onset tail:", focus["recent_onset_tail"])
    print("replay offset supervision:", focus["offset_state_on_replay"])
    print("forward onset contract conflicts:", focus["forward_onset_contract_conflicts"])
    acoustic = report["acoustic_selected_replay_vs_matched_controls"]
    for field in ("fixed_positive_flux_fraction", "fixed_positive_flux_enrichment", "positive_flux_over_pre_energy", "rms_delta_db"):
        print("acoustic", field,
              "replay p50", acoustic["selected"][field]["p50"],
              "control p50", acoustic["matched_controls"][field]["p50"])
    print("output:", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
