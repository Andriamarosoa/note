"""Audit explicit end-of-stream control closures without loading a model."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys
from typing import Dict, List, Mapping, Sequence, Tuple


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from causal_note.detector import BoundaryType, LiveEventTracker  # noqa: E402
from causal_note.guitarset import (  # noqa: E402
    ALLOWED_PLAYERS,
    BoundarySlots,
    GuitarSetTrack,
    index_guitarset,
    load_boundary_slots,
)
from scripts.train_boundaries import (  # noqa: E402
    inspect_pcm16_mono_wav,
    split_tracks_by_group,
)


DEFAULT_PROTOCOL = (
    REPOSITORY_ROOT
    / "model"
    / "causal-boundaries-end-of-stream-finalization-protocol.json"
)
DEFAULT_PRIOR_AUDIT = (
    REPOSITORY_ROOT
    / "model"
    / "causal-boundaries-type-position-target-audit.json"
)
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT
    / "model"
    / "causal-boundaries-end-of-stream-finalization-audit.json"
)
EXPECTED_PROTOCOL_SHA256 = (
    "68DD2207BB3F84A9688750613D26ADE8CFE3CAC852832E0879408EF232003E41"
)
EXPECTED_PRIOR_AUDIT_SHA256 = (
    "D1B311666BAFE8191D6CE3B655EE6CFB8B140F053741B7DF9D98613D0E93CA14"
)
SEED = 1337
VALIDATION_FRACTION = 0.2


class EndOfStreamAuditError(ValueError):
    """Raised when the locked audit or oracle replay is inconsistent."""


@dataclass(frozen=True)
class OracleTrackResult:
    member: str
    player: str
    arrangement: str
    frame_count: int
    notes: int
    internal_reference_offsets: int
    terminal_reference_offsets: int
    open_events_before_finalization: int
    terminal_control_offsets_emitted: int
    missing_terminal_offsets: int
    extra_terminal_offsets: int
    open_events_after_finalization: int
    last_real_sample_active_terminal_notes: int


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
        raise EndOfStreamAuditError(f"cannot read JSON: {path}") from exc
    if not isinstance(value, dict):
        raise EndOfStreamAuditError(f"JSON root must be an object: {path}")
    return value


def _arrangement(member: str) -> str:
    stem = PurePosixPath(member).stem
    if stem.endswith("_comp"):
        return "comp"
    if stem.endswith("_solo"):
        return "solo"
    return "other"


def replay_oracle_track(
    slots: BoundarySlots,
    *,
    frame_count: int,
    member: str,
    player: str,
) -> OracleTrackResult:
    """Replay annotated identity, then close remaining events at exclusive EOF."""

    boundaries: List[Tuple[int, int, int, int]] = []
    notes_by_identity = {}
    for slot_index, notes in enumerate(slots):
        for note_index, note in enumerate(notes):
            identity = (slot_index, note_index)
            notes_by_identity[identity] = note
            boundaries.append((note.onset_sample, 1, slot_index, note_index))
            if note.offset_sample < frame_count:
                boundaries.append((note.offset_sample, 0, slot_index, note_index))
            elif note.offset_sample != frame_count:
                raise EndOfStreamAuditError(
                    f"offset exceeds EOF in {member!r}: {note.offset_sample}"
                )

    tracker = LiveEventTracker(event_prefix="oracle")
    event_id_by_identity: Dict[Tuple[int, int], str] = {}
    for sample, kind_order, slot_index, note_index in sorted(boundaries):
        identity = (slot_index, note_index)
        if kind_order == 0:
            event_id = event_id_by_identity.get(identity)
            if event_id is None:
                raise EndOfStreamAuditError(
                    f"oracle offset precedes its onset in {member!r}"
                )
            tracker.finish_event(event_id, sample)
        else:
            event = tracker.start_event(sample)
            event_id_by_identity[identity] = event.event_id

    terminal_identities = {
        identity
        for identity, note in notes_by_identity.items()
        if note.offset_sample == frame_count
    }
    expected_terminal_ids = {
        event_id_by_identity[identity] for identity in terminal_identities
    }
    active_before = tracker.active_events()
    active_before_ids = {event.event_id for event in active_before}
    if active_before_ids != expected_terminal_ids:
        raise EndOfStreamAuditError(
            f"oracle open set differs from EOF references in {member!r}"
        )

    terminal_events = tracker.finish_all(frame_count)
    emitted_ids = {event.event_id for event in terminal_events}
    if len(emitted_ids) != len(terminal_events):
        raise EndOfStreamAuditError(
            f"terminal finalization duplicated an event ID in {member!r}"
        )
    if any(
        event.kind is not BoundaryType.OFFSET or event.sample != frame_count
        for event in terminal_events
    ):
        raise EndOfStreamAuditError(
            f"terminal event has the wrong type or position in {member!r}"
        )

    last_sample_active = sum(
        int(note.onset_sample <= frame_count - 1 < note.offset_sample)
        for identity, note in notes_by_identity.items()
        if identity in terminal_identities
    )
    internal_offsets = sum(
        int(note.offset_sample < frame_count) for note in notes_by_identity.values()
    )
    missing = len(expected_terminal_ids - emitted_ids)
    extra = len(emitted_ids - expected_terminal_ids)
    return OracleTrackResult(
        member=PurePosixPath(member).name,
        player=player,
        arrangement=_arrangement(member),
        frame_count=frame_count,
        notes=len(notes_by_identity),
        internal_reference_offsets=internal_offsets,
        terminal_reference_offsets=len(terminal_identities),
        open_events_before_finalization=len(active_before),
        terminal_control_offsets_emitted=len(terminal_events),
        missing_terminal_offsets=missing,
        extra_terminal_offsets=extra,
        open_events_after_finalization=len(tracker.active_events()),
        last_real_sample_active_terminal_notes=last_sample_active,
    )


def _prepare_and_replay(tracks: Sequence[GuitarSetTrack]) -> Tuple[OracleTrackResult, ...]:
    results = []
    for track in tracks:
        if track.player_id not in ALLOWED_PLAYERS or track.player_id == "05":
            raise EndOfStreamAuditError("Player05 content must remain locked")
        slots = load_boundary_slots(track.annotation_zip, track.annotation_member)
        info = inspect_pcm16_mono_wav(track.audio_zip, track.audio_member)
        results.append(
            replay_oracle_track(
                slots,
                frame_count=info.frame_count,
                member=track.annotation_member,
                player=track.player_id,
            )
        )
    return tuple(results)


def _histogram(values: Sequence[int]) -> Mapping[str, int]:
    counts = Counter(values)
    return {str(key): counts[key] for key in sorted(counts)}


def aggregate_split(results: Sequence[OracleTrackResult]) -> Mapping[str, object]:
    terminal_multiplicities = [
        item.terminal_reference_offsets
        for item in results
        if item.terminal_reference_offsets
    ]
    tracks_with_terminal = sum(
        int(item.terminal_reference_offsets > 0) for item in results
    )
    return {
        "tracks": len(results),
        "players": dict(Counter(item.player for item in results)),
        "tracks_with_terminal_offsets": tracks_with_terminal,
        "tracks_without_terminal_offsets": len(results) - tracks_with_terminal,
        "affected_tracks_by_arrangement": dict(
            Counter(
                item.arrangement
                for item in results
                if item.terminal_reference_offsets
            )
        ),
        "notes": sum(item.notes for item in results),
        "internal_reference_offsets": sum(
            item.internal_reference_offsets for item in results
        ),
        "terminal_reference_offsets": sum(
            item.terminal_reference_offsets for item in results
        ),
        "raw_reference_offsets": sum(item.notes for item in results),
        "terminal_multiplicity_per_affected_track": _histogram(
            terminal_multiplicities
        ),
        "maximum_terminal_multiplicity": max(terminal_multiplicities, default=0),
        "before": {
            "open_events_at_eof": sum(
                item.open_events_before_finalization for item in results
            ),
            "acoustic_offset_targets": sum(
                item.internal_reference_offsets for item in results
            ),
        },
        "after": {
            "terminal_control_offsets_emitted": sum(
                item.terminal_control_offsets_emitted for item in results
            ),
            "missing_terminal_offsets": sum(
                item.missing_terminal_offsets for item in results
            ),
            "extra_terminal_offsets": sum(item.extra_terminal_offsets for item in results),
            "open_events_after_finalization": sum(
                item.open_events_after_finalization for item in results
            ),
            "total_offsets_acoustic_plus_terminal_control": sum(
                item.internal_reference_offsets
                + item.terminal_control_offsets_emitted
                for item in results
            ),
        },
        "last_real_sample_active_terminal_notes": sum(
            item.last_real_sample_active_terminal_notes for item in results
        ),
        "all_tracks_exact": all(
            item.open_events_before_finalization
            == item.terminal_reference_offsets
            == item.terminal_control_offsets_emitted
            == item.last_real_sample_active_terminal_notes
            and item.missing_terminal_offsets == 0
            and item.extra_terminal_offsets == 0
            and item.open_events_after_finalization == 0
            for item in results
        ),
    }


def _expected_histogram(protocol: Mapping[str, object], split_name: str) -> Mapping[str, int]:
    split = protocol["data"][split_name]
    return {
        str(key): int(value)
        for key, value in split["terminal_multiplicity_per_affected_track"].items()
        if int(value)
    }


def _validate_split_against_protocol(
    name: str,
    report: Mapping[str, object],
    protocol: Mapping[str, object],
) -> None:
    expected = protocol["data"][name]
    checks = {
        "tracks": expected["tracks"],
        "tracks_with_terminal_offsets": expected["tracks_with_terminal_offsets"],
        "internal_reference_offsets": expected["internal_offsets"],
        "terminal_reference_offsets": expected["terminal_offsets"],
    }
    for key, expected_value in checks.items():
        if report[key] != expected_value:
            raise EndOfStreamAuditError(
                f"{name}.{key} differs: {report[key]} != {expected_value}"
            )
    if report["terminal_multiplicity_per_affected_track"] != _expected_histogram(
        protocol, name
    ):
        raise EndOfStreamAuditError(
            f"{name} terminal multiplicity differs from the protocol"
        )


def run_audit(
    dataset_dir: Path,
    *,
    output_path: Path,
    protocol_path: Path = DEFAULT_PROTOCOL,
    prior_audit_path: Path = DEFAULT_PRIOR_AUDIT,
) -> Mapping[str, object]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing audit: {output_path}")
    if sha256_file(protocol_path) != EXPECTED_PROTOCOL_SHA256:
        raise EndOfStreamAuditError("end-of-stream protocol SHA-256 changed")
    if sha256_file(prior_audit_path) != EXPECTED_PRIOR_AUDIT_SHA256:
        raise EndOfStreamAuditError("anonymous target audit SHA-256 changed")
    protocol = _load_json(protocol_path)
    prior_audit = _load_json(prior_audit_path)

    tracks = index_guitarset(dataset_dir)
    if not tracks or any(track.player_id not in ALLOWED_PLAYERS for track in tracks):
        raise EndOfStreamAuditError("GuitarSet index escaped players 00 through 04")
    train_tracks, validation_tracks = split_tracks_by_group(
        tracks,
        validation_fraction=VALIDATION_FRACTION,
        seed=SEED,
    )
    train_results = _prepare_and_replay(train_tracks)
    validation_results = _prepare_and_replay(validation_tracks)
    split_reports = {
        "train": aggregate_split(train_results),
        "validation": aggregate_split(validation_results),
    }
    for split_name, split_report in split_reports.items():
        _validate_split_against_protocol(split_name, split_report, protocol)

    prior_train_offset = prior_audit["full_stream"]["train"]["heads"]["offset"]
    prior_validation_offset = prior_audit["full_stream"]["validation"]["heads"][
        "offset"
    ]
    acoustic_targets_unchanged = (
        split_reports["train"]["internal_reference_offsets"]
        == prior_train_offset["exact_anonymous_count"]["event_instances"]
        and split_reports["validation"]["internal_reference_offsets"]
        == prior_validation_offset["exact_anonymous_count"]["event_instances"]
    )
    raw_references_unchanged = (
        split_reports["train"]["raw_reference_offsets"]
        == prior_train_offset["raw_annotations"]["event_instances"]
        and split_reports["validation"]["raw_reference_offsets"]
        == prior_validation_offset["raw_annotations"]["event_instances"]
    )

    total_terminal = sum(
        split_reports[name]["terminal_reference_offsets"]
        for name in ("train", "validation")
    )
    total_open_before = sum(
        split_reports[name]["before"]["open_events_at_eof"]
        for name in ("train", "validation")
    )
    total_emitted = sum(
        split_reports[name]["after"]["terminal_control_offsets_emitted"]
        for name in ("train", "validation")
    )
    total_missing = sum(
        split_reports[name]["after"]["missing_terminal_offsets"]
        for name in ("train", "validation")
    )
    total_extra = sum(
        split_reports[name]["after"]["extra_terminal_offsets"]
        for name in ("train", "validation")
    )
    total_open_after = sum(
        split_reports[name]["after"]["open_events_after_finalization"]
        for name in ("train", "validation")
    )
    accepted = (
        all(split_reports[name]["all_tracks_exact"] for name in split_reports)
        and acoustic_targets_unchanged
        and raw_references_unchanged
        and total_terminal == total_open_before == total_emitted == 408
        and total_missing == total_extra == total_open_after == 0
    )

    report: Mapping[str, object] = {
        "schema_version": 1,
        "status": "completed_audit_only" if accepted else "failed_audit",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "classification": "end-of-stream control finalization",
            "acoustic_prediction": False,
            "acoustic_targets_changed": False,
            "model_changed": False,
            "training_run": False,
            "sampler_changed": False,
            "loss_changed": False,
            "player_05_content_opened": False,
        },
        "locked_inputs": {
            "protocol": str(protocol_path.resolve()),
            "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
            "prior_anonymous_target_audit": str(prior_audit_path.resolve()),
            "prior_anonymous_target_audit_sha256": EXPECTED_PRIOR_AUDIT_SHA256,
            "audit_source_sha256": sha256_file(Path(__file__)),
            "runtime_source_sha256": {
                relative: sha256_file(REPOSITORY_ROOT / relative)
                for relative in (
                    "src/causal_note/detector.py",
                    "src/causal_note/scheduler.py",
                    "src/causal_note/pipeline.py",
                    "scripts/detect_live.py",
                )
            },
        },
        "data_guard": {
            "players": sorted(ALLOWED_PLAYERS),
            "locked_test_player": "05",
            "player_05_content_opened": False,
            "train_tracks": len(train_tracks),
            "validation_tracks": len(validation_tracks),
        },
        "splits": split_reports,
        "total": {
            "tracks": len(train_results) + len(validation_results),
            "tracks_with_terminal_offsets": sum(
                split_reports[name]["tracks_with_terminal_offsets"]
                for name in split_reports
            ),
            "terminal_reference_offsets": total_terminal,
            "open_events_before_finalization": total_open_before,
            "terminal_control_offsets_emitted": total_emitted,
            "missing_terminal_offsets": total_missing,
            "extra_terminal_offsets": total_extra,
            "open_events_after_finalization": total_open_after,
        },
        "before_after": {
            "before": {
                "acoustic_offsets": sum(
                    split_reports[name]["internal_reference_offsets"]
                    for name in split_reports
                ),
                "terminal_references_not_on_sample_grid": total_terminal,
                "oracle_events_left_open": total_open_before,
            },
            "after": {
                "acoustic_offsets_unchanged": sum(
                    split_reports[name]["internal_reference_offsets"]
                    for name in split_reports
                ),
                "terminal_control_offsets": total_emitted,
                "oracle_events_left_open": total_open_after,
            },
        },
        "integrity": {
            "all_tracks_exact": all(
                split_reports[name]["all_tracks_exact"] for name in split_reports
            ),
            "acoustic_targets_unchanged_from_experiment_10": acoustic_targets_unchanged,
            "raw_references_unchanged_from_experiment_10": raw_references_unchanged,
            "terminal_closures_are_reported_separately": True,
        },
        "decision": {
            "oracle_policy_accepted": accepted,
            "terminal_policy": "close every still-open eventId at exclusive EOF",
            "classification": "control finalization, not acoustic detection",
            "runtime_contract_status": "requires separate test-suite evidence",
            "terminal_restart_frames_by_policy": 0,
            "continuous_live_before_eof": "not measured by this data-only oracle replay",
            "training_ready": False,
            "training_started": False,
            "next_step_requires_user_approval": True,
            "real_prediction_limit": (
                "a false predicted onset still open at EOF is also closed; "
                "oracle exactness is not predicted-association accuracy"
            ),
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
    parser.add_argument("--prior-audit", default=str(DEFAULT_PRIOR_AUDIT))
    return parser


def main(argv: Sequence[str] = ()) -> int:
    arguments = create_argument_parser().parse_args(argv or None)
    report = run_audit(
        Path(arguments.dataset_dir),
        output_path=Path(arguments.output),
        protocol_path=Path(arguments.protocol),
        prior_audit_path=Path(arguments.prior_audit),
    )
    print(json.dumps(report["decision"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
