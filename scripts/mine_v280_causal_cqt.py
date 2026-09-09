"""Mine and verify label-free V28 causal-CQT track caches.

This stage is deliberately acoustic-only.  It reads the canonical V10 cache
bundle solely to recover the 240 outer-clean ``track_members`` identities; it
never deserializes V10 targets, exact counts, string slots, or candidate rows.
GuitarSet JAMS payloads are likewise never opened.  Historical validation
member names are indexed only to prove exclusion, and validation audio is not
decoded.

The output is one independently checksummed float16 cache per track plus an
atomic progress checkpoint and a complete manifest.  ``--resume`` validates
every existing track before continuing, so a failed single-worker Actions job
can be resumed without silently accepting a partial or mismatched cache.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import time
from typing import Iterable, Optional, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for _path in (ROOT, SRC):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from causal_note.guitarset import ALLOWED_PLAYERS, GuitarSetTrack, index_guitarset
from causal_note.v280_causal_cqt import (
    IMPLEMENTATION,
    CausalCQTConfig,
    causal_cqt,
    read_track_cache,
    write_track_cache,
)
from scripts.train_boundaries import decode_pcm16_mono_wav, group_stem, split_tracks_by_group


MANIFEST_SCHEMA_VERSION = 1
PROGRESS_SCHEMA_VERSION = 1
V100_CACHE_SCHEMA_VERSION = 1
EXPECTED_V100_SHARD_COUNT = 8
EXPECTED_INDEXED_TRACK_COUNT = 300
EXPECTED_OUTER_CLEAN_TRACK_COUNT = 240
EXPECTED_HISTORICAL_VALIDATION_TRACK_COUNT = 60
SPLIT_SEED = 1337
VALIDATION_FRACTION = 0.20

V100_ARRAY_NAMES = frozenset(
    {
        "schema_version",
        "spectral",
        "sequence",
        "mask",
        "stats",
        "target",
        "exact",
        "members",
        "top_samples",
        "slot_targets",
        "track_members",
    }
)
V100_ARRAYS_DESERIALIZED = ("schema_version", "track_members")
V100_SUPERVISION_ARRAYS_FORBIDDEN = (
    "exact",
    "members",
    "slot_targets",
    "target",
    "top_samples",
)

V104_SOURCE_RUN_ID = 33_647_694_565
V104_SOURCE_HEAD_SHA = "345dea1e92f6281583c58590a0b1286ce4df459b"
V104_SOURCE_ARTIFACT = "v104-nested-spectral-bundle"
V104_SOURCE_ARTIFACT_DIGEST = (
    "sha256:d54768cae40a9cf9b2c97a1be58fb6f52a681af3d79b79b148b5c92a427a9d99"
)
V272_SOURCE_RUN_ID = 34_287_254_337
V272_SOURCE_HEAD_SHA = "d11f73079a9eb12b2ebce3c8d1df8209b0bd6bdf"
V272_SOURCE_ARTIFACT = "v272-verified-guitarset"
V272_SOURCE_ARTIFACT_DIGEST = (
    "sha256:e90bd513877f8592c7f749410167aa581feb7a4b5d3250f12c980a3b8dc74587"
)
GUITARSET_ANNOTATION_MD5 = "b39b78e63d3446f2e54ddb7a54df9b10"
GUITARSET_AUDIO_MD5 = "aecce79f425a44e2055e46f680e10f6a"


class V280CacheMiningError(RuntimeError):
    """Raised when the V28 cache-mining protocol is violated."""


@dataclass(frozen=True)
class V100Membership:
    track_members: tuple[str, ...]
    shard_paths: tuple[str, ...]
    sha256: str


@dataclass(frozen=True)
class OuterCleanSelection:
    tracks: tuple[GuitarSetTrack, ...]
    indexed_track_count: int
    validation_track_count: int
    membership_sha256: str
    train_groups: tuple[str, ...]
    validation_groups: tuple[str, ...]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _member_digest(members: Iterable[str]) -> str:
    canonical = json.dumps(tuple(sorted(members)), ensure_ascii=True, separators=(",", ":"))
    return _sha256_bytes(canonical.encode("utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def track_cache_filename(annotation_member: str) -> str:
    """Map an annotation identity to a deterministic path-safe cache name."""
    if not isinstance(annotation_member, str) or not annotation_member:
        raise V280CacheMiningError("annotation member must be a non-empty string")
    if "\x00" in annotation_member:
        raise V280CacheMiningError("annotation member contains NUL")
    digest = _sha256_bytes(annotation_member.encode("utf-8"))[:24]
    return f"track-{digest}.npz"


def _validated_member_array(array: np.ndarray, source: Path) -> tuple[str, ...]:
    values = np.asarray(array)
    if values.ndim != 1 or values.dtype.kind != "U":
        raise V280CacheMiningError(f"track_members must be a 1-D Unicode array in {source}")
    members = tuple(str(value) for value in values)
    if not members:
        raise V280CacheMiningError(f"empty track_members in {source}")
    for member in members:
        name = PurePosixPath(member).name
        if not member or "\x00" in member or not re.fullmatch(r"\d{2}_.+\.jams", name):
            raise V280CacheMiningError(f"invalid V10 track member {member!r} in {source}")
        if name[:2] not in ALLOWED_PLAYERS:
            raise V280CacheMiningError(f"forbidden GuitarSet player in {member!r}")
    if len(set(members)) != len(members):
        raise V280CacheMiningError(f"duplicate track member inside {source}")
    return members


def load_v100_track_members(
    cache_dir: Path,
    *,
    expected_shard_count: int = EXPECTED_V100_SHARD_COUNT,
) -> V100Membership:
    """Read only V10 schema and track identity arrays from canonical shards."""
    cache_dir = Path(cache_dir)
    paths = tuple(sorted(cache_dir.rglob("v100-spectral-shard-*.npz")))
    if len(paths) != expected_shard_count:
        raise V280CacheMiningError(
            f"expected {expected_shard_count} V10 spectral shards under {cache_dir}, got {len(paths)}"
        )
    all_members: list[str] = []
    relative_paths: list[str] = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            if set(data.files) != V100_ARRAY_NAMES:
                raise V280CacheMiningError(f"unexpected V10 cache schema in {path}")
            version = np.asarray(data["schema_version"])
            if version.shape != (1,) or int(version[0]) != V100_CACHE_SCHEMA_VERSION:
                raise V280CacheMiningError(f"unsupported V10 cache version in {path}")
            # Protocol-critical: no other array is indexed in this function.
            members = _validated_member_array(data["track_members"], path)
        all_members.extend(members)
        relative_paths.append(path.relative_to(cache_dir).as_posix())
    if len(set(all_members)) != len(all_members):
        raise V280CacheMiningError("duplicate track member across V10 shards")
    ordered = tuple(sorted(all_members))
    return V100Membership(ordered, tuple(relative_paths), _member_digest(ordered))


def _dataset_split(dataset_dir: Path):
    """Reproduce the frozen V8/V10 composition-safe GuitarSet split."""
    indexed = tuple(t for t in index_guitarset(dataset_dir) if t.player_id in ALLOWED_PLAYERS)
    train, validation = split_tracks_by_group(
        indexed,
        validation_fraction=VALIDATION_FRACTION,
        seed=SPLIT_SEED,
    )
    return indexed, train, validation


def validate_outer_clean_selection(
    indexed: Sequence[GuitarSetTrack],
    train: Sequence[GuitarSetTrack],
    validation: Sequence[GuitarSetTrack],
    membership: V100Membership,
) -> OuterCleanSelection:
    """Require exact V10/train identity and explicit validation exclusion."""
    indexed_members = tuple(t.annotation_member for t in indexed)
    train_members = tuple(t.annotation_member for t in train)
    validation_members = tuple(t.annotation_member for t in validation)
    if len(indexed_members) != EXPECTED_INDEXED_TRACK_COUNT:
        raise V280CacheMiningError(
            f"expected {EXPECTED_INDEXED_TRACK_COUNT} indexed tracks, got {len(indexed_members)}"
        )
    if len(train_members) != EXPECTED_OUTER_CLEAN_TRACK_COUNT:
        raise V280CacheMiningError(
            f"expected {EXPECTED_OUTER_CLEAN_TRACK_COUNT} outer-clean tracks, got {len(train_members)}"
        )
    if len(validation_members) != EXPECTED_HISTORICAL_VALIDATION_TRACK_COUNT:
        raise V280CacheMiningError(
            "expected "
            f"{EXPECTED_HISTORICAL_VALIDATION_TRACK_COUNT} historical-validation tracks, "
            f"got {len(validation_members)}"
        )
    for name, members in (
        ("indexed", indexed_members),
        ("train", train_members),
        ("validation", validation_members),
    ):
        if len(set(members)) != len(members):
            raise V280CacheMiningError(f"duplicate member in {name} split")
    train_set = set(train_members)
    validation_set = set(validation_members)
    if train_set & validation_set:
        raise V280CacheMiningError("outer-clean and historical-validation tracks overlap")
    if set(indexed_members) != train_set | validation_set:
        raise V280CacheMiningError("train/validation split does not cover the indexed tracks")
    cached_set = set(membership.track_members)
    if cached_set != train_set:
        missing = sorted(train_set - cached_set)[:5]
        unexpected = sorted(cached_set - train_set)[:5]
        raise V280CacheMiningError(
            f"V10 membership differs from outer-clean split: missing={missing}, unexpected={unexpected}"
        )
    if cached_set & validation_set:
        raise V280CacheMiningError("V10 membership overlaps historical validation")
    train_groups = tuple(sorted({group_stem(track) for track in train}))
    validation_groups = tuple(sorted({group_stem(track) for track in validation}))
    if set(train_groups) & set(validation_groups):
        raise V280CacheMiningError("composition group leakage in frozen split")
    by_member = {track.annotation_member: track for track in train}
    ordered_tracks = tuple(by_member[member] for member in membership.track_members)
    if _member_digest(track.annotation_member for track in ordered_tracks) != membership.sha256:
        raise V280CacheMiningError("selected track membership digest mismatch")
    return OuterCleanSelection(
        tracks=ordered_tracks,
        indexed_track_count=len(indexed_members),
        validation_track_count=len(validation_members),
        membership_sha256=membership.sha256,
        train_groups=train_groups,
        validation_groups=validation_groups,
    )


def select_outer_clean_tracks(dataset_dir: Path, cache_dir: Path) -> tuple[OuterCleanSelection, V100Membership]:
    membership = load_v100_track_members(cache_dir)
    indexed, train, validation = _dataset_split(dataset_dir)
    return validate_outer_clean_selection(indexed, train, validation, membership), membership


def preflight(args) -> dict[str, object]:
    """Validate the real source membership without decoding audio or labels."""
    selection, membership = select_outer_clean_tracks(
        Path(args.dataset_dir), Path(args.v100_cache_dir)
    )
    report = {
        "schema_version": 1,
        "experiment": "v280_causal_cqt_cache_preflight",
        "passed": True,
        "checked_at_utc": _utc_now(),
        "protocol": _protocol_flags(),
        "provenance": _provenance(),
        "feature_sha256": CausalCQTConfig().sha256,
        "indexed_track_count": selection.indexed_track_count,
        "outer_clean_track_count": len(selection.tracks),
        "historical_validation_track_count": selection.validation_track_count,
        "outer_clean_members_sha256": selection.membership_sha256,
        "v100_spectral_shard_count": len(membership.shard_paths),
        "v100_spectral_shard_paths": list(membership.shard_paths),
        "train_composition_group_count": len(selection.train_groups),
        "historical_validation_composition_group_count": len(selection.validation_groups),
        "composition_group_leakage": False,
        "audio_decoded": False,
    }
    if args.report is not None:
        report_path = Path(args.report)
        if report_path.exists():
            raise FileExistsError(f"refusing to overwrite {report_path}")
        _atomic_json(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return report


def _protocol_flags() -> dict[str, object]:
    return {
        "label_free_acoustic_cache_mining": True,
        "v100_arrays_deserialized": list(V100_ARRAYS_DESERIALIZED),
        "v100_supervision_arrays_deserialized": False,
        "v100_supervision_arrays_forbidden": list(V100_SUPERVISION_ARRAYS_FORBIDDEN),
        "annotation_archive_member_names_indexed": True,
        "jams_payloads_read": False,
        "historical_validation_members_used_for_exclusion_only": True,
        "historical_validation_audio_decoded": False,
        "historical_validation_labels_read_or_evaluated": False,
        "locked12_indexed_or_evaluated": False,
        "outer_fold_trained_or_evaluated": False,
        "training_started": False,
        "external_checkpoint_loaded": False,
        "teacher_model_loaded": False,
        "offset_stream_executed": False,
    }


def _provenance() -> dict[str, object]:
    return {
        "v100_membership": {
            "run_id": V104_SOURCE_RUN_ID,
            "head_sha": V104_SOURCE_HEAD_SHA,
            "artifact": V104_SOURCE_ARTIFACT,
            "artifact_digest": V104_SOURCE_ARTIFACT_DIGEST,
        },
        "guitarset": {
            "run_id": V272_SOURCE_RUN_ID,
            "head_sha": V272_SOURCE_HEAD_SHA,
            "artifact": V272_SOURCE_ARTIFACT,
            "artifact_digest": V272_SOURCE_ARTIFACT_DIGEST,
            "annotation_md5": GUITARSET_ANNOTATION_MD5,
            "audio_md5": GUITARSET_AUDIO_MD5,
        },
    }


def _initial_progress(
    selection: OuterCleanSelection,
    membership: V100Membership,
    config: CausalCQTConfig,
    planned_tracks: Sequence[GuitarSetTrack],
    formal: bool,
) -> dict:
    return {
        "schema_version": PROGRESS_SCHEMA_VERSION,
        "experiment": "v280_causal_cqt_cache_mining",
        "status": "in_progress",
        "started_at_utc": _utc_now(),
        "updated_at_utc": _utc_now(),
        "formal": formal,
        "config_sha256": config.sha256,
        "outer_clean_members_sha256": selection.membership_sha256,
        "v100_shard_paths": list(membership.shard_paths),
        "planned_members": [track.annotation_member for track in planned_tracks],
        "tracks": [],
    }


def _validate_progress(
    progress: dict,
    selection: OuterCleanSelection,
    membership: V100Membership,
    config: CausalCQTConfig,
    planned_tracks: Sequence[GuitarSetTrack],
    formal: bool,
) -> None:
    expected_members = [track.annotation_member for track in planned_tracks]
    checks = {
        "schema_version": PROGRESS_SCHEMA_VERSION,
        "experiment": "v280_causal_cqt_cache_mining",
        "formal": formal,
        "config_sha256": config.sha256,
        "outer_clean_members_sha256": selection.membership_sha256,
        "v100_shard_paths": list(membership.shard_paths),
        "planned_members": expected_members,
    }
    for key, expected in checks.items():
        if progress.get(key) != expected:
            raise V280CacheMiningError(f"resume checkpoint mismatch for {key}")
    if not isinstance(progress.get("tracks"), list):
        raise V280CacheMiningError("resume checkpoint tracks must be a list")


def _cache_record(
    output_dir: Path,
    track: GuitarSetTrack,
    cache_path: Path,
    config: CausalCQTConfig,
    *,
    processing_seconds: Optional[float],
    recovered: bool,
) -> dict[str, object]:
    cached = read_track_cache(cache_path, config)
    relative = cache_path.relative_to(output_dir).as_posix()
    expected_relative = f"tracks/{track_cache_filename(track.annotation_member)}"
    if relative != expected_relative:
        raise V280CacheMiningError(f"unexpected cache path for {track.annotation_member}")
    audio_seconds = cached.sample_count / float(config.sample_rate)
    realtime_factor = None
    if processing_seconds is not None and processing_seconds > 0.0:
        realtime_factor = audio_seconds / processing_seconds
    return {
        "annotation_member": track.annotation_member,
        "audio_member": track.audio_member,
        "cache_path": relative,
        "cache_sha256": _sha256_file(cache_path),
        "cache_bytes": cache_path.stat().st_size,
        "sample_count": cached.sample_count,
        "frame_count": int(cached.magnitude.shape[0]),
        "frequency_bin_count": int(cached.magnitude.shape[1]),
        "audio_seconds": audio_seconds,
        "processing_seconds": processing_seconds,
        "realtime_factor": realtime_factor,
        "recovered_from_existing_cache": recovered,
    }


def _cache_set_digest(records: Sequence[dict[str, object]]) -> str:
    lines = [f"{record['cache_path']}\0{record['cache_sha256']}\n" for record in records]
    return _sha256_bytes("".join(lines).encode("utf-8"))


def _manifest(
    progress: dict,
    selection: OuterCleanSelection,
    membership: V100Membership,
    config: CausalCQTConfig,
) -> dict:
    records = list(progress["tracks"])
    processing = [float(r["processing_seconds"]) for r in records if r["processing_seconds"] is not None]
    total_processing_seconds = float(sum(processing))
    total_audio_seconds = float(sum(float(r["audio_seconds"]) for r in records))
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "experiment": "v280_causal_cqt_cache_mining",
        "status": "complete",
        "formal": bool(progress["formal"]),
        "started_at_utc": progress["started_at_utc"],
        "completed_at_utc": _utc_now(),
        "protocol": _protocol_flags(),
        "provenance": _provenance(),
        "configuration": {
            "feature": config.serializable(),
            "feature_sha256": config.sha256,
            "split_seed": SPLIT_SEED,
            "validation_fraction": VALIDATION_FRACTION,
            "cache_dtype": "float16",
            "cache_container": "numpy_npz_uncompressed",
        },
        "data": {
            "indexed_track_count": selection.indexed_track_count,
            "outer_clean_track_count": len(selection.tracks),
            "historical_validation_track_count": selection.validation_track_count,
            "mined_track_count": len(records),
            "outer_clean_members_sha256": selection.membership_sha256,
            "mined_members_sha256": _member_digest(
                str(record["annotation_member"]) for record in records
            ),
            "v100_spectral_shard_count": len(membership.shard_paths),
            "v100_spectral_shard_paths": list(membership.shard_paths),
            "train_composition_group_count": len(selection.train_groups),
            "historical_validation_composition_group_count": len(selection.validation_groups),
            "composition_group_leakage": False,
        },
        "cache": {
            "track_count": len(records),
            "total_frames": int(sum(int(r["frame_count"]) for r in records)),
            "total_audio_samples": int(sum(int(r["sample_count"]) for r in records)),
            "total_audio_seconds": total_audio_seconds,
            "total_cache_bytes": int(sum(int(r["cache_bytes"]) for r in records)),
            "measured_processing_seconds": total_processing_seconds,
            "measured_realtime_factor": (
                total_audio_seconds / total_processing_seconds
                if total_processing_seconds > 0.0 and len(processing) == len(records)
                else None
            ),
            "cache_set_sha256": _cache_set_digest(records),
            "tracks": records,
        },
    }


def mine(args, config: CausalCQTConfig = CausalCQTConfig()) -> dict:
    output_dir = Path(args.output_dir)
    tracks_dir = output_dir / "tracks"
    progress_path = output_dir / "progress.json"
    manifest_path = output_dir / "manifest.json"
    resume = bool(args.resume)
    if output_dir.exists() and not resume:
        raise FileExistsError(f"refusing to overwrite {output_dir}; pass --resume to validate and continue")
    if resume and not output_dir.exists():
        raise V280CacheMiningError(f"cannot resume missing output directory {output_dir}")

    selection, membership = select_outer_clean_tracks(Path(args.dataset_dir), Path(args.v100_cache_dir))
    limit = args.limit_tracks
    if limit is not None and (
        isinstance(limit, bool) or limit <= 0 or limit > EXPECTED_OUTER_CLEAN_TRACK_COUNT
    ):
        raise V280CacheMiningError("limit-tracks must be in [1, 240]")
    planned_tracks = selection.tracks if limit is None else selection.tracks[:limit]
    formal = limit is None
    if formal and len(planned_tracks) != EXPECTED_OUTER_CLEAN_TRACK_COUNT:
        raise V280CacheMiningError("formal mining must cover all 240 outer-clean tracks")

    if manifest_path.exists():
        if not resume:
            raise FileExistsError(f"refusing to overwrite complete cache {output_dir}")
        report = verify_cache(
            output_dir,
            expected_track_count=len(planned_tracks),
            expected_formal=formal,
            config=config,
        )
        print(json.dumps({"status": "already_complete", **report}, sort_keys=True), flush=True)
        return json.loads(manifest_path.read_text())

    if resume:
        if not progress_path.is_file():
            raise V280CacheMiningError("resume requested without progress.json")
        progress = json.loads(progress_path.read_text())
        _validate_progress(progress, selection, membership, config, planned_tracks, formal)
    else:
        tracks_dir.mkdir(parents=True)
        progress = _initial_progress(selection, membership, config, planned_tracks, formal)
        _atomic_json(progress_path, progress)

    by_record = {str(record.get("annotation_member")): record for record in progress["tracks"]}
    if len(by_record) != len(progress["tracks"]):
        raise V280CacheMiningError("duplicate member in progress checkpoint")
    expected_files = {track_cache_filename(track.annotation_member) for track in planned_tracks}
    existing_files = {path.name for path in tracks_dir.glob("*.npz")}
    unexpected_files = sorted(existing_files - expected_files)
    if unexpected_files:
        raise V280CacheMiningError(f"unexpected caches in resume directory: {unexpected_files[:5]}")

    records: list[dict[str, object]] = []
    total = len(planned_tracks)
    for ordinal, track in enumerate(planned_tracks, start=1):
        cache_path = tracks_dir / track_cache_filename(track.annotation_member)
        prior = by_record.get(track.annotation_member)
        if cache_path.exists():
            record = _cache_record(
                output_dir,
                track,
                cache_path,
                config,
                processing_seconds=(
                    None if prior is None else prior.get("processing_seconds")
                ),
                recovered=prior is None or bool(prior.get("recovered_from_existing_cache")),
            )
            if prior is not None and record["cache_sha256"] != prior.get("cache_sha256"):
                raise V280CacheMiningError(f"changed cache detected for {track.annotation_member}")
            action = "verified"
        else:
            if prior is not None:
                raise V280CacheMiningError(f"checkpoint cache missing for {track.annotation_member}")
            started = time.perf_counter()
            audio = decode_pcm16_mono_wav(track.audio_zip, track.audio_member)
            if audio.sample_rate != config.sample_rate:
                raise V280CacheMiningError(
                    f"sample-rate mismatch for {track.annotation_member}: {audio.sample_rate}"
                )
            feature = causal_cqt(audio.samples, config)
            write_track_cache(cache_path, feature, config)
            elapsed = time.perf_counter() - started
            del feature, audio
            gc.collect()
            record = _cache_record(
                output_dir,
                track,
                cache_path,
                config,
                processing_seconds=elapsed,
                recovered=False,
            )
            action = "mined"
        records.append(record)
        progress["tracks"] = records
        progress["updated_at_utc"] = _utc_now()
        _atomic_json(progress_path, progress)
        factor = record["realtime_factor"]
        factor_text = "n/a" if factor is None else f"{float(factor):.2f}x"
        print(
            f"{action} {ordinal}/{total}: {track.annotation_member} "
            f"frames={record['frame_count']} realtime={factor_text}",
            flush=True,
        )

    report = _manifest(progress, selection, membership, config)
    if formal and report["data"]["mined_members_sha256"] != selection.membership_sha256:
        raise V280CacheMiningError("formal cache does not cover the exact outer-clean membership")
    _atomic_json(manifest_path, report)
    progress["status"] = "complete"
    progress["updated_at_utc"] = _utc_now()
    _atomic_json(progress_path, progress)
    verification = verify_cache(
        output_dir,
        expected_track_count=len(planned_tracks),
        expected_formal=formal,
        config=config,
    )
    print(json.dumps(verification, indent=2, sort_keys=True), flush=True)
    return report


def _require_protocol_flags(protocol: dict) -> None:
    expected = _protocol_flags()
    if protocol != expected:
        raise V280CacheMiningError("manifest protocol flags differ from the frozen label-free contract")


def verify_cache(
    output_dir: Path,
    *,
    expected_track_count: int = EXPECTED_OUTER_CLEAN_TRACK_COUNT,
    expected_formal: bool = True,
    config: CausalCQTConfig = CausalCQTConfig(),
) -> dict[str, object]:
    """Verify manifest, every file digest, and every embedded cache contract."""
    output_dir = Path(output_dir)
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        raise V280CacheMiningError(f"missing manifest {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise V280CacheMiningError("unsupported V28 manifest version")
    if manifest.get("experiment") != "v280_causal_cqt_cache_mining":
        raise V280CacheMiningError("unexpected V28 manifest experiment")
    if manifest.get("status") != "complete" or manifest.get("formal") is not expected_formal:
        raise V280CacheMiningError("manifest completion/formal status mismatch")
    _require_protocol_flags(manifest.get("protocol", {}))
    if manifest.get("provenance") != _provenance():
        raise V280CacheMiningError("manifest source provenance mismatch")
    configuration = manifest.get("configuration", {})
    if configuration.get("feature") != config.serializable():
        raise V280CacheMiningError("manifest feature configuration mismatch")
    if configuration.get("feature_sha256") != config.sha256:
        raise V280CacheMiningError("manifest feature digest mismatch")
    if configuration.get("split_seed") != SPLIT_SEED:
        raise V280CacheMiningError("manifest split seed mismatch")
    data = manifest.get("data", {})
    if data.get("indexed_track_count") != EXPECTED_INDEXED_TRACK_COUNT:
        raise V280CacheMiningError("manifest indexed-track count mismatch")
    if data.get("outer_clean_track_count") != EXPECTED_OUTER_CLEAN_TRACK_COUNT:
        raise V280CacheMiningError("manifest outer-clean count mismatch")
    if data.get("historical_validation_track_count") != EXPECTED_HISTORICAL_VALIDATION_TRACK_COUNT:
        raise V280CacheMiningError("manifest historical-validation count mismatch")
    if data.get("mined_track_count") != expected_track_count:
        raise V280CacheMiningError("manifest mined-track count mismatch")
    if data.get("composition_group_leakage") is not False:
        raise V280CacheMiningError("manifest composition leakage guard failed")
    if data.get("v100_spectral_shard_count") != EXPECTED_V100_SHARD_COUNT:
        raise V280CacheMiningError("manifest V10 shard count mismatch")

    cache = manifest.get("cache", {})
    records = cache.get("tracks")
    if not isinstance(records, list) or len(records) != expected_track_count:
        raise V280CacheMiningError("manifest track records mismatch")
    members: list[str] = []
    paths: list[str] = []
    verified_records: list[dict[str, object]] = []
    totals = {"frames": 0, "samples": 0, "bytes": 0}
    for record in records:
        member = str(record.get("annotation_member", ""))
        relative = str(record.get("cache_path", ""))
        if relative != f"tracks/{track_cache_filename(member)}":
            raise V280CacheMiningError(f"non-canonical cache path for {member!r}")
        path = output_dir / relative
        if not path.is_file():
            raise V280CacheMiningError(f"missing cache file {relative}")
        digest = _sha256_file(path)
        if digest != record.get("cache_sha256"):
            raise V280CacheMiningError(f"cache digest mismatch for {member}")
        track = read_track_cache(path, config)
        if track.sample_count != record.get("sample_count"):
            raise V280CacheMiningError(f"sample count mismatch for {member}")
        if list(track.magnitude.shape) != [record.get("frame_count"), record.get("frequency_bin_count")]:
            raise V280CacheMiningError(f"cache shape mismatch for {member}")
        size = path.stat().st_size
        if size != record.get("cache_bytes"):
            raise V280CacheMiningError(f"cache byte count mismatch for {member}")
        members.append(member)
        paths.append(relative)
        totals["frames"] += int(track.magnitude.shape[0])
        totals["samples"] += int(track.sample_count)
        totals["bytes"] += int(size)
        verified_records.append(record)
    if len(set(members)) != len(members) or len(set(paths)) != len(paths):
        raise V280CacheMiningError("duplicate member or path in manifest")
    if members != sorted(members):
        raise V280CacheMiningError("manifest tracks are not in canonical member order")
    mined_digest = _member_digest(members)
    if mined_digest != data.get("mined_members_sha256"):
        raise V280CacheMiningError("mined membership digest mismatch")
    if expected_formal and mined_digest != data.get("outer_clean_members_sha256"):
        raise V280CacheMiningError("formal membership does not equal outer-clean membership")
    if totals["frames"] != cache.get("total_frames"):
        raise V280CacheMiningError("aggregate frame count mismatch")
    if totals["samples"] != cache.get("total_audio_samples"):
        raise V280CacheMiningError("aggregate sample count mismatch")
    if totals["bytes"] != cache.get("total_cache_bytes"):
        raise V280CacheMiningError("aggregate cache byte count mismatch")
    if _cache_set_digest(verified_records) != cache.get("cache_set_sha256"):
        raise V280CacheMiningError("cache-set digest mismatch")
    discovered = sorted(path.relative_to(output_dir).as_posix() for path in (output_dir / "tracks").glob("*.npz"))
    if discovered != sorted(paths):
        raise V280CacheMiningError("cache directory contains missing or unexpected NPZ files")
    return {
        "verified": True,
        "formal": expected_formal,
        "feature_implementation": IMPLEMENTATION,
        "feature_sha256": config.sha256,
        "track_count": len(records),
        "total_frames": totals["frames"],
        "total_audio_samples": totals["samples"],
        "total_cache_bytes": totals["bytes"],
        "cache_set_sha256": cache["cache_set_sha256"],
        "labels_or_checkpoints_loaded": False,
        "training_or_evaluation_started": False,
        "verified_at_utc": _utc_now(),
    }


def verify(args) -> dict[str, object]:
    result = verify_cache(
        Path(args.output_dir),
        expected_track_count=args.expected_track_count,
        expected_formal=not args.allow_diagnostic,
    )
    if args.report is not None:
        report_path = Path(args.report)
        if report_path.exists():
            raise FileExistsError(f"refusing to overwrite {report_path}")
        _atomic_json(report_path, result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)

    preflight_parser = commands.add_parser(
        "preflight", help="validate exact outer-clean membership without decoding audio"
    )
    preflight_parser.add_argument("--dataset-dir", type=Path, required=True)
    preflight_parser.add_argument("--v100-cache-dir", type=Path, required=True)
    preflight_parser.add_argument("--report", type=Path)
    preflight_parser.set_defaults(func=preflight)

    mine_parser = commands.add_parser("mine", help="mine causal CQT caches from outer-clean audio")
    mine_parser.add_argument("--dataset-dir", type=Path, required=True)
    mine_parser.add_argument("--v100-cache-dir", type=Path, required=True)
    mine_parser.add_argument("--output-dir", type=Path, required=True)
    mine_parser.add_argument("--limit-tracks", type=int)
    mine_parser.add_argument("--resume", action="store_true")
    mine_parser.set_defaults(func=mine)

    verify_parser = commands.add_parser("verify", help="verify every cache and manifest invariant")
    verify_parser.add_argument("--output-dir", type=Path, required=True)
    verify_parser.add_argument(
        "--expected-track-count",
        type=int,
        default=EXPECTED_OUTER_CLEAN_TRACK_COUNT,
    )
    verify_parser.add_argument("--allow-diagnostic", action="store_true")
    verify_parser.add_argument("--report", type=Path)
    verify_parser.set_defaults(func=verify)
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
