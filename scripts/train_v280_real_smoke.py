"""V28.0-D real-data plumbing and bounded internal mini-overfit.

This stage consumes the frozen V10 cluster rows and the label-free V28 causal
CQT track cache.  It deliberately trains on a tiny, identity-selected set of
outer-clean composition groups only to prove that real features and GuitarSet
targets are wired correctly.  It does not create a holdout score, open an
outer fold, inspect the historical validation split, or select an architecture.

The command also emits a compact, checksummed copy of the V10 arrays that V28
still needs.  The 1.7 GB V10 spectral maps are intentionally omitted because
V28 replaces them with its own causal CQT representation.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Mapping, Optional, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for _path in (ROOT, SRC):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from causal_note.guitarset import GuitarSetTrack, SLOT_COUNT
from causal_note.guitarset_acoustics import load_rich_annotations
from causal_note.v280_causal_cqt import (
    CausalCQTConfig,
    cluster_feature_map,
    read_track_cache,
)
from scripts import mine_v280_causal_cqt as cqt_mining
from scripts import train_v92_string_factorized_cardinality as v92
from scripts.train_boundaries import group_stem
from scripts.train_v280_harmonic_count import (
    CARDINALITY_CLASSES,
    FRETS_PER_STRING,
    MODEL_KEY,
    SEED,
    STANDARD_TUNING_MIDI,
    build_model,
    guitar_pitch_count,
)


DISTILLED_SCHEMA_VERSION = 1
REPORT_SCHEMA_VERSION = 1
DEFAULT_GROUP_COUNT = 3
DEFAULT_ROWS = 28
DEFAULT_STEPS = 160
MINIMUM_LOSS_REDUCTION = 0.20
MINIMUM_CARDINALITY_ACCURACY = 0.70
MIDI_TIE_TOLERANCE_CENTS = 1e-3
V104_RUN_ID = cqt_mining.V104_SOURCE_RUN_ID
V104_HEAD_SHA = cqt_mining.V104_SOURCE_HEAD_SHA
V104_ARTIFACT = cqt_mining.V104_SOURCE_ARTIFACT
V104_ARTIFACT_DIGEST = cqt_mining.V104_SOURCE_ARTIFACT_DIGEST
V272_RUN_ID = cqt_mining.V272_SOURCE_RUN_ID
V272_HEAD_SHA = cqt_mining.V272_SOURCE_HEAD_SHA
V272_ARTIFACT = cqt_mining.V272_SOURCE_ARTIFACT
V272_ARTIFACT_DIGEST = cqt_mining.V272_SOURCE_ARTIFACT_DIGEST
V280_CQT_RUN_ID = 34_323_015_277
V280_CQT_HEAD_SHA = "3d2fe62b4527236ab91e86dad3cc9c5d81e67b08"
V280_CQT_ARTIFACT = "v280-causal-cqt-outer-clean-cache"
V280_CQT_ARTIFACT_DIGEST = (
    "sha256:2cb10b3042a0ff77ca6e7f59aacbef7a43ff04690aa6f7476a376f3c39eee2ff"
)
V280_CLUSTER_RUN_ID = 34_344_366_846
V280_CLUSTER_HEAD_SHA = "c4fbfc0572cfe704afa2efde149be489fe418829"
V280_CLUSTER_ARTIFACT = "v280-cluster-metadata"
V280_CLUSTER_ARTIFACT_DIGEST = (
    "sha256:22d850ee60978e31f964694859c7b12dd6547ce5115b9406e189f6c48305a8dd"
)


class V280RealSmokeError(RuntimeError):
    """Raised when a real-data smoke guard or gate fails."""


@dataclass(frozen=True)
class ClusterMetadata:
    """Only the frozen V10 row arrays required by the V28 count expert."""

    sequence: np.ndarray
    mask: np.ndarray
    exact: np.ndarray
    members: np.ndarray
    top_samples: np.ndarray
    slot_targets: np.ndarray
    track_members: tuple[str, ...]
    shard_paths: tuple[str, ...]

    @property
    def row_count(self) -> int:
        return int(len(self.exact))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _validate_cluster_metadata(metadata: ClusterMetadata) -> None:
    n = metadata.row_count
    if n <= 0:
        raise V280RealSmokeError("V10 cluster metadata is empty")
    if metadata.sequence.ndim != 3:
        raise V280RealSmokeError("sequence must be a 3-D row/candidate/feature array")
    if metadata.sequence.shape[0] != n:
        raise V280RealSmokeError("sequence row count differs from exact cardinality")
    if metadata.mask.shape != metadata.sequence.shape[:2]:
        raise V280RealSmokeError("candidate mask shape differs from sequence")
    if metadata.exact.shape != (n,):
        raise V280RealSmokeError("exact cardinality must be one-dimensional")
    if metadata.members.shape != (n,):
        raise V280RealSmokeError("members must be one-dimensional")
    if metadata.top_samples.ndim != 2 or metadata.top_samples.shape[0] != n:
        raise V280RealSmokeError("top_samples must be a row-aligned matrix")
    if metadata.slot_targets.shape != (n, SLOT_COUNT):
        raise V280RealSmokeError("slot_targets must have shape (rows, 6)")
    if not metadata.track_members or len(set(metadata.track_members)) != len(metadata.track_members):
        raise V280RealSmokeError("track_members must be non-empty and unique")
    if metadata.track_members != tuple(sorted(metadata.track_members)):
        raise V280RealSmokeError("track_members must use canonical sorted order")
    known = set(metadata.track_members)
    if any(str(member) not in known for member in metadata.members):
        raise V280RealSmokeError("a cluster row references a track outside track_members")
    if not np.isfinite(np.asarray(metadata.sequence, dtype=np.float32)).all():
        raise V280RealSmokeError("sequence contains non-finite values")
    if np.any((metadata.mask != 0) & (metadata.mask != 1)):
        raise V280RealSmokeError("candidate mask must be binary")
    if np.any((metadata.slot_targets != 0) & (metadata.slot_targets != 1)):
        raise V280RealSmokeError("slot targets must be binary")
    exact = np.asarray(metadata.exact, dtype=np.int64)
    if np.any((exact < 0) | (exact >= CARDINALITY_CLASSES)):
        raise V280RealSmokeError("exact cardinality must be in 0..6")


def load_v100_cluster_metadata(
    cache_dir: Path,
    *,
    expected_shard_count: int = cqt_mining.EXPECTED_V100_SHARD_COUNT,
    expected_track_count: int = cqt_mining.EXPECTED_OUTER_CLEAN_TRACK_COUNT,
) -> ClusterMetadata:
    """Load V10 row metadata without deserializing its obsolete spectral maps."""
    cache_dir = Path(cache_dir)
    paths = tuple(sorted(cache_dir.rglob("v100-spectral-shard-*.npz")))
    if len(paths) != expected_shard_count:
        raise V280RealSmokeError(
            f"expected {expected_shard_count} V10 shards under {cache_dir}, got {len(paths)}"
        )

    arrays: dict[str, list[np.ndarray]] = {
        key: [] for key in ("sequence", "mask", "exact", "members", "top_samples", "slot_targets")
    }
    track_members: list[str] = []
    relative_paths: list[str] = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            if set(data.files) != cqt_mining.V100_ARRAY_NAMES:
                raise V280RealSmokeError(f"unexpected V10 cache schema in {path}")
            version = np.asarray(data["schema_version"])
            if version.shape != (1,) or int(version[0]) != cqt_mining.V100_CACHE_SCHEMA_VERSION:
                raise V280RealSmokeError(f"unsupported V10 cache version in {path}")
            shard_members = cqt_mining._validated_member_array(data["track_members"], path)
            track_members.extend(shard_members)
            for key in arrays:
                arrays[key].append(np.asarray(data[key]))
        relative_paths.append(path.relative_to(cache_dir).as_posix())

    if len(track_members) != expected_track_count or len(set(track_members)) != expected_track_count:
        raise V280RealSmokeError(
            f"expected {expected_track_count} unique V10 tracks, got {len(set(track_members))}"
        )
    merged = ClusterMetadata(
        sequence=np.concatenate(arrays["sequence"], axis=0).astype(np.float16, copy=False),
        mask=np.concatenate(arrays["mask"], axis=0).astype(np.uint8, copy=False),
        exact=np.concatenate(arrays["exact"], axis=0).astype(np.int16, copy=False),
        members=np.concatenate(arrays["members"], axis=0).astype(str),
        top_samples=np.concatenate(arrays["top_samples"], axis=0).astype(np.int32, copy=False),
        slot_targets=np.concatenate(arrays["slot_targets"], axis=0).astype(np.uint8, copy=False),
        track_members=tuple(sorted(track_members)),
        shard_paths=tuple(relative_paths),
    )
    _validate_cluster_metadata(merged)
    return merged


def write_distilled_cluster_cache(output_dir: Path, metadata: ClusterMetadata) -> dict[str, object]:
    """Persist a compact V28-owned replacement for the expiring V10 bundle."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    _validate_cluster_metadata(metadata)
    output_dir.mkdir(parents=True)
    data_path = output_dir / "v280-cluster-metadata.npz"
    temporary = data_path.with_name(data_path.name + ".tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                schema_version=np.asarray([DISTILLED_SCHEMA_VERSION], dtype=np.int16),
                sequence=np.asarray(metadata.sequence, dtype=np.float16),
                mask=np.asarray(metadata.mask, dtype=np.uint8),
                exact=np.asarray(metadata.exact, dtype=np.int16),
                members=np.asarray(metadata.members, dtype=str),
                top_samples=np.asarray(metadata.top_samples, dtype=np.int32),
                slot_targets=np.asarray(metadata.slot_targets, dtype=np.uint8),
                track_members=np.asarray(metadata.track_members, dtype=str),
                source_shards=np.asarray(metadata.shard_paths, dtype=str),
            )
        os.replace(temporary, data_path)
    finally:
        if temporary.exists():
            temporary.unlink()

    report: dict[str, object] = {
        "schema_version": DISTILLED_SCHEMA_VERSION,
        "experiment": "v280_distilled_cluster_metadata",
        "created_at_utc": _utc_now(),
        "source": {
            "run_id": V104_RUN_ID,
            "head_sha": V104_HEAD_SHA,
            "artifact": V104_ARTIFACT,
            "artifact_digest": V104_ARTIFACT_DIGEST,
        },
        "data": {
            "row_count": metadata.row_count,
            "track_count": len(metadata.track_members),
            "track_members_sha256": cqt_mining._member_digest(metadata.track_members),
            "source_shard_count": len(metadata.shard_paths),
            "included_arrays": [
                "sequence",
                "mask",
                "exact",
                "members",
                "top_samples",
                "slot_targets",
                "track_members",
            ],
            "excluded_arrays": ["spectral", "stats", "target"],
        },
        "file": {
            "path": data_path.name,
            "size_bytes": data_path.stat().st_size,
            "sha256": _sha256_file(data_path),
        },
        "protocol": {
            "historical_validation_rows_present": False,
            "locked12_rows_present": False,
            "outer_clean_supervision_arrays_present": True,
            "obsolete_v10_spectral_maps_copied": False,
            "external_checkpoint_copied": False,
        },
    }
    _atomic_json(output_dir / "manifest.json", report)
    return report


def load_distilled_cluster_cache(output_dir: Path) -> ClusterMetadata:
    """Reload and independently verify a compact V28 cluster cache."""
    output_dir = Path(output_dir)
    manifest_path = output_dir / "manifest.json"
    data_path = output_dir / "v280-cluster-metadata.npz"
    if not manifest_path.is_file() or not data_path.is_file():
        raise V280RealSmokeError("distilled cluster cache is incomplete")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != DISTILLED_SCHEMA_VERSION:
        raise V280RealSmokeError("unsupported distilled manifest version")
    if manifest.get("source") != {
        "run_id": V104_RUN_ID,
        "head_sha": V104_HEAD_SHA,
        "artifact": V104_ARTIFACT,
        "artifact_digest": V104_ARTIFACT_DIGEST,
    }:
        raise V280RealSmokeError("distilled source provenance differs from the frozen V10 source")
    if _sha256_file(data_path) != manifest.get("file", {}).get("sha256"):
        raise V280RealSmokeError("distilled cluster-cache digest mismatch")
    with np.load(data_path, allow_pickle=False) as data:
        required = {
            "schema_version",
            "sequence",
            "mask",
            "exact",
            "members",
            "top_samples",
            "slot_targets",
            "track_members",
            "source_shards",
        }
        if set(data.files) != required or int(data["schema_version"][0]) != DISTILLED_SCHEMA_VERSION:
            raise V280RealSmokeError("unexpected distilled cluster-cache schema")
        metadata = ClusterMetadata(
            sequence=np.asarray(data["sequence"], dtype=np.float16),
            mask=np.asarray(data["mask"], dtype=np.uint8),
            exact=np.asarray(data["exact"], dtype=np.int16),
            members=np.asarray(data["members"]).astype(str),
            top_samples=np.asarray(data["top_samples"], dtype=np.int32),
            slot_targets=np.asarray(data["slot_targets"], dtype=np.uint8),
            track_members=tuple(str(value) for value in data["track_members"]),
            shard_paths=tuple(str(value) for value in data["source_shards"]),
        )
    _validate_cluster_metadata(metadata)
    expected = manifest["data"]
    if metadata.row_count != expected.get("row_count"):
        raise V280RealSmokeError("distilled row count differs from manifest")
    if len(metadata.track_members) != expected.get("track_count"):
        raise V280RealSmokeError("distilled track count differs from manifest")
    if cqt_mining._member_digest(metadata.track_members) != expected.get("track_members_sha256"):
        raise V280RealSmokeError("distilled track membership digest mismatch")
    if len(metadata.shard_paths) != expected.get("source_shard_count"):
        raise V280RealSmokeError("distilled source shard count differs from manifest")
    if manifest.get("protocol") != {
        "historical_validation_rows_present": False,
        "locked12_rows_present": False,
        "outer_clean_supervision_arrays_present": True,
        "obsolete_v10_spectral_maps_copied": False,
        "external_checkpoint_copied": False,
    }:
        raise V280RealSmokeError("distilled protocol flags differ from the frozen contract")
    return metadata


def _subset(metadata: ClusterMetadata, indices: np.ndarray) -> ClusterMetadata:
    index = np.asarray(indices, dtype=np.int64)
    return ClusterMetadata(
        sequence=metadata.sequence[index],
        mask=metadata.mask[index],
        exact=metadata.exact[index],
        members=metadata.members[index],
        top_samples=metadata.top_samples[index],
        slot_targets=metadata.slot_targets[index],
        track_members=tuple(sorted({str(metadata.members[i]) for i in index})),
        shard_paths=metadata.shard_paths,
    )


def select_development_groups(
    metadata: ClusterMetadata,
    *,
    group_count: int,
) -> tuple[tuple[str, ...], np.ndarray]:
    """Choose groups from identities only, before consulting any target value."""
    if isinstance(group_count, bool) or not isinstance(group_count, int) or group_count <= 0:
        raise V280RealSmokeError("group_count must be a positive integer")
    group_by_member = {member: group_stem(member) for member in metadata.track_members}
    groups = tuple(sorted(set(group_by_member.values())))
    if group_count >= len(groups):
        raise V280RealSmokeError("real smoke must leave at least one composition group untouched")
    selected_groups = groups[:group_count]
    selected_set = set(selected_groups)
    indices = np.asarray(
        [
            row
            for row, member in enumerate(metadata.members)
            if group_by_member[str(member)] in selected_set
        ],
        dtype=np.int64,
    )
    if not len(indices):
        raise V280RealSmokeError("identity-selected development groups contain no cluster rows")
    return selected_groups, indices


def derive_midi_supervision(
    tracks_by_member: Mapping[str, GuitarSetTrack],
    members: Sequence[str],
    candidate_samples: Sequence[np.ndarray],
    expected_slot_targets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Assign rich note labels to the nearest causal cluster within 20 ms."""
    if len(members) != len(candidate_samples):
        raise V280RealSmokeError("members/candidate_samples length mismatch")
    expected = np.asarray(expected_slot_targets, dtype=np.float32) > 0.5
    if expected.shape != (len(members), SLOT_COUNT):
        raise V280RealSmokeError("expected slot targets have the wrong shape")

    midi = np.zeros((len(members), SLOT_COUNT), dtype=np.float32)
    mask = np.zeros((len(members), SLOT_COUNT), dtype=np.float32)
    assignment_distance = np.full((len(members), SLOT_COUNT), np.inf, dtype=np.float64)
    rows_by_member: dict[str, list[int]] = defaultdict(list)
    for row, member in enumerate(members):
        rows_by_member[str(member)].append(row)

    matched_events = 0
    unassigned_events = 0
    collisions = 0
    distances: list[int] = []
    midi_values: list[float] = []
    read_members: list[str] = []
    for member, rows in sorted(rows_by_member.items()):
        track = tracks_by_member.get(member)
        if track is None:
            raise V280RealSmokeError(f"selected cache member is absent from GuitarSet: {member}")
        sample_parts: list[np.ndarray] = []
        row_parts: list[np.ndarray] = []
        for row in rows:
            samples = np.asarray(candidate_samples[row], dtype=np.int32)
            if len(samples):
                sample_parts.append(samples)
                row_parts.append(np.full(len(samples), row, dtype=np.int32))
        if not sample_parts:
            raise V280RealSmokeError(f"selected track has no reconstructed candidates: {member}")
        flat_samples = np.concatenate(sample_parts)
        flat_rows = np.concatenate(row_parts)
        order = np.argsort(flat_samples, kind="stable")
        flat_samples = flat_samples[order]
        flat_rows = flat_rows[order]

        annotations = load_rich_annotations(track.annotation_zip, track.annotation_member)
        read_members.append(member)
        for note in annotations.notes:
            onset = int(note.onset_sample)
            left = int(
                np.searchsorted(flat_samples, onset - v92.LOCAL_RADIUS_SAMPLES, side="left")
            )
            right = int(
                np.searchsorted(flat_samples, onset + v92.LOCAL_RADIUS_SAMPLES, side="right")
            )
            if right <= left:
                unassigned_events += 1
                continue
            local_samples = flat_samples[left:right]
            local_rows = flat_rows[left:right]
            delta = np.abs(local_samples.astype(np.int64) - onset)
            best_distance = int(np.min(delta))
            row = int(np.min(local_rows[delta == best_distance]))
            slot = int(note.slot)
            matched_events += 1
            distances.append(best_distance)
            if mask[row, slot] > 0.5:
                collisions += 1
                if best_distance >= assignment_distance[row, slot]:
                    continue
            midi[row, slot] = float(note.midi)
            mask[row, slot] = 1.0
            assignment_distance[row, slot] = best_distance
            midi_values.append(float(note.midi))

    seen = mask > 0.5
    agreement = float(np.mean(seen == expected))
    active_coverage = float(np.mean(seen[expected])) if np.any(expected) else 1.0
    if agreement < 0.999 or active_coverage < 0.995:
        raise V280RealSmokeError(
            "rich-note assignment does not reproduce frozen slot supervision: "
            f"agreement={agreement} active={active_coverage}"
        )
    diagnostics: dict[str, object] = {
        "annotation_members_read": len(read_members),
        "annotation_member_names_sha256": cqt_mining._member_digest(read_members),
        "matched_events": matched_events,
        "unassigned_events": unassigned_events,
        "same_slot_collisions": collisions,
        "unique_assigned_string_births": int(np.sum(seen)),
        "slot_mask_agreement": agreement,
        "active_slot_pitch_coverage": active_coverage,
        "nearest_candidate_distance_median_ms": (
            float(np.median(distances) * 1000.0 / 44_100.0) if distances else None
        ),
        "nearest_candidate_distance_p90_ms": (
            float(np.percentile(distances, 90) * 1000.0 / 44_100.0) if distances else None
        ),
        "midi_min": min(midi_values) if midi_values else None,
        "midi_max": max(midi_values) if midi_values else None,
    }
    return midi, mask, diagnostics


def real_targets(
    exact: np.ndarray,
    slot_targets: np.ndarray,
    midi: np.ndarray,
    midi_mask: np.ndarray,
    config: CausalCQTConfig = CausalCQTConfig(),
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, object]]:
    """Build discrete V28 targets from GuitarSet's continuous MIDI labels."""
    exact_array = np.asarray(exact, dtype=np.int32)
    strings = np.asarray(slot_targets, dtype=np.float32)
    midi_array = np.asarray(midi, dtype=np.float32)
    mask = np.asarray(midi_mask, dtype=np.float32) > 0.5
    n = len(exact_array)
    if exact_array.shape != (n,) or strings.shape != (n, SLOT_COUNT):
        raise V280RealSmokeError("real target cardinality/string shapes differ")
    if midi_array.shape != strings.shape or mask.shape != strings.shape:
        raise V280RealSmokeError("MIDI supervision shape differs from string targets")
    if not np.array_equal(mask, strings > 0.5):
        raise V280RealSmokeError("MIDI mask must exactly equal frozen string occupancy")

    raw_active = np.asarray(midi_array[mask], dtype=np.float64)
    if not np.isfinite(raw_active).all():
        raise V280RealSmokeError("active MIDI supervision contains a non-finite value")
    rounded_active = np.rint(raw_active)
    signed_cents = 100.0 * (raw_active - rounded_active)
    absolute_cents = np.abs(signed_cents)
    ambiguous = np.flatnonzero(absolute_cents >= 50.0 - MIDI_TIE_TOLERANCE_CENTS)
    if len(ambiguous):
        value = float(raw_active[int(ambiguous[0])])
        raise V280RealSmokeError(
            f"MIDI label {value} is ambiguous at a half-semitone quantization boundary"
        )

    k = np.minimum(exact_array, SLOT_COUNT).astype(np.int32)
    fret = np.zeros((n, SLOT_COUNT, FRETS_PER_STRING), dtype=np.float32)
    pitch = np.zeros((n, guitar_pitch_count(config)), dtype=np.float32)
    pitch_low = int(round(config.input_min_midi))
    for row, slot in np.argwhere(mask):
        raw = float(midi_array[row, slot])
        rounded = int(np.rint(raw))
        fret_index = rounded - int(STANDARD_TUNING_MIDI[slot])
        pitch_index = rounded - pitch_low
        if not 0 <= fret_index < FRETS_PER_STRING:
            raise V280RealSmokeError(
                f"MIDI {rounded} is outside the 20-position grid for string {slot}"
            )
        if not 0 <= pitch_index < pitch.shape[1]:
            raise V280RealSmokeError(f"MIDI {rounded} is outside the V28 pitch range")
        fret[row, slot, fret_index] = 1.0
        pitch[row, pitch_index] = 1.0

    occupancy = np.sum(strings > 0.5, axis=1).astype(np.int32)
    consistent = occupancy == k
    targets = {
        "cardinality": k,
        "string_birth": strings,
        "string_fret_onset": fret,
        "pitch_onset": pitch,
        "poibin_cardinality": k,
    }
    fractional = absolute_cents > 0.01
    quantization: dict[str, object] = {
        "rule": "nearest_equal_tempered_semitone",
        "half_semitone_ties_rejected": True,
        "label_count": int(len(raw_active)),
        "fractional_label_count": int(np.sum(fractional)),
        "fractional_label_fraction": (
            float(np.mean(fractional)) if len(raw_active) else 0.0
        ),
        "absolute_cents_median": (
            float(np.median(absolute_cents)) if len(absolute_cents) else None
        ),
        "absolute_cents_p90": (
            float(np.percentile(absolute_cents, 90)) if len(absolute_cents) else None
        ),
        "absolute_cents_max": (
            float(np.max(absolute_cents)) if len(absolute_cents) else None
        ),
        "signed_cents_min": float(np.min(signed_cents)) if len(signed_cents) else None,
        "signed_cents_max": float(np.max(signed_cents)) if len(signed_cents) else None,
    }
    return targets, consistent, quantization


def select_balanced_rows(k: np.ndarray, eligible: np.ndarray, rows: int) -> np.ndarray:
    """Round-robin exact-K classes without using a metric or random search."""
    values = np.asarray(k, dtype=np.int32)
    allowed = np.asarray(eligible, dtype=bool)
    if values.shape != allowed.shape or values.ndim != 1:
        raise V280RealSmokeError("balanced-row inputs must be aligned vectors")
    if isinstance(rows, bool) or not isinstance(rows, int) or rows < CARDINALITY_CLASSES:
        raise V280RealSmokeError("rows must be an integer >= 7")
    queues = {
        count: list(np.flatnonzero(allowed & (values == count)).astype(int))
        for count in range(CARDINALITY_CLASSES)
    }
    active = [count for count, queue in queues.items() if queue]
    if len(active) < 2 or not any(count >= 2 for count in active):
        raise V280RealSmokeError("development groups lack diverse real cardinality targets")
    selected: list[int] = []
    cursor = {count: 0 for count in active}
    while len(selected) < rows:
        progressed = False
        for count in active:
            position = cursor[count]
            if position < len(queues[count]):
                selected.append(queues[count][position])
                cursor[count] += 1
                progressed = True
                if len(selected) == rows:
                    break
        if not progressed:
            break
    if len(selected) != rows:
        raise V280RealSmokeError(f"requested {rows} smoke rows but only {len(selected)} are eligible")
    return np.asarray(selected, dtype=np.int64)


def extract_selected_features(
    cqt_cache_dir: Path,
    cqt_manifest: Mapping[str, object],
    members: Sequence[str],
    candidate_samples: Sequence[np.ndarray],
    selected_rows: np.ndarray,
    config: CausalCQTConfig = CausalCQTConfig(),
) -> tuple[np.ndarray, dict[str, object]]:
    """Read only selected track caches and materialize causal cluster crops."""
    records = cqt_manifest.get("cache", {}).get("tracks", [])
    if not isinstance(records, list):
        raise V280RealSmokeError("V28 CQT manifest lacks track records")
    by_member = {str(record["annotation_member"]): record for record in records}
    rows = np.asarray(selected_rows, dtype=np.int64)
    features = np.zeros(
        (len(rows), config.cluster_frames, len(config.center_frequencies_hz), 3),
        dtype=np.float32,
    )
    output_positions: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for output_row, source_row in enumerate(rows):
        member = str(members[int(source_row)])
        output_positions[member].append((output_row, int(source_row)))

    decision_ends: list[int] = []
    cluster_starts: list[int] = []
    for member, positions in sorted(output_positions.items()):
        record = by_member.get(member)
        if record is None:
            raise V280RealSmokeError(f"no causal CQT cache record for {member}")
        path = Path(cqt_cache_dir) / str(record["cache_path"])
        track = read_track_cache(path, config)
        for output_row, source_row in positions:
            samples = np.asarray(candidate_samples[source_row], dtype=np.int32)
            if not len(samples):
                raise V280RealSmokeError("cannot crop a cluster without candidates")
            cluster_start = int(np.min(samples))
            decision_end = cluster_start + config.cluster_post_samples
            if cluster_start < 0 or decision_end > track.sample_count:
                raise V280RealSmokeError(
                    f"cluster decision interval exceeds causal cache for {member}: "
                    f"start={cluster_start} end={decision_end} samples={track.sample_count}"
                )
            features[output_row] = cluster_feature_map(
                track,
                cluster_start,
                config,
                decision_end=decision_end,
            )
            cluster_starts.append(cluster_start)
            decision_ends.append(decision_end)
    if not np.isfinite(features).all():
        raise V280RealSmokeError("selected real features contain non-finite values")
    return features, {
        "selected_track_count": len(output_positions),
        "cluster_start_min": min(cluster_starts),
        "cluster_start_max": max(cluster_starts),
        "decision_end_min": min(decision_ends),
        "decision_end_max": max(decision_ends),
        "feature_min": float(np.min(features)),
        "feature_max": float(np.max(features)),
        "feature_mean": float(np.mean(features)),
    }


def _prediction_accuracy(prediction: Mapping[str, object], target: np.ndarray) -> float:
    probability = np.asarray(prediction["cardinality"])
    if probability.shape != (len(target), CARDINALITY_CLASSES):
        raise V280RealSmokeError("unexpected cardinality prediction shape")
    if not np.isfinite(probability).all():
        raise V280RealSmokeError("cardinality prediction contains non-finite values")
    np.testing.assert_allclose(np.sum(probability, axis=1), 1.0, atol=1e-5)
    return float(np.mean(np.argmax(probability, axis=1) == np.asarray(target)))


def bounded_real_overfit(
    features: np.ndarray,
    targets: Mapping[str, np.ndarray],
    *,
    steps: int,
    seed: int,
) -> tuple[object, dict[str, object]]:
    """Fit one fixed real mini-batch; this intentionally yields no holdout metric."""
    if not 1 <= steps <= 400:
        raise V280RealSmokeError("steps must be in [1, 400]")
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required for the real V28 smoke") from exc

    tf.keras.utils.set_random_seed(seed)
    model = build_model(CausalCQTConfig(), seed=seed, harmonic=True)
    initial_prediction = model(features, training=False)
    initial_accuracy = _prediction_accuracy(initial_prediction, targets["cardinality"])
    initial = model.test_on_batch(features, targets, return_dict=True)
    history: list[dict[str, float]] = []
    for _ in range(steps):
        current = model.train_on_batch(features, targets, return_dict=True)
        history.append({key: float(value) for key, value in current.items()})
    final_prediction = model(features, training=False)
    final_accuracy = _prediction_accuracy(final_prediction, targets["cardinality"])
    final = model.test_on_batch(features, targets, return_dict=True)
    all_values = [*initial.values(), *final.values(), *(v for row in history for v in row.values())]
    if not np.isfinite(np.asarray(all_values, dtype=np.float64)).all():
        raise V280RealSmokeError("real mini-overfit produced a non-finite loss")
    initial_loss = float(initial["loss"])
    final_loss = float(final["loss"])
    reduction = (initial_loss - final_loss) / max(abs(initial_loss), 1e-12)
    gates = {
        "finite_losses": True,
        "loss_reduction_at_least_20_percent": bool(reduction >= MINIMUM_LOSS_REDUCTION),
        "cardinality_accuracy_at_least_70_percent": bool(
            final_accuracy >= MINIMUM_CARDINALITY_ACCURACY
        ),
        "cardinality_accuracy_improved": bool(final_accuracy > initial_accuracy),
    }
    return model, {
        "initial_losses": {key: float(value) for key, value in initial.items()},
        "final_losses": {key: float(value) for key, value in final.items()},
        "initial_cardinality_accuracy": initial_accuracy,
        "final_cardinality_accuracy": final_accuracy,
        "total_loss_reduction_fraction": float(reduction),
        "steps": int(steps),
        "gates": gates,
        "passed": bool(all(gates.values())),
    }


def run(args) -> dict[str, object]:
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")

    if args.v100_cache_dir is not None:
        if args.distilled_output_dir is None:
            raise V280RealSmokeError(
                "--distilled-output-dir is required with --v100-cache-dir"
            )
        distilled_dir = Path(args.distilled_output_dir)
        if distilled_dir.exists():
            raise FileExistsError(f"refusing to overwrite {distilled_dir}")
        metadata = load_v100_cluster_metadata(Path(args.v100_cache_dir))
        distilled_report = None
        cluster_source_mode = "frozen_v104_shards_distilled_in_run"
    else:
        if args.distilled_output_dir is not None:
            raise V280RealSmokeError(
                "--distilled-output-dir cannot be used with --distilled-source-dir"
            )
        distilled_dir = Path(args.distilled_source_dir)
        metadata = load_distilled_cluster_cache(distilled_dir)
        distilled_report = json.loads((distilled_dir / "manifest.json").read_text())
        cluster_source_mode = "preserved_v280_cluster_metadata"

    membership = cqt_mining.V100Membership(
        track_members=metadata.track_members,
        shard_paths=metadata.shard_paths,
        sha256=cqt_mining._member_digest(metadata.track_members),
    )
    indexed_tracks, outer_clean_tracks, historical_validation_tracks = (
        cqt_mining._dataset_split(Path(args.dataset_dir))
    )
    selection = cqt_mining.validate_outer_clean_selection(
        indexed_tracks,
        outer_clean_tracks,
        historical_validation_tracks,
        membership,
    )
    if args.v100_cache_dir is not None:
        distilled_report = write_distilled_cluster_cache(distilled_dir, metadata)
        reloaded = load_distilled_cluster_cache(distilled_dir)
        if reloaded.row_count != metadata.row_count:
            raise V280RealSmokeError("distilled cache round-trip changed row count")
    cqt_verification = cqt_mining.verify_cache(Path(args.v280_cache_dir))
    cqt_manifest = json.loads((Path(args.v280_cache_dir) / "manifest.json").read_text())
    cqt_members = tuple(str(row["annotation_member"]) for row in cqt_manifest["cache"]["tracks"])
    if cqt_members != metadata.track_members:
        raise V280RealSmokeError("V10 rows and V28 causal CQT cache membership differ")
    historical_validation_members = {
        track.annotation_member for track in historical_validation_tracks
    }
    if set(metadata.track_members) & historical_validation_members:
        raise V280RealSmokeError("historical validation track entered V28 metadata")

    groups, group_rows = select_development_groups(metadata, group_count=args.group_count)
    development = _subset(metadata, group_rows)
    candidate_samples, reconstruction = v92._reconstruct_candidates(
        {
            "sequence": development.sequence,
            "mask": development.mask,
            "top_samples": development.top_samples,
        }
    )
    if (
        reconstruction["top_sample_match_fraction"] is None
        or reconstruction["top_sample_match_fraction"] < 0.999
    ):
        raise V280RealSmokeError(
            f"candidate position reconstruction is too weak: {reconstruction}"
        )
    outer_tracks = {track.annotation_member: track for track in selection.tracks}
    midi, midi_mask, supervision = derive_midi_supervision(
        outer_tracks,
        [str(member) for member in development.members],
        candidate_samples,
        development.slot_targets,
    )
    targets, consistent, midi_quantization = real_targets(
        development.exact,
        development.slot_targets,
        midi,
        midi_mask,
    )
    selected = select_balanced_rows(targets["cardinality"], consistent, args.rows)
    features, feature_diag = extract_selected_features(
        Path(args.v280_cache_dir),
        cqt_manifest,
        [str(member) for member in development.members],
        candidate_samples,
        selected,
    )
    selected_targets = {key: np.asarray(value)[selected] for key, value in targets.items()}
    model, overfit = bounded_real_overfit(
        features,
        selected_targets,
        steps=args.steps,
        seed=args.seed,
    )

    output_dir.mkdir(parents=True)
    weights_path = output_dir / "v280-chec-real-smoke.weights.h5"
    model.save_weights(weights_path)
    k = np.asarray(selected_targets["cardinality"], dtype=np.int32)
    report: dict[str, object] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "experiment": "v280_chec_real_internal_smoke",
        "created_at_utc": _utc_now(),
        "status": "passed" if overfit["passed"] else "failed",
        "sources": {
            "v104_cluster_rows": {
                "run_id": V104_RUN_ID,
                "head_sha": V104_HEAD_SHA,
                "artifact": V104_ARTIFACT,
                "artifact_digest": V104_ARTIFACT_DIGEST,
            },
            "v272_guitarset": {
                "run_id": V272_RUN_ID,
                "head_sha": V272_HEAD_SHA,
                "artifact": V272_ARTIFACT,
                "artifact_digest": V272_ARTIFACT_DIGEST,
            },
            "v280_causal_cqt": {
                "run_id": V280_CQT_RUN_ID,
                "head_sha": V280_CQT_HEAD_SHA,
                "artifact": V280_CQT_ARTIFACT,
                "artifact_digest": V280_CQT_ARTIFACT_DIGEST,
                "cache_set_sha256": cqt_verification["cache_set_sha256"],
            },
        },
        "protocol": {
            "purpose": "real feature/target wiring and fixed-batch mini-overfit only",
            "development_groups_selected_from_identities_only": True,
            "target_values_used_to_select_groups": False,
            "target_values_used_to_balance_training_rows": True,
            "full_outer_clean_supervision_arrays_deserialized": True,
            "only_identity_selected_development_rows_used_for_training": True,
            "only_selected_development_group_jams_read": True,
            "historical_validation_member_names_indexed_for_exclusion_only": True,
            "historical_validation_jams_read_or_evaluated": False,
            "locked12_indexed_or_evaluated": False,
            "outer_fold_opened": False,
            "holdout_or_generalization_metric_computed": False,
            "architecture_or_hyperparameter_selected": False,
            "external_checkpoint_or_teacher_loaded": False,
            "v27_reference_modified": False,
        },
        "configuration": {
            "model": MODEL_KEY,
            "feature_sha256": CausalCQTConfig().sha256,
            "seed": int(args.seed),
            "group_count": int(args.group_count),
            "requested_rows": int(args.rows),
            "steps": int(args.steps),
            "minimum_loss_reduction": MINIMUM_LOSS_REDUCTION,
            "minimum_cardinality_accuracy": MINIMUM_CARDINALITY_ACCURACY,
            "cluster_source_mode": cluster_source_mode,
        },
        "data": {
            "outer_clean_track_count": len(metadata.track_members),
            "historical_validation_track_count": len(historical_validation_members),
            "full_cluster_row_count": metadata.row_count,
            "development_groups": list(groups),
            "development_group_members_sha256": cqt_mining._member_digest(
                development.track_members
            ),
            "development_track_count": len(development.track_members),
            "development_cluster_row_count": development.row_count,
            "eligible_consistent_row_count": int(np.sum(consistent)),
            "selected_row_count": len(selected),
            "selected_exact_k_histogram": {
                str(count): int(np.sum(k == count)) for count in range(CARDINALITY_CLASSES)
            },
            "selected_member_count": len(set(str(development.members[i]) for i in selected)),
            "distilled_cache": distilled_report,
        },
        "cluster_reconstruction": reconstruction,
        "supervision": {**supervision, "midi_quantization": midi_quantization},
        "features": {"shape": list(features.shape), **feature_diag},
        "training": overfit,
        "model": {
            "trainable_parameters": int(model.count_params()),
            "weights_path": weights_path.name,
            "weights_sha256": _sha256_file(weights_path),
        },
    }
    if args.distilled_source_dir is not None:
        report["sources"]["v280_cluster_metadata"] = {
            "run_id": V280_CLUSTER_RUN_ID,
            "head_sha": V280_CLUSTER_HEAD_SHA,
            "artifact": V280_CLUSTER_ARTIFACT,
            "artifact_digest": V280_CLUSTER_ARTIFACT_DIGEST,
            "ultimate_v104_source": report["sources"]["v104_cluster_rows"],
        }
    _atomic_json(output_dir / "report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if not overfit["passed"]:
        raise V280RealSmokeError(f"real mini-overfit gates failed: {overfit['gates']}")
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset-dir", type=Path, required=True)
    source = result.add_mutually_exclusive_group(required=True)
    source.add_argument("--v100-cache-dir", type=Path)
    source.add_argument("--distilled-source-dir", type=Path)
    result.add_argument("--v280-cache-dir", type=Path, required=True)
    result.add_argument("--distilled-output-dir", type=Path)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--group-count", type=int, default=DEFAULT_GROUP_COUNT)
    result.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    result.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    result.add_argument("--seed", type=int, default=SEED + 3)
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    run(parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
