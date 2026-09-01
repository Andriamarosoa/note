"""Small, extraction-free GuitarSet ZIP index and boundary loader.

The public workflow is deliberately narrow::

    tracks = index_guitarset("data/GuitarSet")
    six_string_slots = load_boundary_slots(
        tracks[0].annotation_zip,
        tracks[0].annotation_member,
    )

Only players ``00`` through ``04`` are admitted.  Player ``05`` is excluded
from the index and is rejected if a caller tries to load it directly.
"""

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
from typing import Any, Dict, List, Tuple, Union
import zipfile


SAMPLE_RATE = 44_100
ANNOTATION_ARCHIVE_NAME = "annotation.zip"
AUDIO_ARCHIVE_NAME = "audio_mono-pickup_mix.zip"
ALLOWED_PLAYERS = frozenset(("00", "01", "02", "03", "04"))
SLOT_COUNT = 6


class GuitarSetError(Exception):
    """Base error for an unreadable or inconsistent GuitarSet archive."""


class GuitarSetFormatError(GuitarSetError, ValueError):
    """Raised when archive names or JAMS boundary values are invalid."""


@dataclass(frozen=True)
class GuitarSetTrack:
    """One allowed JAMS member and its mono pickup-mix WAV member.

    The paths point to ZIP files; this object does not extract or decode audio.
    """

    player_id: str
    annotation_zip: Path
    annotation_member: str
    audio_zip: Path
    audio_member: str

    def __post_init__(self) -> None:
        if self.player_id not in ALLOWED_PLAYERS:
            raise GuitarSetFormatError(
                f"player {self.player_id!r} is not allowed; expected 00 through 04"
            )
        if not isinstance(self.annotation_member, str) or not self.annotation_member:
            raise GuitarSetFormatError("annotation_member must be a non-empty string")
        if not isinstance(self.audio_member, str) or not self.audio_member:
            raise GuitarSetFormatError("audio_member must be a non-empty string")
        object.__setattr__(self, "annotation_zip", Path(self.annotation_zip))
        object.__setattr__(self, "audio_zip", Path(self.audio_zip))


@dataclass(frozen=True, order=True)
class NoteBoundary:
    """Sample-accurate onset and offset for one note in one string slot."""

    onset_sample: int
    offset_sample: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.onset_sample, bool)
            or not isinstance(self.onset_sample, int)
            or self.onset_sample < 0
        ):
            raise GuitarSetFormatError("onset_sample must be an integer >= 0")
        if (
            isinstance(self.offset_sample, bool)
            or not isinstance(self.offset_sample, int)
            or self.offset_sample <= self.onset_sample
        ):
            raise GuitarSetFormatError(
                "offset_sample must be an integer after onset_sample"
            )


BoundarySlots = Tuple[Tuple[NoteBoundary, ...], ...]
PathInput = Union[str, os.PathLike]


def _member_player(member: str) -> str:
    name = PurePosixPath(member).name
    match = re.fullmatch(r"(\d{2})_.+\.jams", name)
    if match is None:
        raise GuitarSetFormatError(
            f"invalid GuitarSet annotation member name: {member!r}"
        )
    return match.group(1)


def _zip_members(path: Path) -> Tuple[str, ...]:
    try:
        with zipfile.ZipFile(path, "r") as archive:
            return tuple(info.filename for info in archive.infolist() if not info.is_dir())
    except (OSError, zipfile.BadZipFile) as exc:
        raise GuitarSetError(f"cannot read ZIP archive: {path}") from exc


def index_guitarset(dataset_dir: PathInput) -> Tuple[GuitarSetTrack, ...]:
    """Index allowed GuitarSet tracks without extracting either ZIP.

    ``dataset_dir`` must contain ``annotation.zip`` and
    ``audio_mono-pickup_mix.zip``.  A JAMS member named ``<track>.jams`` maps
    to the WAV basename ``<track>_mix.wav``.  Returned tracks are sorted by
    annotation member, independently of ZIP entry order.
    """

    root = Path(dataset_dir).resolve()
    annotation_zip = root / ANNOTATION_ARCHIVE_NAME
    audio_zip = root / AUDIO_ARCHIVE_NAME
    annotation_members = _zip_members(annotation_zip)
    audio_members = _zip_members(audio_zip)

    audio_by_name: Dict[str, str] = {}
    for member in audio_members:
        basename = PurePosixPath(member).name
        if not basename.endswith(".wav"):
            continue
        previous = audio_by_name.get(basename)
        if previous is not None:
            raise GuitarSetFormatError(
                f"duplicate audio basename {basename!r} in {audio_zip}"
            )
        audio_by_name[basename] = member

    tracks: List[GuitarSetTrack] = []
    seen_annotation_names = set()
    for member in annotation_members:
        if not member.endswith(".jams"):
            continue
        player_id = _member_player(member)
        if player_id not in ALLOWED_PLAYERS:
            # This is the hard dataset split guard, including player 05.
            continue

        annotation_name = PurePosixPath(member).name
        if annotation_name in seen_annotation_names:
            raise GuitarSetFormatError(
                f"duplicate annotation basename {annotation_name!r}"
            )
        seen_annotation_names.add(annotation_name)
        expected_audio_name = f"{annotation_name[:-5]}_mix.wav"
        audio_member = audio_by_name.get(expected_audio_name)
        if audio_member is None:
            raise GuitarSetFormatError(
                f"missing {expected_audio_name!r} in {audio_zip} for {member!r}"
            )
        tracks.append(
            GuitarSetTrack(
                player_id=player_id,
                annotation_zip=annotation_zip,
                annotation_member=member,
                audio_zip=audio_zip,
                audio_member=audio_member,
            )
        )

    return tuple(sorted(tracks, key=lambda track: track.annotation_member))


def _seconds(value: Any, field: str, *, positive: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        qualifier = "> 0" if positive else ">= 0"
        raise GuitarSetFormatError(f"{field} must be a finite number {qualifier}")
    converted = float(value)
    valid_sign = converted > 0 if positive else converted >= 0
    if not math.isfinite(converted) or not valid_sign:
        qualifier = "> 0" if positive else ">= 0"
        raise GuitarSetFormatError(f"{field} must be a finite number {qualifier}")
    return converted


def _sample_positions(observation: Dict[str, Any]) -> NoteBoundary:
    time = _seconds(observation.get("time"), "time", positive=False)
    duration = _seconds(observation.get("duration"), "duration", positive=True)
    end = time + duration
    onset_position = time * SAMPLE_RATE
    offset_position = end * SAMPLE_RATE
    if not math.isfinite(onset_position) or not math.isfinite(offset_position):
        raise GuitarSetFormatError("time and duration exceed the sample range")
    onset = round(onset_position)
    offset = round(offset_position)
    return NoteBoundary(onset, offset)


def load_boundary_slots(
    annotation_zip: PathInput,
    annotation_member: str,
) -> BoundarySlots:
    """Read one JAMS member in-place and return exactly six sorted slots.

    Only ``note_midi`` annotations are used, solely for their ``time`` and
    ``duration`` fields.  ``annotation_metadata.data_source`` selects an
    internal string slot from 0 through 5.  Each timestamp is converted using
    ``round(seconds * 44100)``; invalid, non-finite, negative, zero-length, or
    out-of-range values are rejected.
    """

    if not isinstance(annotation_member, str) or not annotation_member:
        raise GuitarSetFormatError("annotation_member must be a non-empty string")
    player_id = _member_player(annotation_member)
    if player_id not in ALLOWED_PLAYERS:
        raise GuitarSetFormatError(
            f"player {player_id!r} is not allowed; expected 00 through 04"
        )

    archive_path = Path(annotation_zip)
    try:
        with zipfile.ZipFile(archive_path, "r") as archive:
            with archive.open(annotation_member, "r") as member_stream:
                raw_jams = member_stream.read()
    except (OSError, KeyError, zipfile.BadZipFile) as exc:
        raise GuitarSetError(
            f"cannot read {annotation_member!r} from {archive_path}"
        ) from exc

    try:
        document = json.loads(raw_jams.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GuitarSetFormatError(
            f"invalid JAMS JSON in {annotation_member!r}"
        ) from exc
    if not isinstance(document, dict):
        raise GuitarSetFormatError("JAMS root must be an object")
    annotations = document.get("annotations")
    if not isinstance(annotations, list):
        raise GuitarSetFormatError("JAMS annotations must be a list")

    slots: List[List[NoteBoundary]] = [[] for _ in range(SLOT_COUNT)]
    for annotation in annotations:
        if not isinstance(annotation, dict):
            raise GuitarSetFormatError("each JAMS annotation must be an object")
        if annotation.get("namespace") != "note_midi":
            continue
        metadata = annotation.get("annotation_metadata")
        if not isinstance(metadata, dict):
            raise GuitarSetFormatError(
                "note_midi annotation_metadata must be an object"
            )
        source = metadata.get("data_source")
        if isinstance(source, str) and source in ("0", "1", "2", "3", "4", "5"):
            slot = int(source)
        elif (
            not isinstance(source, bool)
            and isinstance(source, int)
            and 0 <= source < SLOT_COUNT
        ):
            slot = source
        else:
            raise GuitarSetFormatError(
                "note_midi data_source must identify slot 0..5"
            )
        observations = annotation.get("data")
        if not isinstance(observations, list):
            raise GuitarSetFormatError("note_midi data must be a list")
        for observation in observations:
            if not isinstance(observation, dict):
                raise GuitarSetFormatError("each note_midi observation must be an object")
            slots[slot].append(_sample_positions(observation))

    return tuple(tuple(sorted(slot)) for slot in slots)


__all__ = [
    "ALLOWED_PLAYERS",
    "ANNOTATION_ARCHIVE_NAME",
    "AUDIO_ARCHIVE_NAME",
    "BoundarySlots",
    "GuitarSetError",
    "GuitarSetFormatError",
    "GuitarSetTrack",
    "NoteBoundary",
    "SAMPLE_RATE",
    "SLOT_COUNT",
    "index_guitarset",
    "load_boundary_slots",
]
