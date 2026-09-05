"""Rich GuitarSet annotation access for acoustic label-confidence audits.

The narrow boundary loader intentionally discards pitch and contour details.
This module keeps those fields so training diagnostics can compare note labels
against the underlying per-string pitch evidence without extracting the JAMS
archive.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path, PurePosixPath
import re
from typing import Any, Dict, List, Tuple, Union
import zipfile

from .guitarset import (
    ALLOWED_PLAYERS,
    GuitarSetError,
    GuitarSetFormatError,
    SAMPLE_RATE,
    SLOT_COUNT,
)


PathInput = Union[str, Path]


@dataclass(frozen=True, order=True)
class RichNote:
    slot: int
    onset_sample: int
    offset_sample: int
    midi: float

    def __post_init__(self) -> None:
        if isinstance(self.slot, bool) or not isinstance(self.slot, int) or not 0 <= self.slot < SLOT_COUNT:
            raise GuitarSetFormatError("slot must be an integer in 0..5")
        if isinstance(self.onset_sample, bool) or not isinstance(self.onset_sample, int) or self.onset_sample < 0:
            raise GuitarSetFormatError("onset_sample must be an integer >= 0")
        if isinstance(self.offset_sample, bool) or not isinstance(self.offset_sample, int) or self.offset_sample <= self.onset_sample:
            raise GuitarSetFormatError("offset_sample must be after onset_sample")
        value = float(self.midi)
        if not math.isfinite(value):
            raise GuitarSetFormatError("midi must be finite")
        object.__setattr__(self, "midi", value)

    @property
    def frequency_hz(self) -> float:
        return 440.0 * (2.0 ** ((self.midi - 69.0) / 12.0))


@dataclass(frozen=True, order=True)
class PitchContourPoint:
    slot: int
    sample: int
    frequency_hz: float
    voiced: bool

    def __post_init__(self) -> None:
        if isinstance(self.slot, bool) or not isinstance(self.slot, int) or not 0 <= self.slot < SLOT_COUNT:
            raise GuitarSetFormatError("slot must be an integer in 0..5")
        if isinstance(self.sample, bool) or not isinstance(self.sample, int) or self.sample < 0:
            raise GuitarSetFormatError("sample must be an integer >= 0")
        frequency = float(self.frequency_hz)
        if not math.isfinite(frequency) or frequency < 0.0:
            raise GuitarSetFormatError("frequency_hz must be finite and >= 0")
        if not isinstance(self.voiced, bool):
            raise GuitarSetFormatError("voiced must be boolean")
        if self.voiced and frequency <= 0.0:
            raise GuitarSetFormatError("voiced contour points require frequency_hz > 0")
        object.__setattr__(self, "frequency_hz", frequency)


@dataclass(frozen=True)
class RichAnnotations:
    notes_by_slot: Tuple[Tuple[RichNote, ...], ...]
    contours_by_slot: Tuple[Tuple[PitchContourPoint, ...], ...]

    def __post_init__(self) -> None:
        if len(self.notes_by_slot) != SLOT_COUNT or len(self.contours_by_slot) != SLOT_COUNT:
            raise GuitarSetFormatError("rich annotations must contain exactly six slots")

    @property
    def notes(self) -> Tuple[RichNote, ...]:
        return tuple(sorted(note for slot in self.notes_by_slot for note in slot))



def _member_player(member: str) -> str:
    name = PurePosixPath(member).name
    match = re.fullmatch(r"(\d{2})_.+\.jams", name)
    if match is None:
        raise GuitarSetFormatError(f"invalid GuitarSet annotation member name: {member!r}")
    return match.group(1)



def _slot(annotation: Dict[str, Any]) -> int:
    metadata = annotation.get("annotation_metadata")
    if not isinstance(metadata, dict):
        raise GuitarSetFormatError("annotation_metadata must be an object")
    source = metadata.get("data_source")
    if isinstance(source, str) and source in tuple(str(index) for index in range(SLOT_COUNT)):
        return int(source)
    if not isinstance(source, bool) and isinstance(source, int) and 0 <= source < SLOT_COUNT:
        return source
    raise GuitarSetFormatError("data_source must identify slot 0..5")



def _finite_number(value: Any, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GuitarSetFormatError(f"{name} must be numeric")
    converted = float(value)
    if not math.isfinite(converted):
        raise GuitarSetFormatError(f"{name} must be finite")
    if minimum is not None and converted < minimum:
        raise GuitarSetFormatError(f"{name} must be >= {minimum}")
    return converted



def _read_document(annotation_zip: PathInput, annotation_member: str) -> Dict[str, Any]:
    if not isinstance(annotation_member, str) or not annotation_member:
        raise GuitarSetFormatError("annotation_member must be a non-empty string")
    player = _member_player(annotation_member)
    if player not in ALLOWED_PLAYERS:
        raise GuitarSetFormatError(f"player {player!r} is not allowed; expected 00 through 04")
    archive_path = Path(annotation_zip)
    try:
        with zipfile.ZipFile(archive_path, "r") as archive:
            raw = archive.read(annotation_member)
    except (OSError, KeyError, zipfile.BadZipFile) as exc:
        raise GuitarSetError(f"cannot read {annotation_member!r} from {archive_path}") from exc
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GuitarSetFormatError(f"invalid JAMS JSON in {annotation_member!r}") from exc
    if not isinstance(document, dict) or not isinstance(document.get("annotations"), list):
        raise GuitarSetFormatError("JAMS root must contain an annotations list")
    return document



def _observations(annotation: Dict[str, Any], namespace: str) -> Tuple[Dict[str, Any], ...]:
    """Normalize JAMS row and vectorized-column serializations to row objects."""
    data = annotation.get("data")
    if isinstance(data, list):
        if any(not isinstance(observation, dict) for observation in data):
            raise GuitarSetFormatError(f"{namespace} observations must be objects")
        return tuple(data)
    if not isinstance(data, dict):
        raise GuitarSetFormatError(f"{namespace} data must be a list or column object")
    required = ("time", "duration", "value")
    if any(not isinstance(data.get(field), list) for field in required):
        raise GuitarSetFormatError(f"{namespace} column data must contain time/duration/value lists")
    length = len(data["time"])
    if len(data["duration"]) != length or len(data["value"]) != length:
        raise GuitarSetFormatError(f"{namespace} column lengths must match")
    confidence = data.get("confidence")
    if confidence is not None and (not isinstance(confidence, list) or len(confidence) != length):
        raise GuitarSetFormatError(f"{namespace} confidence column length must match")
    return tuple(
        {
            "time": data["time"][index],
            "duration": data["duration"][index],
            "value": data["value"][index],
            "confidence": None if confidence is None else confidence[index],
        }
        for index in range(length)
    )



def load_rich_annotations(annotation_zip: PathInput, annotation_member: str) -> RichAnnotations:
    """Load note MIDI values and six per-string pitch contours from one JAMS file."""
    document = _read_document(annotation_zip, annotation_member)
    notes: List[List[RichNote]] = [[] for _ in range(SLOT_COUNT)]
    contours: List[List[PitchContourPoint]] = [[] for _ in range(SLOT_COUNT)]

    for annotation in document["annotations"]:
        if not isinstance(annotation, dict):
            raise GuitarSetFormatError("each annotation must be an object")
        namespace = annotation.get("namespace")
        if namespace not in ("note_midi", "pitch_contour"):
            continue
        slot = _slot(annotation)
        observations = _observations(annotation, namespace)

        if namespace == "note_midi":
            for observation in observations:
                time_s = _finite_number(observation.get("time"), "note time", minimum=0.0)
                duration_s = _finite_number(observation.get("duration"), "note duration", minimum=0.0)
                if duration_s <= 0.0:
                    raise GuitarSetFormatError("note duration must be > 0")
                midi = _finite_number(observation.get("value"), "note midi")
                onset = round(time_s * SAMPLE_RATE)
                offset = round((time_s + duration_s) * SAMPLE_RATE)
                notes[slot].append(RichNote(slot, onset, offset, midi))
            continue

        for observation in observations:
            time_s = _finite_number(observation.get("time"), "contour time", minimum=0.0)
            value = observation.get("value")
            if not isinstance(value, dict):
                raise GuitarSetFormatError("pitch_contour value must be an object")
            voiced = value.get("voiced")
            if not isinstance(voiced, bool):
                raise GuitarSetFormatError("pitch_contour voiced must be boolean")
            frequency = _finite_number(value.get("frequency"), "contour frequency", minimum=0.0)
            contours[slot].append(
                PitchContourPoint(
                    slot=slot,
                    sample=round(time_s * SAMPLE_RATE),
                    frequency_hz=frequency,
                    voiced=voiced and frequency > 0.0,
                )
            )

    return RichAnnotations(
        notes_by_slot=tuple(tuple(sorted(slot)) for slot in notes),
        contours_by_slot=tuple(tuple(sorted(slot)) for slot in contours),
    )


__all__ = [
    "PitchContourPoint",
    "RichAnnotations",
    "RichNote",
    "load_rich_annotations",
]
