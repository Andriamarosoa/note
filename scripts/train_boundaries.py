"""Reproducible CPU training for the six-slot causal boundary model.

The dataset is read directly from the two GuitarSet ZIP archives.  No archive
member is extracted to disk, and importing this module does not import NumPy or
TensorFlow.  A normal invocation from the repository root is::

    python scripts/train_boundaries.py data/GuitarSet --output model/boundaries.keras

Use ``--smoke`` for a bounded one-epoch wiring check.
"""

from array import array
import argparse
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import random
import re
import sys
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple, Union
import wave
import zipfile


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from causal_note.guitarset import (  # noqa: E402 - local source bootstrap above
    ALLOWED_PLAYERS,
    BoundarySlots,
    GuitarSetTrack,
    SAMPLE_RATE,
    SLOT_COUNT,
    index_guitarset,
    load_boundary_slots,
)
from causal_note.neural_model import (  # noqa: E402
    DEFAULT_DILATION_RATES,
    OUTPUT_NAMES,
    build_causal_boundary_model,
    calculate_receptive_field,
)


PathInput = Union[str, os.PathLike]
DEFAULT_POSITIVE_WEIGHTS = {
    "onset": 64.0,
    "offset": 64.0,
}


class TrainingDataError(ValueError):
    """Raised when training audio or the leakage-safe split is invalid."""


@dataclass(frozen=True)
class Pcm16Info:
    """Validated mono PCM16 WAV metadata."""

    sample_rate: int
    frame_count: int


@dataclass(frozen=True)
class Pcm16Audio:
    """Validated mono PCM16 samples in their signed integer representation."""

    sample_rate: int
    samples: array

    @property
    def frame_count(self) -> int:
        return len(self.samples)


@dataclass(frozen=True)
class _TrainingTrack:
    track: GuitarSetTrack
    slots: BoundarySlots
    frame_count: int


def _validate_wave(reader: wave.Wave_read, source: str) -> Pcm16Info:
    if reader.getcomptype() != "NONE":
        raise TrainingDataError(f"{source} must contain uncompressed PCM audio")
    if reader.getnchannels() != 1:
        raise TrainingDataError(f"{source} must be mono")
    if reader.getsampwidth() != 2:
        raise TrainingDataError(f"{source} must use 16-bit PCM samples")
    if reader.getframerate() != SAMPLE_RATE:
        raise TrainingDataError(
            f"{source} must use {SAMPLE_RATE} Hz, got {reader.getframerate()} Hz"
        )
    frame_count = reader.getnframes()
    if frame_count <= 0:
        raise TrainingDataError(f"{source} contains no audio frames")
    return Pcm16Info(reader.getframerate(), frame_count)


def inspect_pcm16_mono_wav(audio_zip: PathInput, audio_member: str) -> Pcm16Info:
    """Validate one WAV member in-place and return its format and length."""

    archive_path = Path(audio_zip)
    source = f"{audio_member!r} in {archive_path}"
    try:
        with zipfile.ZipFile(archive_path, "r") as archive:
            with archive.open(audio_member, "r") as member_stream:
                with wave.open(member_stream, "rb") as reader:
                    return _validate_wave(reader, source)
    except TrainingDataError:
        raise
    except (EOFError, KeyError, OSError, wave.Error, zipfile.BadZipFile) as exc:
        raise TrainingDataError(f"cannot read PCM16 WAV {source}") from exc


def decode_pcm16_mono_wav(audio_zip: PathInput, audio_member: str) -> Pcm16Audio:
    """Decode one 44.1 kHz mono PCM16 WAV member without extracting it.

    Samples remain signed 16-bit integers.  Conversion to floating point is
    delayed until a selected training window is assembled.
    """

    archive_path = Path(audio_zip)
    source = f"{audio_member!r} in {archive_path}"
    try:
        with zipfile.ZipFile(archive_path, "r") as archive:
            with archive.open(audio_member, "r") as member_stream:
                with wave.open(member_stream, "rb") as reader:
                    info = _validate_wave(reader, source)
                    raw_samples = reader.readframes(info.frame_count)
    except TrainingDataError:
        raise
    except (EOFError, KeyError, OSError, wave.Error, zipfile.BadZipFile) as exc:
        raise TrainingDataError(f"cannot read PCM16 WAV {source}") from exc

    expected_bytes = info.frame_count * 2
    if len(raw_samples) != expected_bytes:
        raise TrainingDataError(
            f"truncated PCM16 WAV {source}: expected {expected_bytes} sample bytes, "
            f"got {len(raw_samples)}"
        )
    samples = array("h")
    samples.frombytes(raw_samples)
    if sys.byteorder != "little":
        samples.byteswap()
    return Pcm16Audio(info.sample_rate, samples)


def group_stem(track_or_member: Union[GuitarSetTrack, str]) -> str:
    """Return a composition group without player or comp/solo suffix."""

    member = (
        track_or_member.annotation_member
        if isinstance(track_or_member, GuitarSetTrack)
        else track_or_member
    )
    if not isinstance(member, str) or not member:
        raise TrainingDataError("annotation member must be a non-empty string")
    basename = PurePosixPath(member).name
    stem = basename[:-5] if basename.endswith(".jams") else Path(basename).stem
    match = re.fullmatch(r"(\d{2})_(.+)", stem)
    if match is None:
        raise TrainingDataError(
            f"annotation stem {stem!r} must start with a two-digit player prefix"
        )
    player_id, unprefixed = match.groups()
    if player_id not in ALLOWED_PLAYERS:
        raise TrainingDataError(
            f"player {player_id!r} is outside the training guard 00 through 04"
        )
    # ``comp`` and ``solo`` are two arrangements of the same underlying
    # composition.  Keeping them together avoids repertoire leakage.
    composition = re.sub(r"_(?:comp|solo)$", "", unprefixed)
    if not composition:
        raise TrainingDataError("composition group must not be empty")
    return composition


def split_tracks_by_group(
    tracks: Sequence[GuitarSetTrack],
    *,
    validation_fraction: float = 0.2,
    seed: int = 1337,
) -> Tuple[Tuple[GuitarSetTrack, ...], Tuple[GuitarSetTrack, ...]]:
    """Split tracks without placing one composition group on both sides."""

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TrainingDataError("seed must be an integer")
    if (
        isinstance(validation_fraction, bool)
        or not isinstance(validation_fraction, (int, float))
        or not math.isfinite(float(validation_fraction))
        or not 0.0 < float(validation_fraction) < 1.0
    ):
        raise TrainingDataError("validation_fraction must be strictly between 0 and 1")

    grouped: Dict[str, List[GuitarSetTrack]] = defaultdict(list)
    for track in tracks:
        if track.player_id not in ALLOWED_PLAYERS:
            raise TrainingDataError(
                f"player {track.player_id!r} is outside the training guard 00 through 04"
            )
        grouped[group_stem(track)].append(track)
    if len(grouped) < 2:
        raise TrainingDataError(
            "at least two distinct unprefixed stems are required for train/validation"
        )

    group_names = sorted(grouped)
    random.Random(seed).shuffle(group_names)
    validation_count = int(len(group_names) * float(validation_fraction) + 0.5)
    validation_count = max(1, min(len(group_names) - 1, validation_count))
    validation_groups = frozenset(group_names[:validation_count])

    train = tuple(
        sorted(
            (
                track
                for group, group_tracks in grouped.items()
                if group not in validation_groups
                for track in group_tracks
            ),
            key=lambda item: item.annotation_member,
        )
    )
    validation = tuple(
        sorted(
            (
                track
                for group, group_tracks in grouped.items()
                if group in validation_groups
                for track in group_tracks
            ),
            key=lambda item: item.annotation_member,
        )
    )
    return train, validation


def _validate_window(slots: BoundarySlots, start_sample: int, window_samples: int) -> None:
    if len(slots) != SLOT_COUNT:
        raise TrainingDataError(f"boundary slots must contain exactly {SLOT_COUNT} slots")
    if isinstance(start_sample, bool) or not isinstance(start_sample, int) or start_sample < 0:
        raise TrainingDataError("start_sample must be an integer >= 0")
    if (
        isinstance(window_samples, bool)
        or not isinstance(window_samples, int)
        or window_samples <= 0
    ):
        raise TrainingDataError("window_samples must be an integer > 0")


def _validate_onset_target_width(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TrainingDataError(
            "onset_target_width_samples must be an integer > 0"
        )
    return value


def _validate_offset_target_width(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TrainingDataError(
            "offset_target_width_samples must be an integer > 0"
        )
    return value


def make_window_targets(
    slots: BoundarySlots,
    *,
    start_sample: int,
    window_samples: int,
    onset_target_width_samples: int = 1,
    offset_target_width_samples: int = 1,
) -> Dict[str, List[List[float]]]:
    """Build causal onset and offset windows for six slots.

    With the legacy onset width of one, a note ``[100, 700)`` has targets only
    at samples 100 and 700. Wider targets occupy the absolute causal intervals
    ``[onset, onset + onset_width)`` and ``[offset, offset + offset_width)``
    wherever they intersect this crop.
    """

    _validate_window(slots, start_sample, window_samples)
    onset_width = _validate_onset_target_width(onset_target_width_samples)
    offset_width = _validate_offset_target_width(offset_target_width_samples)
    targets = {
        name: [[0.0] * SLOT_COUNT for _ in range(window_samples)]
        for name in OUTPUT_NAMES
    }
    end_sample = start_sample + window_samples
    for slot_index, notes in enumerate(slots):
        for note in notes:
            first_onset_sample = max(start_sample, note.onset_sample)
            last_onset_sample = min(
                end_sample,
                note.onset_sample + onset_width,
            )
            for absolute_sample in range(first_onset_sample, last_onset_sample):
                targets["onset"][absolute_sample - start_sample][slot_index] = 1.0
            first_offset_sample = max(start_sample, note.offset_sample)
            last_offset_sample = min(
                end_sample,
                note.offset_sample + offset_width,
            )
            for absolute_sample in range(first_offset_sample, last_offset_sample):
                targets["offset"][absolute_sample - start_sample][slot_index] = 1.0
    return targets


# Readable aliases for callers that think in labels rather than model targets.
build_sample_labels = make_window_targets
load_pcm16_mono = decode_pcm16_mono_wav


def make_positive_sample_weights(
    targets: Mapping[str, Sequence[Sequence[float]]],
    positive_weights: Mapping[str, float] = DEFAULT_POSITIVE_WEIGHTS,
) -> Dict[str, List[List[float]]]:
    """Return strictly positive per-element weights for both targets."""

    result: Dict[str, List[List[float]]] = {}
    for name in OUTPUT_NAMES:
        if name not in targets or name not in positive_weights:
            raise TrainingDataError(f"missing target or positive weight for {name!r}")
        positive_weight = float(positive_weights[name])
        if not math.isfinite(positive_weight) or positive_weight <= 0.0:
            raise TrainingDataError(f"positive weight for {name!r} must be finite and > 0")
        result[name] = [
            [positive_weight if float(value) > 0.0 else 1.0 for value in row]
            for row in targets[name]
        ]
    return result


def _prepare_tracks(tracks: Sequence[GuitarSetTrack]) -> Tuple[_TrainingTrack, ...]:
    prepared: List[_TrainingTrack] = []
    for track in tracks:
        slots = load_boundary_slots(track.annotation_zip, track.annotation_member)
        info = inspect_pcm16_mono_wav(track.audio_zip, track.audio_member)
        for notes in slots:
            for note in notes:
                if note.offset_sample > info.frame_count:
                    raise TrainingDataError(
                        f"boundary {note} exceeds {info.frame_count} frames in "
                        f"{track.audio_member!r}"
                    )
        prepared.append(_TrainingTrack(track, slots, info.frame_count))
    if not prepared:
        raise TrainingDataError("the selected split contains no tracks")
    return tuple(prepared)


class _PcmCache:
    def __init__(self, capacity: int = 2) -> None:
        self._capacity = capacity
        self._items: "OrderedDict[Tuple[Path, str], Pcm16Audio]" = OrderedDict()

    def get(self, item: _TrainingTrack) -> Pcm16Audio:
        key = (item.track.audio_zip, item.track.audio_member)
        cached = self._items.pop(key, None)
        if cached is None:
            cached = decode_pcm16_mono_wav(*key)
        self._items[key] = cached
        while len(self._items) > self._capacity:
            self._items.popitem(last=False)
        if cached.frame_count != item.frame_count:
            raise TrainingDataError(f"WAV length changed for {item.track.audio_member!r}")
        return cached


def _load_numpy():
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError(
            "NumPy is required for training; install requirements-train.txt"
        ) from exc
    return np


class BalancedWindowBatcher:
    """Infinite deterministic batches cycling onset, offset and random windows.

    Each selected boundary is placed in the second half of its causal window.
    The remainder of every sample-wise target naturally supplies abundant
    negative examples.
    """

    def __init__(
        self,
        tracks: Sequence[_TrainingTrack],
        *,
        window_samples: int,
        batch_size: int,
        seed: int,
        positive_weights: Mapping[str, float],
        onset_target_width_samples: int = 1,
        offset_target_width_samples: int = 1,
        warmup_samples: int = 0,
        numpy_module=None,
    ) -> None:
        if not tracks:
            raise TrainingDataError("cannot batch an empty track split")
        if window_samples <= 0 or batch_size <= 0:
            raise TrainingDataError("window_samples and batch_size must be > 0")
        self._tracks = tuple(tracks)
        self._window_samples = window_samples
        self._batch_size = batch_size
        self._onset_target_width_samples = _validate_onset_target_width(
            onset_target_width_samples
        )
        self._offset_target_width_samples = _validate_offset_target_width(
            offset_target_width_samples
        )
        if (
            isinstance(warmup_samples, bool)
            or not isinstance(warmup_samples, int)
            or not 0 <= warmup_samples < window_samples
        ):
            raise TrainingDataError(
                "warmup_samples must be an integer in [0, window_samples)"
            )
        self._warmup_samples = warmup_samples
        self._rng = random.Random(seed)
        self._np = numpy_module if numpy_module is not None else _load_numpy()
        self._weights = {
            name: float(positive_weights[name]) for name in OUTPUT_NAMES
        }
        for name, value in self._weights.items():
            if not math.isfinite(value) or value <= 0.0:
                raise TrainingDataError(
                    f"positive weight for {name!r} must be finite and > 0"
                )
        self._anchors: Dict[str, List[Tuple[int, int]]] = {
            "onset": [],
            "offset": [],
        }
        for track_index, item in enumerate(self._tracks):
            if item.frame_count < self._window_samples:
                raise TrainingDataError(
                    f"{item.track.audio_member!r} is shorter than one "
                    "training window"
                )
            for notes in item.slots:
                for note in notes:
                    if note.onset_sample < item.frame_count:
                        self._anchors["onset"].append(
                            (track_index, note.onset_sample)
                        )
                    # An offset exactly at the exclusive WAV end has no audio
                    # sample to supervise and must not create synthetic padding.
                    if note.offset_sample < item.frame_count:
                        self._anchors["offset"].append(
                            (track_index, note.offset_sample)
                        )
        if not self._anchors["onset"] or not self._anchors["offset"]:
            raise TrainingDataError("the selected split contains no note boundaries")
        self._next_kind = 0
        self._cache = _PcmCache()

    def __iter__(self) -> "BalancedWindowBatcher":
        return self

    def _window_start(self, boundary_sample: int, frame_count: int) -> int:
        earliest_local_boundary = max(
            self._window_samples // 2,
            self._warmup_samples,
        )
        local_boundary = self._rng.randint(
            earliest_local_boundary,
            self._window_samples - 1,
        )
        desired = max(0, boundary_sample - local_boundary)
        return min(desired, frame_count - self._window_samples)

    def _fill_targets(self, targets, item: _TrainingTrack, row: int, start: int) -> None:
        end = start + self._window_samples
        for slot_index, notes in enumerate(item.slots):
            for note in notes:
                first_onset_sample = max(start, note.onset_sample)
                last_onset_sample = min(
                    end,
                    note.onset_sample + self._onset_target_width_samples,
                )
                if first_onset_sample < last_onset_sample:
                    targets["onset"][
                        row,
                        first_onset_sample - start : last_onset_sample - start,
                        slot_index,
                    ] = 1.0
                first_offset_sample = max(start, note.offset_sample)
                last_offset_sample = min(
                    end,
                    note.offset_sample + self._offset_target_width_samples,
                )
                if first_offset_sample < last_offset_sample:
                    targets["offset"][
                        row,
                        first_offset_sample - start : last_offset_sample - start,
                        slot_index,
                    ] = 1.0

    def __next__(self):
        np = self._np
        audio_batch = np.zeros(
            (self._batch_size, self._window_samples, 1),
            dtype=np.float32,
        )
        targets = {
            name: np.zeros(
                (self._batch_size, self._window_samples, SLOT_COUNT),
                dtype=np.float32,
            )
            for name in OUTPUT_NAMES
        }
        valid_loss_starts = [0] * self._batch_size

        for row in range(self._batch_size):
            kind = ("onset", "offset", "random")[self._next_kind]
            self._next_kind = (self._next_kind + 1) % 3
            if kind == "random":
                track_index = self._rng.randrange(len(self._tracks))
                item = self._tracks[track_index]
                start = self._rng.randint(
                    0,
                    item.frame_count - self._window_samples,
                )
            else:
                track_index, boundary_sample = self._rng.choice(
                    self._anchors[kind]
                )
                item = self._tracks[track_index]
                start = self._window_start(boundary_sample, item.frame_count)
            decoded = self._cache.get(item)
            end = start + self._window_samples
            integer_window = decoded.samples[start:end]
            if len(integer_window) != self._window_samples:
                raise TrainingDataError("training windows must never be padded")
            float_window = np.asarray(integer_window, dtype=np.float32) / 32768.0
            audio_batch[row, :, 0] = float_window
            self._fill_targets(targets, item, row, start)
            if start > 0:
                valid_loss_starts[row] = self._warmup_samples

        # Keep the slot axis so a positive target weights only that boundary
        # element, not the five negative slots at the same sample.
        sample_weights = {
            name: np.where(
                targets[name] > 0.0,
                self._weights[name],
                1.0,
            ).astype(np.float32, copy=False)
            for name in OUTPUT_NAMES
        }
        for weights in sample_weights.values():
            for row, valid_start in enumerate(valid_loss_starts):
                if valid_start:
                    weights[row, :valid_start, :] = 0.0
        return audio_batch, targets, sample_weights


def _infinite_batches(batcher: BalancedWindowBatcher) -> Iterator[object]:
    while True:
        yield next(batcher)


def _repeat_fixed_batches(batches: Sequence[object]) -> Iterator[object]:
    if not batches:
        raise TrainingDataError("validation requires at least one fixed batch")
    while True:
        for batch in batches:
            yield batch


def _positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be an integer > 0")
    return value


def _fraction(text: str) -> float:
    value = float(text)
    if not 0.0 < value < 1.0:
        raise argparse.ArgumentTypeError("must be strictly between 0 and 1")
    return value


def _positive_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError("must be finite and > 0")
    return value


def _dilation_rates(text: str) -> Tuple[int, ...]:
    try:
        values = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be comma-separated positive integers") from exc
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("must be comma-separated positive integers")
    return values


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the six-slot causal onset/offset model on GuitarSet ZIPs."
    )
    parser.add_argument(
        "dataset_dir",
        nargs="?",
        type=Path,
        default=REPOSITORY_ROOT / "data" / "GuitarSet",
        help="directory containing annotation.zip and audio_mono-pickup_mix.zip",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / "model" / "causal-boundaries.keras",
        help="destination .keras model",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="JSON metadata path (default: beside --output)",
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--players",
        nargs="+",
        choices=sorted(ALLOWED_PLAYERS),
        default=sorted(ALLOWED_PLAYERS),
        help="allowed development players to include (00 through 04 only)",
    )
    parser.add_argument("--validation-fraction", type=_fraction, default=0.2)
    parser.add_argument("--window-samples", type=_positive_int, default=8192)
    parser.add_argument(
        "--onset-target-width-samples",
        type=_positive_int,
        default=1,
        help=(
            "causal binary onset target width; 1 preserves exact-sample labels"
        ),
    )
    parser.add_argument(
        "--offset-target-width-samples",
        type=_positive_int,
        default=1,
        help=(
            "causal binary offset target width; 1 preserves exact-sample labels"
        ),
    )
    parser.add_argument("--batch-size", type=_positive_int, default=8)
    parser.add_argument("--epochs", type=_positive_int, default=20)
    parser.add_argument("--steps-per-epoch", type=_positive_int, default=200)
    parser.add_argument("--validation-steps", type=_positive_int, default=50)
    parser.add_argument("--learning-rate", type=_positive_float, default=1e-3)
    parser.add_argument("--filters", type=_positive_int, default=24)
    parser.add_argument("--kernel-size", type=_positive_int, default=5)
    parser.add_argument(
        "--dilation-rates",
        type=_dilation_rates,
        default=DEFAULT_DILATION_RATES,
        help="comma-separated causal convolution dilation rates",
    )
    parser.add_argument(
        "--onset-positive-weight",
        type=_positive_float,
        default=DEFAULT_POSITIVE_WEIGHTS["onset"],
    )
    parser.add_argument(
        "--offset-positive-weight",
        type=_positive_float,
        default=DEFAULT_POSITIVE_WEIGHTS["offset"],
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="bounded CPU wiring check: 1 epoch, 2 train steps, 1 validation step",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="allow replacing an existing model and metadata file",
    )
    return parser


def _configure_cpu_dependencies(seed: int):
    # These must be set before TensorFlow is imported.
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    np = _load_numpy()
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is required for training; install requirements-train.txt"
        ) from exc

    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.set_visible_devices([], "GPU")
    except RuntimeError as exc:
        raise RuntimeError("TensorFlow was initialized before CPU-only setup") from exc
    try:
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)
    except RuntimeError:
        # A host may have fixed its thread pools before calling main(); the
        # random and deterministic-op guards above still apply.
        pass
    enable_determinism = getattr(tf.config.experimental, "enable_op_determinism", None)
    if enable_determinism is not None:
        enable_determinism()
    return np, tf


def _effective_settings(arguments: argparse.Namespace) -> Dict[str, object]:
    settings = {
        "window_samples": arguments.window_samples,
        "batch_size": arguments.batch_size,
        "epochs": arguments.epochs,
        "steps_per_epoch": arguments.steps_per_epoch,
        "validation_steps": arguments.validation_steps,
        "filters": arguments.filters,
        "kernel_size": arguments.kernel_size,
        "dilation_rates": tuple(arguments.dilation_rates),
    }
    if arguments.smoke:
        settings.update(
            window_samples=min(arguments.window_samples, 2048),
            batch_size=min(arguments.batch_size, 2),
            epochs=1,
            steps_per_epoch=min(arguments.steps_per_epoch, 2),
            validation_steps=min(arguments.validation_steps, 1),
            filters=min(arguments.filters, 4),
            dilation_rates=tuple(arguments.dilation_rates[:2]),
        )
    return settings


def _metadata_path(output: Path, explicit: Optional[Path]) -> Path:
    if explicit is not None:
        return explicit
    if output.suffix:
        return output.with_suffix(".metadata.json")
    return Path(str(output) + ".metadata.json")


def _count_notes(items: Sequence[_TrainingTrack]) -> int:
    return sum(len(notes) for item in items for notes in item.slots)


def _count_exclusive_end_offsets(items: Sequence[_TrainingTrack]) -> int:
    return sum(
        note.offset_sample == item.frame_count
        for item in items
        for notes in item.slots
        for note in notes
    )


def run_training(arguments: argparse.Namespace) -> Dict[str, object]:
    """Execute one configured training run and return its JSON metadata."""

    dataset_dir = Path(arguments.dataset_dir).resolve()
    output = Path(arguments.output).resolve()
    metadata_path = _metadata_path(output, arguments.metadata).resolve()
    history_path = output.with_suffix(".history.csv")
    checkpoint_dir = output.parent / f"{output.stem}.epochs"
    if output == metadata_path:
        raise TrainingDataError("model and metadata paths must be different")
    existing = tuple(
        path
        for path in (output, metadata_path, history_path, checkpoint_dir)
        if path.exists()
    )
    if existing and not arguments.overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"refusing to replace existing training artifacts: {joined}; "
            "pass --overwrite explicitly"
        )
    selected_players = tuple(dict.fromkeys(arguments.players))
    if not selected_players or any(
        player not in ALLOWED_PLAYERS for player in selected_players
    ):
        raise TrainingDataError("players must be selected from 00 through 04")
    settings = _effective_settings(arguments)
    receptive_field = calculate_receptive_field(
        int(settings["kernel_size"]),
        settings["dilation_rates"],
    )
    if int(settings["window_samples"]) < receptive_field:
        raise TrainingDataError(
            "window_samples must be at least the model receptive field "
            f"({receptive_field})"
        )
    positive_weights = {
        "onset": arguments.onset_positive_weight,
        "offset": arguments.offset_positive_weight,
    }
    indexed = tuple(
        track
        for track in index_guitarset(dataset_dir)
        if track.player_id in selected_players
    )
    if not indexed:
        raise TrainingDataError("no tracks match the selected players")
    train_tracks, validation_tracks = split_tracks_by_group(
        indexed,
        validation_fraction=arguments.validation_fraction,
        seed=arguments.seed,
    )
    train_items = _prepare_tracks(train_tracks)
    validation_items = _prepare_tracks(validation_tracks)

    np, tf = _configure_cpu_dependencies(arguments.seed)
    from causal_note.training_losses import elementwise_binary_crossentropy_v1

    train_batcher = BalancedWindowBatcher(
        train_items,
        window_samples=int(settings["window_samples"]),
        batch_size=int(settings["batch_size"]),
        seed=arguments.seed,
        positive_weights=positive_weights,
        onset_target_width_samples=arguments.onset_target_width_samples,
        offset_target_width_samples=arguments.offset_target_width_samples,
        warmup_samples=receptive_field - 1,
        numpy_module=np,
    )
    validation_batcher = BalancedWindowBatcher(
        validation_items,
        window_samples=int(settings["window_samples"]),
        batch_size=int(settings["batch_size"]),
        seed=arguments.seed + 1,
        positive_weights=positive_weights,
        onset_target_width_samples=arguments.onset_target_width_samples,
        offset_target_width_samples=arguments.offset_target_width_samples,
        warmup_samples=receptive_field - 1,
        numpy_module=np,
    )
    fixed_validation_batches = tuple(
        next(validation_batcher)
        for _ in range(int(settings["validation_steps"]))
    )

    model = build_causal_boundary_model(
        filters=int(settings["filters"]),
        kernel_size=int(settings["kernel_size"]),
        dilation_rates=settings["dilation_rates"],
    )
    if int(model.receptive_field) != receptive_field:
        raise AssertionError("model receptive-field metadata is inconsistent")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=arguments.learning_rate),
        loss={
            name: elementwise_binary_crossentropy_v1
            for name in OUTPUT_NAMES
        },
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=arguments.overwrite)
    history = model.fit(
        _infinite_batches(train_batcher),
        steps_per_epoch=int(settings["steps_per_epoch"]),
        epochs=int(settings["epochs"]),
        validation_data=_repeat_fixed_batches(fixed_validation_batches),
        validation_steps=int(settings["validation_steps"]),
        callbacks=[
            tf.keras.callbacks.TerminateOnNaN(),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(checkpoint_dir / "epoch-{epoch:02d}.keras"),
                monitor="val_loss",
                save_best_only=False,
                save_weights_only=False,
                verbose=1,
            ),
            tf.keras.callbacks.CSVLogger(
                str(history_path),
                append=False,
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=3,
                restore_best_weights=True,
            ),
        ],
        workers=0,
        use_multiprocessing=False,
        verbose=2,
    )

    model.save(output)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    train_groups = sorted({group_stem(track) for track in train_tracks})
    validation_groups = sorted({group_stem(track) for track in validation_tracks})
    metadata: Dict[str, object] = {
        "schema_version": 5,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": str(output),
        "metadata_path": str(metadata_path),
        "artifacts": {
            "history_csv": str(history_path),
            "epoch_checkpoint_dir": str(checkpoint_dir),
            "epoch_checkpoint_pattern": "epoch-{epoch:02d}.keras",
        },
        "dataset_dir": str(dataset_dir),
        "archives_read_without_extraction": True,
        "audio": {"sample_rate": SAMPLE_RATE, "channels": 1, "pcm_bits": 16},
        "dataset_player_guard": sorted(ALLOWED_PLAYERS),
        "selected_players": list(selected_players),
        "seed": arguments.seed,
        "cpu_only": True,
        "smoke": bool(arguments.smoke),
        "split": {
            "validation_fraction": arguments.validation_fraction,
            "group_key": (
                "annotation stem without player prefix and terminal comp/solo"
            ),
            "train_groups": train_groups,
            "validation_groups": validation_groups,
            "train_members": [track.annotation_member for track in train_tracks],
            "validation_members": [
                track.annotation_member for track in validation_tracks
            ],
            "train_tracks": len(train_tracks),
            "validation_tracks": len(validation_tracks),
            "train_notes": _count_notes(train_items),
            "validation_notes": _count_notes(validation_items),
            "train_offsets_at_exclusive_audio_end_not_supervised": (
                _count_exclusive_end_offsets(train_items)
            ),
            "validation_offsets_at_exclusive_audio_end_not_supervised": (
                _count_exclusive_end_offsets(validation_items)
            ),
        },
        "training": {
            **settings,
            "dilation_rates": list(settings["dilation_rates"]),
            "learning_rate": arguments.learning_rate,
            "sampling_cycle": ["onset", "offset", "random"],
            "unscored_crop_warmup_samples": receptive_field - 1,
            "fixed_validation_batches": True,
            "early_stopping": {
                "monitor": "val_loss",
                "patience": 3,
                "restore_best_weights": True,
            },
            "positive_sample_weights": positive_weights,
            "targets": {
                "shape": ["batch", "time", "slots"],
                "onset": {
                    "encoding": "causal_binary_window",
                    "support": "[onset, onset + width_samples)",
                    "width_samples": arguments.onset_target_width_samples,
                },
                "offset": {
                    "encoding": "causal_binary_window",
                    "support": "[offset, offset + width_samples)",
                    "width_samples": arguments.offset_target_width_samples,
                },
            },
            "sample_weights": {
                "shape": ["batch", "time", "slots"],
                "semantics": "elementwise boundary-slot weights before loss reduction",
                "positive_by_output": positive_weights,
                "negative": 1.0,
                "warmup": 0.0,
            },
        },
        "model": {
            "outputs": list(OUTPUT_NAMES),
            "slots": SLOT_COUNT,
            "receptive_field_samples": int(model.receptive_field),
        },
        "history": {
            name: [float(value) for value in values]
            for name, values in history.history.items()
        },
    }
    with metadata_path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return metadata


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = create_argument_parser().parse_args(argv)
    metadata = run_training(arguments)
    print(
        json.dumps(
            {
                "model_path": metadata["model_path"],
                "metadata_path": metadata["metadata_path"],
                "train_tracks": metadata["split"]["train_tracks"],
                "validation_tracks": metadata["split"]["validation_tracks"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
