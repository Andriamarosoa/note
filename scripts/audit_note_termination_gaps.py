"""Audit GuitarSet note endings relative to the next onset on the same string.

GuitarSet does not label performer intent for offsets. This script therefore
reports observable timing relations only. A positive gap means the estimated
note offset occurs before the next onset on the same string; zero means the
note ends exactly at retrigger; a negative gap means the next onset arrives
before the previous estimated offset.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from causal_note.guitarset import ALLOWED_PLAYERS, SAMPLE_RATE, index_guitarset, load_boundary_slots
from scripts.train_boundaries import inspect_pcm16_mono_wav, split_tracks_by_group


def _gap_bin(samples: int) -> str:
    if samples < 0:
        return "overlap_negative"
    if samples == 0:
        return "same_sample"
    ms = samples * 1000.0 / SAMPLE_RATE
    if ms <= 5:
        return "0_to_5_ms"
    if ms <= 10:
        return "5_to_10_ms"
    if ms <= 20:
        return "10_to_20_ms"
    if ms <= 50:
        return "20_to_50_ms"
    if ms <= 100:
        return "50_to_100_ms"
    if ms <= 250:
        return "100_to_250_ms"
    if ms <= 500:
        return "250_to_500_ms"
    return "over_500_ms"


def _audit(tracks):
    bins = Counter()
    total_notes = 0
    successive_pairs = 0
    nonempty_string_sequences = 0
    terminal_before_audio_end = 0
    terminal_at_or_after_audio_end = 0
    terminal_gap_bins = Counter()

    for track in tracks:
        slots = load_boundary_slots(track.annotation_zip, track.annotation_member)
        frame_count = inspect_pcm16_mono_wav(track.audio_zip, track.audio_member).frame_count
        for notes in slots:
            notes = tuple(notes)
            total_notes += len(notes)
            if not notes:
                continue
            nonempty_string_sequences += 1
            for previous, current in zip(notes, notes[1:]):
                successive_pairs += 1
                bins[_gap_bin(current.onset_sample - previous.offset_sample)] += 1

            final_gap = frame_count - notes[-1].offset_sample
            if final_gap > 0:
                terminal_before_audio_end += 1
                terminal_gap_bins[_gap_bin(final_gap)] += 1
            else:
                terminal_at_or_after_audio_end += 1

    positive_gap = sum(
        count for name, count in bins.items()
        if name not in ("overlap_negative", "same_sample")
    )
    return {
        "tracks": len(tracks),
        "total_notes": total_notes,
        "successive_pairs": successive_pairs,
        "nonempty_string_sequences": nonempty_string_sequences,
        "relations": dict(sorted(bins.items())),
        "offset_before_next_onset": positive_gap,
        "offset_before_next_onset_fraction": positive_gap / successive_pairs if successive_pairs else 0.0,
        "same_sample": bins["same_sample"],
        "overlap_next_onset_before_offset": bins["overlap_negative"],
        "terminal_notes": nonempty_string_sequences,
        "terminal_before_audio_end": terminal_before_audio_end,
        "terminal_at_or_after_audio_end": terminal_at_or_after_audio_end,
        "terminal_gap_bins": dict(sorted(terminal_gap_bins.items())),
    }


def main():
    indexed = tuple(
        track for track in index_guitarset(ROOT / "data" / "GuitarSet")
        if track.player_id in ALLOWED_PLAYERS
    )
    train, validation = split_tracks_by_group(indexed, validation_fraction=0.2, seed=1337)
    report = {
        "schema_version": 1,
        "scope": "GuitarSet players 00-04; player 05 locked",
        "sample_rate": SAMPLE_RATE,
        "train": _audit(train),
        "validation": _audit(validation),
    }
    combined = Counter(report["train"]["relations"])
    combined.update(report["validation"]["relations"])
    total_pairs = report["train"]["successive_pairs"] + report["validation"]["successive_pairs"]
    positive = sum(
        count for name, count in combined.items()
        if name not in ("overlap_negative", "same_sample")
    )
    report["combined"] = {
        "tracks": report["train"]["tracks"] + report["validation"]["tracks"],
        "total_notes": report["train"]["total_notes"] + report["validation"]["total_notes"],
        "successive_pairs": total_pairs,
        "relations": dict(sorted(combined.items())),
        "offset_before_next_onset": positive,
        "offset_before_next_onset_fraction": positive / total_pairs if total_pairs else 0.0,
        "same_sample": combined["same_sample"],
        "overlap_next_onset_before_offset": combined["overlap_negative"],
        "terminal_notes": report["train"]["terminal_notes"] + report["validation"]["terminal_notes"],
        "terminal_before_audio_end": report["train"]["terminal_before_audio_end"] + report["validation"]["terminal_before_audio_end"],
        "terminal_at_or_after_audio_end": report["train"]["terminal_at_or_after_audio_end"] + report["validation"]["terminal_at_or_after_audio_end"],
    }
    output = ROOT / "model" / "note-termination-gap-audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
