"""Render S613 JSONL NOTE events using FluidSynth OFFLINE to a WAV file.

This isolates MIDI event timing/pitch errors from Realtek/PortAudio callback
underruns. A different transcription model is NOT used. No audio device opens.

Generate the events first (do NOT use --soundfont with fast --wav replay):
  python -m scripts.live_strict_notes --wav test_04_E2_puis_E3.wav --out s613_test04.jsonl
  python -m scripts.render_live_midi_offline --events s613_test04.jsonl --soundfont piano.sf2 --out s613_piano_offline.wav
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import wave

import numpy as np

SR = 44100


def load_events(path):
    events = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("type") not in ("note_on", "note_off"):
                continue
            sample = int(event.get("decision_sample", -1))
            if sample < 0 or not 0 <= int(event["pitch"]) <= 127:
                raise ValueError("Invalid note event time/pitch")
            events.append(event)
    if any(events[i]["decision_sample"] > events[i+1]["decision_sample"]
           for i in range(len(events)-1)):
        raise ValueError("Event stream is not time-sorted")
    return events


def render_midi_sequence(synth, events, end_sample, *, chunk=512):
    """Yield mono? No: stereo signed PCM16 arrays honoring sample timestamps.

    All acoustic samples BEFORE an event are rendered before the event is
    applied; unlike a callback draining a batch, this preserves note lengths.
    """
    cursor = 0
    for event in [*events, dict(type="end", decision_sample=end_sample)]:
        target = int(event["decision_sample"])
        if target < cursor:
            raise ValueError("Events out of order")
        while cursor < target:
            frames = min(chunk, target - cursor)
            values = np.asarray(synth.get_samples(frames))
            if values.size != frames * 2:
                raise RuntimeError("FluidSynth did not return stereo interleaved samples")
            if np.issubdtype(values.dtype, np.floating):
                values = np.round(np.clip(values, -1, 1) * 32767).astype("<i2")
            else:
                values = np.asarray(np.clip(values, -32768, 32767), dtype="<i2")
            yield values
            cursor += frames
        pitch = event.get("pitch")
        if event["type"] == "note_on":
            synth.noteon(0, int(pitch), int(event.get("velocity", 96)))
        elif event["type"] == "note_off":
            synth.noteoff(0, int(pitch))


def render(events_path, soundfont, output, *, tail_s=1.5, gain=.35):
    try:
        import fluidsynth
    except (OSError, ImportError) as exc:
        raise RuntimeError("Install pyfluidsynth and native FluidSynth DLL") from exc
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    if not 0 <= tail_s <= 10 or not 0 < gain <= 5:
        raise ValueError("Invalid tail/gain")
    events = load_events(events_path)
    last = int(events[-1]["decision_sample"]) if events else 0
    target = last + int(round(tail_s * SR))
    synth = fluidsynth.Synth(samplerate=SR, gain=gain)
    blocks = 0
    sample_count = 0
    near_clip = 0
    try:
        sfid = synth.sfload(str(soundfont))
        if sfid < 0:
            raise RuntimeError("SoundFont loading failed")
        synth.program_select(0, sfid, 0, 0)
        with wave.open(str(output), "wb") as w:
            w.setframerate(SR)
            w.setnchannels(2)
            w.setsampwidth(2)
            for chunk in render_midi_sequence(synth, events, target):
                w.writeframes(chunk.astype("<i2", copy=False).tobytes())
                near_clip += int((np.abs(chunk.astype(np.int32)) > 32720).sum())
                sample_count += len(chunk)
                blocks += 1
    finally:
        synth.delete()
    return {
        "events": str(events_path), "wav": str(output),
        "duration_s": round(target/SR, 3), "notes_on": sum(e["type"] == "note_on" for e in events),
        "soundfont": str(soundfont), "gain": gain,
        "near_clipped_samples": near_clip, "stereo_samples": sample_count,
        "rendered_blocks": blocks,
        "note": "Non-real-time FluidSynth: no PortAudio callback. If this is clean but live piano glitches, investigate output callback and playback device.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", required=True, type=Path)
    parser.add_argument("--soundfont", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--tail-s", type=float, default=1.5)
    parser.add_argument("--gain", type=float, default=.35)
    args = parser.parse_args()
    print(json.dumps(render(args.events, args.soundfont, args.out,
                            tail_s=args.tail_s, gain=args.gain), indent=2))


if __name__ == "__main__":
    main()
