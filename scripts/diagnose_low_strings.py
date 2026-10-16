"""Diagnose low guitar-string capture separately from NOTE prediction and piano output.

This is a diagnostic only: it never normalizes/multiplies the iRig signal
and does not alter any model, decoder, head or selection in NOTE.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time
import wave

import numpy as np

from scripts.live_audio_io import managed_input_stream, parse_device

SR = 44100
STRINGS = {
    "E2": 40, "A2": 45, "D3": 50,
    "G3": 55, "B3": 59, "E4": 64,
    "D2": 38,
}


def db(value):
    return float(20 * np.log10(max(float(value), 1e-12)))


def read_wav(path):
    with wave.open(str(path), "rb") as reader:
        if reader.getnchannels() != 1 or reader.getframerate() != SR or reader.getsampwidth() != 2:
            raise ValueError("Requires mono 16-bit PCM WAV at 44100 Hz")
        samples = np.frombuffer(reader.readframes(reader.getnframes()), dtype="<i2")
    return np.asarray(samples, dtype=np.float32) / 32768.


def write_wav(path, samples):
    data = np.rint(np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes()
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(SR)
        out.writeframes(data)


def capture_audio(device, seconds):
    pieces = []
    callback_status = []
    blocksize = 256
    max_pieces = int(np.ceil(seconds * SR / blocksize)) + 50

    def callback(indata, frames, _timing, status):
        if status:
            callback_status.append(str(status))
        if len(pieces) < max_pieces:
            pieces.append(np.asarray(indata[:, 0], np.float32).copy())

    with managed_input_stream(
        preferred_device=parse_device(device),
        channels=1,
        samplerate=SR,
        blocksize=blocksize,
        dtype="float32",
        latency="low",
        callback=callback,
    ) as (_stream, info):
        print(f"CAPTURE: {info.device_name} (index={info.device_index}) / "
              f"{info.host_api} / latency={info.latency_ms:.2f} ms")
        print(f"Play a SINGLE guitar string now for {seconds:g} seconds.")
        time.sleep(seconds)

    if not pieces:
        raise RuntimeError("No microphone audio callback arrived")
    samples = np.concatenate(pieces)
    print(f"Callbacks={len(pieces)} statuses={len(callback_status)} samples={samples.size}")
    return samples


def diagnostics(samples, note):
    x = np.asarray(samples, np.float32)
    if not np.isfinite(x).all() or x.size < 16384:
        raise ValueError("Need at least 16384 finite captured PCM samples")
    f0 = 440 * 2 ** ((STRINGS[note] - 69) / 12)
    peak = float(np.max(np.abs(x)))
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    clipped = int(np.count_nonzero(np.abs(x) >= .999))
    print(f"Requested={note} MIDI={STRINGS[note]} f0={f0:.2f}Hz | "
          f"peak={db(peak):.1f}dBFS RMS={db(rms):.1f}dBFS "
          f"clipped_samples={clipped}")
    n = 16384
    # Use the loudest full window to focus on the attack/sustain, not long silence.
    windows = np.lib.stride_tricks.sliding_window_view(x, n)[::4096]
    window_rms = np.mean(windows.astype(np.float64) ** 2, axis=1)
    strongest = windows[int(np.argmax(window_rms))]
    tapered = strongest * np.hanning(n)
    spectrum = np.abs(np.fft.rfft(tapered)) / (np.hanning(n).sum() / 2)
    freq = np.fft.rfftfreq(n, 1. / SR)
    print("Harmonics near the expected low-string pitch (raw spectrum):")
    contrasts = []
    for h in range(1, 6):
        target = f0 * h
        band = (freq >= target - max(4., .025 * target)) & (freq <= target + max(4., .025 * target))
        observed = float(spectrum[band].max()) if np.any(band) else 0.
        baseline = spectrum[(freq >= target - 50) & (freq <= target + 50) & (~band)]
        noise = float(np.median(baseline)) if baseline.size else 1e-12
        contrast = 20 * np.log10(max(observed, 1e-12) / max(noise, 1e-12))
        contrasts.append(contrast)
        print(f"  H{h}: target={target:6.1f}Hz amplitude={db(observed):6.1f}dBFS "
              f"local-contrast={contrast:5.1f}dB")
    local = (freq >= 55) & (freq <= 700)
    strongest_peaks = np.flatnonzero(local)
    chosen = []
    for idx in strongest_peaks[np.argsort(spectrum[strongest_peaks])[::-1]]:
        if any(abs(freq[idx] - old) < 12 for old in chosen):
            continue
        chosen.append(float(freq[idx]))
        if len(chosen) == 6:
            break
    print("Dominant peaks (55-700 Hz): " + ", ".join(f"{v:.1f}Hz" for v in chosen))
    if clipped:
        print("Warning: ADC saturation/clipping can corrupt the spectrum.")
    elif db(rms) < -45:
        print("Warning: very low capture level; check iRig gain and Windows input settings.")
    elif max(contrasts[:3]) < 6:
        print("Warning: expected string harmonics are not clearly above local spectral background.")
    else:
        print("Expected pitch has spectral evidence; if NOTE emits no note_on, investigate model/selection.")
    print("These are spectral diagnostics, NOT ground-truth note-detection accuracy.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--mic", action="store_true", help="Record unprocessed raw iRig audio")
    source.add_argument("--wav", type=Path, help="Analyze an existing mono PCM16 WAV")
    source.add_argument("--test-piano", action="store_true", help="Play the chosen pitch directly with FluidSynth")
    parser.add_argument("--note", choices=tuple(STRINGS), default="E2")
    parser.add_argument("--device", help="sounddevice input name/index; use the exact iRig if needed")
    parser.add_argument("--duration-s", type=float, default=4.0)
    parser.add_argument("--record-wav", type=Path, help="Save raw iRig input as mono WAV for NOTE replay")
    parser.add_argument("--soundfont", type=Path)
    parser.add_argument("--piano-output-device")
    args = parser.parse_args(argv)
    if args.test_piano:
        if args.soundfont is None or not args.soundfont.is_file():
            parser.error("--test-piano needs --soundfont pointing to an existing .sf2")
        from scripts.live_piano_output import PianoOutput
        piano = PianoOutput(args.soundfont, output_device=args.piano_output_device)
        try:
            p = STRINGS[args.note]
            print(f"Direct piano output test: {args.note}, MIDI={p}; bypass NOTE and iRig.")
            piano.send({"type": "note_on", "pitch": p, "velocity": 110})
            time.sleep(1.2)
            piano.send({"type": "note_off", "pitch": p, "velocity": 0})
            time.sleep(.4)
            if piano.health_error():
                raise RuntimeError(piano.health_error())
        finally:
            piano.close()
        return
    if args.mic:
        if not 0 < args.duration_s <= 30:
            parser.error("--duration-s must be in (0,30]")
        samples = capture_audio(args.device, args.duration_s)
        if args.record_wav is not None:
            args.record_wav.parent.mkdir(parents=True, exist_ok=True)
            write_wav(args.record_wav, samples)
            print(f"Recorded unnormalized PCM at {args.record_wav}")
    else:
        samples = read_wav(args.wav)
    diagnostics(samples, args.note)


if __name__ == "__main__":
    main()
