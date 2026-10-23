"""NOTE-only microphone Live: causal Onset NoteOn at every 256 PCM samples.

Runs the signed *research* models from repository note. No NoteOff, MIDI output,
cross-task heads, future audio, attack queue or added confirmation hop.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

SR = 44100
HOP = 256
FOLDS = ("BN1", "Funk2", "Jazz1", "Rock3", "SS1")


def load_engine(args):
    # Avoid importing sounddevice (hardware) or expensive models for --list-devices.
    if args.variant == "recovery-hgb":
        from onset256.acoustic_live import current_reference
        return current_reference(
            fold=args.fold, base_bundle=args.base_bundle,
            recovery_bundle=args.recovery_bundle, native_path=args.native_model)
    if args.variant == "research-original":
        from onset256.live_event_stream import load_research_stream
        return load_research_stream(
            args.base_bundle, fold=args.fold, native_path=args.native_model)
    raise ValueError("Only signed NOTE research variants are allowed")


def device_index(sd, selected):
    if selected is None:
        return None
    devices = sd.query_devices()
    if str(selected).isdigit():
        index = int(selected)
        if index >= len(devices) or int(devices[index]["max_input_channels"]) < 1:
            raise ValueError("The specified audio device has no input channels")
        return index
    found = [(i, d) for i, d in enumerate(devices)
             if str(selected).casefold() in d["name"].casefold()
             and int(d["max_input_channels"]) > 0]
    if len(found) != 1:
        raise ValueError("Input name matches %d devices: %s; use the numeric index" %
                         (len(found), [i for i,_ in found]))
    return found[0][0]


def list_devices(sd):
    print("AUDIO_INPUTS (index | name | host API | channels | default rate)")
    for i, d in enumerate(sd.query_devices()):
        if int(d["max_input_channels"]) >= 1:
            host = sd.query_hostapis(d["hostapi"])["name"]
            print("%3d | %s | %s | %s | %.0f" %
                  (i, d["name"], host, d["max_input_channels"],
                   d["default_samplerate"]))


def checked_hop(stream, pcm, start_sample):
    samples = np.asarray(pcm, dtype=np.float32)
    if samples.shape != (HOP,) or not np.isfinite(samples).all():
        raise ValueError("One finite mono PCM256 hop is required")
    if start_sample % HOP:
        raise AssertionError("Non-contiguous sample index")
    result = stream.push(samples, start_sample=int(start_sample))
    if result.sample != start_sample + HOP:
        raise AssertionError("NOTE Onset returned a delayed or backdated decision")
    if any(e.kind != "note_on" or e.sample != result.sample for e in result.intents):
        raise AssertionError("Only current-boundary Onset NoteOn intents are allowed")
    if len({e.pitch for e in result.intents}) != len(result.intents):
        raise AssertionError("Repeated pitch emitted within one PCM256 hop")
    return result


def meter_db(samples):
    rms = float(np.sqrt(np.mean(np.square(samples.astype(np.float64)))))
    return 20.0 * math.log10(max(rms, 1e-12))


def run(args, *, sd=None):
    if sd is None:
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise RuntimeError("Install sounddevice: python -m pip install sounddevice") from exc
    if args.list_devices:
        list_devices(sd)
        return 0
    if args.duration_s is not None and (not math.isfinite(args.duration_s)
                                         or args.duration_s <= 0):
        raise ValueError("--duration-s must be positive")
    index = device_index(sd, args.input_device)
    # Compiling/training artifacts are loaded before the microphone is opened.
    # Default fold is a fixed, explicitly disclosed research selection.
    engine = load_engine(args)
    print("NOTE_PCM256_LIVE start | 44100 Hz | block=256 | "
          "fold=%s | variant=%s | Onset ONLY (not full MIDI)" %
          (args.fold, args.variant), flush=True)
    if index is not None:
        print("Selected device:", index, sd.query_devices(index)["name"], flush=True)
    else:
        print("AUTO_MIC_INPUT: Windows default input (check Realtek gain; --list-devices to override)", flush=True)
    print("RESEARCH_FOLD_NOTICE: %s is fixed, not inferred from guitar audio or best Live calibration" % args.fold, flush=True)

    out = None
    if args.jsonl is not None:
        args.jsonl.parent.mkdir(parents=True, exist_ok=True)
        out = args.jsonl.open("x", encoding="utf-8")
    count = 0
    notes = 0
    worst_processing_ms = 0.
    interval_peak = -240.
    last_meter = time.perf_counter()
    max_hops = (int(math.ceil(args.duration_s * SR / HOP))
                if args.duration_s is not None else None)
    try:
        with sd.InputStream(
            samplerate=SR, blocksize=HOP, channels=1,
            dtype="float32", device=index, latency="low") as mic:
            while max_hops is None or count < max_hops:
                block, overflowed = mic.read(HOP)
                if overflowed:
                    # The sample chronology cannot be proven after a PortAudio
                    # overflow: never pretend that skipped PCM never existed.
                    raise BufferError("Microphone overflow: stop, do not bridge missing audio")
                pcm = np.ascontiguousarray(block[:, 0], dtype=np.float32)
                begin = time.perf_counter()
                decision = checked_hop(engine, pcm, count * HOP)
                elapsed_ms = (time.perf_counter() - begin) * 1000.
                worst_processing_ms = max(worst_processing_ms, elapsed_ms)
                notes += len(decision.intents)
                interval_peak = max(interval_peak, meter_db(pcm))
                for e in decision.intents:
                    item = dict(sample=int(e.sample), time_s=e.sample/SR,
                                pitch=int(e.pitch), confidence=float(e.confidence),
                                reason=e.reason, fold=args.fold,
                                variant=args.variant)
                    print("ONSET",json.dumps(item,ensure_ascii=False),flush=True)
                    if out is not None:
                        out.write(json.dumps(item,ensure_ascii=False)+"\n")
                        out.flush()
                count += 1
                now = time.perf_counter()
                if now-last_meter >= 1.:
                    print("INPUT_METER blocks=%d | peak_1s=%.1f dBFS | "
                          "worst_compute=%.2f ms | note_ons=%d" %
                          (count, interval_peak, worst_processing_ms, notes), flush=True)
                    last_meter=now
                    interval_peak=-240.
        if hasattr(engine, "finish") and engine.finish():
            raise AssertionError("Unexpected delayed NoteOn after stream end")
    finally:
        if out is not None:
            out.close()
    print("NOTE_PCM256_LIVE_END",json.dumps(dict(
        hops=count, samples=count*HOP, note_ons=notes,
        worst_compute_ms=worst_processing_ms,
        hardware_latency_measured=False)),flush=True)
    return 0


def make_parser():
    p = argparse.ArgumentParser(description="NOTE Onset PCM256 microphone Live (not MIDI)")
    p.add_argument("--list-devices",action="store_true")
    p.add_argument("--mic", action="store_true",
                   help="Launch automatic microphone input (default input when --device omitted)")
    p.add_argument("--input-device", "--device", dest="input_device", default=None,
                   help="Optional input index or unique substring; if omitted, use Windows default input")
    p.add_argument("--fold",choices=FOLDS,default="BN1",
                   help="Fixed default research fold BN1; not an automatic best-model selection")
    p.add_argument("--variant",choices=("recovery-hgb","research-original"),
                   default="recovery-hgb")
    p.add_argument("--base-bundle",type=Path,
                   default=Path("model/onset256_research_39915038301"))
    p.add_argument("--recovery-bundle",type=Path,
                   default=Path("model/onset256_recovery_20261023"))
    p.add_argument("--native-model",type=Path,
                   default=Path("model/live/strict_note/NOTE_strict_live.npz"))
    p.add_argument("--duration-s",type=float,default=None)
    p.add_argument("--jsonl",type=Path,default=None,
                   help="Create a new JSONL file of real-time Onset events")
    return p


def main(argv=None):
    args = make_parser().parse_args(argv)
    try:
        return run(args)
    except (ValueError,FileNotFoundError,BufferError,RuntimeError) as e:
        print("NOTE_LIVE_ERROR:",e,file=sys.stderr,flush=True)
        return 2
    except KeyboardInterrupt:
        print("NOTE_PCM256_LIVE stopped by user",flush=True)
        return 130


if __name__=="__main__":
    raise SystemExit(main())
