#!/usr/bin/env python3
"""S610: microphone or paced WAV -> named notes and optional real MIDI port."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import importlib.metadata
from pathlib import Path
import queue
import sys
import threading
import time
import wave

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from causal_note.live_engine import (
    BACKEND, DEFAULT_BUNDLE, HOP, SAMPLE_RATE, LatestAudioQueue, LiveEngine,
    TimingStats, note_name,
)
from causal_note.live_reference.backpressure import LiveBackpressure


class EventSink:
    def __init__(self, *, output_format: str, midi_port: str | None = None,
                 channel: int = 0, events_path: Path | None = None):
        self.output_format, self.channel = output_format, channel
        self.port = None
        self.handle = None
        self.sent_pitches: set[int] = set()
        try:
            if midi_port is not None:
                import mido
                self.mido = mido
                self.port = mido.open_output(midi_port)
            if events_path:
                events_path.parent.mkdir(parents=True, exist_ok=True)
                self.handle = events_path.open("w", encoding="utf-8")
        except BaseException:
            self.close()
            raise

    def emit(self, events) -> None:
        for event in events:
            if self.port is not None:
                # Track before send: a driver error may occur after transmission.
                if event.type == "note_on":
                    self.sent_pitches.add(event.pitch)
                self.port.send(self.mido.Message(event.type, note=event.pitch,
                                                velocity=event.velocity, channel=self.channel))
                if event.type == "note_off":
                    self.sent_pitches.discard(event.pitch)
            row = event.json()
            if self.handle is not None:
                self.handle.write(json.dumps(row) + "\n")
                self.handle.flush()
            if self.output_format == "jsonl":
                print(json.dumps(row), flush=True)
            elif self.output_format == "text":
                print(f"{row['time_s']:8.3f}s  {event.type:8s}  {row['note']:3s}  ({event.reason})", flush=True)

    def close(self) -> list[str]:
        errors = []
        if self.port is not None:
            for pitch in sorted(self.sent_pitches):
                try:
                    self.port.send(self.mido.Message("note_off", note=pitch, velocity=0, channel=self.channel))
                except Exception as exc:
                    errors.append(f"NoteOff {pitch}: {exc}")
            # Sustain off, all sound off, all notes off on this channel only.
            for control in (64, 120, 123):
                try:
                    self.port.send(self.mido.Message("control_change", control=control, value=0, channel=self.channel))
                except Exception as exc:
                    errors.append(f"MIDI panic {control}: {exc}")
            try:
                self.port.close()
            except Exception as exc:
                errors.append(f"MIDI close: {exc}")
            self.port = None
        if self.handle is not None:
            try:
                self.handle.close()
            except Exception as exc:
                errors.append(f"events close: {exc}")
            self.handle = None
        return errors


def wav_blocks(path: Path, *, duration_s: float | None = None):
    """Yield complete hops only. The real EOF retains the unpadded tail."""
    with wave.open(str(path), "rb") as source:
        if source.getframerate() != SAMPLE_RATE or source.getsampwidth() != 2 or source.getnchannels() not in (1, 2):
            raise ValueError("WAV requis : PCM16, 44 100 Hz, mono ou stéréo")
        channels = source.getnchannels()
        total = source.getnframes()
        if duration_s is not None:
            total = min(total, int(duration_s * SAMPLE_RATE))
        yield {"sample_count": total, "channels": channels,
               "tail_samples_without_inference": total % HOP,
               "stereo_policy": "mean_channels" if channels == 2 else "mono"}
        for start in range(0, total - HOP + 1, HOP):
            raw = source.readframes(HOP)
            if len(raw) != HOP * channels * 2:
                raise ValueError("WAV tronqué avant sa fin déclarée")
            samples = np.frombuffer(raw, "<i2").astype(np.float32).reshape(HOP, channels)
            yield start, samples.mean(axis=1) / 32768.0


def run(args) -> dict:
    engine = LiveEngine.from_bundle(args.bundle, threads=args.threads, gain=args.gain,
                                    calibration_s=args.calibration_s,
                                    prime_silence=bool(args.wav))
    sink = EventSink(output_format=args.format, midi_port=args.midi_port,
                     channel=args.channel - 1, events_path=args.events)
    capture = LatestAudioQueue()
    backpressure = LiveBackpressure(max_backlog_hops=3)
    stop = threading.Event()
    producer_done = threading.Event()
    producer_errors = []
    producer = None
    input_stream = None
    source_info = {}
    stop_reason = "unknown"
    error = None
    cleanup_errors = []
    callback_timing, cycle_timing = TimingStats(), TimingStats()
    producer_timing = TimingStats()
    delivered = Counter()
    final_sample = 0
    started = time.perf_counter()
    last_status = started
    announced_calibration = bool(args.wav)
    injected = False

    def process(block):
        nonlocal last_status, announced_calibration, backpressure
        cycle_start = time.perf_counter()
        if block.discontinuity or block.start_sample != engine.last_end:
            backpressure = LiveBackpressure(max_backlog_hops=3)
        skip = backpressure.decide(capture.queue.qsize())
        events = engine.process(block.samples, start_sample=block.start_sample,
                                discontinuity=block.discontinuity, skip_inference=skip)
        sink.emit(events)
        delivered.update(event.type for event in events)
        now = time.perf_counter()
        cycle_timing.add((now - cycle_start) * 1000)
        callback_timing.add((now - block.captured_at) * 1000)
        if engine.evidence.calibrated and not announced_calibration:
            print("Calibration terminée. Jouez.", file=sys.stderr, flush=True)
            announced_calibration = True
        if not args.quiet and now - last_status >= 1:
            names = " ".join(note_name(p) for p in sorted(engine.active_ids)) or "—"
            print(f"Notes actives : {names} | reprises après coupure : {engine.recoveries}", file=sys.stderr, flush=True)
            last_status = now

    try:
        print(f"NOTE Live expérimental — {BACKEND} — E2..E5, 6 notes max.", file=sys.stderr)
        if args.wav:
            reader = wav_blocks(args.wav, duration_s=args.duration_s)
            source_info = {"mode": "paced_wav" if args.realtime else "unpaced_wav",
                           "path": str(args.wav), **next(reader),
                           "calibration": "synthetic_silence; not measured microphone noise"}
            if not args.realtime:
                from causal_note.live_engine import AudioBlock
                for sample, hop in reader:
                    process(AudioBlock(sample, hop, time.perf_counter()))
                final_sample = source_info["sample_count"]
                stop_reason = "eof"
            else:
                def replay():
                    origin = time.perf_counter()
                    try:
                        for sample, hop in reader:
                            deadline = origin + (sample + HOP) / SAMPLE_RATE
                            if stop.wait(max(0, deadline - time.perf_counter())):
                                break
                            producer_timing.add(max(0, time.perf_counter() - deadline) * 1000)
                            capture.callback(hop[:, None], HOP)
                    except BaseException as exc:
                        producer_errors.append(exc)
                    finally:
                        reader.close()
                        producer_done.set()
                producer = threading.Thread(target=replay, name="note-wav-capture", daemon=True)
                producer.start()
        else:
            import sounddevice as sd
            device = int(args.device) if args.device and args.device.isdecimal() else args.device
            sd.check_input_settings(device=device, channels=1, samplerate=SAMPLE_RATE, dtype="float32")
            input_stream = sd.InputStream(device=device, channels=1, samplerate=SAMPLE_RATE,
                                          blocksize=HOP, dtype="float32", latency="low",
                                          callback=capture.callback)
            device_info = sd.query_devices(device, "input")
            source_info = {"mode": "microphone", "device": device_info["name"],
                           "calibration": "captured_silence", "requested_sample_rate": SAMPLE_RATE}
            print("Restez silencieux pendant la calibration (1 à 3 secondes). Ctrl+C pour arrêter.", file=sys.stderr, flush=True)
            input_stream.start()
            source_info["reported_device_input_latency_ms"] = float(input_stream.latency) * 1000
        if not args.wav or args.realtime:
            started = time.perf_counter()
            capture.last_callback_at = started
            capture.last_valid_callback_at = started
            while True:
                if not args.wav and args.duration_s and time.perf_counter() - started >= args.duration_s:
                    stop_reason = "duration"
                    break
                if args.wav and producer_done.is_set() and capture.queue.empty():
                    if producer_errors:
                        raise producer_errors[0]
                    stop_reason = "eof"
                    final_sample = source_info["sample_count"]
                    break
                if args.inject_stall_ms and not injected and capture.next_sample / SAMPLE_RATE >= args.inject_stall_at_s:
                    stop.wait(args.inject_stall_ms / 1000)
                    injected = True
                try:
                    block = capture.get(timeout=0.1)
                except queue.Empty:
                    if input_stream is not None and (
                        not input_stream.active or time.perf_counter() - capture.last_valid_callback_at > 1.0
                    ):
                        raise RuntimeError("Flux microphone interrompu (inactif ou sans bloc valide depuis une seconde)")
                    continue
                process(block)
    except KeyboardInterrupt:
        stop_reason = "interrupt"
    except Exception as exc:
        stop_reason = "error"
        error = f"{type(exc).__name__}: {exc}"
    finally:
        stop.set()
        if input_stream is not None:
            try:
                input_stream.stop()
            except Exception as exc:
                cleanup_errors.append(f"input stop: {exc}")
            try:
                input_stream.close()
            except Exception as exc:
                cleanup_errors.append(f"input close: {exc}")
        if producer is not None:
            producer.join(timeout=2)
            if producer.is_alive():
                cleanup_errors.append("WAV producer did not stop")
        final_sample = max(final_sample, engine.last_end, capture.next_sample)
        try:
            events = engine.finalize(final_sample, reason=stop_reason)
            sink.emit(events)
            delivered.update(event.type for event in events)
        except Exception as exc:
            cleanup_errors.append(f"terminal NoteOff: {exc}")
        cleanup_errors.extend(sink.close())

    report = {"schema": 1, "stage": "S610", **engine.report(), "source": source_info,
              "runtime_versions": {name: importlib.metadata.version(name) for name in ("numpy", "tensorflow", "mido")},
              "output": {"midi_port": args.midi_port, "midi_channel": args.channel,
                         "console_format": args.format, "event_log_enabled": args.events is not None},
              "stop_reason": stop_reason, "error": error, "cleanup_errors": cleanup_errors,
              "end_sample": final_sample, "threads": args.threads,
              "wall_duration_s_excluding_model_load": time.perf_counter() - started,
              "successfully_delivered_events": dict(delivered),
              "capture": capture.report(), "worker_cycle_including_output": cycle_timing.report(),
              "paced_wav_delivery_lateness": producer_timing.report(),
              "overload_policy": {"retain_audio_skip_inference_above_queued_hops": 3,
                                  "maximum_skips_in_32_hops": 2,
                                  "discard_stale_audio_above_queued_hops": 8},
              "callback_to_output": callback_timing.report(),
              "timing_scope": "callback/worker to MIDI send return; excludes device ADC, acoustic onset and synth/DAC",
              "physical_end_to_end_latency_measured": False,
              "injected_worker_stall_ms": args.inject_stall_ms if injected else 0}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Arrêt : {stop_reason}. NoteOn={engine.events['note_on']}, NoteOff={engine.events['note_off']}, notes restantes={len(engine.active_ids)}.", file=sys.stderr)
    if error:
        print(error, file=sys.stderr)
    for detail in cleanup_errors:
        print(detail, file=sys.stderr)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--microphone", action="store_true", help="Capturer le microphone (par défaut)")
    source.add_argument("--wav", type=Path, help="Rejouer un WAV PCM16 44,1 kHz")
    parser.add_argument("--realtime", action="store_true", help="Cadencer le WAV comme une capture microphone")
    parser.add_argument("--device", help="Index ou nom du microphone")
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--list-midi-ports", action="store_true")
    parser.add_argument("--midi-port", help="Nom exact du port MIDI de sortie")
    parser.add_argument("--channel", type=int, default=1, choices=range(1, 17))
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--gain", type=float, default=1.0)
    parser.add_argument("--calibration-s", type=float, default=1.0)
    parser.add_argument("--duration-s", type=float)
    parser.add_argument("--format", choices=("text", "jsonl", "none"), default="text")
    parser.add_argument("--events", type=Path, help="Journal JSONL des événements")
    parser.add_argument("--report", type=Path, help="Rapport JSON à l'arrêt")
    parser.add_argument("--quiet", action="store_true", help="Masquer le statut périodique")
    parser.add_argument("--inject-stall-ms", type=float, default=0, help="Test de surcharge, WAV cadencé seulement")
    parser.add_argument("--inject-stall-at-s", type=float, default=2.0)
    args = parser.parse_args(argv)
    if args.duration_s is not None and (not np.isfinite(args.duration_s) or args.duration_s <= 0):
        parser.error("--duration-s doit être positif")
    if not np.isfinite(args.calibration_s) or not 0 < args.calibration_s <= 3:
        parser.error("--calibration-s doit être entre 0 (exclu) et 3")
    if args.threads < 1 or not np.isfinite(args.gain) or args.gain <= 0:
        parser.error("threads et gain doivent être positifs")
    if not np.isfinite(args.inject_stall_ms) or not 0 <= args.inject_stall_ms <= 10000 or not np.isfinite(args.inject_stall_at_s) or args.inject_stall_at_s < 0:
        parser.error("Injection de surcharge invalide")
    if args.realtime and not args.wav:
        parser.error("--realtime accompagne --wav")
    if args.inject_stall_ms and not (args.wav and args.realtime):
        parser.error("La surcharge simulée exige --wav et --realtime")
    try:
        if args.list_devices:
            import sounddevice as sd
            print(sd.query_devices())
        if args.list_midi_ports:
            import mido
            print("\n".join(mido.get_output_names()) or "Aucun port MIDI disponible")
        if args.list_devices or args.list_midi_ports:
            return 0
        report = run(args)
        return 1 if report["error"] or report["cleanup_errors"] else 0
    except Exception as exc:
        print(f"Impossible de démarrer NOTE Live : {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
