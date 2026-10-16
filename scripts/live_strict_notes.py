"""Autonomous causal NOTE Live: microphone, PCM stdin or WAV -> MIDI notes.

No events/K/BP file is accepted. Capture, the native acoustic/count/pitch
heads and note selection run continuously. On an overrun, stop and release
notes instead of silently processing a delayed or discontinuous queue.
"""
from __future__ import annotations
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import queue
import sys
import threading
import time
import wave

import numpy as np
from threadpoolctl import threadpool_limits
from causal_note.strict_live import HOP, SR, load_heads, StrictLiveStream
from scripts.live_audio_io import audio_devices, managed_input_stream, open_output_stream, parse_device

DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "model/live/strict_note/NOTE_strict_live.npz"


def blocks_stdin(stream, block_size):
    pending = b""
    while True:
        data = stream.read(block_size * 2 - len(pending))
        if not data:
            if len(pending) % 2:
                raise ValueError("Truncated signed PCM16 sample")
            if pending:
                yield np.frombuffer(pending, dtype="<i2").astype(np.float32) / 32768
            return
        pending += data
        if len(pending) == block_size * 2:
            yield np.frombuffer(pending, dtype="<i2").astype(np.float32) / 32768
            pending = b""


def blocks_wav(path, block_size):
    with wave.open(str(path), "rb") as reader:
        if reader.getframerate() != SR or reader.getnchannels() != 1 or reader.getsampwidth() != 2:
            raise ValueError("WAV must be mono signed PCM16 at 44.1 kHz")
        while raw := reader.readframes(block_size):
            yield np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768


def capture_deadline_exceeded(captured, now, block_size, blocks):
    """Optional input-worker wall-time guard (0 disables it)."""
    return blocks > 0 and now - captured > blocks * block_size / SR


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--list-midi-ports", action="store_true")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--wav", type=Path)
    source.add_argument("--stdin", action="store_true", help="Raw mono s16le PCM, 44100 Hz")
    source.add_argument("--mic", action="store_true")
    parser.add_argument("--device", help="Microphone name/index; auto WASAPI/system if omitted")
    parser.add_argument("--soundfont", type=Path, help="GM-compatible SF2 piano SoundFont: hear detected notes as acoustic piano")
    parser.add_argument("--piano-output-device", help="Piano audio output name/index; automatically selects low-latency WDM-KS/WASAPI if omitted")
    parser.add_argument("--piano-gain", type=float, default=0.35, help="FluidSynth piano gain, default 0.35")
    parser.add_argument("--monitor-input", action="store_true", help="Listen to raw microphone audio (headphones recommended); independent of note detection")
    parser.add_argument("--output-device", help="sounddevice output device name/index for monitoring; select a WASAPI endpoint via --list-devices")
    parser.add_argument("--monitor-gain", type=float, default=0.5, help="Raw input monitoring gain, 0.0..1.0 (default 0.5)")
    parser.add_argument("--midi-port", help="Exact mido output port name; omit for JSONL events")
    parser.add_argument("--out", type=Path, help="JSONL output; defaults to stdout")
    parser.add_argument("--block-size", type=int, default=HOP, choices=(128, 256, 512, 1024))
    parser.add_argument("--capture-deadline-blocks", type=int, default=0,
                        help="0 = no artificial microphone worker deadline (default)")
    parser.add_argument("--queue-blocks", type=int, default=32,
                        help="Maximum buffered microphone blocks before overflow")
    args = parser.parse_args(argv)
    if args.piano_output_device is not None and args.soundfont is None:
        parser.error("--piano-output-device requires --soundfont")
    if args.soundfont is not None and not args.soundfont.is_file():
        parser.error(f"SoundFont not found: {args.soundfont}")
    if not 0.0 < args.piano_gain <= 5.0:
        parser.error("--piano-gain must be > 0 and <= 5")
    if args.monitor_input and args.soundfont:
        parser.error("Use either --monitor-input or --soundfont; not both")
    if args.monitor_input and not args.mic:
        parser.error("--monitor-input requires --mic")
    if args.output_device is not None and not args.monitor_input:
        parser.error("--output-device requires --monitor-input")
    if not 0.0 <= args.monitor_gain <= 1.0:
        parser.error("--monitor-gain must be in [0.0, 1.0]")
    if args.capture_deadline_blocks < 0:
        parser.error("--capture-deadline-blocks must be >= 0")
    if args.queue_blocks < 1:
        parser.error("--queue-blocks must be >= 1")
    if args.list_devices:
        for direction, title in (("input", "Microphone INPUT"), ("output", "Piano/audio OUTPUT")):
            print(f"{title} (index | name | host API | native Hz | reported low-latency ms):")
            for entry in audio_devices(direction):
                marker = " [system default]" if entry["default"] else ""
                print(f'  {entry["index"]} | {entry["name"]} | {entry["host_api"]} | '
                      f'{entry["native_sample_rate"]} Hz | '
                      f'{entry["low_latency_ms"]:.2f} ms{marker}')
        print("Auto selection probes live streams and chooses lowest *healthy negotiated* latency at launch.")
        return
    if args.list_midi_ports:
        import mido
        print(json.dumps(mido.get_output_names()))
        return
    if args.status:
        print(json.dumps(dict(status="EXPERIMENTAL_AUTONOMOUS_CAUSAL_NOTE_LIVE", sample_rate=SR, hop=HOP,
                             lookahead_samples=0, sidecars_required=False, model=str(args.model),
                             model_available=args.model.is_file(), pitch_range=[40, 88], polyphony=6,
                             original_S80_promoted=False, microphone_requires="sounddevice + PortAudio",
                             midi_requires="mido + python-rtmidi", accuracy="see S613 held-out report")))
        return
    if not (args.wav or args.stdin or args.mic):
        parser.error("Choose --wav, --stdin or --mic")
    if args.out and args.out.exists():
        parser.error("Output already exists")
    # Limit BLAS threads throughout the stream; tiny matrices must not spawn
    # a thread pool on the realtime path. No model loading in the callback.
    with ExitStack() as stack:
        stack.enter_context(threadpool_limits(limits=1))
        expected_sha = None
        if args.model.resolve() == DEFAULT_MODEL.resolve():
            expected_sha = json.loads(DEFAULT_MODEL.with_name("provenance.json").read_text())["model_sha256"]
        heads = load_heads(args.model, expected_sha256=expected_sha)
        warm = StrictLiveStream(heads)
        for j in range(8):
            warm.push(np.zeros(HOP, np.float32), start_sample=j * HOP)
        stream = StrictLiveStream(heads)
        output = stack.enter_context(args.out.open("x")) if args.out else sys.stdout
        piano = None
        if args.soundfont:
            from scripts.live_piano_output import PianoOutput
            piano = PianoOutput(args.soundfont, output_device=args.piano_output_device,
                                gain=args.piano_gain, samplerate=SR)
            stack.callback(piano.close)
        port = None
        if args.midi_port:
            import mido
            if args.midi_port not in mido.get_output_names():
                raise ValueError("Requested MIDI port does not exist")
            port = stack.enter_context(mido.open_output(args.midi_port))
        def emit(events):
            for event in events:
                if piano is not None:
                    piano.send(event)
                if port:
                    import mido
                    port.send(mido.Message(event["type"], note=event["pitch"], velocity=event["velocity"], channel=0))
                output.write(json.dumps(event) + "\n")
            if events:
                output.flush()
        failure = "stream_end"
        try:
            if args.mic:
                import sounddevice as sd
                inbox = queue.Queue(maxsize=args.queue_blocks)
                overrun = threading.Event()
                capture_start = [0]
                # A separate bounded monitor queue cannot delay the recognition path.
                monitor_queue = queue.Queue(maxsize=4) if args.monitor_input else None
                monitor_underruns = [0]
                def monitor_callback(outdata, frames, timing, status):
                    if status:
                        monitor_underruns[0] += 1
                    try:
                        chunk = monitor_queue.get_nowait()
                        if len(chunk) != frames:
                            outdata.fill(0.0)
                            monitor_underruns[0] += 1
                        else:
                            outdata[:, 0] = np.clip(chunk * args.monitor_gain, -1.0, 1.0)
                    except queue.Empty:
                        outdata.fill(0.0)
                        monitor_underruns[0] += 1
                def callback(indata, frames, timing, status):
                    if status or overrun.is_set():
                        overrun.set()
                        raise sd.CallbackAbort
                    try:
                        chunk = indata[:, 0].copy()
                        inbox.put_nowait((capture_start[0], chunk, time.monotonic()))
                        capture_start[0] += frames
                        if monitor_queue is not None:
                            try:
                                monitor_queue.put_nowait(chunk)
                            except queue.Full:
                                # Monitoring is best-effort; never block audio capture.
                                try:
                                    monitor_queue.get_nowait()
                                except queue.Empty:
                                    pass
                                try:
                                    monitor_queue.put_nowait(chunk)
                                except queue.Full:
                                    pass
                    except queue.Full:
                        overrun.set()
                        raise sd.CallbackAbort
                capture, capture_info = stack.enter_context(managed_input_stream(
                    preferred_device=parse_device(args.device),
                    samplerate=SR,
                    blocksize=args.block_size,
                    channels=1,
                    dtype="float32",
                    latency="low",
                    callback=callback,
                ))
                if args.monitor_input:
                    playback, playback_info = open_output_stream(
                        preferred_device=parse_device(args.output_device),
                        samplerate=SR,
                        blocksize=args.block_size,
                        channels=1,
                        dtype="float32",
                        latency="low",
                        callback=monitor_callback,
                    )
                    stack.callback(playback.close)
                    print(json.dumps(dict(monitoring="raw_microphone",
                        output_device=playback_info.device_index,
                        output_host_api=playback_info.host_api,
                        output_latency_ms=playback_info.latency_ms,
                        warning="Use headphones to avoid feedback; this is not MIDI synthesis")),
                        file=sys.stderr)
                print(json.dumps(dict(input_device=capture_info.device_name,
                         input_host_api=capture_info.host_api,
                         input_device_index=capture_info.device_index,
                         capture_latency_ms=capture_info.latency_ms,
                         block_ms=args.block_size / SR * 1000,
                         capture_deadline_blocks=args.capture_deadline_blocks,
                         queue_blocks=args.queue_blocks)), file=sys.stderr)
                while True:
                    if overrun.is_set():
                        raise BufferError("Input overrun: stream stopped and notes released")
                    try:
                        start, audio, captured = inbox.get(timeout=.1)
                    except queue.Empty:
                        if not capture.active:
                            raise BufferError("Microphone stopped unexpectedly")
                        continue
                    if capture_deadline_exceeded(captured, time.monotonic(), args.block_size,
                                                 args.capture_deadline_blocks):
                        raise BufferError("Optional capture deadline exceeded before inference")
                    events = stream.push(audio, start_sample=start)
                    if overrun.is_set():
                        raise BufferError("Microphone input overflow; notes released")
                    if capture_deadline_exceeded(captured, time.monotonic(), args.block_size,
                                                 args.capture_deadline_blocks):
                        raise BufferError("Optional capture deadline exceeded after inference")
                    emit(events)
            else:
                blocks = blocks_wav(args.wav, args.block_size) if args.wav else blocks_stdin(sys.stdin.buffer, args.block_size)
                start = 0
                for audio in blocks:
                    emit(stream.push(audio, start_sample=start))
                    start += len(audio)
        except KeyboardInterrupt:
            failure = "shutdown"
        except Exception:
            failure = "stream_error"
            raise
        finally:
            emit(stream.finish(failure))


if __name__ == "__main__":
    main()
