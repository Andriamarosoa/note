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
    parser.add_argument("--device", help="sounddevice input device name/index")
    parser.add_argument("--midi-port", help="Exact mido output port name; omit for JSONL events")
    parser.add_argument("--out", type=Path, help="JSONL output; defaults to stdout")
    parser.add_argument("--block-size", type=int, default=HOP, choices=(128, 256, 512, 1024))
    args = parser.parse_args(argv)
    if args.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
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
        port = None
        if args.midi_port:
            import mido
            if args.midi_port not in mido.get_output_names():
                raise ValueError("Requested MIDI port does not exist")
            port = stack.enter_context(mido.open_output(args.midi_port))
        def emit(events):
            for event in events:
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
                inbox = queue.Queue(maxsize=2)
                overrun = threading.Event()
                capture_start = [0]
                def callback(indata, frames, timing, status):
                    if status or overrun.is_set():
                        overrun.set()
                        raise sd.CallbackAbort
                    try:
                        inbox.put_nowait((capture_start[0], indata[:, 0].copy(), time.monotonic()))
                        capture_start[0] += frames
                    except queue.Full:
                        overrun.set()
                        raise sd.CallbackAbort
                device = int(args.device) if args.device and args.device.isdigit() else args.device
                capture = stack.enter_context(sd.InputStream(samplerate=SR, blocksize=args.block_size,
                         channels=1, dtype="float32", device=device, latency="low", callback=callback))
                print(json.dumps(dict(capture_latency_ms=capture.latency * 1000, block_ms=args.block_size / SR * 1000)), file=sys.stderr)
                while True:
                    if overrun.is_set():
                        raise BufferError("Input overrun: stream stopped and notes released")
                    try:
                        start, audio, captured = inbox.get(timeout=.1)
                    except queue.Empty:
                        if not capture.active:
                            raise BufferError("Microphone stopped unexpectedly")
                        continue
                    if time.monotonic() - captured > 2 * args.block_size / SR:
                        raise BufferError("Live worker exceeded its two-block queue budget")
                    events = stream.push(audio, start_sample=start)
                    if overrun.is_set() or time.monotonic() - captured > 2 * args.block_size / SR:
                        raise BufferError("Live decision missed its deadline; notes released")
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
