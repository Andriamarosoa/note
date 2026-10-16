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
from scripts.live_input_level import NoteAutoLevel
from scripts.live_hop_tuning import parse_hop_size, resolve_hop

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


def parse_block_size(value):
    """0 requests PortAudio host-chosen size; positive values have no CLI cap."""
    try:
        size = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("block size must be an integer") from exc
    if size < 0:
        raise argparse.ArgumentTypeError("block size must be >= 0 (0 = PortAudio auto)")
    return size


def parse_queue_blocks(value):
    if str(value).casefold() == "auto":
        return "auto"
    try:
        blocks = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("queue-blocks must be auto or a positive integer") from exc
    if blocks < 1:
        raise argparse.ArgumentTypeError("queue-blocks must be >= 1")
    return blocks


def resolve_queue_blocks(option, block_size, *, sample_rate=SR, target_ms=75.0):
    """Bound queue occupancy by roughly target_ms of captured PCM.

    This is a capacity, NOT a mandatory delay or adaptive hidden FIFO.
    0 is PortAudio's variable callback size; 256 samples is the nominal
    queue planning size until actual device frames are known.
    """
    if option != "auto":
        return int(option)
    nominal = block_size if block_size > 0 else HOP
    return max(2, int(np.ceil((target_ms / 1000.) * sample_rate / nominal)))


def apply_model_input(leveler, audio):
    """Keep optional gain's hop bound even with variable PortAudio frames."""
    if leveler is None:
        return audio
    hop = leveler.leveler.hop_samples
    if len(audio) <= hop:
        return leveler.apply(audio)
    return np.concatenate([
        leveler.apply(audio[offset:offset + hop])
        for offset in range(0, len(audio), hop)
    ])


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
    parser.add_argument("--dual-hop-mode", choices=("off", "shadow", "guard"), default="off",
                        help="EXPERIMENTAL: 128+256 note heads, shadow leaves original MIDI unchanged; guard fuses fast acoustic novelty notes")
    parser.add_argument("--dual-hop-fast-model", type=Path,
                        help="S621 separately TRAINED hop128 native NOTE candidate checkpoint, required for scientifically matched cadence")
    parser.add_argument("--dual-hop-fast-onset", type=float, default=.50,
                        help="S620 fast 128-hop onset candidate minimum (default .50)")
    parser.add_argument("--dual-hop-novelty-ratio", type=float, default=.55,
                        help="S620 independent spectral novelty ratio (default .55)")
    parser.add_argument("--dual-hop-novelty-min", type=float, default=.04,
                        help="S620 absolute new spectral evidence minimum (default .04)")
    parser.add_argument("--dual-hop-novelty-mode", choices=("lag","recent"), default="lag")
    parser.add_argument("--harmonic-mode", choices=("off", "shadow", "guard"), default="off",
                        help="Experimental native harmonic expert: off (default), shadow (no changed events), guard (requires trained checkpoint)")
    parser.add_argument("--harmonic-checkpoint", type=Path,
                        help="Separately trained harmonic genuine-onset checkpoint; no checkpoint is bundled")
    parser.add_argument("--harmonic-threshold", type=float, default=0.10,
                        help="Experimental independent-onset gating threshold, dev-calibration required")
    parser.add_argument("--auto-level", action="store_true", help="Enable bounded causal input gain (+18dB max), adapted from midi; experimental, default OFF")
    parser.add_argument("--auto-level-max-db", type=float, default=18.0, help="Maximum automatic amplification, 0..24 dB (default: 18)")
    parser.add_argument("--auto-level-peak-gate-dbfs", type=float, default=-55.0, help="Raw peak activity gate in dBFS (default -55)")
    parser.add_argument("--auto-level-rms-gate-dbfs", type=float, default=-65.0, help="Raw RMS activity gate in dBFS (default -65)")
    parser.add_argument("--midi-port", help="Exact mido output port name; omit for JSONL events")
    parser.add_argument("--out", type=Path, help="JSONL output; defaults to stdout")
    parser.add_argument("--lookahead-ms", type=float, default=0.0,
                        help="S628 actual future PCM in ms (0=original, 5=experimental), delays physical notes; must use --hop-size 256")
    parser.add_argument("--lookahead-mode", choices=("delay","average","confirm","max_onset","future"), default="average",
                        help="Frozen NOTE future score fusion, delay=timing-only control; default average")
    parser.add_argument("--lookahead-engine", choices=("shared","phase"),default="shared",
                        help="S629 shared (default) computes ONE frozen S613 head per 256 hop, rounds 5ms->5.80ms and 10ms->11.61ms; phase reproduces S628 exact delay but ~2x CPU")
    parser.add_argument("--hop-size", type=str, default=None,
                        help="Neural decision hop 128/256/512 or auto (experimental CPU preflight). Default 256 preserves validated S613.")
    parser.add_argument("--block-size", type=parse_block_size, default=HOP,
                        help="PortAudio callback samples: any positive integer; 0 = driver auto. Large blocks add output latency.")
    parser.add_argument("--inference-batch-hops", type=int, default=1,
                        help="S630 EXPERIMENT: native 256-hop frozen head frames per vectorized ONNX call (1=original; try 4); batches add up to (N-1)*5.805ms decision wait.")
    parser.add_argument("--capture-deadline-blocks", type=int, default=0,
                        help="0 = no artificial microphone worker deadline (default)")
    parser.add_argument("--queue-blocks", type=parse_queue_blocks, default="auto",
                        help="Input queue capacity: auto (default, about 75ms of PCM), or explicit positive block count")
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
    if not (np.isfinite(args.auto_level_max_db) and 0 <= args.auto_level_max_db <= 24):
        parser.error("--auto-level-max-db must be within 0..24 dB")
    if not (np.isfinite(args.auto_level_peak_gate_dbfs) and -120 <= args.auto_level_peak_gate_dbfs <= 0):
        parser.error("--auto-level-peak-gate-dbfs must be within -120..0")
    if not (np.isfinite(args.auto_level_rms_gate_dbfs) and -120 <= args.auto_level_rms_gate_dbfs <= 0):
        parser.error("--auto-level-rms-gate-dbfs must be within -120..0")
    if args.harmonic_mode == "guard" and args.harmonic_checkpoint is None:
        parser.error("--harmonic-mode guard requires --harmonic-checkpoint with trained weights")
    if args.harmonic_mode == "off" and args.harmonic_checkpoint is not None:
        parser.error("--harmonic-checkpoint requires --harmonic-mode shadow or guard")
    if args.harmonic_checkpoint is not None and not args.harmonic_checkpoint.is_file():
        parser.error(f"Harmonic checkpoint not found: {args.harmonic_checkpoint}")
    if not (np.isfinite(args.harmonic_threshold) and 0 < args.harmonic_threshold < 1):
        parser.error("--harmonic-threshold must be inside (0,1)")
    if not np.isfinite(args.lookahead_ms) or not 0 <= args.lookahead_ms <= 50:
        parser.error("--lookahead-ms must be finite and within 0..50")
    if args.capture_deadline_blocks < 0:
        parser.error("--capture-deadline-blocks must be >= 0")
    if not 1 <= args.inference_batch_hops <= 32:
        parser.error("--inference-batch-hops must be from 1..32")
    if args.inference_batch_hops > 1 and (args.dual_hop_mode!="off" or args.harmonic_mode!="off"
                                          or (args.lookahead_ms>0 and args.lookahead_engine=="phase")):
        parser.error("S630 batched inference currently requires normal S613 or shared lookahead with no dual-hop/harmonic sidecars")
    try:
        hop_request = parse_hop_size(args.hop_size if args.hop_size is not None
                                     else 256)
    except ValueError as exc:
        parser.error(str(exc))
    if args.dual_hop_fast_model is not None:
        if args.dual_hop_mode == "off":
            parser.error("--dual-hop-fast-model requires --dual-hop-mode shadow or guard")
        if not args.dual_hop_fast_model.is_file():
            parser.error("S621 trained hop128 checkpoint not found: " + str(args.dual_hop_fast_model))
    if args.dual_hop_mode != "off":
        if args.harmonic_mode != "off" or args.harmonic_checkpoint is not None:
            parser.error("Dual-hop prototype and harmonic specialist cannot be combined yet")
        if hop_request != 256:
            parser.error("Dual-hop requires --hop-size 256 for its validated reference")
        if (not 0 < args.dual_hop_fast_onset < 1 or
                not 0 <= args.dual_hop_novelty_ratio <= 1 or
                args.dual_hop_novelty_min < 0):
            parser.error("Invalid S620 dual-hop fast-onset/novelty thresholds")
    if args.lookahead_ms > 0:
        if hop_request!=256:
            parser.error("--lookahead-ms >0 requires --hop-size 256 for the frozen S613 selector")
        if args.dual_hop_mode!="off" or args.harmonic_mode!="off":
            parser.error("S628 lookahead cannot be combined with dual-hop or harmonic guard at this stage")
    queue_capacity = resolve_queue_blocks(args.queue_blocks, args.block_size)
    if args.block_size and args.block_size > HOP:
        print(json.dumps(dict(
            warning="A large callback block increases LIVE event-delivery latency",
            block_size=args.block_size, block_ms=round(args.block_size * 1000 / SR, 3),
            note="Model decision cadence is set by --hop-size; events are delivered in callback groups",
        )), file=sys.stderr)
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
                             lookahead_samples=round(args.lookahead_ms*SR/1000),
                             lookahead_ms=args.lookahead_ms,
                             lookahead_mode=args.lookahead_mode if args.lookahead_ms>0 else "off",
                             sidecars_required=False, model=str(args.model),
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
        selected_hop, hop_audit = resolve_hop(
            heads, hop_request, harmonic_mode=args.harmonic_mode,
            harmonic_checkpoint=args.harmonic_checkpoint,
            harmonic_threshold=args.harmonic_threshold,
        )
        print(json.dumps(dict(hop_selection=hop_audit)), file=sys.stderr)
        if args.inference_batch_hops > 1:
            from causal_note.batched_live import BatchedLiveStream
            from causal_note.lookahead import LookaheadConfig
            config=(LookaheadConfig(milliseconds=args.lookahead_ms,
                                    mode=args.lookahead_mode)
                    if args.lookahead_ms>0 else None)
            stream=BatchedLiveStream(
                heads,batch_hops=args.inference_batch_hops,lookahead=config)
            print(json.dumps(dict(
                s630_batched_inference="EXPERIMENTAL",
                batch_hops=args.inference_batch_hops,
                native_hop_samples=HOP,
                requested_lookahead_ms=args.lookahead_ms,
                effective_lookahead_ms=stream.actual_lookahead_samples*1000/SR,
                max_batch_wait_ms=(args.inference_batch_hops-1)*HOP*1000/SR,
                original_model_and_K_frozen=True,
                warning=("Batch several REAL 256-hop feature frames through one "
                         "ONNX/MLP call; no data or notes are dropped, "
                         "but MIDI output latency increases."),
            )),file=sys.stderr)
        elif args.lookahead_ms > 0:
            from causal_note.lookahead import (
                LookaheadConfig,LookaheadLiveStream,SharedHopLookaheadLiveStream,
            )
            cfg=LookaheadConfig(milliseconds=args.lookahead_ms,
                                mode=args.lookahead_mode)
            if args.lookahead_engine=="shared":
                stream=SharedHopLookaheadLiveStream(heads,config=cfg)
                effective_samples=stream.actual_samples
            else:
                stream=LookaheadLiveStream(heads,config=cfg)
                effective_samples=cfg.samples
            print(json.dumps(dict(
                s629_lookahead="EXPERIMENTAL",
                engine=args.lookahead_engine,
                requested_ms=args.lookahead_ms,
                requested_samples=cfg.samples,
                actual_samples=effective_samples,
                actual_ms=round(effective_samples*1000/SR,3),
                quantization_extra_ms=round((effective_samples-cfg.samples)*1000/SR,3),
                mode=cfg.mode,
                single_model_inference_per_hop=args.lookahead_engine=="shared",
                latency_added=True,
                warning=("Only PHYSICALLY received future PCM is used; no event backdating. "
                         "The shared engine is a NEW unpromoted ablation distinct from "
                         "S628 exact-phase F1; benchmark real iRig CPU and false notes."),
            )),file=sys.stderr)
        elif args.dual_hop_mode != "off":
            from causal_note.dual_hop import DualHopConfig, DualHopLiveStream
            dual_config = DualHopConfig(
                mode=args.dual_hop_mode,
                fast_onset=args.dual_hop_fast_onset,
                novelty_min=args.dual_hop_novelty_min,
                novelty_ratio=args.dual_hop_novelty_ratio,
                novelty_mode=args.dual_hop_novelty_mode,
            )
            fast_heads = (load_heads(args.dual_hop_fast_model)
                          if args.dual_hop_fast_model is not None else None)
            stream = DualHopLiveStream(heads,config=dual_config,
                                       fast_heads=fast_heads)
            print(json.dumps(dict(
                s620_dual_hop=dual_config.mode,
                reference_hop=256, fast_hop=128,
                fast_model_trained_at_128=fast_heads is not None,
                warning=("RESEARCH ONLY: 128 path uses an independent trained "
                         "checkpoint when supplied; otherwise reuses unvalidated "
                         "256-trained S613 weights. Extra CPU and false pitches "
                         "remain possible. Only shadow preserves original MIDI."),
            )), file=sys.stderr)
        else:
            stream = StrictLiveStream(
                heads, hop=selected_hop, harmonic_mode=args.harmonic_mode,
                harmonic_checkpoint=args.harmonic_checkpoint,
                harmonic_threshold=args.harmonic_threshold,
            )
        if args.harmonic_mode != "off":
            print(json.dumps(dict(
                harmonic_mode=args.harmonic_mode,
                harmonic_checkpoint=str(args.harmonic_checkpoint) if args.harmonic_checkpoint else None,
                changed_note_events=args.harmonic_mode == "guard",
                note="Experimental harmonic expert: existing NOTE model and frontend are unchanged",
            )), file=sys.stderr)
        leveler = (NoteAutoLevel(
            sample_rate=SR, block_size=(args.block_size or HOP),
            max_gain_db=args.auto_level_max_db,
            peak_gate_dbfs=args.auto_level_peak_gate_dbfs,
            rms_gate_dbfs=args.auto_level_rms_gate_dbfs,
        ) if args.auto_level else None)
        if leveler is not None:
            print(json.dumps(dict(auto_level="enabled", max_gain_db=args.auto_level_max_db,
                 raw_peak_gate_dbfs=args.auto_level_peak_gate_dbfs,
                 raw_rms_gate_dbfs=args.auto_level_rms_gate_dbfs,
                 warning="Gain only; cannot recover missing or quantized-out E2 fundamentals")),
                 file=sys.stderr)
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
                inbox = queue.Queue(maxsize=queue_capacity)
                overrun = threading.Event()
                overrun_reason = [None]
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
                    if status:
                        overrun_reason[0] = f"portaudio_callback_status={status}"
                        overrun.set()
                        raise sd.CallbackAbort
                    if overrun.is_set():
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
                        overrun_reason[0] = (
                            f"input_queue_full(maxsize={queue_capacity}, "
                            f"block_size={args.block_size}, "
                            f"queue_nominal_audio_ms={queue_capacity * max(args.block_size, HOP) * 1000 / SR:.1f})"
                        )
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
                         queue_blocks=queue_capacity,
                         decision_hop_samples=selected_hop,
                         decision_hop_ms=round(selected_hop * 1000 / SR, 4),
                         queue_setting=args.queue_blocks,
                         queue_nominal_capacity_ms=round(
                             queue_capacity * (args.block_size or HOP) * 1000 / SR, 2
                         ))), file=sys.stderr)
                while True:
                    if piano is not None:
                        piano_error = piano.health_error()
                        if piano_error is not None:
                            raise RuntimeError(piano_error)
                    if overrun.is_set():
                        raise BufferError(f"Input overrun: {overrun_reason[0]}; stream stopped and notes released")
                    try:
                        start, audio, captured = inbox.get(timeout=.1)
                    except queue.Empty:
                        if not capture.active:
                            raise BufferError("Microphone stopped unexpectedly")
                        continue
                    if capture_deadline_exceeded(captured, time.monotonic(), len(audio),
                                                 args.capture_deadline_blocks):
                        raise BufferError("Optional capture deadline exceeded before inference")
                    model_audio = apply_model_input(leveler, audio)
                    events = stream.push(model_audio, start_sample=start)
                    if overrun.is_set():
                        raise BufferError(f"Microphone input overflow: {overrun_reason[0]}; notes released")
                    if capture_deadline_exceeded(captured, time.monotonic(), len(audio),
                                                 args.capture_deadline_blocks):
                        raise BufferError("Optional capture deadline exceeded after inference")
                    emit(events)
            else:
                file_block = args.block_size or HOP
                blocks = blocks_wav(args.wav, file_block) if args.wav else blocks_stdin(sys.stdin.buffer, file_block)
                start = 0
                for audio in blocks:
                    model_audio = apply_model_input(leveler, audio)
                    emit(stream.push(model_audio, start_sample=start))
                    start += len(audio)
        except KeyboardInterrupt:
            failure = "shutdown"
        except Exception:
            failure = "stream_error"
            raise
        finally:
            try:
                emit(stream.finish(failure))
            finally:
                if leveler is not None:
                    print(json.dumps(dict(auto_level_diagnostics=leveler.diagnostics())),
                          file=sys.stderr)
                if hasattr(stream, "diagnostics"):
                    field=("s630_batched_diagnostics" if args.inference_batch_hops>1
                           else "s629_lookahead_diagnostics" if args.lookahead_ms>0
                           else "dual_hop_diagnostics")
                    print(json.dumps({field:stream.diagnostics()}),
                          file=sys.stderr)
                if args.lookahead_ms == 0 and hasattr(stream,"timings_ms") and stream.timings_ms:
                    elapsed=np.asarray(stream.timings_ms,np.float64)
                    total_samples=stream.frontend.received
                    print(json.dumps(dict(
                        s629_original_worker_diagnostics=dict(
                            callback_count=int(len(elapsed)),
                            worker_p95_ms=round(float(np.percentile(elapsed,95)),4),
                            worker_max_ms=round(float(np.max(elapsed)),4),
                            worker_total_ms=round(float(np.sum(elapsed)),4),
                            model_audio_processed_ms=round(total_samples*1000/SR,4),
                            worker_over_audio_ratio=round(float(np.sum(elapsed))/
                                max(1.,total_samples*1000/SR),4),
                            nominal_callback_ms=round((args.block_size or HOP)*1000/SR,4),
                            warning="Includes acoustic/model CPU, not piano synthesis, and does not prove end-to-end latency.",
                        )
                    )),file=sys.stderr)
                if stream.harmonic_selector is not None:
                    from dataclasses import asdict
                    print(json.dumps(dict(harmonic_expert_diagnostics=asdict(
                        stream.harmonic_selector.stats,
                    ))), file=sys.stderr)
                if piano is not None:
                    print(json.dumps(dict(piano_output_diagnostics=piano.diagnostics())),
                          file=sys.stderr)


if __name__ == "__main__":
    main()
