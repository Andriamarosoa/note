"""S635 microphone-to-three-view NOTE diagnosis WITHOUT any WAV file.

The sounddevice callback records a bounded number of raw PCM samples IN RAM.
After capture stops, the exact frozen NOTE S634 0/x2/half evaluation runs
off the microphone callback thread. This avoids the previously observed
input_queue_full from trying three expensive neural windows in real time.

This is a safe *mic direct* diagnostic, NOT a three-stream low-latency MIDI
engine and NOT a verified F1 evaluation without ground-truth annotations.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import threading
import time

import numpy as np
from threadpoolctl import threadpool_limits

from causal_note.strict_live import SR, load_heads
from scripts.live_audio_io import managed_input_stream, audio_devices, parse_device
from scripts.live_strict_notes import DEFAULT_MODEL
from scripts.s613_prepare_strict_live import digest
from scripts.s634_three_view_wav import diagnostic


def capture_pcm(*, seconds=8., device=None, block_size=1024,
                stream_factory=managed_input_stream):
    """Capture exactly requested PCM into memory; reject gaps and errors.

    IMPORTANT: No FFT, ONNX, MIDI, stdout or file writes in PortAudio callback.
    A user-selected 44.1-kHz input endpoint is opened through NOTE's existing
    input selector. Stream is always stopped/closed before expensive inference.
    """
    if not np.isfinite(seconds) or not 1. <= seconds <= 120.:
        raise ValueError("seconds must be within 1..120")
    if type(block_size) is not int or not 256 <= block_size <= 4096:
        raise ValueError("block_size must be between 256 and 4096")
    n=int(round(seconds*SR))
    data=[]
    stats=dict(callbacks=0, received=0, max_callback_frames=0, error=None)
    done=threading.Event()

    def callback(indata, frames, timing, status):
        if done.is_set():
            return
        if status:
            stats["error"]=str(status)
            done.set()
            return
        if int(frames)<1 or indata.ndim!=2 or indata.shape!=(frames,1):
            stats["error"]="PortAudio produced invalid microphone block shape"
            done.set()
            return
        if not np.isfinite(indata).all():
            stats["error"]="Microphone delivered NaN or Inf"
            done.set()
            return
        remaining=n-stats["received"]
        if remaining<=0:
            done.set()
            return
        count=min(remaining,int(frames))
        data.append(np.array(indata[:count,0],dtype=np.float32,copy=True))
        stats["received"]+=count
        stats["callbacks"]+=1
        stats["max_callback_frames"]=max(stats["max_callback_frames"],int(frames))
        if stats["received"]>=n:
            done.set()

    started=time.monotonic()
    with stream_factory(
            preferred_device=parse_device(device),
            samplerate=SR,blocksize=block_size,channels=1,
            dtype="float32",latency="low",callback=callback) as (_,info):
        # stop() and close() run on exit, before returning the collected PCM.
        completed=done.wait(seconds+max(8.,seconds*.5))
    stats["wall_seconds"]=round(time.monotonic()-started,3)
    if stats["error"]:
        raise BufferError("S635 microphone capture failed: "+stats["error"])
    if not completed or stats["received"]!=n:
        raise TimeoutError(f"Microphone capture incomplete: {stats['received']}/{n} samples")
    audio=np.concatenate(data)
    assert len(audio)==n
    rms=float(np.sqrt(np.mean(audio.astype(np.float64)**2)))
    stats["rms_dbfs"]=round(20*np.log10(max(rms,1e-12)),2)
    stats["input_device"]=info.device_name
    stats["input_host_api"]=info.host_api
    stats["actual_latency_ms"]=float(info.latency_ms)
    stats["sample_rate"]=SR
    return audio,stats


def run(*, seconds, block_size, device, confidence, out,
        stream_factory=managed_input_stream):
    if out.exists():
        raise FileExistsError("Diagnostic JSON exists; refusing to overwrite: "+str(out))
    frozen=json.loads(DEFAULT_MODEL.with_name("provenance.json").read_text())
    model_sha=frozen["model_sha256"]
    if digest(DEFAULT_MODEL.read_bytes())!=model_sha:
        raise ValueError("S614 frozen NOTE checkpoint integrity mismatch")
    # Capture before loading and running three acoustic heads: the worker
    # consumes NO extra CPU during physical microphone recording.
    print(f"S635 microphone capture: {seconds:g}s, no WAV needed, three views "
          "will be evaluated from RAM after the microphone closes.",file=sys.stderr,
          flush=True)
    audio,cap=capture_pcm(seconds=seconds,block_size=block_size,
                          device=device,stream_factory=stream_factory)
    print(json.dumps(dict(s635_microphone_capture=cap)),file=sys.stderr,flush=True)
    with threadpool_limits(limits=1):
        model=load_heads(DEFAULT_MODEL,expected_sha256=model_sha)
        report=diagnostic(audio,model,confidence=confidence)
    report["source"]="microphone_capture_in_ram"
    report["capture"]=cap
    report["reference_sha256"]=model_sha
    report["accuracy_not_yet_measured"]=True
    report["not_live_midi_output"]=True
    report["warning"]="Proposals, not note correctness. No inference ran during capture."
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(dict(
        result=str(out),capture=cap,
        candidate_counts=report["candidate_counts"],
        analysis_CPU_over_audio_ratio=report["worker_over_audio_ratio"],
        warning=report["warning"],
    ),indent=2))
    return report


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mic",action="store_true",
                    help="Microphone capture into RAM, no WAV file required")
    ap.add_argument("--list-devices",action="store_true",
                    help="List available microphone inputs")
    ap.add_argument("--device",default=None,
                    help="Microphone name/index or 'WASAPI:Microphone name'")
    ap.add_argument("--seconds",type=float,default=8.)
    ap.add_argument("--block-size",type=int,default=1024)
    ap.add_argument("--confidence",type=float,default=.6)
    ap.add_argument("--out",type=Path,default=Path("three_views_mic.json"))
    args=ap.parse_args(argv)
    if args.list_devices:
        print(json.dumps(audio_devices("input"),indent=2))
        return
    if not args.mic:
        ap.error("Specify --mic or --list-devices; --wav is not required")
    if not 1.<=args.seconds<=120.:
        ap.error("--seconds must be within 1..120")
    if not 256<=args.block_size<=4096:
        ap.error("--block-size must be 256..4096")
    if not 0.<args.confidence<1.:
        ap.error("--confidence must be within (0,1)")
    run(seconds=args.seconds,block_size=args.block_size,
        device=args.device,confidence=args.confidence,out=args.out)


if __name__=="__main__":
    main()
