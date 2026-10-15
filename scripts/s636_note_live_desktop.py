"""S636 optional Windows microphone -> NOTE S614/S632 -> MIDI host.

Designed from the verified Windows I/O precautions in Andriamarosoa/midi.
No pitch recognition weights, acoustic features or selection thresholds are
copied from MIDI. S634 is the transport and safety layer.

Hardware tests are NOT covered by GitHub CI; run actual mic/loopback checks
on the target machine before claiming end-to-end latency.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

from causal_note.strict_live import HOP,SR,StrictLiveStream,load_heads
from causal_note.s634_safe_live_transport import SafeLiveTransport,TransportConfig

BASELINE_SHA="7ccf046e3a1870eaca3a108e6f9565298aeeadf43732a2f83559542c03f5f17f"
COST_FILES=("S632_Q_emit_train_only.joblib",
            "S632_Q_reject_train_only.joblib",
            "S632_Q_wait_train_only.joblib")
COST_HASHES=("207329be553ad179b461b725749e69244c080ac562b71364df6dc58066620218",
             "e57c818431e2544eba3d0b22d59d6df621cbac7637c7874df98424d8fb0a6a5b",
             "b54f631eaaeb0c31bc9f072bbfe2f6b76b9dcdd795f06401389c76a61824a4c6")


class MidiOutput:
    """Stateful MIDI sink with explicit emergency note-off on shutdown."""

    def __init__(self,port,*,channel=0,message_factory=None):
        if not 0<=channel<=15:raise ValueError("MIDI channel outside 0..15")
        if message_factory is None:
            import mido
            message_factory=mido.Message
        self.port=port
        self.message=message_factory
        self.channel=int(channel)
        self.active=set()
        self.counts=Counter()

    def __call__(self,event):
        pitch=int(event["pitch"])
        if not 0<=pitch<=127:raise ValueError("Invalid MIDI pitch")
        if event["type"]=="note_on":
            if pitch in self.active:
                # An unmatched restrike is invalid; do not transmit it.
                raise RuntimeError("Duplicate MIDI NoteOn without NoteOff")
            self.port.send(self.message("note_on",note=pitch,
                              velocity=int(event["velocity"]),channel=self.channel))
            self.active.add(pitch)
            self.counts["note_on"]+=1
        elif event["type"]=="note_off":
            self.port.send(self.message("note_off",note=pitch,velocity=0,
                                       channel=self.channel))
            self.active.discard(pitch)
            self.counts["note_off"]+=1
        else:
            raise ValueError("Unsupported NOTE event")

    def panic(self):
        errors=[]
        for pitch in sorted(self.active):
            try:
                self.port.send(self.message("note_off",note=pitch,
                                          velocity=0,channel=self.channel))
            except Exception as exc:errors.append(str(exc))
        self.active.clear()
        try:
            self.port.send(self.message("control_change",control=123,value=0,
                                       channel=self.channel))
        except Exception as exc:errors.append(str(exc))
        if errors:
            raise OSError("MIDI port refused emergency all-notes-off: "+"; ".join(errors))


class DiagnosticSink:
    """No MIDI device required; verifies event pairing and exposes counts."""
    def __init__(self):
        self.active=set()
        self.counts=Counter()

    def __call__(self,event):
        pitch=int(event["pitch"])
        if event["type"]=="note_on":
            if pitch in self.active:raise RuntimeError("Duplicate NoteOn")
            self.active.add(pitch)
        elif event["type"]=="note_off":
            self.active.discard(pitch)
        self.counts[event["type"]]+=1

    def panic(self):
        self.active.clear()


def engine_factory(model,cost_dir=None):
    from scripts.live_strict_notes import DEFAULT_MODEL
    native=load_heads(model or DEFAULT_MODEL,expected_sha256=BASELINE_SHA)
    if cost_dir is None:
        return lambda:StrictLiveStream(native),"S614"
    from causal_note.s632_streaming_runtime import S632Live
    from causal_note.s632_risk_policy import RiskPolicy
    names=[Path(cost_dir)/name for name in COST_FILES]
    for name,digest in zip(names,COST_HASHES):
        if not name.is_file() or hashlib.sha256(name.read_bytes()).hexdigest()!=digest:
            raise ValueError("S632 weights must match the frozen trained SHA256")
    return (lambda:S632Live(native,names,COST_HASHES,
             RiskPolicy(mode="assist",extra_wait_cost=.10,max_emit_cost=.65))),"S632-assist"


def port_option(text):
    if text is None:return None
    return int(text) if str(text).isdigit() else text


def run_live(args):
    try:
        import numpy as np
        import sounddevice as sd
    except ImportError as exc:
        raise RuntimeError("Install sounddevice: pip install sounddevice") from exc
    if args.list_devices:
        print(sd.query_devices())
        return 0
    if args.list_midi:
        import mido
        print("\n".join(mido.get_output_names()))
        return 0
    factory,mode=engine_factory(args.model,args.s632_dir)
    if args.no_midi:
        sink=DiagnosticSink()
        port=None
    else:
        try:import mido
        except ImportError as exc:
            raise RuntimeError("Install MIDI drivers: pip install mido python-rtmidi") from exc
        port=mido.open_output(args.midi_port)
        sink=MidiOutput(port,channel=args.channel)

    host=SafeLiveTransport(factory,sink,
        TransportConfig(queue_capacity=16,max_backlog_blocks=3,stall_seconds=1.0))
    clock_sample=[0]
    callback_errors=[]
    current_device=port_option(args.audio_device)
    shutdown_reason="duration" if args.duration_s else "user_interrupt"
    io_latency=None
    started=time.monotonic()

    def callback(indata,frames,timing,status):
        # Sounddevice callback: NEVER infer, create MIDI notes, or access disk.
        absolute=clock_sample[0]
        clock_sample[0]+=frames
        try:
            pcm=np.asarray(indata[:,0],np.float32).copy()
            host.capture(pcm,start_sample=absolute,
                         status_ok=not bool(status))
        except Exception as exc:
            callback_errors.append(type(exc).__name__+": "+str(exc))
            raise sd.CallbackAbort

    try:
        with sd.InputStream(device=current_device,samplerate=SR,
                 channels=1,dtype="float32",blocksize=HOP,
                 latency="low",callback=callback) as microphone:
            io_latency=float(microphone.latency)
            while True:
                if args.duration_s is not None and time.monotonic()-started>=args.duration_s:
                    break
                if callback_errors:
                    shutdown_reason="capture_callback_error"
                    break
                if host.check_stall():
                    shutdown_reason="microphone_stall"
                    break
                processed=host.poll(max_blocks=2)
                if not processed:
                    time.sleep(.001)
    except KeyboardInterrupt:
        shutdown_reason="user_interrupt"
    except Exception:
        shutdown_reason="runtime_error"
        raise
    finally:
        try:
            host.close(drain=False)
        finally:
            try:sink.panic()
            finally:
                if port is not None:port.close()
        report=dict(status="S636_OPTIONAL_DESKTOP_NOTE_LIVE",
                    model=mode,original_S80_S614_unchanged=True,
                    shutdown_reason=shutdown_reason,
                    audio_device=str(current_device),
                    audio_io_negotiated_input_latency_ms=(
                        io_latency*1000 if io_latency is not None else None),
                    transport=host.report(),midi_events=dict(sink.counts),
                    all_notes_closed=not sink.active,
                    callback_errors=callback_errors,
                    acoustic_to_MIDI_physical_latency_measured=False,
                    caution="I/O latency, model time, candidate waiting and real playback latency are separate.")
        if args.report_json is not None:
            out=Path(args.report_json)
            out.parent.mkdir(parents=True,exist_ok=True)
            out.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
        print(json.dumps(report,indent=2))
    return 0


def main():
    parser=argparse.ArgumentParser(description="NOTE causal MIDI LIVE host (S634/S636)")
    parser.add_argument("--model",type=Path,default=None)
    parser.add_argument("--s632-dir",type=Path,default=None,
                       help="Optional verified frozen S632 Q models")
    parser.add_argument("--audio-device",default=None,
                       help="Input index or name; 44.1 kHz mono supported device")
    parser.add_argument("--midi-port",default=None,
                       help="Mido output name; defaults to configured system port")
    parser.add_argument("--channel",type=int,default=0)
    parser.add_argument("--no-midi",action="store_true",
                       help="Diagnostic-only; no physical MIDI output")
    parser.add_argument("--duration-s",type=float,default=None)
    parser.add_argument("--report-json",type=Path,default=None)
    parser.add_argument("--list-devices",action="store_true")
    parser.add_argument("--list-midi",action="store_true")
    args=parser.parse_args()
    if args.duration_s is not None and args.duration_s<=0:
        parser.error("--duration-s must be positive")
    return run_live(args)


if __name__=="__main__":
    raise SystemExit(main())
