"""S637: real recorded WAV, exact same MIDI V2.2.1 and NOTE S614 audio.

Uses a real ~60 sec WAV already committed in public Andriamarosoa/midi.
NOTE native S614 receives that raw PCM through the new S634 queue bridge.
The pre-existing MIDI V2.2.1 '.mid' (and its report) was made from the same
recorded WAV; this script does not retrain MIDI or simulate its predictor.

NO ANNOTATED TRUTH on this recording => NO F1 / accuracy claim.
Only event density, pitch coverage, timing and CPU pipeline are checked.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.io import wavfile
import mido
from causal_note.strict_live import HOP,SR,StrictLiveStream,load_heads
from causal_note.s634_safe_live_transport import SafeLiveTransport
from scripts.s636_note_live_desktop import BASELINE_SHA


def midi_noteons(path):
    source=mido.MidiFile(str(path))
    total=0.
    tempo=500000
    onset=[]
    for ev in mido.merge_tracks(source.tracks):
        total+=mido.tick2second(ev.time,source.ticks_per_beat,tempo)
        if ev.type=="set_tempo":
            tempo=ev.tempo
        elif ev.type=="note_on" and ev.velocity>0:
            onset.append((int(ev.note),float(total)))
    return onset


def run(wav_path,midi_path,original_report,out):
    if out.exists():raise FileExistsError("S637 A/B report is immutable")
    audio_bytes=wav_path.read_bytes()
    audio_sha=hashlib.sha256(audio_bytes).hexdigest()
    rate,pcm=wavfile.read(wav_path)
    if rate!=SR or pcm.ndim!=1:
        raise AssertionError("Original 44.1kHz single-channel PCM required")
    if pcm.dtype==np.int16:
        signal=pcm.astype(np.float32)/32768.
    elif pcm.dtype==np.float32:
        signal=pcm.copy()
    else:
        raise ValueError("Unexpected original input capture format")
    if len(signal)!=2636800:
        raise AssertionError("Recorded MIDI 60s reference sample count changed")
    midi_report=json.loads(original_report.read_text())
    assert midi_report["input_wav"].replace("\\","/").endswith("runs/polyphonic/live_poly_v2.wav")
    before=midi_noteons(midi_path)
    assert len(before)==midi_report["note_on_events"]==121,(len(before),midi_report["note_on_events"])

    from scripts.live_strict_notes import DEFAULT_MODEL
    native=load_heads(DEFAULT_MODEL,expected_sha256=BASELINE_SHA)
    events=[]
    stream=SafeLiveTransport(lambda:StrictLiveStream(native),events.append)
    start=time.perf_counter()
    for position in range(0,len(signal),HOP):
        block=signal[position:position+HOP]
        assert len(block)==HOP
        stream.capture(block,start_sample=position)
        stream.poll()
    stream.close()
    elapsed=time.perf_counter()-start
    report=stream.report()
    assert report["recoveries"]==report["queue_drops"]==report["backlog_drops"]==0
    assert all(ev["available_sample"]>=ev["decision_sample"] for ev in events)
    note=[(e["pitch"],e["time"]) for e in events if e["type"]=="note_on"]
    paired=Counter(e["type"] for e in events)
    assert paired["note_on"]==paired["note_off"],paired

    common=lambda seq:Counter(pitch for pitch,time_s in seq if 40<=pitch<=76)
    midi_range=common(before)
    note_range=common(note)
    report_json=dict(status="S637_RECORDED_SAME_WAV_ENGINE_AB_NO_LABELS",
        original_audio_sha256=audio_sha,
        source_filename=wav_path.name,samples=len(signal),sample_rate=SR,
        audio_duration_s=len(signal)/SR,
        MIDI_original_decoder="V2.2.1 frozen previous transcript",
        NOTE_reference="S614 original SHA256 "+BASELINE_SHA,
        MIDI_noteon_all=len(before),
        NOTE_noteon_all=len(note),
        MIDI_noteon_common_range=sum(midi_range.values()),
        NOTE_noteon_common_range=sum(note_range.values()),
        NOTE_noteon_outside_common_range=len(note)-sum(note_range.values()),
        MIDI_pitch_histogram={str(k):v for k,v in sorted(midi_range.items())},
        NOTE_pitch_histogram={str(k):v for k,v in sorted(note_range.items())},
        S634_transport=report,simulation_processing_wall_seconds=elapsed,
        no_f1_without_annotations=True,
        not_hardware_microphone_to_midi_measurement=True,
        no_accuracy_winner_declared=True,
        original_S80_S614_unchanged=True,
        source_midi_report_original_noteon=121)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report_json,indent=2)+"\n")
    print("S637 REAL SAME WAV",json.dumps({k:report_json[k] for k in (
        "original_audio_sha256","MIDI_noteon_all","NOTE_noteon_all",
        "MIDI_noteon_common_range","NOTE_noteon_common_range",
        "NOTE_noteon_outside_common_range",
        "simulation_processing_wall_seconds","no_f1_without_annotations")},indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--wav",type=Path,required=True)
    p.add_argument("--midi",type=Path,required=True)
    p.add_argument("--midi-report",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    o=p.parse_args()
    run(o.wav,o.midi,o.midi_report,o.out)
