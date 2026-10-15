"""S638: genuine same-WAV NOTE S614 vs MIDI V2.2.1 GuitarSet dev A/B.

Runs ORIGINAL published MIDI TFLite checkpoint and ORIGINAL NOTE S614
native checkpoint on exactly the same real GuitarSet annotated 24s WAV.
Three original NOTE DEV compositions, 1 fixed recording each; NO threshold
tuning. The resulting F1 is comparable under S635 but NOT an independent
holdout: NOTE's DEV is historically exposed and MIDI train overlap unknown.

The user-visible project keeps S80, S614, S632, and MIDI V2.2.1 unchanged.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import time
import zipfile

import numpy as np
from scipy.io import wavfile
from causal_note.strict_live import SR,HOP,StrictLiveStream,load_heads
from causal_note.s635_shared_midi_metric import compare_manifest
from scripts.s613_prepare_strict_live import download,labels,digest
from scripts.s613_train_strict_live import real_notes
from scripts.s616_live_dev_ablation import ROOT,MODEL

MIDI_TFLITE_SHA="4a4df49d34c1fd359fab98c20bb598f37c7f42f22493c70923a2e829176df744"
S614_SHA="7ccf046e3a1870eaca3a108e6f9565298aeeadf43732a2f83559542c03f5f17f"


def convert_same_wav(spec,source_zip,temp):
    original_bytes=source_zip.read(spec["audio"])
    rate,pcm=wavfile.read(io.BytesIO(original_bytes))
    if rate!=SR or pcm.dtype not in (np.int16,np.int32):
        raise ValueError("S638 only supports pinned original 44.1kHz PCM GuitarSet")
    if pcm.ndim==2:
        # Single reproducible conversion for both engines, not two
        # independent channel-mixing/normalisation schemes.
        data=pcm.astype(np.float64).mean(axis=1)
        if pcm.dtype==np.int32:data=np.clip(np.rint(data/65536.),-32768,32767)
        data=np.clip(np.rint(data),-32768,32767).astype(np.int16)
    elif pcm.dtype==np.int16:
        data=pcm
    else:
        data=np.clip(np.rint(pcm.astype(np.float64)/65536.),-32768,32767).astype(np.int16)
    data=data[:int(spec["seconds"]*SR)].copy()
    if not SR<=len(data)<=int(spec["seconds"]*SR):
        raise AssertionError("Invalid real input duration")
    dst=temp/(spec["name"]+".wav")
    wavfile.write(dst,SR,data)
    encoded=dst.read_bytes()
    return dst,digest(encoded),digest(original_bytes),data


def s614_events(pcm,heads):
    audio=pcm.astype(np.float32)/32768.
    stream=StrictLiveStream(heads)
    out=[]
    for start in range(0,len(audio),HOP):
        out.extend(stream.push(audio[start:start+HOP],start_sample=start))
    out.extend(stream.finish())
    assert sum(x["type"]=="note_on" for x in out)==sum(x["type"]=="note_off" for x in out)
    return out


def midi_events(wav_path,artifact_dir,midi_root,temp):
    if not (midi_root/"src/polyphonic/transcribe.py").is_file():
        raise FileNotFoundError("Pinned original midi source not checked out")
    model=artifact_dir/"guitar_midi_polyphonic.tflite"
    if digest(model.read_bytes())!=MIDI_TFLITE_SHA:
        raise ValueError("Original MIDI 2.2.1 model SHA256 mismatch")
    from src.polyphonic import transcribe as midi_transcribe
    original_write=midi_transcribe.write_midi
    captured=[]
    def save_and_record(output,events,*,channel,program):
        captured.extend(dict(x) for x in events)
        return original_write(output,events,channel=channel,program=program)
    midi_transcribe.write_midi=save_and_record
    try:
        report=midi_transcribe.transcribe(
            wav_path,temp/(wav_path.stem+".mid"),artifact_dir,
            channel=0,program=0,auto_level=False,audio_gain=1.)
    finally:
        midi_transcribe.write_midi=original_write
    on=[dict(type="note_on",pitch=int(x["pitch"]),time_s=float(x["time_s"]))
        for x in captured if x["kind"]=="note_on"]
    if len(on)!=report["note_on_events"]:
        raise AssertionError("Raw causal decoder events do not match transcript")
    return on,report


def main(out,midi_root,all_dev=False):
    if out.exists():raise FileExistsError("S638 evidence is immutable")
    out.mkdir(parents=True)
    frozen=json.loads((ROOT/"selection_frozen.json").read_text())
    protocol=json.loads((ROOT/"protocol.json").read_text())
    assert digest((ROOT/"protocol.json").read_bytes())==frozen["protocol_sha256"]
    assert digest(MODEL.read_bytes())==S614_SHA==frozen["model_sha256"]
    items=[s for s in protocol["sources"] if s["split"]=="dev" and s["corpus"]=="GuitarSet"]
    assert len(items)==12
    groups={}
    for s in sorted(items,key=lambda x:x["name"]):
        groups.setdefault(s["composition"],s)
    assert len(groups)==3
    chosen=(sorted(items,key=lambda x:x["name"]) if all_dev
            else [groups[key] for key in sorted(groups)])
    assert len(chosen)==(12 if all_dev else 3)
    assert all(s["name"].startswith("04_") for s in chosen)
    artifact_dir=midi_root/"artifacts/guitar_midi_polyphonic_v2_2_0"
    heads=load_heads(MODEL,expected_sha256=S614_SHA)
    records=[]
    started=time.perf_counter()
    with tempfile.TemporaryDirectory() as d:
        cache=Path(d)
        (cache/"data").mkdir()
        archive=protocol["guitar_archives"]
        for key in ("GuitarSet_audio","GuitarSet_annotations"):
            a=archive[key]
            download(a["url"],cache/"data"/(key+".zip"),a["size"],a["sha256"])
        with zipfile.ZipFile(cache/"data/GuitarSet_audio.zip") as zipped:
            for spec in chosen:
                wav,sha,original_sha,pcm=convert_same_wav(spec,zipped,cache)
                duration=len(pcm)/SR
                annotated=labels(spec,cache)
                truth=real_notes(annotated["notes"],duration)
                reference=[dict(pitch=int(n["pitch"]),start_s=float(n["time"]),
                                end_s=min(duration,float(n["time"]+n["duration"])))
                           for n in truth if n["duration"]>0]
                note=s614_events(pcm,heads)
                print("S638 NOTE",spec["name"],sum(x["type"]=="note_on" for x in note),flush=True)
                midi,report=midi_events(wav,artifact_dir,midi_root,cache)
                print("S638 MIDI",spec["name"],len(midi),flush=True)
                record=dict(clip_id=spec["name"],corpus="GuitarSet",
                    duration_s=duration,audio_sha256=sha,reference=reference,
                    note=dict(audio_sha256=sha,audio_sample_rate=SR,events=note),
                    midi=dict(audio_sha256=sha,audio_sample_rate=SR,events=midi))
                records.append(record)
                print("S638 COMMON PCM",spec["name"],sha,flush=True)
    combined=compare_manifest(dict(protocol="NOTE_MIDI_SHARED_ONSET_V1",clips=records))
    # Large records contain events/matches; save separately.
    (out/"common_metric.json").write_text(json.dumps(combined,indent=2)+"\n")
    (out/"frozen_inputs_and_emissions.json").write_text(
        json.dumps(dict(protocol="NOTE_MIDI_SHARED_ONSET_V1",clips=records),indent=2)+"\n")
    brief=dict(status="S638_REAL_CHECKPOINT_SAME_GUITARSET_DEV_PCM_AB",
        original_note_sha256=S614_SHA,original_midi_tflite_sha256=MIDI_TFLITE_SHA,
        source_count=len(records),composition_count=len(groups),
        shared_pitch_range=[40,76],
        causal_deadlines_ms=combined["common_deadlines_ms"],
        note=combined["overall"]["note"],midi=combined["overall"]["midi"],
        clips=[dict(id=c["clip_id"],audio_sha256=c["audio_sha256"],
                    ref=len(c["reference"]),note_on=sum(e["type"]=="note_on" for e in c["note"]["events"]),
                    midi_on=len(c["midi"]["events"])) for c in records],
        notes_tested_on_previous_NOTE_DEV=True,
        midi_training_overlap_not_proven_absent=True,
        no_new_independent_test_claim=True,
        no_promotion=True,elapsed_seconds=round(time.perf_counter()-started,2))
    (out/"report.json").write_text(json.dumps(brief,indent=2)+"\n")
    print("S638 REAL AUDIO METRIC",json.dumps({
       "clips":brief["clips"],
       "note_80":brief["note"]["80"],"midi_80":brief["midi"]["80"],
       "note_35":brief["note"]["35"],"midi_35":brief["midi"]["35"],
       "provisional_not_independent":True},indent=2),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--midi-root",type=Path,required=True)
    p.add_argument("--all-dev",action="store_true",
                   help="Score the complete 12-recording original GuitarSet DEV, without tuning")
    a=p.parse_args()
    main(a.out,a.midi_root,all_dev=a.all_dev)
