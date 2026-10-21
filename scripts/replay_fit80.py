"""Self-contained signed GuitarSet FIT80 replay: Exact-K ONLY, no other axes.

GT is read only after a track's complete causal PCM->K predictions.
The S834 native256 benchmark baseline is 232295/306536.
"""
import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path
import numpy as np
from scipy.io import wavfile

from exact_k.acoustics import SR,HOP,load_native_heads
from exact_k.temporal import load_frozen_source
from exact_k.runtime import ExactKOnly256,SCHEMA

COMPOSITIONS=("BN1","Funk2","Jazz1","Rock3","SS1")
SOURCE_SCHEMA="S800_FIT80_OOF_PER_FRAGMENT_MIDI256_V1"


def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def truth_k(notes,ends):
    result=np.zeros(len(ends),np.int16)
    times=ends/SR
    for note in notes:
        p=int(note["pitch"])
        if not 40<=p<=88:raise ValueError("GT pitch out of 40..88")
        occupied=(times>=float(note["time"]))&(
            times<float(note["time"])+float(note["duration"]))
        result+=occupied
    return np.minimum(result,6).astype(np.uint8)


def run(audio_zip,traces,manifest,checkpoint,source_report,native_model,out,tracks_per_composition=16):
    if not 1<=tracks_per_composition<=16:raise ValueError("track cap invalid")
    if out.exists():raise FileExistsError("do not overwrite scientific evidence")
    source=json.loads(manifest.read_text())
    if source.get("schema")!=SOURCE_SCHEMA or not source.get("complete") or len(source.get("fit_files",[]))!=80:
        raise ValueError("unsigned or incomplete FIT80 manifest")
    if any(source.get(s) for s in ("CAL_opened","HELD_opened","TEST_opened")):
        raise ValueError("protected test/CAL data exposed")
    native=load_native_heads(native_model)
    model,s708_sha=load_frozen_source(checkpoint,source_report)
    selected=[entry for comp in COMPOSITIONS for entry in sorted((
      f for f in source["fit_files"] if f["composition"]==comp),
      key=lambda r:r["index"])[:tracks_per_composition]]
    if len(selected)!=5*tracks_per_composition:raise ValueError("wrong compositions")
    out.mkdir(parents=True)
    report=dict(schema=SCHEMA,completed=False,promoted=False,
        only_axis="exact_k",native_hop=HOP,tracks=[],frames=0,correct=0,
        undercount=0,overcount=0,cal_held_test_unopened=True,
        s708_source_original_training_hop=1024,
        s708_source_checkpoint_sha256=s708_sha,
        source_native_model_sha256=digest(native_model),
        source_manifest_sha256=digest(manifest),
        midi_production_modified=False)
    with zipfile.ZipFile(audio_zip) as archive:
        for entry in selected:
            mpath=traces/f"fit_{entry['index']:03d}.json"
            if digest(mpath)!=entry["json_sha256"]:raise ValueError("signed track metadata mismatch")
            meta=json.loads(mpath.read_text())
            if meta["name"]!=entry["name"] or meta["composition"]!=entry["composition"]:
                raise ValueError("signed track identity mismatch")
            rate,data=wavfile.read(io.BytesIO(archive.read(meta["name"]+"_mic.wav")))
            if rate!=SR or data.ndim!=1 or data.dtype!=np.int16:
                raise ValueError("unsigned audio format")
            audio=data[:meta["samples"]].astype(np.float32)/32768.
            ends=np.arange(HOP,len(audio)+1,HOP,dtype=np.int64)
            stream=ExactKOnly256(native,model)
            predicted=[]
            for start in range(0,len(audio),HOP):
                output=stream.push(audio[start:start+HOP],start_sample=start)
                if len(output)!=(1 if start+HOP<=len(audio) else 0):
                    raise AssertionError("missing or extra PCM256 decision")
                if output:
                    if output[0].sample!=start+HOP:
                        raise AssertionError("causal timestamp mismatch")
                    predicted.append(output[0].count)
            if len(predicted)!=len(ends) or stream.model_calls!=len(ends):
                raise AssertionError("exact count of real S708 neural evaluations")
            pred=np.asarray(predicted,np.uint8)
            truth=truth_k(meta["fit_truth"],ends)
            correct=int((pred==truth).sum())
            under=int((pred<truth).sum())
            over=int((pred>truth).sum())
            assert correct+under+over==len(ends)
            report["frames"]+=len(ends)
            report["correct"]+=correct
            report["undercount"]+=under
            report["overcount"]+=over
            report["tracks"].append(dict(name=meta["name"],composition=entry["composition"],
                frames=len(ends),correct=correct,undercount=under,overcount=over,
                diagnostics=stream.diagnostics()))
            (out/"report.json").write_text(json.dumps(report,indent=2))
            print("EXACT_K_ONLY_256_TRACK",entry["index"],entry["composition"],correct,len(ends),flush=True)
    report["accuracy"]=report["correct"]/report["frames"]
    report["completed"]=True
    if tracks_per_composition==16 and (
         report["frames"]!=306536 or report["correct"]!=232295):
        raise AssertionError("Exact-K separation regressed from S834 232295/306536")
    (out/"report.json").write_text(json.dumps(report,indent=2))
    print("EXACT_K_ONLY_NATIVE256_FIT80_FINAL",report["correct"],report["frames"],report["accuracy"],flush=True)
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for field in ("audio-zip","traces","manifest","checkpoint","source-report","native-model","out"):
        p.add_argument("--"+field,type=Path,required=True)
    p.add_argument("--tracks-per-composition",type=int,default=16)
    a=p.parse_args()
    run(a.audio_zip,a.traces,a.manifest,a.checkpoint,
        a.source_report,a.native_model,a.out,a.tracks_per_composition)
