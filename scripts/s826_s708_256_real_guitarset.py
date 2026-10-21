"""S826 real GuitarSet SHA-pinned pilot: SAME S708 weights at 256 vs 1024.

DRAFT/SHADOW only. Scores measured on 1 signed track per each of the five
GuitarSet FIT80 compositions, not on official CAL32 or previously untouched
held-out test. This verifies the actual unmodified neural architecture
is called every 256, not 4x256; no promise of unchanged 1024-trained F1.
"""
from __future__ import annotations
import argparse,hashlib,io,json,zipfile
from pathlib import Path
from time import perf_counter
import numpy as np
from scipy.io import wavfile

from causal_note.strict_live import HOP,SR,load_heads
from causal_note.s712_s708_live import S708StrictLiveStream,load_s708_checkpoint
from causal_note.s826_s708_whole_native256 import S826S708Native256Stream
from scripts.s704_joint_metrics import metric,aggregate,SCORES
from scripts.s813_prepare_native256 import save_json,digest_file

SCHEMA="S826_S708_ORIGINAL_WEIGHTS_FULL_ARCH_FIRST_WAV_PILOT_STRICT256_V1"
COMPS=("BN1","Funk2","Jazz1","Rock3","SS1")

def replay(audio,heads,model,policy,*,strict256):
    stream=(S826S708Native256Stream(heads,model,policy)
            if strict256 else S708StrictLiveStream(heads,model,policy))
    events=[]
    begin=perf_counter()
    for start in range(0,len(audio),HOP):
        chunk=audio[start:start+HOP]
        events.extend(stream.push(chunk,start_sample=start))
    events.extend(stream.finish())
    elapsed=perf_counter()-begin
    stride=HOP if strict256 else 4*HOP
    diag=stream.diagnostics()
    if diag["decision_hop"]!=stride:
        raise AssertionError("unpermitted hidden 1024 decision cadence")
    expected_calls=len(audio)//stride
    if stream.model_calls!=expected_calls:
        raise AssertionError("S708 neural infer calls less frequent than requested")
    if any(e["available_sample"]<e["decision_sample"] for e in events):
        raise AssertionError("backdated emitted event")
    if any(e["decision_sample"]%stride for e in events if e["reason"]!="stream_end"):
        raise AssertionError("non-grid NoteOn/NoteOff")
    active=set();peak=0
    for e in events:
        p=e["pitch"]
        if e["type"]=="note_on":
            if p in active:raise AssertionError("duplicate NoteOn")
            active.add(p)
        else:
            if p not in active:raise AssertionError("orphan NoteOff")
            active.remove(p)
        peak=max(peak,len(active))
        if peak>6:raise AssertionError("six voice limit breached")
    if active:raise AssertionError("event stream left open")
    return events,dict(
       decision_hop_samples=stride,torch_inference_calls=stream.model_calls,
       elapsed_seconds=elapsed,frames=int(len(audio)//stride),
       audio_seconds=float(len(audio)/SR),peak_polyphony=peak,
       diagnostic_only_not_real_mic_latency=True,
       no_future_samples=True)

def run(audio_zip,traces,manifest_path,source_checkpoint,source_report,native_model,out):
    if out.exists():raise FileExistsError("S826 output must remain immutable")
    manifest=json.loads(manifest_path.read_text())
    if len(manifest.get("fit_files",[]))!=80 or not manifest.get("complete"):
        raise AssertionError("original signed S800 FIT80 missing")
    model,policy,model_sha=load_s708_checkpoint(source_checkpoint,source_report)
    heads=load_heads(native_model)
    if (heads.metadata.get("selection") is None or
        not set(COMPS).issubset({x["composition"] for x in manifest["fit_files"]})):
        raise AssertionError("S708 source or original composition missing")
    selected={comp:sorted((x for x in manifest["fit_files"]
         if x["composition"]==comp),key=lambda z:z["index"])[0]
         for comp in COMPS}
    out.mkdir(parents=True);(out/"events").mkdir()
    report=dict(schema=SCHEMA,complete=False,promotion=False,
        source_S708_sha256=model_sha,source_native_sha256=digest_file(native_model),
        original_architecture=True,weights_trained_at_1024=True,
        source_neural_architecture_retrained_at_256=False,
        source_dataset="GuitarSet FIT80 original S800 signed subset 5",
        source_manifest_sha256=digest_file(manifest_path),
        selected_fit_compositions=list(COMPS),selected_fit_recordings=5,
        independent_external_generalization_verified=False,
        note_count_six_metrics_in_single_MIDI_stream=True,
        original_1024_stream_unchanged=True,
        production_LIVE_modified=False,shadow_only=True,
        decision_hop_samples=256,feature_hop_samples=256,metric_hop_samples=256,
        added_lookahead_samples=0,added_hold_samples=0,rows=[])
    with zipfile.ZipFile(audio_zip) as audio_archive:
        for comp in COMPS:
            spec=selected[comp]
            stem=f"fit_{spec['index']:03d}"
            meta_file=traces/(stem+".json")
            if digest_file(meta_file)!=spec["json_sha256"]:
                raise AssertionError("signed S800 source metadata SHA drift")
            meta=json.loads(meta_file.read_text())
            if meta["name"]!=spec["name"] or meta["composition"]!=comp:
                raise AssertionError("source identity changed")
            member=spec["name"]+"_mic.wav"
            wave_bytes=audio_archive.read(member)
            rate,pcm=wavfile.read(io.BytesIO(wave_bytes))
            if rate!=SR or pcm.ndim!=1 or pcm.dtype!=np.int16:
                raise AssertionError("original 44.1kHz mono signed 16-bit GuitarSet mic required")
            pcm=(pcm.astype(np.float32)/32768)[:meta["samples"]]
            if len(pcm)!=meta["samples"] or len(pcm)<HOP*4:
                raise AssertionError("GuitarSet PCM or scientific clipping not original")
            ends=np.arange(HOP,len(pcm)+1,HOP,dtype=np.int64)
            truth=meta["fit_truth"]
            original=meta["source_events"]
            reference=metric(original,truth,ends)
            for key,field in (("active_k_correct","active_k_correct"),
                        ("active_frames","active_frames"),("k_correct","k_correct")):
                if reference[key]!=meta["source_counts"][field]:
                    raise AssertionError("S800 original MIDI scientific GT source mismatch")
            a,trace_a=replay(pcm,heads,model,policy,strict256=True)
            b,trace_b=replay(pcm,heads,model,policy,strict256=False)
            scores=dict(S826_256=metric(a,truth,ends),S712_1024=metric(b,truth,ends),
                 signed_S800_original=reference)
            safe={k:{z:float(row[z]) for z in SCORES} for k,row in scores.items()}
            track=dict(composition=comp,index=spec["index"],name=spec["name"],
                original_pcm_samples=len(pcm),true_metric_ends=len(ends),
                scores=scores,metrics_six=safe,
                strict256=trace_a,original1024=trace_b,
                original_source_json_sha256=spec["json_sha256"])
            (out/"events"/f"{stem}_strict256.json").write_text(json.dumps(a)+"\n")
            (out/"events"/f"{stem}_original1024.json").write_text(json.dumps(b)+"\n")
            report["rows"].append(track)
            save_json(out/"report.json",report)
            print("S826_SOURCE_S708_PER_COMPOSITION",comp,
                  json.dumps(dict(strict256=safe["S826_256"],legacy1024=safe["S712_1024"],
                                  H0=safe["signed_S800_original"],
                                  calls256=trace_a["torch_inference_calls"],
                                  calls1024=trace_b["torch_inference_calls"])),
                  flush=True)
    report["aggregate"]={name:aggregate([r["scores"][name] for r in report["rows"]])
        for name in ("S826_256","S712_1024","signed_S800_original")}
    report["complete"]=True
    save_json(out/"report.json",report)
    print("S826_REAL_PILOT_COMPLETE",json.dumps({
      name:{k:round(v[k],6) for k in SCORES}
      for name,v in report["aggregate"].items()},sort_keys=True),flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    for arg in ("audio-zip","traces","manifest-path","source-checkpoint",
                "source-report","native-model","out"):
        ap.add_argument("--"+arg,type=Path,required=True)
    a=ap.parse_args()
    run(a.audio_zip,a.traces,a.manifest_path,a.source_checkpoint,
        a.source_report,a.native_model,a.out)
