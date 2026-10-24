"""S841 forensic audit: genuine GuitarSet E2/E3 and simultaneous attacks.

Pure metadata plus REAL 256-sample original WAV blocks. No DSP shift, model
predictions, synthesis, S708 input, annotations during inference, or Onset
head edits. Does NOT confuse nearby NoteOn with known simultaneous sustain.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
from collections import Counter,defaultdict
from pathlib import Path
import zipfile

import numpy as np
from scipy.io import wavfile

SR,HOP=44100,256
E2,E3=40,52
COMP=("BN1","Funk2","Jazz1","Rock3","SS1")


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as fd:
        for chunk in iter(lambda:fd.read(1048576),b""):
            h.update(chunk)
    return h.hexdigest()


def audit(root,out):
    from scripts.onset_event_integrity256 import verify_inputs
    root=Path(root);out=Path(out)
    if out.exists():raise FileExistsError("audit report is immutable")
    manifest=verify_inputs(root)
    trace_dir=root/"traces"
    report=dict(schema="S841_REAL_GUITARSET_E2_E3_PCM256_V1",
                complete=False,hop_samples=HOP,source="signed GuitarSet original mono WAV",
                onset_only_truth_not_proof_of_simultaneous_sustain=True,
                pairs_are_independently_recorded=False,
                piano_pitch_labels={"E2":40,"E3":52},
                tracks=0,frames=0,counts=dict(),by_composition=dict(),
                by_texture=dict(),examples=[],errors=[])
    global_counts=Counter();by_comp=defaultdict(Counter);by_texture=defaultdict(Counter)
    with zipfile.ZipFile(root/"GuitarSet_audio.zip") as archive:
        for item in sorted(manifest["fit_files"],key=lambda x:int(x["index"])):
            path=trace_dir/("fit_%03d.json"%int(item["index"]))
            if sha(path)!=item["json_sha256"]:
                raise ValueError("S841 signed annotation mismatch")
            meta=json.loads(path.read_text(encoding="utf8"))
            if meta["name"]!=item["name"] or meta["composition"]!=item["composition"]:
                raise ValueError("S841 signed track identity mismatch")
            sr,pcm=wavfile.read(io.BytesIO(archive.read(meta["name"]+"_mic.wav")))
            if sr!=SR or pcm.dtype!=np.int16 or pcm.ndim!=1:
                raise ValueError("S841 requires real 44100 int16 mono guitar recording")
            end=int(meta["samples"])
            if end>len(pcm) or end<=HOP:
                raise ValueError("bad signed sample count")
            notes=meta["fit_truth"]
            onset=[(float(n["time"]),int(n["pitch"])) for n in notes]
            if any(not (0<=t<end/SR) for t,p in onset):
                raise ValueError("invalid GT onset timestamp")
            starts={E2:[],E3:[]}
            for sec,pitch in onset:
                if pitch in starts:starts[pitch].append(sec)
            within_hop=0
            within_80ms=0
            for t in starts[E2]:
                if any(abs(t-v)<=HOP/SR for v in starts[E3]):
                    within_hop+=1
                if any(abs(t-v)<=.08 for v in starts[E3]):
                    within_80ms+=1
            texture="solo" if meta["name"].endswith("_solo") else (
                    "comp" if meta["name"].endswith("_comp") else "other")
            summary=dict(index=item["index"],name=meta["name"],
                         composition=item["composition"],texture=texture,
                         e2_onsets=len(starts[E2]),e3_onsets=len(starts[E3]),
                         both_onsets_within_one_256_hop=within_hop,
                         both_onsets_within_80ms=within_80ms)
            report["examples"].append(summary)
            metrics=dict(tracks=1,e2_onsets=len(starts[E2]),e3_onsets=len(starts[E3]),
                near_256_e2_e3=within_hop,near_80ms_e2_e3=within_80ms,
                e2_tracks=int(bool(starts[E2])),e3_tracks=int(bool(starts[E3])))
            global_counts.update(metrics)
            by_comp[item["composition"]].update(metrics)
            by_texture[texture].update(metrics)
            report["frames"]+=end//HOP
            report["tracks"]+=1
            print("S841_REAL_E2E3_TRACK",meta["name"],
                  "E2",len(starts[E2]),"E3",len(starts[E3]),
                  "near256",within_hop,"near80ms",within_80ms,flush=True)
    if report["tracks"]!=80 or report["frames"]!=306536:
        raise AssertionError("not 80 signed PCM256 tracks")
    report["complete"]=True
    report["counts"]=dict(global_counts)
    report["by_composition"]={k:dict(by_comp[k]) for k in COMP}
    report["by_texture"]={k:dict(v) for k,v in by_texture.items()}
    out.mkdir(parents=True)
    (out/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("S841_E2E3_REAL_AUDIT_TOTAL",json.dumps({
        "tracks":report["tracks"],"frames":report["frames"],
        "counts":report["counts"],"compositions":report["by_composition"]},
        sort_keys=True),flush=True)
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    audit(args.root,args.out)
