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
                gt_note_duration_verified_overlap=True,
                near_onsets_not_equivalent_to_simultaneous_sustain=True,
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
            intervals={E2:[],E3:[]}
            gt_has_note_durations=all("duration" in n for n in notes)
            if not gt_has_note_durations:
                raise ValueError("signed GT lacks note duration; cannot prove overlap")
            for n in notes:
                sec,pitch,duration=float(n["time"]),int(n["pitch"]),float(n["duration"])
                if duration<0 or not np.isfinite(duration):
                    raise ValueError("invalid GT note length")
                if pitch in starts:
                    starts[pitch].append(sec)
                    intervals[pitch].append((sec,min(end/SR,sec+duration)))
            actual_intersections=[]
            for begin_e2,end_e2 in intervals[E2]:
                for begin_e3,end_e3 in intervals[E3]:
                    overlap_start=max(begin_e2,begin_e3)
                    overlap_stop=min(end_e2,end_e3)
                    if overlap_stop-overlap_start >= HOP/SR:
                        actual_intersections.append((overlap_start,overlap_stop))
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
                         both_onsets_within_80ms=within_80ms,
                         simultaneous_sustain_gt_intersections=len(actual_intersections),
                         simultaneous_sustain_gt_seconds=sum(stop-start for start,stop in actual_intersections))
            report["examples"].append(summary)
            metrics=dict(tracks=1,e2_onsets=len(starts[E2]),e3_onsets=len(starts[E3]),
                near_256_e2_e3=within_hop,near_80ms_e2_e3=within_80ms,
                sustained_e2_e3_gt=len(actual_intersections),
                e2_tracks=int(bool(starts[E2])),e3_tracks=int(bool(starts[E3])),
                sustained_e2_e3_tracks=int(bool(actual_intersections)))
            global_counts.update(metrics)
            by_comp[item["composition"]].update(metrics)
            by_texture[texture].update(metrics)
            report["frames"]+=end//HOP
            report["tracks"]+=1
            print("S841_REAL_E2E3_TRACK",meta["name"],
                  "E2",len(starts[E2]),"E3",len(starts[E3]),
                  "near256",within_hop,"near80ms",within_80ms,
                  "actual_sustain_overlap",len(actual_intersections),flush=True)
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
