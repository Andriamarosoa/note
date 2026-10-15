"""S629: event-K informed ATTACK RESCUE of true native NOTE sustain-only pitches.

Uses S627 shared learned onset/event cardinality head frozen from TRAIN,
without retraining on development. Tries fixed policies on existing 17 DEV
compositions, never on published S614 test. S80 and S614 unaffected.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict,dataclass
import hashlib
import io
import json
from pathlib import Path
import tempfile
import time
import zipfile
import numpy as np
from scipy.io import wavfile

from causal_note.strict_live import SR,Selection,feature_track,load_heads,CausalSelection
from causal_note.s627_joint_heads import frames,SCHEMA,INPUT_DIM
from scripts.s627_train_joint_eventK_pitch import export_forward
from scripts.s623_mixed_source_event_head import pick_original_guitar_training
from scripts.s626_train_sharp_native_onset import get_source
from scripts.s616_live_dev_ablation import ROOT,MODEL
from scripts.s613_prepare_strict_live import download,labels,digest
from scripts.s613_train_strict_live import decode,event_score,offset_audit,summary

CONTROL={"GuitarSet":dict(TP=998,FP=585,FN=582),
         "URMP":dict(TP=243,FP=149,FN=256)}


@dataclass(frozen=True)
class Rescue:
    onset_prob:float=.8
    event_conf:float=.6
    native_attack_floor:float=.1
    sustain_floor:float=.5
    novelty:float=0.


def candidates():
    return [
      ("S614_original",None),
      ("event035_pitch065",Rescue(.65,.35,.1,.5,0)),
      ("event050_pitch065",Rescue(.65,.5,.1,.5,0)),
      ("event065_pitch065",Rescue(.65,.65,.1,.5,0)),
      ("event035_pitch080",Rescue(.8,.35,.1,.5,0)),
      ("event050_pitch080",Rescue(.8,.5,.1,.5,0)),
      ("event065_pitch080",Rescue(.8,.65,.1,.5,0)),
      ("event050_pitch090",Rescue(.9,.5,.1,.5,0)),
      ("event065_pitch090",Rescue(.9,.65,.1,.5,0)),
      ("event050_pitch080_sust035",Rescue(.8,.5,.1,.35,0)),
      ("event065_pitch080_sust035",Rescue(.8,.65,.1,.35,0)),
      ("event050_pitch080_novelty020",Rescue(.8,.5,.1,.5,.2)),
      ("event065_pitch080_novelty020",Rescue(.8,.65,.1,.5,.2)),
    ]


def run_gate(ends,x,p,k,q,ek,selection,total_samples,policy):
    state=CausalSelection(selection)
    events=[];rescue_count=0
    lastq=np.zeros(49,np.float32)
    for end,pnative,dist,newpitch,newK in zip(ends,p,k,q,ek):
        novelty=newpitch-lastq
        # K0 is probability of *no new onset*; do not substitute S614 K active.
        eligible=(
            (1-newK[0]>=policy.event_conf) &
            (newpitch>=policy.onset_prob) &
            (pnative[:,0]>=policy.native_attack_floor) &
            (pnative[:,1]>=policy.sustain_floor) &
            (novelty>=policy.novelty) &
            (~state.active)&state.armed)
        modified=pnative.copy()
        modified[eligible,0]=np.maximum(modified[eligible,0],selection.onset)
        modified[eligible,1]=np.maximum(modified[eligible,1],selection.active_on)
        rescue_count+=int(eligible.sum())
        events.extend(state.step(modified,dist,int(end),int(end)))
        lastq=newpitch.copy()
    events.extend(state.panic(int(total_samples)))
    return events,rescue_count


def evaluation(row,arr,cfg):
    ends,x,p,k=row["ends"],row["x"],row["p"],row["k"]
    q,ek=export_forward(arr,frames(x,p))
    baseline=decode(ends,p,k,cfg,row["n"])
    results={}
    for name,policy in candidates():
        if policy is None: events=baseline;opportunities=0
        else: events,opportunities=run_gate(ends,x,p,k,q,ek,cfg,row["n"],policy)
        score,pairs=event_score(events,row["truth"])
        offsets=offset_audit(events,row["truth"],pairs,row["n"]/SR)
        results[name]=dict(score=score,opportunities=opportunities,
            offset_conditional=offsets["conditional_offset_accuracy"])
    return results


def run(checkpoint,out):
    if out.exists():raise FileExistsError("Immutable S629 report")
    out.mkdir(parents=True)
    if not checkpoint.is_file() or not checkpoint.stat().st_size:
        raise FileNotFoundError("Frozen trained S627 checkpoint missing")
    original=json.loads((ROOT/"selection_frozen.json").read_text())
    baseline_sha=digest(MODEL.read_bytes())
    assert baseline_sha==original["model_sha256"]
    with np.load(checkpoint,allow_pickle=False) as z:
        assert str(z["schema"])==SCHEMA
        arr={k:z[k].copy() for k in z.files if k not in ("schema","metadata")}
    assert arr["mean"].shape==(INPUT_DIM,)
    protocol=json.loads((ROOT/"protocol.json").read_text())
    devU=[s for s in protocol["sources"] if s["corpus"]=="URMP" and s["split"]=="dev"]
    devG=[s for s in protocol["sources"] if s["corpus"]=="GuitarSet" and s["split"]=="dev"]
    assert len(devU)==5 and len(devG)==12
    native=load_heads(MODEL,expected_sha256=baseline_sha)
    cfg=Selection(**native.metadata["selection"])
    start=time.perf_counter()
    scores=[]
    with tempfile.TemporaryDirectory() as td:
        root=Path(td);(root/"data").mkdir()
        for spec in devU:
            data=get_source(spec,root,native)
            scores.append(dict(name=spec["name"],corpus="URMP",eval=evaluation(data,arr,cfg)))
            print("S629 SOURCE URMP",spec["name"],flush=True)
        a=protocol["guitar_archives"]["GuitarSet_audio"]
        b=protocol["guitar_archives"]["GuitarSet_annotations"]
        download(a["url"],root/"data/GuitarSet_audio.zip",a["size"],a["sha256"])
        download(b["url"],root/"data/GuitarSet_annotations.zip",b["size"],b["sha256"])
        with zipfile.ZipFile(root/"data/GuitarSet_audio.zip") as arc:
            for spec in devG:
                data=get_source(spec,root,native,arc)
                scores.append(dict(name=spec["name"],corpus="GuitarSet",eval=evaluation(data,arr,cfg)))
                print("S629 SOURCE GUITAR",spec["name"],flush=True)
    variants=[]
    for name,params in candidates():
        by={}
        for corpus,control in CONTROL.items():
            ss=[r for r in scores if r["corpus"]==corpus]
            result=summary([r["eval"][name]["score"] for r in ss])
            if name=="S614_original":
                assert all(result[k]==v for k,v in control.items()),(corpus,result)
            control_f1=2*control["TP"]/(2*control["TP"]+control["FP"]+control["FN"])
            by[corpus]=dict(TP=result["TP"],FP=result["FP"],FN=result["FN"],
                F1=result["F1"],delta_F1_points=100*(result["F1"]-control_f1),
                wins=sum(r["eval"][name]["score"]["F1"]>r["eval"]["S614_original"]["score"]["F1"] for r in ss),
                losses=sum(r["eval"][name]["score"]["F1"]<r["eval"]["S614_original"]["score"]["F1"] for r in ss),
                attempted_rescues=sum(r["eval"][name]["opportunities"] for r in ss))
        variants.append(dict(name=name,configuration=asdict(params) if params else None,by_corpus=by))
        print("S629 VARIANT",name,"G",round(by["GuitarSet"]["F1"],5),
              "U",round(by["URMP"]["F1"],5),flush=True)
    report=dict(status="S629_FIXED_EVENTK_GUIDED_ONSET_RESCUE_DEV_ONLY",
        original_S80_and_S614_unchanged=True,
        original_S614_sha256=baseline_sha,S627_joint_sha256=digest(checkpoint.read_bytes()),
        candidate_policy_count=len(variants),dev_source_count=len(scores),
        results=variants,both_corpus_gain=[v["name"] for v in variants if all(
            v["by_corpus"][c]["delta_F1_points"]>0 for c in ("GuitarSet","URMP"))],
        test_not_loaded=True,dev_historically_opened=True,official_promotion=False,
        elapsed_seconds=round(time.perf_counter()-start,2))
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("S629 SUMMARY",json.dumps({k:report[k] for k in ("S627_joint_sha256",
        "both_corpus_gain","elapsed_seconds")},indent=2),flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    run(args.checkpoint,args.out)
