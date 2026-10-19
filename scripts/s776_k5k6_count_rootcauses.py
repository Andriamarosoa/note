"""S776: explain why independent K heads fail on K5/K6 decision fragments.

FIT/CAL composition separation preserved. CAL32 diagnostic only, no tuning,
no predictions altered, no LIVE change, no promotion. Causal K heads
and signed final MIDI are compared to the same truth at 1024-sample clock.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
import tempfile
import zipfile
import numpy as np
import torch
from causal_note.strict_live import SR
from causal_note.s708_cardinality_selection import GuidedPolicy
from scripts.s613_prepare_strict_live import download,labels,digest
from scripts.s613_train_strict_live import real_notes
from scripts.s699_player_held_full_train_K import expanded_train_specs
from scripts.s705_train_temporal_loops import setup
from scripts.s706_train_sharp_temporal_loops import CAL_COMPS,load_model
from scripts.s733_train_onset_only import baseline_input,active_count_from_events

def truth_count(clock,truth):
    """Reference active note instances at a causal decision end."""
    t=np.asarray(clock,np.int64)/SR
    count=np.zeros(len(t),np.int16)
    for n in truth:
        count+=((t>=float(n["time"]))&
                (t<float(n["time"])+float(n["duration"]))).astype(np.int16)
    return np.minimum(6,count).astype(np.int8)

def confusion(truth,pred):
    t=np.asarray(truth,dtype=np.int32)
    p=np.asarray(pred,dtype=np.int32)
    if t.shape!=p.shape or np.any((t<0)|(t>6)|(p<0)|(p>6)):
        raise ValueError("S776 K0..K6 count ABI")
    return np.bincount(t*7+p,minlength=49).reshape(7,7).astype(np.int64)

def measurements(table,prob,gt):
    mat=table.tolist()
    recalls=[]
    for k in range(7):
        support=int(table[k].sum())
        recalls.append(dict(k=k,n=support,correct=int(table[k,k]),
                            accuracy=float(table[k,k]/support) if support else None,
                            under=int(table[k,:k].sum()),
                            over=int(table[k,k+1:].sum()),
                            major_under=int(table[k,:max(0,k-1)].sum())))
    return dict(frames=int(table.sum()),exact_accuracy=float(np.trace(table)/max(1,table.sum())),
                off_by_at_most_one=float(sum(table[i,j] for i in range(7)
                    for j in range(7) if abs(i-j)<=1)/max(1,table.sum())),
                per_k=recalls,confusion=mat)

def evaluate(args):
    if args.out.exists():
        raise FileExistsError("S776 immutable report")
    args.out.mkdir(parents=True)
    signed=json.loads((args.s708_dir/"report.json").read_text())
    sha=digest((args.s708_dir/"s708_selected.pt").read_bytes())
    assert sha==signed["checkpoint_sha256"]
    choice=signed["frozen_candidate"]["choice"]
    policy=GuidedPolicy(**choice["policy"])
    net=load_model(args.s708_dir/"s708_selected.pt")[0]
    torch.set_num_threads(2)
    report=dict(schema="S776_CAUSAL_EXACTK_TAIL_DIAG",complete=False,
        promoted=False,source="S708_frozen",LIVE_changed=False,
        HELD_opened=False,TEST_opened=False,no_oracle_at_inference=True,
        ONSET_OFFSET_unchanged=True,decision_clock_samples=1024,
        cal_recordings_done=0,groups={},overall={})
    def save():
        (args.out/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    save()
    rows=[]
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);(root/"data").mkdir()
        protocol,heads,selection,offset=setup(root,args.offset_checkpoint)
        for key,f in (("GuitarSet_audio","GuitarSet_audio.zip"),
                      ("GuitarSet_annotations","GuitarSet_annotations.zip")):
            remote=protocol["guitar_archives"][key]
            download(remote["url"],root/"data"/f,remote["size"],remote["sha256"])
        train=[s for s in protocol["sources"] if s["corpus"]=="GuitarSet" and s["split"]=="train"]
        with zipfile.ZipFile(root/"data/GuitarSet_audio.zip") as wav,\
             zipfile.ZipFile(root/"data/GuitarSet_annotations.zip") as ann:
            specs=[s for s in expanded_train_specs(train,wav,ann)
                   if s["composition"] in CAL_COMPS]
            assert len(specs)==32
            for i,spec in enumerate(specs):
                item,post,kprob,source,acoustic,ctx,eligible=baseline_input(
                    spec,wav,root,heads,selection,offset,net,policy,choice["release"])
                # No labels available at extraction/prediction time.
                head=np.argmax(kprob,axis=1).astype(np.int8)
                active=np.rint(active_count_from_events(source,item["decision_ends"])*6).astype(np.int8)
                truth=real_notes(labels(spec,root)["notes"],item["samples"]/SR)
                gt=truth_count(item["decision_ends"],truth)
                if not (len(gt)==len(head)==len(active)):
                    raise AssertionError("S776 causal clock mismatch")
                top2=np.argsort(kprob,axis=1)[:,-2:]
                top3=np.argsort(kprob,axis=1)[:,-3:]
                rows.append(dict(name=spec["name"],texture=spec["name"].rsplit("_",1)[-1],
                   composition=spec["composition"],gt=gt,head=head,active=active,
                   top2=int(np.count_nonzero(np.any(top2==gt[:,None],axis=1))),
                   top3=int(np.count_nonzero(np.any(top3==gt[:,None],axis=1)))))
                report["cal_recordings_done"]=i+1
                save()
                print("S776 CAL",i+1,32,spec["name"],"tail",int((gt>=5).sum()),flush=True)
    for group,names in (("all",None),("comp",["comp"]),("solo",["solo"])):
        chosen=[r for r in rows if names is None or r["texture"] in names]
        gt=np.concatenate([r["gt"] for r in chosen])
        head=np.concatenate([r["head"] for r in chosen])
        active=np.concatenate([r["active"] for r in chosen])
        hm=confusion(gt,head);am=confusion(gt,active)
        report["groups"][group]=dict(
            raw_head=measurements(hm,None,gt),source_active=measurements(am,None,gt),
            top2_head_recall=float(sum(r["top2"] for r in chosen)/len(gt)),
            top3_head_recall=float(sum(r["top3"] for r in chosen)/len(gt)),
            positive_K5_K6=int(np.count_nonzero(gt>=5)),
            exact_tail_raw=int(np.count_nonzero((gt>=5)&(head==gt))),
            exact_tail_source=int(np.count_nonzero((gt>=5)&(active==gt))))
    report["overall"]=report["groups"]["all"]
    report["complete"]=True
    save()
    print("S776_K5K6_AUDIT",json.dumps(dict(cal_recordings=32,
        raw_tail=report["overall"]["exact_tail_raw"],
        source_tail=report["overall"]["exact_tail_source"],
        tail_support=report["overall"]["positive_K5_K6"],
        top2=report["overall"]["top2_head_recall"])),flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--s708-dir",type=Path,required=True)
    parser.add_argument("--offset-checkpoint",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    evaluate(parser.parse_args())
