"""S800 reproduce frozen S795 FIT80 OOF, exporting full held-out evidence.

NO CAL32 evaluation/selection and NO model/decoder modifications.
Rerunning S795 FIT train is unavoidable: its original archive lacked
per-fragment predictions and saved trained expert banks.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import tempfile
import zipfile
from collections import Counter
import joblib
import numpy as np
import torch
from causal_note.strict_live import SR
from causal_note.s708_cardinality_selection import GuidedPolicy
from causal_note.s793_zero_added_latency import SchedulingContract
from causal_note.s795_activek_experts import HEADS
from scripts.s613_prepare_strict_live import download,digest
from scripts.s699_player_held_full_train_K import expanded_train_specs
from scripts.s704_joint_metrics import SCORES,metric,ratios
from scripts.s705_train_temporal_loops import setup
from scripts.s706_train_sharp_temporal_loops import FIT_COMPS,CAL_COMPS,load_model
from scripts.s795_train_activek_midi256 import (
    COUNT_KEYS,SEED,SCHEMA as S795_SCHEMA,feature_record,
    extract_fit_training,train_bank,predictions
)
from scripts.s800_oof_trace import SCHEMA,snapshot,verify_snapshot

def save(path,data):
    path.write_text(json.dumps(data,sort_keys=True,indent=2)+"\n",
                    encoding="utf-8")

def run(args):
    if args.out.exists():
        raise FileExistsError("S800 immutable root must not preexist")
    args.out.mkdir(parents=True)
    snapshot_dir=args.out/"fit_oof"
    snapshot_dir.mkdir()
    s708=json.loads((args.s708/"report.json").read_text(encoding="utf-8"))
    s795=json.loads(args.s795_report.read_text(encoding="utf-8"))
    sha=digest((args.s708/"s708_selected.pt").read_bytes())
    if s708["schema"]!="S708_K_GUIDED_S706_CAL_LOOPS" or sha!=s708["checkpoint_sha256"]:
        raise AssertionError("S708 signed hash/ABI mismatch")
    if s795["schema"]!=S795_SCHEMA or s795["fit_done"]!=80:
        raise AssertionError("S795 FIT80 original source not finished")
    if s795["signed_source_sha256"]!=sha:
        raise AssertionError("S795/S708 source checkpoint differs")
    choice=s708["frozen_candidate"]["choice"]
    if choice["release"]!=dict(gate=.8,persistence=1,grace=24):
        raise AssertionError("S707 frozen offset contract changed")
    SchedulingContract().validate()
    torch.set_num_threads(2)
    net=load_model(args.s708/"s708_selected.pt")[0]
    policy=GuidedPolicy(**choice["policy"])
    expected={k:int(s795["FIT"]["source_counts"][k]) for k in COUNT_KEYS}
    manifest=dict(schema=SCHEMA,complete=False,promoted=False,
        official_reference="S708",signed_source_sha256=sha,
        S795_archived_fit_counts=expected,
        decision_clock_samples=1024,metric_clock_samples=256,
        FIT_only=True,fit_done=0,folds_done=[],
        fit_files=[],head_order=list(HEADS),
        CAL_opened=False,HELD_opened=False,TEST_opened=False,
        cal_used_for_training=False,cal_used_for_selection=False,
        source_midi_unchanged=True,production_LIVE_modified=False,
        physical_mic_latency_measured=False,
        selected_from_CAL=False,retrained_S795_FIT_experts=True,
        recomputation_reason="S795 archive lacks OOF probabilities/model bank")
    def checkpoint():save(args.out/"report.json",manifest)
    checkpoint()
    with tempfile.TemporaryDirectory() as td:
        root=Path(td);(root/"data").mkdir()
        protocol,heads,selection,offset=setup(root,args.offset_checkpoint)
        for key,name in (("GuitarSet_audio","GuitarSet_audio.zip"),
                         ("GuitarSet_annotations","GuitarSet_annotations.zip")):
            val=protocol["guitar_archives"][key]
            download(val["url"],root/"data"/name,val["size"],val["sha256"])
        specs=[s for s in protocol["sources"]
               if s["corpus"]=="GuitarSet" and s["split"]=="train"]
        with zipfile.ZipFile(root/"data/GuitarSet_audio.zip") as audio, \
             zipfile.ZipFile(root/"data/GuitarSet_annotations.zip") as annot:
            expanded=expanded_train_specs(specs,audio,annot)
            fit=[s for s in expanded if s["composition"] in FIT_COMPS]
            cal=[s for s in expanded if s["composition"] in CAL_COMPS]
            if len(fit)!=80 or len(cal)!=32 or set(FIT_COMPS)&set(CAL_COMPS):
                raise AssertionError("FIT80/CAL32 frozen compositions changed")
            rows=[]
            for i,spec in enumerate(fit):
                row=feature_record(spec,audio,root,heads,selection,
                                   offset,net,policy,choice)
                xs,ys,counts=extract_fit_training(row,SEED+i*71)
                row["X"]=row["X"].astype(np.float16)
                path=root/f"fit_{i:03d}.joblib"
                joblib.dump(row,path,compress=0)
                rows.append(dict(name=row["name"],composition=row["composition"],
                    cache=path,train_x=xs,train_y=ys,counts=counts,index=i))
                manifest["fit_done"]=i+1
                checkpoint()
                print("S800_FIT_EXTRACT",i+1,"/80",row["name"],flush=True)
            folds=sorted({r["composition"] for r in rows})
            if len(folds)!=5 or len({r["name"] for r in rows})!=80:
                raise AssertionError("S800 invalid composition/record cardinality")
            if Counter(r["composition"] for r in rows)!={f:16 for f in folds}:
                raise AssertionError("S800 unexpected tracks per held-out composition")
            all_counts={k:0 for k in COUNT_KEYS}
            for fold in folds:
                models,training=train_bank(rows,omit=fold)
                for name in HEADS:
                    if training[name].get("trained"):
                        if fold in training[name].get("compositions",[]):
                            raise AssertionError("S800 OOF training leakage")
                        if len(training[name].get("compositions",[]))!=4:
                            raise AssertionError("S800 OOF trained on unexpected splits")
                for meta in (r for r in rows if r["composition"]==fold):
                    record=joblib.load(meta["cache"])
                    P=predictions(models,record["X"])
                    trace=snapshot(meta["index"],record,P,training,fold,snapshot_dir)
                    verify_snapshot(trace,snapshot_dir)
                    manifest["fit_files"].append(trace)
                    for key in COUNT_KEYS:
                        all_counts[key]+=trace["source_counts"][key]
                    checkpoint()
                    print("S800_OOF_SAVED",meta["index"],fold,
                          "frames",trace["fragment_frames"],flush=True)
                manifest["folds_done"].append(fold)
                checkpoint()
                print("S800_OOF_FOLD_COMPLETE",fold,flush=True)
    if any(all_counts[k]!=expected[k] for k in COUNT_KEYS):
        differences={k:dict(expected=expected[k],actual=all_counts[k])
                     for k in COUNT_KEYS if all_counts[k]!=expected[k]}
        manifest["fit_source_parity_failures"]=differences
        checkpoint()
        raise AssertionError("S800 original S795 FIT count PARITY FAILED: "+str(differences))
    scores=ratios(all_counts)
    archived=s795["FIT"]["baseline"]
    for metric_name in SCORES:
        if abs(scores[metric_name]-archived[metric_name])>1e-12:
            raise AssertionError("S800 original S795 FIT metric PARITY FAILED: "+metric_name)
    manifest["FIT_source_counts"]=all_counts
    manifest["FIT_source_scores"]={k:scores[k] for k in SCORES}
    manifest["FIT_source_parity_S795"]=True
    manifest["fit_files_count"]=len(manifest["fit_files"])
    manifest["complete"]=len(manifest["fit_files"])==80 and len(manifest["folds_done"])==5
    checkpoint()
    if not manifest["complete"]:
        raise AssertionError("S800 incomplete FIT OOF evidence")
    print("S800_OOF_FIT_REPRODUCED",json.dumps(dict(
        count=80,source_counts=all_counts,scores=manifest["FIT_source_scores"],
        CAL_opened=False,promoted=False)),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--s708",type=Path,required=True)
    p.add_argument("--s795-report",type=Path,required=True)
    p.add_argument("--offset-checkpoint",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    run(p.parse_args())
