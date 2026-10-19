"""S794 compare unchanged ORIGINAL S708 versus newer S733 convenience wrapper.

CAL32 signed baseline parity requires real S708 native decoding, not
assuming that newer wrappers have identical floating point behavior.
Both inference paths use precisely the same input and frozen source models.
Record every per-track event-hash difference and aggregate MIDI256 scores.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import zipfile
import numpy as np
import torch

from causal_note.strict_live import SR
from causal_note.s707_causal_release import prepare_hazards,render_release
from causal_note.s708_cardinality_selection import GuidedPolicy
from scripts.s613_prepare_strict_live import download,labels,digest
from scripts.s613_train_strict_live import real_notes
from scripts.s699_player_held_full_train_K import expanded_train_specs
from scripts.s704_joint_metrics import SCORES,metric,aggregate
from scripts.s705_train_temporal_loops import setup
from scripts.s706_train_sharp_temporal_loops import CAL_COMPS,load_model
from scripts.s733_train_onset_only import baseline_input
from scripts.s708_count_guided_joint_loops import (
    source as s708_source, evaluate_policy as s708_decode
)

COUNT_KEYS=("tp","fp","fn","frames","k_correct","active_frames","active_k_correct",
    "frame_pitch_tp","frame_pitch_fp","frame_pitch_fn","offset_eligible","offset_correct")

def fingerprint(events):
    z=[(e['type'],int(e['pitch']),int(e['decision_sample']),
        int(e['available_sample']),str(e['reason'])) for e in events]
    return hashlib.sha256(json.dumps(z,sort_keys=True).encode()).hexdigest()

def run(args):
    if args.out.exists():raise FileExistsError("immutable output")
    args.out.mkdir(parents=True)
    report=json.loads((args.s708/"report.json").read_text())
    signed=report["frozen_candidate"]["CAL"]
    digest_current=digest((args.s708/"s708_selected.pt").read_bytes())
    if digest_current!=report["checkpoint_sha256"]:
        raise AssertionError("signed source sha mismatch")
    choice=report["frozen_candidate"]["choice"]
    policy=GuidedPolicy(**choice["policy"])
    model_name="s708_selected.pt"
    net=load_model(args.s708/"s708_selected.pt")[0]
    torch.set_num_threads(2)
    saved=dict(schema="S794_S708_SOURCE_PATH_PARITY",complete=False,
         original_source_sha256=digest_current,
         wrapper_vs_original=[],original_signed_drift={},
         modern_signed_drift={},signed_reference=39333999180,
         promoted=False,HELD_opened=False,TEST_opened=False)

    def store(): (args.out/"report.json").write_text(json.dumps(saved,indent=2,sort_keys=True)+"\n")
    store()
    with tempfile.TemporaryDirectory() as td:
        root=Path(td);(root/"data").mkdir()
        proto,heads,selection,offset=setup(root,args.offset_checkpoint)
        for key,name in (("GuitarSet_audio","GuitarSet_audio.zip"),
                         ("GuitarSet_annotations","GuitarSet_annotations.zip")):
            meta=proto["guitar_archives"][key]
            download(meta["url"],root/"data"/name,meta["size"],meta["sha256"])
        original_specs=[p for p in proto["sources"] if p["corpus"]=="GuitarSet"
                        and p["split"]=="train"]
        with zipfile.ZipFile(root/"data/GuitarSet_audio.zip") as audio,\
             zipfile.ZipFile(root/"data/GuitarSet_annotations.zip") as ann:
            cal=[s for s in expanded_train_specs(original_specs,audio,ann)
                 if s["composition"] in CAL_COMPS]
            if len(cal)!=32:raise AssertionError("not CAL32")
            original_rows=[];wrapper_rows=[]
            for i,spec in enumerate(cal):
                # Exact archived S708 code path; source() builds all causal inputs.
                item=s708_source(spec,audio,heads,selection,offset,
                                 {model_name:net},706)
                native=s708_decode(item,model_name,policy)
                source_events,_=render_release(
                    prepare_hazards(item,native,offset),
                    item["ends"],item["samples"],**choice["release"])
                wrapper_item,wrapper_post,k,wrapper_events,*_=baseline_input(
                    spec,audio,root,heads,selection,offset,net,policy,choice["release"])
                truth=real_notes(labels(spec,root)["notes"],item["samples"]/SR)
                before=metric(source_events,truth,item["ends"])
                after=metric(wrapper_events,truth,wrapper_item["ends"])
                original_rows.append(before);wrapper_rows.append(after)
                diffs={field:[int(before[field]),int(after[field])]
                       for field in COUNT_KEYS
                       if before[field]!=after[field]}
                if fingerprint(source_events)!=fingerprint(wrapper_events):
                    saved["wrapper_vs_original"].append(dict(
                        name=spec["name"],index=i+1,
                        original_hash=fingerprint(source_events),
                        modern_hash=fingerprint(wrapper_events),count_diffs=diffs,
                        source_event_count=len(source_events),
                        wrapper_event_count=len(wrapper_events)))
                saved["done"]=i+1;store()
                print("S794_PATH_PARITY",i+1,"/32",spec["name"],
                    "differing_counts",diffs,flush=True)
    for key,rows in (("original",original_rows),("modern",wrapper_rows)):
        totals=aggregate(rows)
        saved[key+"_aggregate"]={p:totals[p] for p in COUNT_KEYS+SCORES}
        saved[key+"_signed_drift"]={
            p:dict(signed=signed[p],observed=totals[p],
                   delta=totals[p]-signed[p])
            for p in COUNT_KEYS if totals[p]!=signed[p]}
    saved["complete"]=True;store()
    print("S794_PATH_PROBE_RESULT",json.dumps(dict(
       original_signed_drift=saved["original_signed_drift"],
       modern_signed_drift=saved["modern_signed_drift"],
       distinct_recordings=len(saved["wrapper_vs_original"]),
       discrepancies=saved["wrapper_vs_original"])),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--s708",type=Path,required=True)
    p.add_argument("--offset-checkpoint",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    run(p.parse_args())
