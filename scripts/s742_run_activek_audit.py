"""S742 real GuitarSet S708 active Exact-K root-cause audit, NO tuning.

Classifies K-active under/over errors on actual emitted MIDI note intervals,
and compares them with frozen neural K heads and pre-event acoustic gates.
Every metric must replay exactly the signed S708 reference. Cannot silently
promote a policy, train on CAL, open HELD/DEV or modify microphone LIVE.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import tempfile
import zipfile

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from causal_note.s708_cardinality_selection import GuidedPolicy
from causal_note.strict_live import SR
from scripts.s613_prepare_strict_live import download,labels,digest
from scripts.s613_train_strict_live import real_notes
from scripts.s699_player_held_full_train_K import expanded_train_specs
from scripts.s704_joint_metrics import SCORES,metric,aggregate
from scripts.s705_train_temporal_loops import setup
from scripts.s706_train_sharp_temporal_loops import (
    FIT_COMPS,CAL_COMPS,HELD_COMPS,load_model)
from scripts.s735_audit_32_onset_combinations import load_s734
from scripts.s741_train_exactk_fragments import one_k_track
from scripts.s742_activek_forensics import full_track_forensics,aggregate_forensics

def write(path,result):
    path.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")


def run(args):
    if args.out.exists():
        raise FileExistsError("Forensic report is immutable")
    args.out.mkdir(parents=True)
    frozen=json.loads((args.s708_dir/"report.json").read_text())
    sha=digest((args.s708_dir/"s708_selected.pt").read_bytes())
    if sha!=frozen["checkpoint_sha256"]:
        raise AssertionError("Wrong original S708 checkpoint")
    h5report=json.loads(args.s734_report.read_text())
    if h5report["base_sha256"]!=sha or not h5report["complete"]:
        raise AssertionError("Original S734 H5 mismatch")
    torch.set_num_threads(2)
    model=load_model(args.s708_dir/"s708_selected.pt")[0]
    h5=load_s734(args.s734_checkpoint,sha)
    policy=GuidedPolicy(**frozen["frozen_candidate"]["choice"]["policy"])
    release=frozen["frozen_candidate"]["choice"]["release"]
    if release!=dict(gate=.8,persistence=1,grace=24):
        raise AssertionError("Unexpected S707 release policy")
    output=dict(schema="S742_S708_EMITTED_ACTIVE_K_CAUSAL_FORENSICS",
        source_sha256=sha,split="CAL32",research_only=True,
        completed=False,promoted=False,LIVE_changed=False,TEST_opened=False,
        no_oracle_inference=True,
        notes="S742 reports actual MIDI active-K K1..K6, not standalone neural K accuracy",
        records=[],summary={})
    write(args.out/"report.json",output)
    with tempfile.TemporaryDirectory() as td:
        root=Path(td);(root/"data").mkdir()
        proto,heads,selection,offset=setup(root,args.offset_checkpoint)
        for kind,name in (("GuitarSet_audio","GuitarSet_audio.zip"),
                          ("GuitarSet_annotations","GuitarSet_annotations.zip")):
            source=proto["guitar_archives"][kind]
            download(source["url"],root/"data"/name,source["size"],source["sha256"])
        old=[s for s in proto["sources"] if
             s["corpus"]=="GuitarSet" and s["split"]=="train"]
        with zipfile.ZipFile(root/"data/GuitarSet_audio.zip") as audio, \
             zipfile.ZipFile(root/"data/GuitarSet_annotations.zip") as annotations:
            expanded=expanded_train_specs(old,audio,annotations)
            cal=[s for s in expanded if s["composition"] in CAL_COMPS]
            if len(cal)!=32:
                raise AssertionError("CAL32 split is missing files")
            scored=[]
            records=[]
            for idx,spec in enumerate(cal):
                # First freeze all S708 output and five causal K proposals.
                track=one_k_track(spec,audio,root,heads,selection,offset,
                                  model,policy,release,h5)
                track["policy"]=policy
                # Only NOW open annotations for retrospective diagnosis.
                truth=real_notes(labels(spec,root)["notes"],
                                 track["item"]["samples"]/SR)
                track["truth"]=truth
                row=full_track_forensics(track)
                records.append(row)
                scored.append(metric(track["baseline"],truth,
                                     track["item"]["ends"]))
                print("S742 REAL AUDIT",idx+1,32,spec["name"],
                      "activeK",round(row["native_frames"]["active_exact_k"],5),
                      "under",row["native_frames"]["under_active"],
                      "over",row["native_frames"]["over_active"],flush=True)
                output["records"].append(row)
                write(args.out/"report.json",output)
            baseline=aggregate(scored)
            original=frozen["frozen_candidate"]["CAL"]
            for name in SCORES:
                if abs(baseline[name]-original[name])>1e-8:
                    raise AssertionError("Signed S708 CAL score replay failure "+name)
            summary=aggregate_forensics(records)
            if abs(summary["active_exact_k"]-
                   baseline["active_exact_k"])>1e-12:
                raise AssertionError("S742 active-K aggregate mismatch")
            summary["scores_replayed_exactly"]={
                k:float(baseline[k]) for k in SCORES}
            summary["source"]={
                "S708":sha,"CAL_compositions":sorted(set(
                    x["composition"] for x in records)),
                "new_neural_training":False,"LIVE_updated":False}
            by_texture={}
            for texture in ("comp","solo","unknown"):
                group=[x for x in records if x["texture"]==texture]
                if group:
                    by_texture[texture]=aggregate_forensics(group)
            output["summary"]=summary
            output["by_texture"]=by_texture
            output["completed"]=True
            write(args.out/"report.json",output)
            print("S742_FINAL",json.dumps(dict(
                active_exact_k=summary["active_exact_k"],
                under_active=summary["under_active"],
                over_active=summary["over_active"],
                by_true_k=summary["by_true_k"],
                weak_onset_gate=summary["weak_onset_gate"],
                K_heads_wrong_with_some_correct_alternative=
                    summary["K_heads_wrong_with_some_correct_alternative"],
                promoted=False)),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--s708-dir",required=True,type=Path)
    p.add_argument("--s734-checkpoint",required=True,type=Path)
    p.add_argument("--s734-report",required=True,type=Path)
    p.add_argument("--offset-checkpoint",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    arguments=p.parse_args()
    with threadpool_limits(limits=2):
        run(arguments)
