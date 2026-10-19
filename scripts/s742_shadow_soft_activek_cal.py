"""S742 shadow acoustic-confirmed soft-K MIDI rescue on CAL32.

Ablation of six PREDECLARED, label-free causal decoders; baseline S708 signed.
The score is from real NoteOn/NoteOff with S707 original release, NOT from
K0..K6 argmax. Full six-score paired no-regression gate and preserve all
previously correct Offset and long Onset IDs. No auto-promotion/LIVE switch.
CAL is for selection; a future genuinely blind split is still necessary.
"""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import zipfile

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from causal_note.strict_live import SR
from causal_note.s708_cardinality_selection import GuidedPolicy
from causal_note.s707_causal_release import prepare_hazards,render_release
from causal_note.s742_soft_activek_decoder import SoftTailPolicy,decode_soft_tail
from scripts.s613_prepare_strict_live import download,labels,digest
from scripts.s613_train_strict_live import real_notes
from scripts.s699_player_held_full_train_K import expanded_train_specs
from scripts.s704_joint_metrics import SCORES,metric,aggregate
from scripts.s705_train_temporal_loops import setup
from scripts.s706_train_sharp_temporal_loops import (
    CAL_COMPS,load_model)
from scripts.s735_audit_32_onset_combinations import load_s734
from scripts.s741_train_exactk_fragments import one_k_track
from scripts.s742_activek_forensics import exact_active_breakdown

# Research presets. Not tuned on test labels, no K hard target or oracle.
CHOICES={
    "soft055_on025_s065":SoftTailPolicy(.55,.25,.65),
    "soft065_on025_s065":SoftTailPolicy(.65,.25,.65),
    "soft055_on030_s075":SoftTailPolicy(.55,.30,.75),
    "soft065_on030_s075":SoftTailPolicy(.65,.30,.75),
    "soft070_on020_s060":SoftTailPolicy(.70,.20,.60),
    "soft075_on025_s065":SoftTailPolicy(.75,.25,.65),
}

def save(path,obj):
    path.write_text(json.dumps(obj,indent=2,sort_keys=True)+"\n")

def candidate(track,extra):
    item=track["item"]
    src,diag=decode_soft_tail(item["decision_ends"],track["post"],
                              track["k"],track["policy"],
                              item["samples"],extra)
    events,skipped=render_release(
        prepare_hazards(item,src,track["offset"]),item["ends"],
        item["samples"],**track["release"])
    return events,dict(**diag,skipped_polyphony=int(skipped))

def replay(args):
    if args.out.exists():
        raise FileExistsError("Immutable S742 shadow output")
    args.out.mkdir(parents=True)
    frozen=json.loads((args.s708_dir/"report.json").read_text())
    sha=digest((args.s708_dir/"s708_selected.pt").read_bytes())
    if sha!=frozen["checkpoint_sha256"]:
        raise AssertionError("Official S708 signature mismatch")
    h5report=json.loads(args.s734_report.read_text())
    if h5report["base_sha256"]!=sha or not h5report["complete"]:
        raise AssertionError("S734 source changed")
    torch.set_num_threads(2)
    network=load_model(args.s708_dir/"s708_selected.pt")[0]
    h5=load_s734(args.s734_checkpoint,sha)
    policy=GuidedPolicy(**frozen["frozen_candidate"]["choice"]["policy"])
    release=frozen["frozen_candidate"]["choice"]["release"]
    if release!=dict(gate=.8,persistence=1,grace=24):
        raise AssertionError("Official S707 release changed")
    candidates=list(CHOICES)
    names=["S708_frozen",*candidates]
    rows={name:[] for name in names}
    lost={name:Counter() for name in candidates}
    births={name:Counter() for name in candidates}
    k_deltas={name:Counter() for name in names}
    report=dict(schema="S742_CAUSAL_SOFT_TAIL_ACTIVE_K_SHADOW_CAL32",
        source_sha256=sha,promoted=False,LIVE_changed=False,TEST_opened=False,
        selected="S708_frozen",completed=False,
        choices={k:asdict(v) for k,v in CHOICES.items()},
        no_new_training=True,split="CAL32",baseline="S708_frozen",
        objective="80 percent on all six metrics with paired zero regression",
        report_by_choice={})
    save(args.out/"report.json",report)
    with tempfile.TemporaryDirectory() as td:
        root=Path(td);(root/"data").mkdir()
        proto,heads,selection,offset=setup(root,args.offset_checkpoint)
        for typ,name in (("GuitarSet_audio","GuitarSet_audio.zip"),
                         ("GuitarSet_annotations","GuitarSet_annotations.zip")):
            url=proto["guitar_archives"][typ]
            download(url["url"],root/"data"/name,url["size"],url["sha256"])
        training=[s for s in proto["sources"] if
                  s["corpus"]=="GuitarSet" and s["split"]=="train"]
        with zipfile.ZipFile(root/"data/GuitarSet_audio.zip") as audio, \
             zipfile.ZipFile(root/"data/GuitarSet_annotations.zip") as annotations:
            expanded=expanded_train_specs(training,audio,annotations)
            cal=[x for x in expanded if x["composition"] in CAL_COMPS]
            if len(cal)!=32:
                raise AssertionError("CAL32 input mismatch")
            for i,spec in enumerate(cal):
                tr=one_k_track(spec,audio,root,heads,selection,offset,
                               network,policy,release,h5)
                tr["offset"]=offset;tr["policy"]=policy;tr["release"]=release
                all_events={"S708_frozen":tr["baseline"]}
                diags={}
                # All decisions frozen BEFORE annotations are read.
                for name,extra in CHOICES.items():
                    result,diag=candidate(tr,extra)
                    all_events[name]=result;diags[name]=diag
                truth=real_notes(labels(spec,root)["notes"],
                                 tr["item"]["samples"]/SR)
                measured={name:metric(ev,truth,tr["item"]["ends"])
                          for name,ev in all_events.items()}
                old=measured["S708_frozen"]
                for name in names:
                    now=measured[name]
                    rows[name].append(now)
                    kstats=exact_active_breakdown(
                        all_events[name],truth,tr["item"]["ends"])
                    k_deltas[name]["under_active"]+=kstats["under_active"]
                    k_deltas[name]["over_active"]+=kstats["over_active"]
                for name in candidates:
                    now=measured[name]
                    lost[name]["correct_offsets"]+=len(
                        set(old["correct_offset_truth_ids"])-
                        set(now["correct_offset_truth_ids"]))
                    lost[name]["long_onsets"]+=len(
                        set(old["long_onset_truth_ids"])-
                        set(now["long_onset_truth_ids"]))
                    births[name].update(diags[name])
                print("S742 SOFT-K REAL MIDI",i+1,32,spec["name"],
                      {k:round(measured[k]["active_exact_k"],5)
                       for k in names},flush=True)
            scored={n:aggregate(rows[n]) for n in names}
            original=frozen["frozen_candidate"]["CAL"]
            for m in SCORES:
                if abs(original[m]-scored["S708_frozen"][m])>1e-8:
                    raise AssertionError("Original S708 CAL mismatch "+m)
            passing=[]
            for name in candidates:
                measured=scored[name]
                safe=(all(measured[m]>=scored["S708_frozen"][m]-1e-12
                          for m in SCORES) and
                      lost[name]["correct_offsets"]==0 and
                      lost[name]["long_onsets"]==0 and
                      measured["tp"]>=scored["S708_frozen"]["tp"])
                result=dict(
                    actual={m:float(measured[m]) for m in SCORES},
                    reference={m:float(scored["S708_frozen"][m]) for m in SCORES},
                    delta={m:float(measured[m]-scored["S708_frozen"][m])
                           for m in SCORES},
                    active_K_under=int(k_deltas[name]["under_active"]),
                    active_K_over=int(k_deltas[name]["over_active"]),
                    acoustic_rescue_count=int(births[name]["s742_soft_births"]),
                    suppressed_by_soft_K=int(births[name]["low_K_tail_veto"]),
                    no_acoustic_evidence=int(births[name]["no_acoustic_veto"]),
                    lost_correct_offsets=int(lost[name]["correct_offsets"]),
                    lost_long_onsets=int(lost[name]["long_onsets"]),
                    safe=bool(safe),
                    all_six_gt80=bool(safe and all(measured[m]>.80 for m in SCORES)))
                report["report_by_choice"][name]=result
                if safe and any(result["delta"][m]>1e-10 for m in SCORES):
                    passing.append((name,result))
                print("S742 SOFT K RESULT",json.dumps(dict(
                    name=name,actual=result["actual"],safe=safe,
                    acoustic_rescue_count=result["acoustic_rescue_count"],
                    all_six_gt80=result["all_six_gt80"])),flush=True)
            if passing:
                best=max(passing,key=lambda row:(
                    row[1]["all_six_gt80"],
                    min(row[1]["delta"].values()),
                    row[1]["delta"]["active_exact_k"]))
                report["CAL_candidate"]=best[0]
                report["CAL_candidate_scores"]=best[1]
                # No promotion until external independent evaluation.
            else:
                report["CAL_no_safe_improvement"]=True
            report["completed"]=True
            report["reference_scores"]={
                m:float(scored["S708_frozen"][m]) for m in SCORES}
            report["selected"]="S708_frozen"
            report["promoted"]=False
            save(args.out/"report.json",report)
            print("S742_SOFT_FINAL",json.dumps({
                "finished":True,
                "selected":"S708_frozen",
                "safe_improvement":bool(passing),
                "CAL_candidate":report.get("CAL_candidate"),
                "target_80_achieved":bool(any(
                    v["all_six_gt80"] for v in report["report_by_choice"].values())),
                "no_LIVE_update":True}),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--s708-dir",required=True,type=Path)
    p.add_argument("--s734-checkpoint",required=True,type=Path)
    p.add_argument("--s734-report",required=True,type=Path)
    p.add_argument("--offset-checkpoint",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    args=p.parse_args()
    with threadpool_limits(limits=2):
        replay(args)
