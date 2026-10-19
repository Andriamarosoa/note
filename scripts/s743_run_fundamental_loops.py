"""S743: genuine 12-loop GuitarSet CAL32 multi-fundamental active-K study.

No oracle, no new trained fake '80%'. Six current/past acoustic heads,
32 fragment-local possible subsets, three label-free dynamic mask policies
x two K-tail x two Sustain gates = 12 independent complete MIDI decoders.
All six official metrics and previously correct S707 offsets/long onsets.
S708 and LIVE remain immutable; no HELD/DEV tuning, TEST unopened.
"""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import asdict
from itertools import product
import json
from pathlib import Path
import tempfile
import zipfile
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from causal_note.strict_live import SR
from causal_note.s731_auditory_scene import scene_scores
from causal_note.s707_causal_release import prepare_hazards,render_release
from causal_note.s708_cardinality_selection import GuidedPolicy
from causal_note.s743_fundamental_router import (
    MultiHeadBirthPolicy,candidate_heads,decode_s743)
from scripts.s613_prepare_strict_live import download,labels,digest
from scripts.s613_train_strict_live import real_notes
from scripts.s699_player_held_full_train_K import expanded_train_specs
from scripts.s704_joint_metrics import SCORES,metric,aggregate
from scripts.s705_train_temporal_loops import setup
from scripts.s706_train_sharp_temporal_loops import CAL_COMPS,load_model
from scripts.s735_audit_32_onset_combinations import load_s734
from scripts.s741_train_exactk_fragments import one_k_track
from scripts.s742_activek_forensics import exact_active_breakdown

CHOICES={
    f"{mode}_tail{str(tail).replace('.','p')}_s{str(sus).replace('.','p')}":
    MultiHeadBirthPolicy(k_tail=tail,min_onset=.16,
        min_sustain=sus,mask_delta=.06,mask_strategy=mode,
        min_rise=.02,max_births=1)
    for mode,tail,sus in product(("adaptive","physical","conservative"),
                                (.30,.55),(.30,.55))
}
assert len(CHOICES)==12

def save(path,row):
    path.write_text(json.dumps(row,indent=2,sort_keys=True)+"\n")


def replay(args):
    if args.out.exists():
        raise FileExistsError("S743 report is immutable")
    args.out.mkdir(parents=True)
    old=json.loads((args.s708_dir/"report.json").read_text())
    sha=digest((args.s708_dir/"s708_selected.pt").read_bytes())
    if sha!=old["checkpoint_sha256"]:
        raise AssertionError("S743 S708 signed source mismatch")
    h5report=json.loads(args.s734_report.read_text())
    if h5report["base_sha256"]!=sha or not h5report["complete"]:
        raise AssertionError("S743 S734 source mismatch")
    torch.set_num_threads(2)
    network=load_model(args.s708_dir/"s708_selected.pt")[0]
    h5=load_s734(args.s734_checkpoint,sha)
    policy=GuidedPolicy(**old["frozen_candidate"]["choice"]["policy"])
    release=old["frozen_candidate"]["choice"]["release"]
    if release!=dict(gate=.8,persistence=1,grace=24):
        raise AssertionError("Must preserve original S707 release")
    report=dict(schema="S743_REAL_MULTI_FUNDAMENTAL_12_LOOPS_CAL32",
        source_sha256=sha,completed=False,promoted=False,
        LIVE_changed=False,TEST_opened=False,selected="S708_frozen",
        loop_count=len(CHOICES),split="CAL32",
        baseline="S708_frozen",loop_params={k:asdict(v) for k,v in CHOICES.items()},
        metrics={},candidate_rows={})
    save(args.out/"report.json",report)
    keys=("S708_frozen",*CHOICES)
    measured={name:[] for name in keys}
    lost={name:Counter() for name in CHOICES}
    diag={name:Counter() for name in CHOICES}
    masks={name:Counter() for name in CHOICES}
    byk={name:Counter() for name in keys}
    with tempfile.TemporaryDirectory() as work:
        root=Path(work);(root/"data").mkdir()
        proto,heads,selection,offset=setup(root,args.offset_checkpoint)
        for kind,name in (("GuitarSet_audio","GuitarSet_audio.zip"),
                          ("GuitarSet_annotations","GuitarSet_annotations.zip")):
            metadata=proto["guitar_archives"][kind]
            download(metadata["url"],root/"data"/name,
                     metadata["size"],metadata["sha256"])
        train=[s for s in proto["sources"] if
               s["corpus"]=="GuitarSet" and s["split"]=="train"]
        with zipfile.ZipFile(root/"data/GuitarSet_audio.zip") as audio, \
             zipfile.ZipFile(root/"data/GuitarSet_annotations.zip") as annotations:
            expanded=expanded_train_specs(train,audio,annotations)
            cal=[s for s in expanded if s["composition"] in CAL_COMPS]
            if len(cal)!=32:
                raise AssertionError("Only known composition-disjoint CAL32")
            for i,spec in enumerate(cal):
                tr=one_k_track(spec,audio,root,heads,selection,offset,
                               network,policy,release,h5)
                item=tr["item"]
                scene=scene_scores(item["x"],tr["post"],tr["k"],
                                   tr["baseline"],item["decision_ends"])
                alternative=candidate_heads(tr["post"],item["native_post"],
                          tr["context"],item["x"],scene)
                predictions={"S708_frozen":tr["baseline"]}
                local={}
                for key,parameters in CHOICES.items():
                    ons,extra=decode_s743(
                        item["decision_ends"],tr["post"],tr["k"],
                        alternative,tr["context"],policy,parameters,item["samples"])
                    events,skipped=render_release(
                        prepare_hazards(item,ons,offset),
                        item["ends"],item["samples"],**release)
                    predictions[key]=events
                    local[key]=extra
                    diag[key].update(dict(
                        added_births=extra["added_births"],
                        blocked_K=extra["blocked_K"],
                        blocked_acoustic=extra["blocked_acoustic"],
                        skipped_polyphony=int(skipped)))
                    masks[key].update(extra["masks"])
                # Ground truth is opened ONLY after every source MIDI
                # NoteOn/Off event and K mask decision is already frozen.
                truth=real_notes(labels(spec,root)["notes"],item["samples"]/SR)
                result={name:metric(events,truth,item["ends"])
                        for name,events in predictions.items()}
                baseline=result["S708_frozen"]
                for key in keys:
                    measured[key].append(result[key])
                    breakdown=exact_active_breakdown(predictions[key],
                                                    truth,item["ends"])
                    for k,row in breakdown["by_true_k"].items():
                        for stat in ("frames","correct","under","over"):
                            byk[key][f"K{k}_{stat}"]+=row[stat]
                for name in CHOICES:
                    row=result[name]
                    lost[name]["correct_offsets"]+=len(
                        set(baseline["correct_offset_truth_ids"])-
                        set(row["correct_offset_truth_ids"]))
                    lost[name]["long_onsets"]+=len(
                        set(baseline["long_onset_truth_ids"])-
                        set(row["long_onset_truth_ids"]))
                print("S743 REAL TRACK",i+1,32,spec["name"],
                      "baseline_activeK",round(baseline["active_exact_k"],5),
                      "best_trial_activeK",round(max(
                          result[name]["active_exact_k"]
                          for name in CHOICES),5),flush=True)
                report["tracks_done"]=i+1
                save(args.out/"report.json",report)
    all_scores={name:aggregate(rows) for name,rows in measured.items()}
    official=old["frozen_candidate"]["CAL"]
    for m in SCORES:
        if abs(all_scores["S708_frozen"][m]-official[m])>1e-8:
            raise AssertionError("S743 changed signed S708 CAL "+m)
    passing=[]
    for name in CHOICES:
        row=all_scores[name]
        baseline=all_scores["S708_frozen"]
        safe=bool(all(row[k]>=baseline[k]-1e-12 for k in SCORES) and
                  lost[name]["correct_offsets"]==0 and
                  lost[name]["long_onsets"]==0 and
                  row["tp"]>=baseline["tp"])
        entry=dict(scores={m:float(row[m]) for m in SCORES},
                   delta={m:float(row[m]-baseline[m]) for m in SCORES},
                   acoustic_births=diag[name]["added_births"],
                   blocked_K=diag[name]["blocked_K"],
                   blocked_acoustic=diag[name]["blocked_acoustic"],
                   lost_correct_offsets=lost[name]["correct_offsets"],
                   lost_long_onsets=lost[name]["long_onsets"],
                   mask_hist=dict(masks[name]),
                   safe=safe,all_six_above_80=bool(safe and
                                     all(row[k]>.80 for k in SCORES)))
        entry["by_true_k"]={
            str(k):{stat:byk[name][f"K{k}_{stat}"]
                    for stat in ("frames","correct","under","over")}
            for k in range(7)}
        report["candidate_rows"][name]=entry
        if safe and any(entry["delta"][k]>1e-12 for k in SCORES):
            passing.append((name,entry))
        print("S743 LOOP",json.dumps(dict(
            loop=name,acoustic_births=entry["acoustic_births"],
            active_exact_k=entry["scores"]["active_exact_k"],
            onset=entry["scores"]["onset_f1"],
            offset=entry["scores"]["offset_accuracy"],
            full_note=entry["scores"]["full_note_f1"],
            safe=entry["safe"],six_gt80=entry["all_six_above_80"])),flush=True)
    report["reference_scores"]={
        m:float(all_scores["S708_frozen"][m]) for m in SCORES}
    report["completed"]=True
    report["CAL_candidate"]=max(passing,key=lambda r:(
        r[1]["all_six_above_80"],
        min(r[1]["delta"].values()),
        r[1]["delta"]["active_exact_k"]))[0] if passing else None
    report["no_accepted_candidate"]=not bool(passing)
    report["promoted"]=False
    report["selected"]="S708_frozen"
    save(args.out/"report.json",report)
    print("S743_FINAL",json.dumps({
        "loops":len(CHOICES),"selected":"S708_frozen",
        "CAL_candidate":report["CAL_candidate"],
        "best_active_K":max(x["scores"]["active_exact_k"]
                             for x in report["candidate_rows"].values()),
        "six_gt80":any(x["all_six_above_80"]
                       for x in report["candidate_rows"].values()),
        "no_LIVE_change":True}),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--s708-dir",required=True,type=Path)
    p.add_argument("--s734-checkpoint",required=True,type=Path)
    p.add_argument("--s734-report",required=True,type=Path)
    p.add_argument("--offset-checkpoint",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    args=p.parse_args()
    with threadpool_limits(limits=2):
        replay(args)
