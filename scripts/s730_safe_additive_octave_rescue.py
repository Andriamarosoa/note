"""S730: CAL-only additive K-gated recovery without touching original S708 notes.

Unlike head replacement S721 and 320D full decoder S725/S727, the S730
rescue keeps every original S708 note-on/off identical. Candidate S727
emitted events are inserted *only* for independent pitches when neural K0..6
of the specialist predicts more simultaneous notes than the current emitted
source state. No future PCM, true labels, or backdating are used to admit
a new note. Six metrics and paired correct-offset guards still required.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import zipfile
import torch
import numpy as np
from threadpoolctl import threadpool_limits

from causal_note.strict_live import SR
from causal_note.s725_source_memory import augmented_track
from causal_note.s705_temporal_heads import predict_track
from causal_note.s708_cardinality_selection import GuidedPolicy
from causal_note.s730_additive_k_guard import additive_rescue
from scripts.s613_prepare_strict_live import download,labels,digest
from scripts.s613_train_strict_live import real_notes
from scripts.s699_player_held_full_train_K import expanded_train_specs
from scripts.s706_train_sharp_temporal_loops import record,load_model
from scripts.s705_train_temporal_loops import setup
from scripts.s727_train_harmonic_negative_loops import score_events
from scripts.s704_joint_metrics import SCORES,metric,aggregate,target_rank

MODELS=("harmonic_1p0_epoch1","harmonic_1p8_epoch1",
        "harmonic_1p8_epoch2","harmonic_2p6_epoch1","harmonic_2p6_epoch2")
CONFIDENCES=(.35,.50,.65)
REF="S708_frozen"


def options(models):
    choices={}
    for name in models:
        for confidence in CONFIDENCES:
            choices[f"{name}_Kguard{str(confidence).replace('.','p')}"]=(name,confidence)
    return choices


def guard(baseline,candidate,loss):
    return (all(candidate[m]>=baseline[m]-1e-12 for m in SCORES)
           and candidate["tp"]>=baseline["tp"] and
           loss["lost_long"]==0 and loss["lost_correct_offsets"]==0)


def score_phase(specs,archive,root,heads,selector,offset,old_model,models,
                policy,release,choices,phase):
    outputs={REF:[]}
    paired={}
    for name in choices:
        outputs[name]=[]
        paired[name]=dict(lost_correct_offsets=0,lost_long=0,added=0,rejected=0)
    tracks=[]
    for idx,spec in enumerate(specs):
        item=record(spec,archive,heads,selector,offset)
        native=item["x"];item["native_features"]=native[:,:,:126].copy()
        p0,k0=predict_track(old_model,native)
        baseline=score_events(item,p0,k0,policy,offset,release)
        raw=augmented_track(native)
        predictions={}
        for name in set(modelname for modelname,conf in choices.values()):
            p,k=predict_track(models[name],raw)
            events=score_events(item,p,k,policy,offset,release)
            predictions[name]=(events,k)
        emitted={REF:baseline}
        stats={}
        for label,(name,confidence) in choices.items():
            extra,k=predictions[name]
            merged,state=additive_rescue(baseline,extra,
                              item["decision_ends"],k,confidence=confidence)
            emitted[label]=merged;stats[label]=state
        # No true label enters K merge or event construction.
        truth=real_notes(labels(spec,root)["notes"],item["samples"]/SR)
        rows={name:metric(events,truth,item["ends"])
              for name,events in emitted.items()}
        base=rows[REF]
        old_off=set(base["correct_offset_truth_ids"])
        old_long=set(base["long_onset_truth_ids"])
        outputs[REF].append(base)
        report=dict(name=spec["name"],sha256=item["sha256"],
            base={m:base[m] for m in SCORES},trials={})
        for name in choices:
            metric_row=rows[name];loss=paired[name]
            outputs[name].append(metric_row)
            loss["lost_correct_offsets"]+=len(old_off-set(metric_row["correct_offset_truth_ids"]))
            loss["lost_long"]+=len(old_long-set(metric_row["long_onset_truth_ids"]))
            loss["added"]+=stats[name]["added"]
            loss["rejected"]+=stats[name]["rejected"]
            report["trials"][name]=dict(scores={m:metric_row[m] for m in SCORES},
                                         **stats[name])
        tracks.append(report)
        print("S730 REAL EMITTED",phase,idx+1,len(specs),spec["name"],flush=True)
    return ({k:aggregate(rows) for k,rows in outputs.items()},
            paired,tracks)


def run(args):
    if args.out.exists():raise FileExistsError("S730 refuses overwrite")
    args.out.mkdir(parents=True)
    original=json.loads((args.s708_dir/"report.json").read_text())
    specialist=json.loads((args.s727_dir/"report.json").read_text())
    sha=digest((args.s708_dir/"s708_selected.pt").read_bytes())
    if sha!=original["checkpoint_sha256"] or sha!=specialist["s708_sha256"]:
        raise AssertionError("S730 requires original signed S708 and real S727 weights")
    policy=GuidedPolicy(**original["frozen_candidate"]["choice"]["policy"])
    release=original["frozen_candidate"]["choice"]["release"]
    assert release==dict(gate=.8,persistence=1,grace=24)
    models={}
    for name in MODELS:
        spec=specialist["train"][name]
        path=args.s727_dir/spec["path"]
        if digest(path.read_bytes())!=spec["sha256"]:
            raise AssertionError("S730 experimental checkpoint SHA mismatch")
        models[name]=load_model(path)[0]
    old_model=load_model(args.s708_dir/"s708_selected.pt")[0]
    choices=options(models)
    report=dict(schema="S730_ADDITIVE_SAFE_EVENT_RESCUE_CAUSAL_K_GUARD",
          promoted=False,LIVE_changed=False,TEST_opened=False,
          CAL_only_selection=True,HELD_DEV_historically_exposed=True,
          s708_sha256=sha,source_branch_S727=True,
          original_events_preserved_identically=True,
          shared_release=release,shared_selection=asdict(policy),
          candidate_grid=choices,
          safeguard="all six scores nonregressive; zero lost original correct offsets/long true onsets")
    def save():
        (args.out/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    save()
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp);(root/"data").mkdir()
        proto,heads,selector,offset=setup(root,args.offset_checkpoint)
        for key,file in (("GuitarSet_audio","GuitarSet_audio.zip"),
                         ("GuitarSet_annotations","GuitarSet_annotations.zip")):
            meta=proto["guitar_archives"][key]
            download(meta["url"],root/"data"/file,meta["size"],meta["sha256"])
        original_train=[s for s in proto["sources"] if s["corpus"]=="GuitarSet" and s["split"]=="train"]
        dev=[s for s in proto["sources"] if s["corpus"]=="GuitarSet" and s["split"]=="dev"]
        with zipfile.ZipFile(root/"data/GuitarSet_audio.zip") as archive, zipfile.ZipFile(root/"data/GuitarSet_annotations.zip") as ann:
            expanded=expanded_train_specs(original_train,archive,ann)
            cal=[s for s in expanded if s["composition"] in original["cal_compositions"]]
            held=[s for s in expanded if s["composition"] in original["held_compositions"]]
            assert (len(cal),len(held),len(dev))==(32,32,12)
            scores,paired,tracks=score_phase(cal,archive,root,heads,selector,offset,
                              old_model,models,policy,release,choices,"CAL")
            baseline=scores[REF]
            eligible=[REF]+[name for name in choices if guard(baseline,scores[name],paired[name])]
            chosen=max(eligible,key=lambda name:target_rank(scores[name]))
            report["CAL"]=dict(scores=scores,paired=paired,tracks=tracks,
                                eligible=eligible,selected=chosen)
            report["frozen_choice"]=dict(selected=chosen,
                     policy=choices.get(chosen),no_promote=True)
            save()
            print("S730 CAL FROZEN",json.dumps(dict(selected=chosen,eligible=eligible,
                 scores={n:{m:scores[n][m] for m in SCORES} for n in scores})),flush=True)
            selected={} if chosen==REF else {chosen:choices[chosen]}
            for phase,specs in (("HELD",held),("DEV",dev)):
                row,loss,records=score_phase(specs,archive,root,heads,selector,offset,
                                 old_model,models,policy,release,selected,phase)
                for met in SCORES:
                    if abs(row[REF][met]-original[phase]["candidate"][met])>1e-8:
                        raise AssertionError("S730 original benchmark mismatch "+phase+" "+met)
                safe=True if chosen==REF else guard(row[REF],row[chosen],loss[chosen])
                report[phase]=dict(baseline={m:row[REF][m] for m in SCORES},
                    candidate={m:row[chosen][m] for m in SCORES},
                    safe=safe,loss=loss,tracks=records)
                save()
                print("S730",phase,"GUARDED",json.dumps(dict(chosen=chosen,
                     safe=safe,baseline=report[phase]["baseline"],
                     candidate=report[phase]["candidate"])),flush=True)
    report["shadow_qualifies"]=bool(chosen!=REF and
                                  report["HELD"]["safe"] and report["DEV"]["safe"])
    report["promoted"]=False
    save()
    print("S730 FINAL",json.dumps(dict(chosen=chosen,
          shadow_qualifies=report["shadow_qualifies"],promoted=False)),flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--s708-dir",required=True,type=Path)
    ap.add_argument("--s727-dir",required=True,type=Path)
    ap.add_argument("--offset-checkpoint",required=True,type=Path)
    ap.add_argument("--out",required=True,type=Path)
    with threadpool_limits(limits=2):run(ap.parse_args())
