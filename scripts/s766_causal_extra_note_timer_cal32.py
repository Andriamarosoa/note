"""S766: causal timer closure for ADDITIONAL notes, independently scored on CAL32.

This is a diagnostic study, NOT a selected or promoted version.
- Frozen signed S708/S707 never modified, even under source collisions.
- Every additional note's maximum lifetime is scheduled at its NoteOn
  (no future audio, source events or annotations required).
- A genuine causal release, if earlier, still terminates it.
- FIT80-only grouped cross-fit heads and OOF risk policy from S765.
- Four duration budgets x three predeclared onset-risk settings = 12
  complete MIDI streams. Scores are audited on already-explored CAL32.
- Results must not be marketed as unseen/independent final testing.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import tempfile
import zipfile

import joblib
import numpy as np
import torch

from causal_note.strict_live import SR
from causal_note.s708_cardinality_selection import GuidedPolicy
from causal_note.s746_instance_guard import InstanceGate, fit_event_labels, merge_events
from scripts.s613_prepare_strict_live import download, labels, digest
from scripts.s613_train_strict_live import real_notes
from scripts.s699_player_held_full_train_K import expanded_train_specs
from scripts.s704_joint_metrics import SCORES, metric, aggregate
from scripts.s705_train_temporal_loops import setup
from scripts.s706_train_sharp_temporal_loops import FIT_COMPS, CAL_COMPS, load_model
from scripts.s761_fit_dualquality_birth_routing import onset_labels
from scripts.s765_fit_independent_arbiter import (
    DIM, collect, train_one, grouped_quality, positive_probability)

RISKS=(.50,.60,.70)
CAP_BLOCKS=(4,8,16,32)
BLOCK=4*256
GRID={f"risk{int(risk*100)}_timer{cap}":(risk,cap)
      for risk in RISKS for cap in CAP_BLOCKS}
assert len(GRID)==12


def timer_limited_pairs(pairs,cap_blocks):
    """Causal note-off deadline fixed when NoteOn is emitted.

    The date of candidate future NoteOff is NOT used to accept or reject
    the NoteOn; min(original causal release, scheduled causal deadline)
    determines actual note termination once those events become available.
    """
    if cap_blocks not in CAP_BLOCKS:
        raise ValueError("S766 unregistered causal timer duration")
    budget=int(cap_blocks*BLOCK)
    limited=[]
    counts=Counter(timer_capped=0,natural_release=0)
    for on,off in pairs:
        sample=int(on["available_sample"])
        natural=int(off["available_sample"])
        if natural<=sample:
            raise AssertionError("S766 non-positive candidate lifetime")
        deadline=sample+budget
        if natural>deadline:
            finish=off.copy()
            finish.update(type="note_off",available_sample=deadline,
                          decision_sample=deadline,
                          time=deadline/SR,
                          reason="s766_causal_timer_release",
                          velocity=0)
            counts["timer_capped"]+=1
        else:
            finish=off.copy()
            counts["natural_release"]+=1
        limited.append((on,finish))
    counts["proposed"]=len(pairs)
    return limited,dict(counts)


def emit(source,pairs,features,indices,k,heads,risk,cap_blocks):
    if risk not in RISKS:
        raise ValueError("S766 unregistered risk")
    candidate,pair_stats=timer_limited_pairs(pairs,cap_blocks)
    quality=np.minimum(
        positive_probability(heads["onset"],features),
        positive_probability(heads["complete"],features))
    output,details=merge_events(
        source,candidate,quality,indices,k,
        InstanceGate(risk=risk,k_tail=0.0))
    detail=Counter(details)
    detail.update(pair_stats)
    return output,dict(detail)


def write(folder,report):
    (folder/"report.json").write_text(
        json.dumps(report,sort_keys=True,indent=2)+"\n")


def evaluate(args):
    if args.out.exists():
        raise FileExistsError("S766 immutable result folder already exists")
    args.out.mkdir(parents=True)
    signed=json.loads((args.s708_dir/"report.json").read_text())
    sha=digest((args.s708_dir/"s708_selected.pt").read_bytes())
    if sha!=signed["checkpoint_sha256"]:
        raise AssertionError("S766 S708 signed SHA mismatch")
    choice=signed["frozen_candidate"]["choice"]
    release=choice["release"]
    if release!=dict(gate=.8,persistence=1,grace=24):
        raise AssertionError("S766 changed signed S707")
    policy=GuidedPolicy(**choice["policy"])
    network=load_model(args.s708_dir/"s708_selected.pt")[0]
    torch.set_num_threads(2)
    report=dict(
        schema="S766_CAUSAL_EXTRA_NOTE_TIMERS_TWELVE_CAL32",
        checkpoint_sha256=sha,complete=False,promoted=False,
        LIVE_changed=False,HELD_opened=False,TEST_opened=False,
        original_S708_S707_unchanged=True,
        no_future_audio_at_noteon=True,
        no_truth_at_inference=True,
        timebase_samples_per_decision=BLOCK,
        duration_caps_ms={str(c):1000*c*BLOCK/SR for c in CAP_BLOCKS},
        compositions_fit=list(FIT_COMPS),
        compositions_cal=list(CAL_COMPS),
        parameter_grid={n:dict(risk=r,cap_blocks=c)
                        for n,(r,c) in GRID.items()},
        fit_recordings_done=0,cal_recordings_done=0,
        FIT={},CAL={},source_reference_parity=None,diagnostic_winner=None)
    write(args.out,report)
    fit_x,fit_on,fit_good,groups=[],[],[],[]
    source_rows=[]
    trials={name:[] for name in GRID}
    losses={name:Counter() for name in GRID}
    events_count={name:Counter() for name in GRID}
    anomaly={name:[] for name in GRID}
    census=Counter()
    with tempfile.TemporaryDirectory() as td:
        root=Path(td);(root/"data").mkdir()
        protocol,heads,selection,offset=setup(root,args.offset_checkpoint)
        for archive,name in (("GuitarSet_audio","GuitarSet_audio.zip"),
                             ("GuitarSet_annotations","GuitarSet_annotations.zip")):
            remote=protocol["guitar_archives"][archive]
            download(remote["url"],root/"data"/name,
                     remote["size"],remote["sha256"])
        tr=[s for s in protocol["sources"]
            if s["corpus"]=="GuitarSet" and s["split"]=="train"]
        with zipfile.ZipFile(root/"data/GuitarSet_audio.zip") as wav, \
             zipfile.ZipFile(root/"data/GuitarSet_annotations.zip") as ann:
            expanded=expanded_train_specs(tr,wav,ann)
            fit=[s for s in expanded if s["composition"] in FIT_COMPS]
            cal=[s for s in expanded if s["composition"] in CAL_COMPS]
            if len(fit)!=80 or len(cal)!=32 or set(FIT_COMPS)&set(CAL_COMPS):
                raise AssertionError("S766 FIT/CAL split integrity")
            for i,spec in enumerate(fit):
                item,source,k,pairs,X,indices,stats=collect(
                    spec,wav,root,heads,selection,offset,network,policy,release)
                # At training time only, after fully causal candidate emission.
                truth=real_notes(labels(spec,root)["notes"],item["samples"]/SR)
                original=metric(source,truth,item["ends"])
                onset=onset_labels(pairs,truth,original["matched_truth_ids"])
                complete=fit_event_labels(pairs,truth,original["matched_truth_ids"])
                if np.any(complete>onset):
                    raise AssertionError("S766 full labels not onset subset")
                if len(X):
                    fit_x.append(X);fit_on.append(onset);fit_good.append(complete)
                    groups.extend([spec["composition"]]*len(X))
                report["fit_recordings_done"]=i+1
                write(args.out,report)
                print("S766 FIT",i+1,80,spec["name"],"eligible",len(X),flush=True)
            X=np.concatenate(fit_x) if fit_x else np.empty((0,DIM),np.float32)
            yon=np.concatenate(fit_on) if fit_on else np.empty(0,np.int32)
            yg=np.concatenate(fit_good) if fit_good else np.empty(0,np.int32)
            fit_risk,cv=grouped_quality(X,yon,yg,groups)
            learned=dict(onset=train_one(X,yon),complete=train_one(X,yg))
            joblib.dump(dict(training="FIT80_only_group_CV",source_sha256=sha,
                             models=learned,folds=cv["folds"],risk=fit_risk),
                        args.out/"s766_fit_heads.joblib")
            report["FIT"]=dict(cv,risk_frozen=fit_risk)
            write(args.out,report)
            for i,spec in enumerate(cal):
                item,source,k,pairs,features,indices,diag=collect(
                    spec,wav,root,heads,selection,offset,network,policy,release)
                emitted={}
                info={}
                # All 12 complete MIDI event streams emitted BEFORE
                # any CAL ground truth is read.
                for name,(risk,cap) in GRID.items():
                    stream,details=emit(
                        source,pairs,features,indices,k,learned,risk,cap)
                    emitted[name]=stream
                    info[name]=details
                truth=real_notes(labels(spec,root)["notes"],item["samples"]/SR)
                original=metric(source,truth,item["ends"])
                source_rows.append(original)
                for name in GRID:
                    result=metric(emitted[name],truth,item["ends"])
                    trials[name].append(result)
                    correct_source=set(original["correct_offset_truth_ids"])
                    lost_off=correct_source-set(result["correct_offset_truth_ids"])
                    lost_matched=set(original["matched_truth_ids"])-set(
                        result["matched_truth_ids"])
                    recovered=set(result["matched_truth_ids"])-set(
                        original["matched_truth_ids"])
                    losses[name].update(dict(
                        correct_offsets=len(lost_off),
                        onset_tp=max(0,original["tp"]-result["tp"]),
                        long_onsets=len(set(original["long_onset_truth_ids"])-
                                        set(result["long_onset_truth_ids"])),
                        stolen_truth_matches=len(lost_matched),
                        recovered_truth_matches=len(recovered)))
                    events_count[name].update(info[name])
                    if lost_off:
                        anomaly[name].append(dict(
                            recording=spec["name"],
                            source_correct_offset_lost=len(lost_off),
                            original_tp=original["tp"],trial_tp=result["tp"],
                            source_matches_displaced=len(lost_matched),
                            recovered_matches=len(recovered),
                            preemptions=info[name]["preempted"],
                            accepted=info[name]["accepted"]))
                census.update(diag)
                report["cal_recordings_done"]=i+1
                write(args.out,report)
                print("S766 CAL",i+1,32,spec["name"],
                      "candidates",len(pairs),flush=True)
    baseline=aggregate(source_rows)
    signed_scores=signed["frozen_candidate"]["CAL"]
    delta={key:float(baseline[key]-signed_scores[key]) for key in SCORES}
    parity=all(abs(v)<1e-8 for v in delta.values())
    report.update(source_reference_parity=dict(valid=parity,delta=delta),
                  source={m:float(baseline[m]) for m in SCORES},
                  source_counts={m:int(baseline[m]) for m in (
                      "tp","fp","fn","offset_correct","offset_eligible",
                      "frames","active_frames")},
                  CAL_proposals=dict(census))
    for name in GRID:
        result=aggregate(trials[name])
        movement={k:float(result[k]-baseline[k]) for k in SCORES}
        miss=dict(losses[name])
        safe=(parity and result["tp"]>=baseline["tp"]
              and all(v>=-1e-12 for v in movement.values())
              and not any(miss[k] for k in (
                  "correct_offsets","onset_tp","long_onsets")))
        report["CAL"][name]=dict(
            metrics={k:float(result[k]) for k in SCORES},
            delta=movement,losses=miss,
            events=dict(events_count[name]),
            failures=anomaly[name],
            safe=safe,
            six_gt80=bool(safe and all(result[k]>.8 for k in SCORES)))
    best=max(GRID,key=lambda x:(
        report["CAL"][x]["safe"],
        min(report["CAL"][x]["delta"].values()),
        report["CAL"][x]["metrics"]["full_note_f1"]))
    report["diagnostic_winner"]=best
    report["complete"]=parity
    write(args.out,report)
    if not parity:
        raise AssertionError("S766 signed S708 parity drift")
    print("S766_CAUSAL_TIMER_RESULTS",json.dumps({
        "source":report["source"],"FIT":report["FIT"],
        "candidate_counts":report["CAL_proposals"],
        "diagnostic_winner":best,"trials":report["CAL"],
        "promoted":False}),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s708-dir",type=Path,required=True)
    parser.add_argument("--offset-checkpoint",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    evaluate(parser.parse_args())
