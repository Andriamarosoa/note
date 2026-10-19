"""S769: FIT80 grouped source-Onset false-positive classifier (shadow-only).

This is a COMPLEMENTARY precision direction to S768's missing-onset recall.
S708 signed output is never modified, no audio or label oracle at inference.
We score source NoteOn only in a SHADOW stream; this is not complete MIDI,
thus never promote solely because the shadow onset F1 rises above 80%.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import tempfile
import zipfile
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier

from causal_note.strict_live import SR
from causal_note.s708_cardinality_selection import GuidedPolicy
from scripts.s613_prepare_strict_live import download, labels, digest
from scripts.s613_train_strict_live import real_notes,event_score
from scripts.s699_player_held_full_train_K import expanded_train_specs
from scripts.s704_joint_metrics import SCORES,metric,aggregate
from scripts.s705_train_temporal_loops import setup
from scripts.s706_train_sharp_temporal_loops import FIT_COMPS,CAL_COMPS,load_model
from scripts.s733_train_onset_only import baseline_input
from scripts.s768_independent_onset_lattice_fit80 import feature_grid,combine

THRESHOLDS=(0.,.10,.20,.30,.40,.50,.60,.70,.80,.90,.95)
DIM=92
SEED=769


def source_rows(events,ends,features):
    """Source MIDI NoteOn decisions and immutable causal feature vectors.

    A decision uses the last frame ALREADY RECEIVED at that event's
    available_sample; it never peeks at the future decision frame.
    """
    stamps=np.asarray(ends,np.int64)
    feat=np.asarray(features,np.float32)
    if feat.shape!=(len(stamps),49,87):
        raise ValueError("S769 onset feature grid ABI mismatch")
    if not np.all(np.diff(stamps)>0):
        raise ValueError("non-monotonic causal clock")
    ons=[]
    rows=[]
    past_pitch={}
    for event in events:
        if event["type"]!="note_on":
            continue
        sample=int(event["available_sample"])
        pitch=int(event["pitch"])
        frame=int(np.searchsorted(stamps,sample,side="right"))-1
        if frame<0 or not 40<=pitch<=88:
            raise AssertionError("S769 source Onset not causally observable")
        reason=str(event.get("reason",""))
        delta=sample-past_pitch.get(pitch,-10**12)
        extra=np.array([
            min(max(delta/SR,0.),2.)/2.,
            float("count_supported" in reason),
            float("joint_" in reason),
            float("retrigger" in reason or "restrike" in reason),
            float("acoustic" in reason)],np.float32)
        row=np.concatenate((feat[frame,pitch-40],extra))
        if row.shape!=(DIM,) or not np.isfinite(row).all():
            raise AssertionError("S769 source feature divergence")
        rows.append(row);ons.append(event)
        past_pitch[pitch]=sample
    return ons,np.stack(rows) if rows else np.empty((0,DIM),np.float32)


def fit(X,y):
    if len(X)<10 or len(np.unique(y))!=2:
        return None
    clf=HistGradientBoostingClassifier(
        max_iter=110,max_leaf_nodes=17,max_depth=5,
        learning_rate=.07,l2_regularization=4.,
        min_samples_leaf=15,random_state=SEED,
        early_stopping=False)
    clf.fit(X,y)
    return clf


def scores(model,X):
    if model is None:
        return np.ones(len(X),np.float32)
    return model.predict_proba(np.asarray(X,np.float32))[:,1].astype(np.float32)


def kept_native(ons,prob,threshold):
    if threshold not in THRESHOLDS:
        raise ValueError("S769 unknown training-only calibration risk")
    if len(ons)!=len(prob):
        raise ValueError("S769 source event/probability length mismatch")
    return [event for event,p in zip(ons,prob) if float(p)>=threshold]


def group_oof(rows):
    compositions=sorted({r["composition"] for r in rows})
    if len(compositions)!=5:
        raise AssertionError("S769 not five independent training compositions")
    X=np.concatenate([r["X"] for r in rows])
    Y=np.concatenate([r["y"] for r in rows])
    group=np.concatenate([
        np.repeat(r["composition"],len(r["y"])) for r in rows])
    if len(X)==0 or not np.isfinite(X).all() or not set(Y)=={0,1}:
        raise AssertionError("S769 missing source True/False positive FIT examples")
    report={t:[] for t in THRESHOLDS}
    fold_info={}
    for comp in compositions:
        model=fit(X[group!=comp],Y[group!=comp])
        if model is None:
            raise AssertionError("S769 degenerate composition holdout")
        val=[r for r in rows if r["composition"]==comp]
        for track in val:
            pr=scores(model,track["X"])
            for threshold in THRESHOLDS:
                prediction=kept_native(track["events"],pr,threshold)
                score,_=event_score(prediction,track["truth"],tolerance=.080)
                report[threshold].append(dict(TP=score["TP"],FP=score["FP"],
                                              FN=score["FN"]))
        fold_info[comp]=dict(training_onsets=int(np.count_nonzero(group!=comp)),
                             held_recordings=len(val),
                             held_onsets=int(sum(len(v["X"]) for v in val)))
        print("S769 OOF",comp,"recordings",len(val),flush=True)
    summary={str(t):combine(v) for t,v in report.items()}
    choice=max(THRESHOLDS,key=lambda t:(summary[str(t)]["f1"],
                                       summary[str(t)]["recall"]))
    return fit(X,Y),choice,summary,fold_info,dict(
        total=int(len(Y)),matched=int(Y.sum()),unmatched=int((1-Y).sum()))


def write(path,data):
    path.write_text(json.dumps(data,sort_keys=True,indent=2)+"\n")


def evaluate(args):
    if args.out.exists():
        raise FileExistsError("S769 immutable audit report already exists")
    args.out.mkdir(parents=True)
    signed=json.loads((args.s708_dir/"report.json").read_text())
    sha=digest((args.s708_dir/"s708_selected.pt").read_bytes())
    if sha!=signed["checkpoint_sha256"]:
        raise AssertionError("S769 signed S708 model hash changed")
    choice=signed["frozen_candidate"]["choice"]
    if choice["release"]!=dict(gate=.8,persistence=1,grace=24):
        raise AssertionError("S769 source release mismatch")
    policy=GuidedPolicy(**choice["policy"])
    network=load_model(args.s708_dir/"s708_selected.pt")[0]
    torch.set_num_threads(2)
    report=dict(schema="S769_FIT80_SOURCE_ONSET_PRECISION_VETO",
                complete=False,selected="S708_frozen",promoted=False,
                LIVE_changed=False,HELD_opened=False,TEST_opened=False,
                source_midi_immutable=True,ONSET_only_not_MIDI=True,
                no_future_audio=True,no_truth_at_inference=True,
                fit_recordings_done=0,cal_recordings_done=0,FIT={},CAL={})
    write(args.out/"report.json",report)
    fit_rows=[]
    observed=[]
    cal_scores={t:[] for t in THRESHOLDS}
    cal_counts=Counter()
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp);(root/"data").mkdir()
        protocol,heads,selection,offset=setup(root,args.offset_checkpoint)
        for key,file in (("GuitarSet_audio","GuitarSet_audio.zip"),
                         ("GuitarSet_annotations","GuitarSet_annotations.zip")):
            remote=protocol["guitar_archives"][key]
            download(remote["url"],root/"data"/file,remote["size"],remote["sha256"])
        train=[s for s in protocol["sources"]
               if s["corpus"]=="GuitarSet" and s["split"]=="train"]
        with zipfile.ZipFile(root/"data/GuitarSet_audio.zip") as wav, \
             zipfile.ZipFile(root/"data/GuitarSet_annotations.zip") as ann:
            expanded=expanded_train_specs(train,wav,ann)
            fit_specs=[s for s in expanded if s["composition"] in FIT_COMPS]
            cal_specs=[s for s in expanded if s["composition"] in CAL_COMPS]
            if len(fit_specs)!=80 or len(cal_specs)!=32 or set(FIT_COMPS)&set(CAL_COMPS):
                raise AssertionError("S769 split composition mismatch")
            for i,spec in enumerate(fit_specs):
                item,post,k,source,_,ctx,_=baseline_input(
                    spec,wav,root,heads,selection,offset,network,
                    policy,choice["release"])
                features=feature_grid(item,post,k,ctx)
                events,X=source_rows(source,item["decision_ends"],features)
                truth=real_notes(labels(spec,root)["notes"],item["samples"]/SR)
                _,matches=event_score(events,truth,tolerance=.080)
                matched={i for i,j in matches}
                Y=np.array([int(n in matched) for n in range(len(events))],np.int8)
                fit_rows.append(dict(composition=spec["composition"],
                                     X=X,y=Y,events=events,truth=truth))
                report["fit_recordings_done"]=i+1
                write(args.out/"report.json",report)
                print("S769 FIT",i+1,80,spec["name"],"source_ons",
                      len(events),"correct",int(Y.sum()),flush=True)
            model,risk,thresholds,folds,census=group_oof(fit_rows)
            report["FIT"]=dict(folds=folds,source_onset_counts=census,
                               risk_frozen=risk,thresholds=thresholds)
            write(args.out/"report.json",report)
            for i,spec in enumerate(cal_specs):
                item,post,k,source,_,ctx,_=baseline_input(
                    spec,wav,root,heads,selection,offset,network,
                    policy,choice["release"])
                features=feature_grid(item,post,k,ctx)
                events,X=source_rows(source,item["decision_ends"],features)
                quality=scores(model,X)
                shadows={t:kept_native(events,quality,t) for t in THRESHOLDS}
                # All decisions finished BEFORE CAL annotations accessed.
                truth=real_notes(labels(spec,root)["notes"],item["samples"]/SR)
                observed.append(metric(source,truth,item["ends"]))
                for t,shadow in shadows.items():
                    score,_=event_score(shadow,truth,tolerance=.080)
                    cal_scores[t].append({k:int(score[k]) for k in ("TP","FP","FN")})
                    cal_counts[str(t)]+=len(shadow)
                report["cal_recordings_done"]=i+1
                write(args.out/"report.json",report)
                print("S769 CAL",i+1,32,spec["name"],
                      "kept_at_frozen_risk",len(shadows[risk]),flush=True)
    original=aggregate(observed)
    delta={m:float(original[m]-signed["frozen_candidate"]["CAL"][m])
           for m in SCORES}
    valid=all(abs(d)<1e-8 for d in delta.values())
    report["source_reference_parity"]=dict(valid=valid,delta=delta)
    report["source"]={m:float(original[m]) for m in SCORES}
    report["CAL"]=dict(
        native_onset=dict(TP=int(original["tp"]),FP=int(original["fp"]),
                          FN=int(original["fn"]),f1=float(original["onset_f1"])),
        precision_veto={str(t):dict(combine(v),kept_native=int(cal_counts[str(t)]))
                        for t,v in cal_scores.items()},
        frozen_risk=risk,frozen_score=combine(cal_scores[risk]),
        frozen_risk_over_80=bool(combine(cal_scores[risk])["f1"]>.80))
    report["complete"]=valid
    write(args.out/"report.json",report)
    if not valid:
        raise AssertionError("S769 S708 signed six-axis parity mismatch")
    print("S769_ONSET_SOURCE_PRECISION",json.dumps(dict(
        source=report["source"],FIT=report["FIT"],CAL=report["CAL"],
        promoted=False)),flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--s708-dir",type=Path,required=True)
    ap.add_argument("--offset-checkpoint",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    evaluate(ap.parse_args())
