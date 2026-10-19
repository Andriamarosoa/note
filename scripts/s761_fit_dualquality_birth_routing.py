"""S761 — FIT80-trained dual quality heads for causal, per-fragment polyphonic births.

The original S708/S707 MIDI stream is immutable. Four pre-trained S755
Exact-K specialists generate candidate MIDI in parallel; S760's causal
one-block confirmation rejects source collisions without backdating.
Two independently FIT80-supervised binary heads learn the chance that a
candidate's attack is new/correct and the chance its whole note is correct.
FIT labels are read only AFTER source/candidate streams have been emitted.
At CAL32 inference features use received PCM, current/past heads and K only.
A candidate's future NoteOff is used strictly to close a previously accepted
note, NEVER to make its NoteOn selection decision.
"""
from __future__ import annotations

import argparse
from collections import Counter
from itertools import product
import json
from pathlib import Path
import tempfile
import zipfile

import joblib
import numpy as np
import torch
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from causal_note.strict_live import SR
from causal_note.s708_cardinality_selection import GuidedPolicy
from causal_note.s746_instance_guard import InstanceGate, fit_event_labels, merge_events
from scripts.s613_prepare_strict_live import download, labels, digest
from scripts.s613_train_strict_live import real_notes
from scripts.s699_player_held_full_train_K import expanded_train_specs
from scripts.s704_joint_metrics import SCORES, metric, aggregate
from scripts.s705_train_temporal_loops import setup
from scripts.s706_train_sharp_temporal_loops import FIT_COMPS, CAL_COMPS, load_model
from scripts.s733_train_onset_only import baseline_input
from scripts.s731_auditory_scene_calibration import emit
from scripts.s755_pitch_coherent_k_fit_cal32 import coherent_k_features, calibrated_k
from scripts.s760_causal_confirmation_delay import FAMILIES, select_novel_births

THRESHOLDS=(.50,.70,.85)
BLEND=.55
CONFIRM_BLOCKS=1
GRID={
    f"{arch}_{weight}_quality{int(t*100)}":(i,t)
    for i,(arch,weight) in enumerate(FAMILIES) for t in THRESHOLDS
}
assert len(GRID)==12

def candidate_quality_features(item,post,native_k,adjusted_k,pairs,ranking,indices,expert):
    """Only information received at each proposed NoteOn decision.

    Never uses candidate off timestamps, durations, source truth, future
    K/posterior frames or notes that have not yet become available.
    """
    x=np.asarray(item["x"],np.float32)
    p=np.asarray(post,np.float32)
    native=np.asarray(item["native_post"],np.float32)
    k=np.asarray(native_k,np.float32)
    adjusted=np.asarray(adjusted_k,np.float32)
    ends=np.asarray(item["decision_ends"],np.int64)
    if (x.shape!=(len(ends),49,288) or p.shape!=(len(ends),49,3)
        or native.shape!=p.shape or k.shape!=(len(ends),7)
        or adjusted.shape!=k.shape or len(pairs)!=len(ranking)
        or len(pairs)!=len(indices) or expert not in range(4)):
        raise ValueError("S761 causal per-fragment feature ABI mismatch")
    rows=[]
    for (on,_),confidence,(frame,ix) in zip(pairs,ranking,indices):
        if not (0<=frame<len(ends) and 0<=ix<49
                and int(on["available_sample"])==int(ends[frame])
                and int(on["decision_sample"])==int(ends[frame])
                and int(on["pitch"])==ix+40):
            raise AssertionError("S761 proposal cannot see future or backdate")
        prev=max(0,frame-1)
        spectra=x[frame,ix,192:216]
        context=p[frame,:,1]
        features=np.concatenate((
            p[frame,ix],p[prev,ix],native[frame,ix],
            k[frame],adjusted[frame],
            np.array([spectra.mean(),spectra.max(),spectra.std(),
                      float(np.mean(spectra[:8])),confidence],np.float32),
            np.array([context.mean(),context.max(),
                      np.count_nonzero(context>.3)/49.,
                      np.count_nonzero(context>.6)/49.,
                      (ix-24.)/24.],np.float32),
            np.eye(4,dtype=np.float32)[expert]
        )).astype(np.float32)
        rows.append(features)
    arr=np.stack(rows) if rows else np.empty((0,37),np.float32)
    if arr.shape!=(len(pairs),37) or not np.isfinite(arr).all():
        raise ValueError("S761 invalid/nonfinite causal quality features")
    return arr

def onset_labels(pairs,truth,baseline_matched):
    """FIT-only target of a NEW note, with pitch and +/-80ms onset."""
    missing=set(range(len(truth)))-set(baseline_matched)
    return np.array([
        int(any(j in missing and int(note["pitch"])==int(on["pitch"])
                and abs(float(note["time"])-float(on["time"]))<=.080
                for j,note in enumerate(truth)))
        for on,_ in pairs
    ],dtype=np.int32)

def train_head(X,y):
    """One-class FIT folds fail closed rather than fabricating positives."""
    if len(y)==0:
        return None
    if len(np.unique(y))==1:
        return None  # Ambiguous FIT evidence: fail closed, do not infer a positive probability
    return make_pipeline(StandardScaler(),LogisticRegression(
        C=.25,class_weight="balanced",max_iter=500,random_state=761
    )).fit(X,y)

def positive_probability(model,X):
    if len(X)==0:
        return np.empty(0,np.float32)
    if model is None:
        return np.zeros(len(X),np.float32)
    proba=model.predict_proba(X)
    classes=model.classes_
    pos=np.flatnonzero(classes==1)
    if len(pos)==0:
        return np.zeros(len(X),np.float32)
    if proba.shape[1]!=len(classes):
        raise AssertionError("S761 classifier ABI mismatch")
    return np.clip(proba[:,int(pos[0])],0,1).astype(np.float32)

def fit_candidate_rows(spec,audio,root,heads,selection,offset,network,policy,
                       release,experts):
    item,post,k,source,_,_,_=baseline_input(
        spec,audio,root,heads,selection,offset,network,policy,release)
    features=coherent_k_features(item["x"],post,k,item["native_post"])
    # Produce all candidate MIDI streams BEFORE opening FIT supervision.
    proposals=[]
    for i,family in enumerate(FAMILIES):
        estimate=experts[family].predict_proba(features).astype(np.float32)
        fused=calibrated_k(k,estimate,BLEND)
        candidate=emit(item,post,fused,policy,offset,release)
        pairs,ranking,indices,_=select_novel_births(
            source,candidate,item["decision_ends"],post,k,fused,
            confirmation_blocks=CONFIRM_BLOCKS)
        X=candidate_quality_features(
            item,post,k,fused,pairs,ranking,indices,i)
        proposals.append((pairs,X))
    truth=real_notes(labels(spec,root)["notes"],item["samples"]/SR)
    matched=metric(source,truth,item["ends"])["matched_truth_ids"]
    X=[]; onset=[]; complete=[]
    for pairs,features in proposals:
        if not len(features):
            continue
        a=onset_labels(pairs,truth,matched)
        b=fit_event_labels(pairs,truth,matched)
        if np.any(b>a):
            raise AssertionError("S761 complete truth must imply onset truth")
        X.append(features);onset.append(a);complete.append(b)
    if not X:
        return np.empty((0,37),np.float32),np.empty(0,np.int32),np.empty(0,np.int32)
    return np.concatenate(X),np.concatenate(onset),np.concatenate(complete)

def score_variants(source,post,k,item,experts,heads,policy,offset,release):
    feats=coherent_k_features(item["x"],post,k,item["native_post"])
    streams={"S708_frozen":source};details={}
    for i,family in enumerate(FAMILIES):
        specialist=experts[family].predict_proba(feats).astype(np.float32)
        fused=calibrated_k(k,specialist,BLEND)
        proposed=emit(item,post,fused,policy,offset,release)
        pairs,ranking,indices,census=select_novel_births(
            source,proposed,item["decision_ends"],post,k,fused,
            confirmation_blocks=CONFIRM_BLOCKS)
        X=candidate_quality_features(item,post,k,fused,pairs,ranking,indices,i)
        p_on=positive_probability(heads["onset"],X)
        p_full=positive_probability(heads["complete"],X)
        quality=np.minimum(p_on,p_full)
        for threshold in THRESHOLDS:
            key=f"{family[0]}_{family[1]}_quality{int(threshold*100)}"
            events,diag=merge_events(
                source,pairs,quality,indices,fused,
                InstanceGate(risk=threshold,k_tail=.10))
            streams[key]=events
            details[key]=dict(candidate=census,gate=diag,
                              quality_max=float(quality.max()) if len(quality) else 0.,
                              quality_mean=float(quality.mean()) if len(quality) else 0.)
    if len(streams)!=13:
        raise AssertionError("S761 must evaluate 12 full MIDI streams")
    return streams,details

def save(folder,report):
    (folder/"report.json").write_text(
        json.dumps(report,sort_keys=True,indent=2)+"\n")

def evaluate(args):
    if args.out.exists():
        raise FileExistsError("S761 requires a fresh immutable output folder")
    args.out.mkdir(parents=True)
    signed=json.loads((args.s708_dir/"report.json").read_text())
    sha=digest((args.s708_dir/"s708_selected.pt").read_bytes())
    if sha!=signed["checkpoint_sha256"]:
        raise AssertionError("S761 signed S708 hash mismatch")
    release=signed["frozen_candidate"]["choice"]["release"]
    if release!=dict(gate=.8,persistence=1,grace=24):
        raise AssertionError("S761 signed release drift")
    policy=GuidedPolicy(**signed["frozen_candidate"]["choice"]["policy"])
    experts={}
    for family in FAMILIES:
        name=f"s755_{family[0]}_{family[1]}.joblib"
        bundle=joblib.load(args.specialist_dir/name)
        if (bundle["source_sha256"]!=sha
            or bundle["training"]!="FIT80_only"
            or bundle["features"]!=240
            or bundle["classes"]!=list(range(7))):
            raise AssertionError("S761 S755 specialist provenance violation")
        experts[family]=bundle["model"]
    torch.set_num_threads(2)
    network=load_model(args.s708_dir/"s708_selected.pt")[0]
    report=dict(
        schema="S761_FIT80_DUAL_ONSET_COMPLETE_QUALITY_CAL32",
        source_sha256=sha,source="S708_frozen",heads="FIT80_onset_and_complete",
        complete=False,promoted=False,selected="S708_frozen",
        original_release_unchanged=True,no_truth_at_inference=True,
        LIVE_changed=False,HELD_opened=False,TEST_opened=False,
        loops=12,FIT_recordings_done=0,CAL_recordings_done=0,
        parameter_grid={name:{"expert":FAMILIES[i],
                               "threshold":threshold,
                               "confirmation_blocks":CONFIRM_BLOCKS}
                        for name,(i,threshold) in GRID.items()},
        FIT={},CAL={},CAL_candidate=None,reference_parity=None)
    save(args.out,report)
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);(root/"data").mkdir()
        protocol,heads,selection,offset=setup(root,args.offset_checkpoint)
        for source,path in (("GuitarSet_audio","GuitarSet_audio.zip"),
                            ("GuitarSet_annotations","GuitarSet_annotations.zip")):
            info=protocol["guitar_archives"][source]
            download(info["url"],root/"data"/path,info["size"],info["sha256"])
        originals=[s for s in protocol["sources"]
                   if s["corpus"]=="GuitarSet" and s["split"]=="train"]
        with zipfile.ZipFile(root/"data/GuitarSet_audio.zip") as audio, \
             zipfile.ZipFile(root/"data/GuitarSet_annotations.zip") as ann:
            expanded=expanded_train_specs(originals,audio,ann)
            fit=[s for s in expanded if s["composition"] in FIT_COMPS]
            cal=[s for s in expanded if s["composition"] in CAL_COMPS]
            if len(fit)!=80 or len(cal)!=32 or set(FIT_COMPS)&set(CAL_COMPS):
                raise AssertionError("S761 FIT/CAL composition leakage")
            fit_x=[];fit_a=[];fit_b=[]
            for index,spec in enumerate(fit):
                x,a,b=fit_candidate_rows(
                    spec,audio,root,heads,selection,offset,network,policy,
                    release,experts)
                if len(x):
                    fit_x.append(x);fit_a.append(a);fit_b.append(b)
                report["FIT_recordings_done"]=index+1
                save(args.out,report)
                print("S761 FIT",index+1,80,spec["name"],"candidates",len(a),flush=True)
            if fit_x:
                X=np.concatenate(fit_x)
                y_on=np.concatenate(fit_a)
                y_full=np.concatenate(fit_b)
            else:
                X=np.empty((0,37),np.float32)
                y_on=y_full=np.empty(0,np.int32)
            heads_fit=dict(onset=train_head(X,y_on),
                           complete=train_head(X,y_full))
            joblib.dump(dict(models=heads_fit,training="FIT80_only",
                             source_sha256=sha,features=37,
                             positives_onset=int(y_on.sum()),
                             positives_complete=int(y_full.sum())),
                        args.out/"s761_fit_dualquality.joblib")
            report["FIT"]=dict(candidate_rows=len(X),
                               onset_positives=int(y_on.sum()),
                               complete_positives=int(y_full.sum()),
                               onset_class_balance=np.bincount(y_on,minlength=2).tolist(),
                               complete_class_balance=np.bincount(y_full,minlength=2).tolist(),
                               trained_composition_count=len(FIT_COMPS))
            save(args.out,report)
            rows={name:[] for name in ("S708_frozen",*GRID)}
            losses={name:Counter() for name in GRID}
            counts={name:Counter() for name in GRID}
            for idx,spec in enumerate(cal):
                item,post,k,source,_,_,_=baseline_input(
                    spec,audio,root,heads,selection,offset,network,policy,release)
                alternatives,details=score_variants(
                    source,post,k,item,experts,heads_fit,policy,offset,release)
                # No CAL annotations are inspected until ALL 12 MIDI streams exist.
                truth=real_notes(labels(spec,root)["notes"],item["samples"]/SR)
                measured={name:metric(stream,truth,item["ends"])
                          for name,stream in alternatives.items()}
                base=measured["S708_frozen"]
                for name,entry in measured.items():
                    rows[name].append(entry)
                for name in GRID:
                    trial=measured[name]
                    losses[name].update({
                        "correct_offsets":len(set(base["correct_offset_truth_ids"])-
                                              set(trial["correct_offset_truth_ids"])),
                        "long_onsets":len(set(base["long_onset_truth_ids"])-
                                           set(trial["long_onset_truth_ids"])),
                        "onset_tp":max(0,base["tp"]-trial["tp"])
                    })
                    counts[name].update(details[name]["gate"])
                report["CAL_recordings_done"]=idx+1
                save(args.out,report)
                print("S761 CAL",idx+1,32,spec["name"],flush=True)
    summary={key:aggregate(value) for key,value in rows.items()}
    source=summary["S708_frozen"]
    signed_cal=signed["frozen_candidate"]["CAL"]
    delta={key:float(source[key]-signed_cal[key]) for key in SCORES}
    parity=all(abs(d)<=1e-8 for d in delta.values())
    report["reference_parity"]=dict(valid=parity,delta=delta)
    report["reference_scores"]={key:float(source[key]) for key in SCORES}
    eligible=[]
    for name in GRID:
        r=summary[name]
        changes={key:float(r[key]-source[key]) for key in SCORES}
        lost=dict(losses[name])
        safe=bool(parity and all(v>=-1e-12 for v in changes.values())
                  and r["tp"]>=source["tp"] and not any(lost.values()))
        report["CAL"][name]=dict(
            scores={key:float(r[key]) for key in SCORES},delta=changes,
            lost=lost,event_counts=dict(counts[name]),
            source_events_preserved=True,safe=safe,
            six_gt80=bool(safe and all(r[key]>.8 for key in SCORES)))
        if safe and any(v>1e-12 for v in changes.values()):
            eligible.append(name)
    report["CAL_candidate"]=max(
        eligible,key=lambda name:(min(report["CAL"][name]["scores"].values()),
                                  sum(report["CAL"][name]["delta"].values())),
        default=None)
    report["complete"]=parity
    save(args.out,report)
    if not parity:
        raise AssertionError("S761 signed CAL32 parity failed")
    print("S761_COMPLETE",json.dumps({
        "reference":report["reference_scores"],
        "FIT":report["FIT"],
        "candidate":report["CAL_candidate"],"safe":eligible,
        "best_active":max(
            ((key,value["scores"]["active_exact_k"])
             for key,value in report["CAL"].items()),
            key=lambda row:(row[1],row[0]))}),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--s708-dir",type=Path,required=True)
    p.add_argument("--offset-checkpoint",type=Path,required=True)
    p.add_argument("--specialist-dir",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    evaluate(p.parse_args())
