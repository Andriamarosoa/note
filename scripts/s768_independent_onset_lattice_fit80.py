"""S768: FIT80-trained high-recall independent ONSET-ONLY rescue head.

S767 proved 1073/1170 missed CAL32 onsets have no matching S765
candidate, so S768 scores the ENTIRE (received decision, MIDI40..88)
acoustic lattice, not the narrow S765 FullNote proposal set.

Each prediction is strictly causal (current/past features only). Training
labels are missing onsets on FIT80 ONLY. An OOF five-composition calibration
freezes the risk threshold before opening CAL labels. S708 MIDI, S707
offsets and LIVE are NOT edited. Onset-only shadows are NOT full MIDI.

No CAL tuning, no HELD/TEST, no post-event oracle at inference.
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
from scripts.s613_train_strict_live import real_notes, event_score
from scripts.s699_player_held_full_train_K import expanded_train_specs
from scripts.s704_joint_metrics import SCORES, metric, aggregate
from scripts.s705_train_temporal_loops import setup
from scripts.s706_train_sharp_temporal_loops import FIT_COMPS,CAL_COMPS,load_model
from scripts.s733_train_onset_only import baseline_input
from scripts.s701_train_causal_onset_shadow import onset_features

PITCHES=49
BLOCK=1024
FEAT=87  # 39 S701 + 16 acoustic context + 4x8 physical spectral
THRESHOLDS=(.05,.10,.15,.20,.30,.40,.50,.60,.70,.80,.90)
REARM_RATIO=.45
REFRACTORY=4  # 92.88 ms, measured as onset-only retrigger guard
SOURCE_COOLDOWN=3  # 69.66 ms AFTER past/current frozen S708 NoteOn
SEED=768


def lag(a,blocks):
    return a[np.maximum(0,np.arange(len(a))-blocks)]


def feature_grid(item,post,k,ctx):
    """Causal pitch-specific 87D features over ALL 49 pitches at every block."""
    x=np.asarray(item["x"],np.float32)
    p=np.asarray(post,np.float32)
    counts=np.asarray(k,np.float32)
    scene=np.asarray(ctx,np.float32)
    n=len(p)
    if (x.shape!=(n,49,288) or p.shape!=(n,49,3)
        or counts.shape!=(n,7) or scene.shape!=(n,49,16)):
        raise ValueError("S768 causal 288D/49 pitch ABI mismatch")
    native=onset_features(x[:,:,:126],p,counts)
    spectral=x[:,:,192:216].reshape(n,49,3,8).mean(axis=2)
    change=np.maximum(spectral-lag(spectral,1),0.)
    long_change=np.maximum(spectral-lag(spectral,4),0.)
    spectral_norm=spectral/np.maximum(
        np.abs(spectral).mean(axis=2,keepdims=True),1e-3)
    out=np.concatenate((
        native,scene,np.clip(spectral,0,16),
        np.clip(change,0,16),np.clip(long_change,0,16),
        np.clip(spectral_norm,0,32)),axis=-1).astype(np.float32)
    if out.shape!=(n,49,FEAT) or not np.isfinite(out).all():
        raise AssertionError("S768 feature schema drift")
    return out


def missing_onset_targets(ends,truth,baseline_matches):
    """TRAIN ONLY: positive 0..2 causal decision blocks after a missed note."""
    clock=np.asarray(ends,np.int64)
    target=np.zeros((len(clock),PITCHES),np.uint8)
    seen=set(baseline_matches)
    for idx,n in enumerate(truth):
        if idx in seen:
            continue
        pitch=int(n["pitch"])-40
        if not 0<=pitch<PITCHES:
            continue
        onset=int(round(float(n["time"])*SR))
        start=int(np.searchsorted(clock,onset,side="left"))
        if start<len(clock):
            target[start:min(start+3,len(clock)),pitch]=1
    return target


def sample_training(feat,y,post,ctx,seed):
    """Fixed subsampling from FIT compositions, labels never enter inference."""
    rng=np.random.default_rng(seed)
    pos=np.flatnonzero(y.reshape(-1).astype(bool))
    negatives=np.flatnonzero(~y.reshape(-1).astype(bool))
    if not len(pos):
        return np.empty((0,FEAT),np.float32),np.empty(0,np.int8)
    n=feat.shape[0]; flat=feat.reshape(n*PITCHES,-1)
    on=np.asarray(post[:,:,0]).reshape(-1)
    sustain=np.asarray(post[:,:,1]).reshape(-1)
    novelty=np.asarray(ctx[:,:,4]).reshape(-1)
    # Hard acoustic negatives and random background selected WITHOUT
    # looking at future waveform. Labels are used only for FIT subsampling.
    hard=(on>.035)|(sustain>.18)|(novelty>.01)|(
        feat[:,:,63:71].mean(axis=-1).reshape(-1)>.025)
    hardneg=negatives[hard[negatives]]
    easyneg=negatives[~hard[negatives]]
    pp=rng.choice(pos,size=min(600,len(pos)),replace=False)
    hh=rng.choice(hardneg,size=min(1300,len(hardneg)),replace=False)
    ee=rng.choice(easyneg,size=min(650,len(easyneg)),replace=False)
    selected=np.concatenate((pp,hh,ee))
    rng.shuffle(selected)
    labels=y.reshape(-1)[selected].astype(np.int8)
    return flat[selected],labels


def fit(X,y):
    """Complementary weak-onset classifier; deterministic, no CAL supervision."""
    if len(X)<10 or len(np.unique(y))!=2:
        return None
    model=HistGradientBoostingClassifier(
        max_iter=100,max_leaf_nodes=23,max_depth=6,
        l2_regularization=2.,learning_rate=.08,
        min_samples_leaf=28,early_stopping=False,random_state=SEED)
    # Aggressively reward otherwise missed FIT onset snippets without
    # pretending that undersampled negative labels are naturally calibrated.
    model.fit(X,y,sample_weight=np.where(y==1,4.,1.))
    return model


def onset_probabilities(model,features,batch=16384):
    n=len(features)
    if model is None:
        return np.zeros((n,PITCHES),np.float32)
    flat=features.reshape(n*PITCHES,FEAT)
    output=np.empty(len(flat),np.float32)
    for st in range(0,len(flat),batch):
        batch_x=np.asarray(flat[st:st+batch],np.float32)
        output[st:st+len(batch_x)]=model.predict_proba(batch_x)[:,1]
    return output.reshape(n,PITCHES)


def proposals(score,ends,source,threshold):
    """Online-equivalent events: hysteretic rise + refractory + past source veto.

    A later S708 event cannot retroactively affect a previously emitted
    candidate; offline CAL ground truth is not an input.
    """
    scores=np.asarray(score,np.float32)
    times=np.asarray(ends,np.int64)
    if scores.shape!=(len(times),PITCHES) or threshold not in THRESHOLDS:
        raise ValueError("S768 model/scoring ABI mismatch")
    originals=sorted((e for e in source if e["type"]=="note_on"),
                     key=lambda e:int(e["available_sample"]))
    last_native=np.full(PITCHES,-10**12,np.int64)
    last_candidate=np.full(PITCHES,-10**12,np.int64)
    armed=np.ones(PITCHES,bool)
    cursor=0
    result=[]
    for i,t in enumerate(times):
        now=int(t)
        while cursor<len(originals) and int(originals[cursor]["available_sample"])<=now:
            e=originals[cursor]
            last_native[int(e["pitch"])-40]=int(e["available_sample"])
            cursor+=1
        armed |= scores[i]<threshold*REARM_RATIO
        for ix in np.flatnonzero(
                (scores[i]>=threshold) & armed &
                (now-last_candidate>=REFRACTORY*BLOCK) &
                (now-last_native>=SOURCE_COOLDOWN*BLOCK)):
            result.append(dict(type="note_on",pitch=int(ix+40),
                 decision_sample=now,available_sample=now,
                 time=now/SR,velocity=96,reason="s768_shadow_new_attack",
                 _s768_shadow=True))
            last_candidate[ix]=now
            armed[ix]=False
    return result


def onset_score(reference,added,truth):
    """Actual per-pitch one-to-one 80ms matching; onset-only stream."""
    stats,_=event_score([*reference,*added],truth,tolerance=.080)
    return {k:int(stats[k]) for k in ("TP","FP","FN")}


def combine(rows):
    result={k:int(sum(x[k] for x in rows)) for k in ("TP","FP","FN")}
    tp,fp,fn=(result[x] for x in ("TP","FP","FN"))
    result.update(f1=2*tp/max(1,2*tp+fp+fn),
                  precision=tp/max(1,tp+fp),
                  recall=tp/max(1,tp+fn))
    return result


def train_fit(records,groups):
    groups=np.asarray(groups)
    unique=sorted(set(groups))
    if len(unique)!=5:
        raise AssertionError("S768 composition split not five-fold")
    X=np.concatenate([r["train_X"] for r in records],axis=0)
    y=np.concatenate([r["train_y"] for r in records],axis=0)
    g=np.concatenate([np.repeat(g,len(r["train_y"]))
                      for g,r in zip(groups,records)])
    print("S768 TRAINING",len(X),"positives",int(y.sum()),flush=True)
    if len(X)==0 or len(np.unique(y))<2:
        raise AssertionError("S768 insufficient FIT training labels")
    scores={th:[] for th in THRESHOLDS}
    logs={}
    for hold in unique:
        mask=g!=hold
        model=fit(X[mask],y[mask])
        if model is None:
            raise AssertionError("S768 degenerate composition-fold model")
        subset=[r for r,group in zip(records,groups) if group==hold]
        for r in subset:
            features=np.load(r["features"],mmap_mode="r")
            p=onset_probabilities(model,features)
            for th in THRESHOLDS:
                shadow=proposals(p,r["ends"],r["source"],th)
                scores[th].append(onset_score(r["source"],shadow,r["truth"]))
        logs[hold]=dict(train_examples=int(mask.sum()),
                        held_recordings=len(subset))
        print("S768 OOF",hold,"records",len(subset),flush=True)
    summary={str(th):combine(rows) for th,rows in scores.items()}
    best=max(THRESHOLDS,key=lambda t:(summary[str(t)]["f1"],
                                     summary[str(t)]["precision"]))
    final_model=fit(X,y)
    return final_model,best,summary,logs


def write(path,report):
    path.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")


def evaluate(args):
    if args.out.exists():
        raise FileExistsError("S768 immutable result path already exists")
    args.out.mkdir(parents=True)
    signed=json.loads((args.s708_dir/"report.json").read_text())
    sha=digest((args.s708_dir/"s708_selected.pt").read_bytes())
    if sha!=signed["checkpoint_sha256"]:
        raise AssertionError("S768 signed S708 hash mismatch")
    choice=signed["frozen_candidate"]["choice"]
    release=choice["release"]
    if release!=dict(gate=.8,persistence=1,grace=24):
        raise AssertionError("S768 frozen S707 release mismatch")
    policy=GuidedPolicy(**choice["policy"])
    network=load_model(args.s708_dir/"s708_selected.pt")[0]
    torch.set_num_threads(2)
    report=dict(schema="S768_FIT80_INDEPENDENT_ONSET_FULL_LATTICE",
                checkpoint_sha256=sha,complete=False,promoted=False,
                LIVE_changed=False,TEST_opened=False,HELD_opened=False,
                ONSET_only_not_MIDI=True,no_future_inputs=True,
                no_truth_at_inference=True,
                fit_recordings_done=0,cal_recordings_done=0,
                selected="S708_frozen",FIT={},CAL={},source_reference_parity=None)
    write(args.out/"report.json",report)
    originals=[]
    selected=[]
    scores={th:[] for th in THRESHOLDS}
    census=Counter()
    with tempfile.TemporaryDirectory() as td:
        root=Path(td);(root/"data").mkdir()
        cached=root/"fit_tensor";cached.mkdir()
        protocol,heads,selection,offset=setup(root,args.offset_checkpoint)
        for key,file in (("GuitarSet_audio","GuitarSet_audio.zip"),
                         ("GuitarSet_annotations","GuitarSet_annotations.zip")):
            remote=protocol["guitar_archives"][key]
            download(remote["url"],root/"data"/file,
                     remote["size"],remote["sha256"])
        train=[spec for spec in protocol["sources"]
               if spec["corpus"]=="GuitarSet" and spec["split"]=="train"]
        with zipfile.ZipFile(root/"data/GuitarSet_audio.zip") as wav, \
             zipfile.ZipFile(root/"data/GuitarSet_annotations.zip") as ann:
            expanded=expanded_train_specs(train,wav,ann)
            fit_specs=[s for s in expanded if s["composition"] in FIT_COMPS]
            cal_specs=[s for s in expanded if s["composition"] in CAL_COMPS]
            if len(fit_specs)!=80 or len(cal_specs)!=32 or set(FIT_COMPS)&set(CAL_COMPS):
                raise AssertionError("S768 split disjointness or file count drift")
            for i,spec in enumerate(fit_specs):
                item,post,k,source,acoustic,ctx,eligible=baseline_input(
                    spec,wav,root,heads,selection,offset,network,policy,release)
                features=feature_grid(item,post,k,ctx)
                # Labels opened ONLY after causal S708 acoustic extraction.
                truth=real_notes(labels(spec,root)["notes"],item["samples"]/SR)
                _,matched=event_score(source,truth)
                y=missing_onset_targets(item["decision_ends"],truth,
                                        [index for _,index in matched])
                train_X,train_y=sample_training(
                    features,y,post,ctx,SEED+i)
                filename=cached/f"{i:03d}.npy"
                np.save(filename,features.astype(np.float16))
                selected.append(dict(features=filename,train_X=train_X,
                                     train_y=train_y,
                                     ends=item["decision_ends"],
                                     source=source,truth=truth))
                report["fit_recordings_done"]=i+1
                write(args.out/"report.json",report)
                print("S768 FIT",i+1,80,spec["name"],
                      "positives",int(train_y.sum()),
                      "sample",len(train_y),flush=True)
            model,threshold,oof,folds=train_fit(
                selected,[s["composition"] for s in fit_specs])
            report["FIT"]=dict(folds=folds,
                               thresholds=oof,risk_frozen=threshold,
                               training_rows=int(sum(len(x["train_y"]) for x in selected)),
                               training_positives=int(sum(x["train_y"].sum()
                                                          for x in selected)))
            write(args.out/"report.json",report)
            for i,spec in enumerate(cal_specs):
                item,post,k,source,acoustic,ctx,eligible=baseline_input(
                    spec,wav,root,heads,selection,offset,network,policy,release)
                features=feature_grid(item,post,k,ctx)
                prob=onset_probabilities(model,features)
                streams={th:proposals(prob,item["decision_ends"],source,th)
                         for th in THRESHOLDS}
                # Fully emitted candidates before any CAL truth access.
                truth=real_notes(labels(spec,root)["notes"],item["samples"]/SR)
                originals.append(metric(source,truth,item["ends"]))
                for th,events in streams.items():
                    scores[th].append(onset_score(source,events,truth))
                    census[str(th)]+=len(events)
                report["cal_recordings_done"]=i+1
                write(args.out/"report.json",report)
                print("S768 CAL",i+1,32,spec["name"],
                      "frozen_risk_proposals",len(streams[threshold]),flush=True)
    native=aggregate(originals)
    drift={m:float(native[m]-signed["frozen_candidate"]["CAL"][m])
           for m in SCORES}
    parity=all(abs(val)<1e-8 for val in drift.values())
    report["source_reference_parity"]=dict(valid=parity,delta=drift)
    report["source"]={m:float(native[m]) for m in SCORES}
    reference=dict(TP=int(native["tp"]),FP=int(native["fp"]),FN=int(native["fn"]),
                   f1=float(native["onset_f1"]))
    report["CAL"]=dict(
        source_onset=reference,
        onset_only={str(th):dict(combine(rows),extra_emitted=int(census[str(th)]))
                    for th,rows in scores.items()},
        frozen_risk=threshold,
        frozen_score=combine(scores[threshold]),
        frozen_risk_over_80=bool(combine(scores[threshold])["f1"]>.8))
    report["complete"]=parity
    write(args.out/"report.json",report)
    if not parity:
        raise AssertionError("S768 signed S708 six-score reference drift")
    print("S768_ONSET_ONLY_RESULTS",json.dumps({
        "source":report["source"],"FIT":report["FIT"],
        "CAL":report["CAL"],"promoted":False}),flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--s708-dir",type=Path,required=True)
    ap.add_argument("--offset-checkpoint",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    evaluate(ap.parse_args())
