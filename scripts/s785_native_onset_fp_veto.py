"""S785: FIT80 cross-composition causal NoteOn false-positive veto.

Adds a supervised *separate* specialist trained exclusively on FIT80
original NoteOn correctness. Its decisions use only available/past acoustic
features. It filters an onset-only SHADOW stream; never changes the final MIDI
stream or NoteOff, and never promotes from CAL32.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import tempfile
import zipfile
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from causal_note.strict_live import SR
from causal_note.s708_cardinality_selection import GuidedPolicy
from scripts.s613_prepare_strict_live import download,labels,digest
from scripts.s613_train_strict_live import real_notes,event_score
from scripts.s699_player_held_full_train_K import expanded_train_specs
from scripts.s705_train_temporal_loops import setup
from scripts.s706_train_sharp_temporal_loops import FIT_COMPS,CAL_COMPS,load_model
from scripts.s733_train_onset_only import baseline_input
from scripts.s768_independent_onset_lattice_fit80 import onset_score,combine

DIM=20
CS=(.1,1.,10.)
CUTS=(.08,.15,.25,.35,.45)
SEED=785

def event_features(item,post,kprob,events):
    """Create features at each NoteOn, never reading acoustic frames after it."""
    clock=np.asarray(item["decision_ends"],np.int64)
    spectra=np.asarray(item["x"][:,:,192:216],np.float32).mean(axis=2)
    post=np.asarray(post,np.float32);kprob=np.asarray(kprob,np.float32)
    n=len(clock)
    if (spectra.shape!=(n,49) or post.shape!=(n,49,3)
        or kprob.shape!=(n,7)):
        raise ValueError("S785 S708 model input ABI mismatch")
    active=set()
    features=[]
    for e in events:
        pitch=int(e["pitch"])
        if e["type"]!="note_on":
            if e["type"]=="note_off":active.discard(pitch)
            continue
        p=pitch-40
        if not 0<=p<49:raise ValueError("S785 outside supported MIDI40..88")
        t=int(e["available_sample"])
        idx=int(np.searchsorted(clock,t,side="right"))-1
        v=np.zeros(DIM,np.float32)
        if idx>=0:
            prev=max(0,idx-1)
            v[:7]=kprob[idx]
            v[7:10]=post[idx,p]
            v[10]=post[idx,p,0]-post[prev,p,0]
            v[11]=post[idx,p,1]-post[prev,p,1]
            energy=spectra[idx,p]
            v[12]=energy
            v[13]=energy-spectra[prev,p]
            v[14]=energy/max(float(np.quantile(spectra[idx],.75)),.01)
            lower=[z for z in (p-12,p-19) if z>=0]
            if lower:
                v[15]=max(float(post[idx,z,1]) for z in lower)
                v[16]=max(float(spectra[idx,z]) for z in lower)
        v[17]=min(len(active),7)/7.
        v[18]=p/48.
        v[19]=float(e.get("velocity",96))/127.
        if not np.isfinite(v).all():raise ValueError("S785 nonfinite event features")
        features.append(v)
        active.add(pitch)
    result=np.asarray(features,np.float32).reshape(-1,DIM)
    return result

def select_events(source,correct_probability,cut):
    """Drop only selected NOTE_ONs from a shadow stream, never its real MIDI."""
    if len(correct_probability)!=sum(e["type"]=="note_on" for e in source):
        raise ValueError("S785 probability/NoteOn count mismatch")
    j=0;keep=[]
    for e in source:
        if e["type"]=="note_on":
            valid=bool(correct_probability[j]>=cut)
            j+=1
            if not valid:continue
        keep.append(e)
    return keep

def train(X,y,C):
    if np.unique(y).size!=2:
        raise ValueError("S785 requires both native TP and FP in FIT")
    model=make_pipeline(StandardScaler(),LogisticRegression(
        C=C,solver="lbfgs",class_weight={0:2.,1:1.},
        max_iter=250,random_state=SEED))
    model.fit(X,y)
    return model

def policy_key(C,cut):
    return f"C{C:g}_cut{cut:.2f}"

def bycomp(records):
    return {comp:combine([r["score"] for r in records if r["composition"]==comp])
            for comp in sorted({r["composition"] for r in records})}

def robust_selection(base,summaries,groups):
    base_all=combine([r["score"] for r in base])
    base_comps=bycomp(base)
    totals={k:combine([r["score"] for r in rows]) for k,rows in summaries.items()}
    comps={k:bycomp(rows) for k,rows in summaries.items()}
    # FIT-only guard: no composition loses >0.5 percentage points onset F1.
    viable=[key for key in summaries
            if totals[key]["f1"]>base_all["f1"] and
            all(comps[key][c]["f1"]>=base_comps[c]["f1"]-.005 for c in groups)]
    best=max(viable,key=lambda k:(
        min(comps[k][c]["f1"]-base_comps[c]["f1"] for c in groups),
        totals[k]["f1"],totals[k]["precision"]),default=None)
    return best,base_all,base_comps,totals,comps,viable

def write(path,report):
    path.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")

def evaluate(args):
    if args.out.exists():raise FileExistsError("S785 immutable result directory")
    args.out.mkdir(parents=True)
    signed=json.loads((args.s708_dir/"report.json").read_text())
    sha=digest((args.s708_dir/"s708_selected.pt").read_bytes())
    if sha!=signed["checkpoint_sha256"]:
        raise AssertionError("S785 signed S708 checkpoint mismatch")
    frozen=signed["frozen_candidate"]["choice"]
    network=load_model(args.s708_dir/"s708_selected.pt")[0]
    policy=GuidedPolicy(**frozen["policy"])
    torch.set_num_threads(2)
    report=dict(schema="S785_FIT80_NATIVE_ONSET_FP_VETO",
        complete=False,promoted=False,LIVE_changed=False,
        final_MIDI_changed=False,ONSET_only_not_MIDI=True,
        labels_only_after_causal_predictions=True,
        no_CAL_training_or_selection=True,TEST_opened=False,HELD_opened=False,
        fit_recordings_done=0,cal_recordings_done=0,
        selected_policy=None,FIT={},CAL={},checkpoint_sha256=sha)
    save=lambda:write(args.out/"report.json",report)
    save()
    fit_rows=[];cal_rows=[]
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp);(root/"data").mkdir()
        protocol,heads,selection,offset=setup(root,args.offset_checkpoint)
        for key,path in (("GuitarSet_audio","GuitarSet_audio.zip"),
                         ("GuitarSet_annotations","GuitarSet_annotations.zip")):
            remote=protocol["guitar_archives"][key]
            download(remote["url"],root/"data"/path,remote["size"],remote["sha256"])
        train_specs=[spec for spec in protocol["sources"]
                     if spec["corpus"]=="GuitarSet" and spec["split"]=="train"]
        with zipfile.ZipFile(root/"data/GuitarSet_audio.zip") as wav,\
             zipfile.ZipFile(root/"data/GuitarSet_annotations.zip") as ann:
            specs=expanded_train_specs(train_specs,wav,ann)
            fit_specs=[s for s in specs if s["composition"] in FIT_COMPS]
            cal_specs=[s for s in specs if s["composition"] in CAL_COMPS]
            if len(fit_specs)!=80 or len(cal_specs)!=32 or set(FIT_COMPS)&set(CAL_COMPS):
                raise AssertionError("S785 split invariant")
            for i,spec in enumerate(fit_specs):
                item,post,k,source,acoustic,ctx,eligible=baseline_input(
                    spec,wav,root,heads,selection,offset,network,policy,frozen["release"])
                feats=event_features(item,post,k,source)
                # FIT labels become available only after constructing features.
                truth=real_notes(labels(spec,root)["notes"],item["samples"]/SR)
                _,pairs=event_score(source,truth,tolerance=.080)
                labels_y=np.zeros(len(feats),np.int8)
                for pred_idx,_ in pairs:labels_y[pred_idx]=1
                fit_rows.append(dict(X=feats,y=labels_y,source=source,truth=truth,
                                     composition=spec["composition"],name=spec["name"]))
                report["fit_recordings_done"]=i+1;save()
                print("S785 FIT",i+1,80,spec["name"],
                      "TP",int(labels_y.sum()),"FP",int((1-labels_y).sum()),flush=True)
            groups=sorted({r["composition"] for r in fit_rows})
            if len(groups)!=5:raise AssertionError("S785 expected five FIT compositions")
            oof={policy_key(C,cut):[] for C in CS for cut in CUTS}
            base=[]
            folds={}
            for group in groups:
                tr=[r for r in fit_rows if r["composition"]!=group]
                val=[r for r in fit_rows if r["composition"]==group]
                X=np.concatenate([r["X"] for r in tr],axis=0)
                y=np.concatenate([r["y"] for r in tr],axis=0)
                for C in CS:
                    model=train(X,y,C)
                    for r in val:
                        probs=model.predict_proba(r["X"])[:,1]
                        for cut in CUTS:
                            key=policy_key(C,cut)
                            shadow=select_events(r["source"],probs,cut)
                            oof[key].append(dict(composition=group,
                                score=onset_score(shadow,[],r["truth"])))
                for r in val:
                    base.append(dict(composition=group,
                        score=onset_score(r["source"],[],r["truth"])))
                folds[group]=dict(train_examples=len(X),
                                  held_tracks=len(val))
                print("S785 OOF",group,len(val),flush=True)
            best,baseline,baseline_comps,totals,comps,eligible=robust_selection(
                base,oof,groups)
            frozen_C,frozen_cut=(
                next((C,cut) for C in CS for cut in CUTS
                     if policy_key(C,cut)==best)
                if best else (None,None))
            report["FIT"]=dict(baseline=baseline,
                by_composition_baseline=baseline_comps,
                thresholds=totals,by_composition=comps,
                eligible=eligible,selected=best,folds=folds,
                training_events=sum(len(r["y"]) for r in fit_rows),
                original_training_tp=sum(int(r["y"].sum()) for r in fit_rows),
                selection_objective="worst_composition_F1_nonregression")
            report["selected_policy"]=best
            save()
            final_model=(train(np.concatenate([r["X"] for r in fit_rows]),
                               np.concatenate([r["y"] for r in fit_rows]),frozen_C)
                         if best else None)
            for i,spec in enumerate(cal_specs):
                item,post,k,source,acoustic,ctx,eligible=baseline_input(
                    spec,wav,root,heads,selection,offset,network,policy,frozen["release"])
                feats=event_features(item,post,k,source)
                scores=(final_model.predict_proba(feats)[:,1] if final_model else None)
                shadow=(select_events(source,scores,frozen_cut)
                        if final_model else source)
                # CAL annotations opened only after frozen acoustic decisions.
                truth=real_notes(labels(spec,root)["notes"],item["samples"]/SR)
                cal_rows.append(dict(composition=spec["composition"],
                    native=onset_score(source,[],truth),
                    trial=onset_score(shadow,[],truth),
                    vetoed=sum(e["type"]=="note_on" for e in source)-
                           sum(e["type"]=="note_on" for e in shadow)))
                report["cal_recordings_done"]=i+1;save()
                print("S785 CAL",i+1,32,spec["name"],flush=True)
    def metric(key):return combine([r[key] for r in cal_rows])
    report["CAL"]=dict(baseline=metric("native"),frozen=metric("trial"),
        vetoed=sum(r["vetoed"] for r in cal_rows),
        by_composition_base={c:combine([r["native"] for r in cal_rows if r["composition"]==c])
            for c in CAL_COMPS},
        by_composition_frozen={c:combine([r["trial"] for r in cal_rows if r["composition"]==c])
            for c in CAL_COMPS})
    report["CAL"]["delta_f1"]=report["CAL"]["frozen"]["f1"]-report["CAL"]["baseline"]["f1"]
    report["complete"]=True;save()
    print("S785_FROZEN_FP_VETO",json.dumps(dict(selected=best,
        fit_base=baseline["f1"],
        fit_selected=totals[best]["f1"] if best else baseline["f1"],
        cal_base=report["CAL"]["baseline"]["f1"],
        cal_selected=report["CAL"]["frozen"]["f1"],
        vetoed=report["CAL"]["vetoed"])),flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--s708-dir",type=Path,required=True)
    parser.add_argument("--offset-checkpoint",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    evaluate(parser.parse_args())
