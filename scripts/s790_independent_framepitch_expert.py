"""S790 independent causal 49-pitch sustain head: FIT80 OOF, CAL32 shadow.

We measure the real S708 MIDI source-occupancy mask on the same 1024
decision grid as each candidate. Candidate pitch masks are diagnostic
ONLY: no new NoteOn/NoteOff MIDI event stream is generated, so a better
shadow F1 is NOT an official 256-hop Frame Pitch F1.
Every feature is from the current or an earlier available frame.
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
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from causal_note.strict_live import SR
from causal_note.s708_cardinality_selection import GuidedPolicy
from scripts.s613_prepare_strict_live import download,labels,digest
from scripts.s613_train_strict_live import real_notes
from scripts.s699_player_held_full_train_K import expanded_train_specs
from scripts.s705_train_temporal_loops import setup
from scripts.s706_train_sharp_temporal_loops import FIT_COMPS,CAL_COMPS,load_model
from scripts.s733_train_onset_only import baseline_input
from scripts.s704_train_joint_loops import targets

INTERVALS=(12,19,24,28,31)
DIM=40
CUTS=(.30,.45,.60)
HARM_PENALTIES=(0.,.08,.16)
SEED=790
MAX_ROWS_PER_RECORDING=1600

def occupancy_from_emitted(events,ends):
    """Match s704_joint_metrics.metric: actual event occupancy, never head."""
    ends=np.asarray(ends,np.int64)
    out=np.zeros((len(ends),49),dtype=bool)
    starts={}
    for e in sorted(events,key=lambda z:(int(z["available_sample"]),
             0 if z["type"]=="note_off" else 1,int(z["pitch"]))):
        pitch=int(e["pitch"])-40
        if not 0<=pitch<49:
            raise ValueError("S790 supports MIDI40..88")
        frame=int(np.searchsorted(ends,int(e["available_sample"]),side="left"))
        if e["type"]=="note_on":
            if pitch in starts:
                raise AssertionError("duplicate NoteOn without NoteOff")
            starts[pitch]=frame
        elif e["type"]=="note_off":
            if pitch in starts:
                out[starts.pop(pitch):frame,pitch]=True
        else:
            raise ValueError("unknown note event")
    for pitch,frame in starts.items():
        out[frame:,pitch]=True
    return out

def causal_features(item,post,kprob,events):
    n=len(item["decision_ends"])
    post=np.asarray(post,np.float32)
    kprob=np.asarray(kprob,np.float32)
    x=np.asarray(item["x"],np.float32)
    if post.shape!=(n,49,3) or kprob.shape!=(n,7) or x.shape!=(n,49,288):
        raise ValueError("S790 frozen S708 acoustic input ABI mismatch")
    active=occupancy_from_emitted(events,item["decision_ends"])
    on=post[:,:,0]; sustain=post[:,:,1]
    old=np.concatenate((sustain[:1],sustain[:-1]),axis=0)
    old2=np.concatenate((sustain[:1],sustain[:1],sustain[:-2]),axis=0)[:n]
    oldon=np.concatenate((on[:1],on[:-1]),axis=0)
    spectrum=np.log1p(np.maximum(x[:,:,192:216],0.)).astype(np.float32)
    parents=[]
    for d in INTERVALS:
        source=np.zeros_like(sustain)
        source[:,d:]=sustain[:,:-d]
        parents.append(source)
    upper=[]
    for d in (12,19):
        source=np.zeros_like(sustain)
        source[:,:-d]=sustain[:,d:]
        upper.append(source)
    harmonic=np.max(np.stack(parents,axis=-1),axis=-1)
    extra=[post,old[:,:,None],old2[:,:,None],
           np.maximum(on-oldon,0)[:,:,None],
           spectrum,*[z[:,:,None] for z in parents],
           *[z[:,:,None] for z in upper],
           active.astype(np.float32)[:,:,None],
           np.broadcast_to(kprob[:,5:6,None],(n,49,1)),
           np.broadcast_to(np.linspace(0,1,49,dtype=np.float32)[None,:,None],
                           (n,49,1))]
    X=np.concatenate(extra,axis=-1).astype(np.float32)
    if X.shape!=(n,49,DIM) or not np.isfinite(X).all():
        raise AssertionError(f"S790 40D causal feature ABI drift: {X.shape}")
    return X,active,harmonic,sustain

def counts(pred,true,sel=None):
    if sel is None:sel=np.ones_like(true,bool)
    tp=int(np.sum(pred&true&sel))
    fp=int(np.sum(pred&~true&sel))
    fn=int(np.sum(~pred&true&sel))
    return {"tp":tp,"fp":fp,"fn":fn}

def summarize(rows,key):
    tp=sum(r[key]["tp"] for r in rows)
    fp=sum(r[key]["fp"] for r in rows)
    fn=sum(r[key]["fn"] for r in rows)
    return dict(TP=tp,FP=fp,FN=fn,
                precision=tp/max(tp+fp,1),recall=tp/max(tp+fn,1),
                f1=2*tp/max(2*tp+fp+fn,1))

def sample_training(X,y,post,seed):
    n,p,_=X.shape
    rng=np.random.default_rng(seed)
    truth=np.flatnonzero(y.ravel())
    neg=np.flatnonzero(~y.ravel())
    hard=((post[:,:,1]>.12)|(X[:,:,-3]>.5)).ravel()
    quota=MAX_ROWS_PER_RECORDING//2
    if len(truth)>quota:
        truth=rng.choice(truth,quota,replace=False)
    left=MAX_ROWS_PER_RECORDING-len(truth)
    hard_neg=neg[hard[neg]]
    picked_hard=rng.choice(hard_neg,min(int(left*.70),len(hard_neg)),
                          replace=False)
    rest=np.setdiff1d(neg,picked_hard,assume_unique=False)
    picked_rest=rng.choice(rest,min(left-len(picked_hard),len(rest)),
                           replace=False)
    selected=np.sort(np.concatenate((truth,picked_hard,picked_rest)))
    return X.reshape(-1,DIM)[selected].astype(np.float32),\
           y.ravel()[selected].astype(np.int8)

def fit_specialist(rows,omit=None):
    xx=[];yy=[]
    for r in rows:
        if r["composition"]==omit:continue
        xx.append(r["train_x"]);yy.append(r["train_y"])
    x=np.concatenate(xx);y=np.concatenate(yy)
    if len(np.unique(y))!=2:
        raise AssertionError("S790 model needs both pitch labels")
    model=make_pipeline(StandardScaler(),
        LogisticRegression(C=.1,solver="liblinear",max_iter=140,
                           class_weight="balanced",random_state=SEED))
    model.fit(x,y)
    return model

def evaluate_policy(prob,base,harm,sustain,cut,penalty):
    """Fragment+pitch adaptive expert arbitration; no labels/future."""
    threshold=cut+penalty*np.clip(harm,0.,1.)
    enough_evidence=(sustain>.14)|(base)
    proposal=(prob>=threshold)&enough_evidence
    return np.where(enough_evidence,proposal,base)

def policy_name(cut,penalty):
    return f"cut{cut:.2f}_harm{penalty:.2f}"

def groups(rows,field,key):
    names=sorted(set(r[field] for r in rows))
    return {v:summarize([r for r in rows if r[field]==v],key) for v in names}

def save(path,report):
    path.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")

def evaluate(args):
    if args.out.exists():raise FileExistsError("S790 output immutable")
    args.out.mkdir(parents=True)
    signed=json.loads((args.s708/"report.json").read_text())
    if digest((args.s708/"s708_selected.pt").read_bytes())!=signed["checkpoint_sha256"]:
        raise AssertionError("S790 signed S708 SHA mismatch")
    choice=signed["frozen_candidate"]["choice"]
    network=load_model(args.s708/"s708_selected.pt")[0]
    policy=GuidedPolicy(**choice["policy"])
    torch.set_num_threads(2)
    report=dict(schema="S790_INDEPENDENT_CAUSAL_FRAMEPITCH_FIT80_CAL32",
        complete=False,promoted=False,LIVE_changed=False,
        final_MIDI_changed=False,ONSET_OFFSET_changed=False,
        no_CAL_training_or_selection=True,HELD_opened=False,TEST_opened=False,
        decision_clock_samples=1024,feature_dimension=DIM,
        fit_recordings_done=0,cal_recordings_done=0,
        FIT={},CAL={},selected_policy=None)
    def commit():save(args.out/"report.json",report)
    commit()
    policies=[(policy_name(a,b),a,b) for a in CUTS for b in HARM_PENALTIES]
    fitrows=[];calrows=[]
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)
        (root/"data").mkdir()
        protocol,heads,selection,offset=setup(root,args.offset_checkpoint)
        for archive,name in (("GuitarSet_audio","GuitarSet_audio.zip"),
                             ("GuitarSet_annotations","GuitarSet_annotations.zip")):
            meta=protocol["guitar_archives"][archive]
            download(meta["url"],root/"data"/name,meta["size"],meta["sha256"])
        train_specs=[s for s in protocol["sources"]
                     if s["corpus"]=="GuitarSet" and s["split"]=="train"]
        with zipfile.ZipFile(root/"data/GuitarSet_audio.zip") as wav,\
             zipfile.ZipFile(root/"data/GuitarSet_annotations.zip") as ann:
            specs=expanded_train_specs(train_specs,wav,ann)
            fit_specs=[s for s in specs if s["composition"] in FIT_COMPS]
            cal_specs=[s for s in specs if s["composition"] in CAL_COMPS]
            if len(fit_specs)!=80 or len(cal_specs)!=32 or set(FIT_COMPS)&set(CAL_COMPS):
                raise AssertionError("S790 signed composition boundaries changed")
            for i,spec in enumerate(fit_specs):
                item,post,k,events,*_=baseline_input(
                    spec,wav,root,heads,selection,offset,network,policy,choice["release"])
                X,active,harm,sust=causal_features(item,post,k,events)
                # Training labels are ONLY read after acoustic features and
                # candidate events have been independently generated.
                truth=real_notes(labels(spec,root)["notes"],item["samples"]/SR)
                label=targets(item["decision_ends"],truth)[0][:,:,1].astype(bool)
                sampled_x,sampled_y=sample_training(X,label,post,SEED+i)
                cache=root/f"fit_{i:03d}.npz"
                np.savez_compressed(cache,X=X.astype(np.float16),
                    y=label,base=active,harm=harm.astype(np.float16),
                    sust=sust.astype(np.float16))
                fitrows.append(dict(cache=cache,train_x=sampled_x,
                    train_y=sampled_y,composition=spec["composition"],
                    texture=spec["name"].rsplit("_",1)[-1],name=spec["name"]))
                report["fit_recordings_done"]=i+1;commit()
                print("S790_FIT_RECORD",i+1,"/80",spec["name"],flush=True)
            comps=sorted(set(r["composition"] for r in fitrows))
            if len(comps)!=5:raise AssertionError("S790 needs five OOF groups")
            oof={name:[] for name,_,_ in policies}
            base_oof=[]
            for comp in comps:
                model=fit_specialist(fitrows,omit=comp)
                for row in (r for r in fitrows if r["composition"]==comp):
                    with np.load(row["cache"]) as z:
                        shape=z["base"].shape
                        scores=model.predict_proba(z["X"].astype(np.float32).reshape(-1,DIM))[:,1].reshape(shape)
                        truth=z["y"].astype(bool);base=z["base"].astype(bool)
                        harm=z["harm"].astype(np.float32)
                        sust=z["sust"].astype(np.float32)
                        baseline=counts(base,truth)
                        base_oof.append(dict(stats=baseline,
                            composition=comp,texture=row["texture"]))
                        for name,cut,penalty in policies:
                            pred=evaluate_policy(scores,base,harm,sust,cut,penalty)
                            oof[name].append(dict(stats=counts(pred,truth),
                                composition=comp,texture=row["texture"]))
                print("S790_OOF",comp,flush=True)
            original=summarize(base_oof,"stats")
            candidates={name:summarize(oof[name],"stats") for name,_,_ in policies}
            by_comp={name:groups(oof[name],"composition","stats") for name in candidates}
            original_by=groups(base_oof,"composition","stats")
            # Never select on CAL. Candidate must raise FIT micro F1 by
            # at least .1 point and not regress a single FIT comp >1 point.
            eligible=[name for name in candidates
                if candidates[name]["f1"] >= original["f1"]+.001
                and all(by_comp[name][c]["f1"]>=original_by[c]["f1"]-.01
                        for c in comps)]
            best=max(eligible,key=lambda n:(candidates[n]["f1"],
                       min(by_comp[n][c]["f1"] for c in comps)),default=None)
            report["FIT"]=dict(baseline=original,
                baseline_by_composition=original_by,candidates=candidates,
                by_composition=by_comp,eligible=eligible,
                min_gain=.001,max_comp_regression=.01,selected=best,
                fit_compositions=comps,number_of_policies=len(policies))
            report["selected_policy"]=best
            commit()
            model=fit_specialist(fitrows) if best else None
            selected=next(((cut,pen) for name,cut,pen in policies
                           if name==best),None)
            for i,spec in enumerate(cal_specs):
                item,post,k,events,*_=baseline_input(
                    spec,wav,root,heads,selection,offset,network,policy,choice["release"])
                X,active,harm,sust=causal_features(item,post,k,events)
                truth=real_notes(labels(spec,root)["notes"],item["samples"]/SR)
                label=targets(item["decision_ends"],truth)[0][:,:,1].astype(bool)
                candidate=active
                if selected is not None:
                    scores=model.predict_proba(X.reshape(-1,DIM))[:,1].reshape(active.shape)
                    candidate=evaluate_policy(scores,active,harm,sust,*selected)
                reg=np.broadcast_to(np.arange(49)[None,:],active.shape)
                bins={"low_E2_B2":(reg<8),
                      "mid_C3_B3":(reg>=8)&(reg<20),
                      "high_C4_E5":(reg>=20)}
                calrows.append(dict(baseline=counts(active,label),
                    trial=counts(candidate,label),
                    by_register={name:dict(baseline=counts(active,label,mask),
                      trial=counts(candidate,label,mask)) for name,mask in bins.items()},
                    composition=spec["composition"],
                    texture=spec["name"].rsplit("_",1)[-1],name=spec["name"]))
                report["cal_recordings_done"]=i+1;commit()
                print("S790_CAL_RECORD",i+1,"/32",spec["name"],flush=True)
    report["CAL"]=dict(baseline=summarize(calrows,"baseline"),
        frozen=summarize(calrows,"trial"),
        by_composition_baseline=groups(calrows,"composition","baseline"),
        by_composition_frozen=groups(calrows,"composition","trial"),
        by_texture_baseline=groups(calrows,"texture","baseline"),
        by_texture_frozen=groups(calrows,"texture","trial"),
        by_register={key:dict(baseline=summarize([
            {"x":row["by_register"][key]["baseline"]} for row in calrows],"x"),
            trial=summarize([{"x":row["by_register"][key]["trial"]}
                            for row in calrows],"x"))
            for key in ("low_E2_B2","mid_C3_B3","high_C4_E5")})
    report["complete"]=True
    commit()
    print("S790_RESULT",json.dumps(dict(selected=best,
        fit_base=original["f1"],
        fit_selected=candidates[best]["f1"] if best else original["f1"],
        cal_base=report["CAL"]["baseline"]["f1"],
        cal_selected=report["CAL"]["frozen"]["f1"],
        cal_by_register=report["CAL"]["by_register"])),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--s708",type=Path,required=True)
    p.add_argument("--offset-checkpoint",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    evaluate(p.parse_args())
