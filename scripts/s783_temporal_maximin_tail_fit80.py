"""S783: temporal and composition-robust dynamic K5/K6 decoder.

Five composition-held-out folds choose tail cost on FIT only.
No TEST, HELD, CAL tuning, LIVE change or final MIDI promotion.
All K scores are at the S708 1024-sample decision cadence, NOT official
256-sample Exact-K. S708 final event stream is unchanged.
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

from causal_note.strict_live import SR
from causal_note.s708_cardinality_selection import GuidedPolicy
from scripts.s613_prepare_strict_live import download,labels,digest
from scripts.s613_train_strict_live import real_notes
from scripts.s699_player_held_full_train_K import expanded_train_specs
from scripts.s705_train_temporal_loops import setup
from scripts.s706_train_sharp_temporal_loops import FIT_COMPS,CAL_COMPS,load_model
from scripts.s733_train_onset_only import baseline_input,active_count_from_events
from scripts.s776_k5k6_count_rootcauses import truth_count,confusion,measurements

WEIGHTS=(1.,2.,4.,8.)
K6_BIASES=(0.,.5,1.,1.5)
SEED=783
DIM=56

def causal_features(item,post,kprob,events):
    """Current/past acoustic composition-specific cardinality evidence."""
    n=len(kprob)
    if np.asarray(post).shape!=(n,49,3) or np.asarray(item["x"]).shape!=(n,49,288):
        raise ValueError("S783 expected frozen acoustic causal ABI")
    on=np.sort(np.asarray(post[:,:,0],np.float32),axis=1)[:,-8:]
    sustain=np.sort(np.asarray(post[:,:,1],np.float32),axis=1)[:,-8:]
    counts=np.stack([(post[:,:,1]>t).sum(1)/49.
                     for t in (.1,.3,.5,.8)],axis=1).astype(np.float32)
    source_active=active_count_from_events(events,item["decision_ends"])[:,None]
    energy=item["x"][:,:,192:216].mean(axis=2)
    spectral=np.stack([np.quantile(energy,q,axis=1)
                      for q in (.5,.75,.9)],axis=1).astype(np.float32)
    # 25 extra per-track causal dynamics; no lookahead and no other track state.
    prev=np.asarray(kprob[np.maximum(0,np.arange(n)-1)],np.float32)
    k_change=np.asarray(kprob,np.float32)-prev
    extra_sustain=np.stack([(post[:,:,1]>t).sum(1)/49.
                            for t in (.05,.2,.4,.65)],axis=1).astype(np.float32)
    onset_counts=np.stack([(post[:,:,0]>t).sum(1)/49.
                            for t in (.1,.3,.5,.8)],axis=1).astype(np.float32)
    prev_energy=energy[np.maximum(0,np.arange(n)-1)]
    energy_change=np.maximum(energy-prev_energy,0.)
    spectral_rises=np.stack([np.quantile(energy_change,q,axis=1)
                              for q in (.5,.75,.9)],axis=1).astype(np.float32)
    result=np.concatenate((kprob,on,sustain,counts,source_active,spectral,
                           prev,k_change,extra_sustain,onset_counts,
                           spectral_rises),axis=1).astype(np.float32)
    if result.shape!=(n,DIM) or not np.isfinite(result).all():
        raise AssertionError("S783 class-tail feature drift")
    return result

def fit(X,y,weight):
    unique=np.unique(y)
    if len(unique)<5 or not set(unique).issubset(set(range(7))):
        raise AssertionError("S783 insufficient training K classes")
    w=np.ones(len(y),np.float32)
    w[y==5]=weight
    w[y==6]=weight*2.
    model=LogisticRegression(C=.25,max_iter=200,solver="lbfgs",
                             random_state=SEED)
    model.fit(X,y,sample_weight=w)
    return model

def predicted(model,features,baseline,k6_bias=0.):
    if model is None:
        return np.argmax(baseline,axis=1).astype(np.int8)
    proba=model.predict_proba(features)
    # FIT-selected tail-prior correction; per-fragment, no future/no labels.
    logits=np.log(np.maximum(proba,1e-8))
    for k,w in ((5,.5),(6,1.)):
        if k in model.classes_:
            logits[:,int(np.where(model.classes_==k)[0][0])]+=float(k6_bias)*w
    est=model.classes_[np.argmax(logits,axis=1)].astype(np.int8)
    # Restrict disruptive tail promotion: permit early-K states to
    # change only if the independent classifier is also <= K4.
    native=np.argmax(baseline,axis=1).astype(np.int8)
    return np.where((native<3)&(est>=5),native,est).astype(np.int8)

def merge_counts(records,key):
    gt=np.concatenate([r["gt"] for r in records])
    pred=np.concatenate([r[key] for r in records])
    return measurements(confusion(gt,pred),None,gt)

def robust_score(group_summaries,overall):
    """Worst-composition K6, then K5 recall, then global accuracy."""
    k6=[v["per_k"][6]["accuracy"] or 0. for v in group_summaries.values()]
    k5=[v["per_k"][5]["accuracy"] or 0. for v in group_summaries.values()]
    return (min(k6),min(k5),overall["exact_accuracy"])


def tail_histogram(records,key):
    """K5 and K6 true frames split by predicted K0..K6 (no oracle at inference)."""
    gt=np.concatenate([r["gt"] for r in records])
    y=np.concatenate([r[key] for r in records])
    return {str(k):np.bincount(y[gt==k],minlength=7).astype(int).tolist() for k in (5,6)}

def by_composition(records,key):
    groups=sorted({r["composition"] for r in records})
    return {c:dict(overall=merge_counts([r for r in records if r["composition"]==c],key),
        tail_histogram=tail_histogram([r for r in records if r["composition"]==c],key),
        texture={
            texture:dict(
               metrics=merge_counts([r for r in records if r["composition"]==c and r["texture"]==texture],key),
               tail_histogram=tail_histogram([r for r in records if r["composition"]==c and r["texture"]==texture],key))
            for texture in ("comp","solo")})
        for c in groups}

def write(path,data):
    path.write_text(json.dumps(data,sort_keys=True,indent=2)+"\n")

def evaluate(args):
    if args.out.exists():
        raise FileExistsError("S783 immutable output")
    args.out.mkdir(parents=True)
    signed=json.loads((args.s708_dir/"report.json").read_text())
    sha=digest((args.s708_dir/"s708_selected.pt").read_bytes())
    assert sha==signed["checkpoint_sha256"]
    frozen=signed["frozen_candidate"]["choice"]
    network=load_model(args.s708_dir/"s708_selected.pt")[0]
    policy=GuidedPolicy(**frozen["policy"])
    torch.set_num_threads(2)
    result=dict(schema="S783_TEMPORAL_MAXIMIN_TAIL_FIT80",
        complete=False,selected="S708_frozen",promoted=False,LIVE_changed=False,
        final_MIDI_changed=False,ONSET_OFFSET_changed=False,
        decision_clock_samples=1024,TEST_opened=False,HELD_opened=False,
        no_CAL_training_or_selection=True,fit_recordings_done=0,cal_recordings_done=0,
        FIT={},CAL={},selected_tail_weight=None,selected_k6_bias=None)
    save=lambda: write(args.out/"report.json",result)
    save()
    fit_rows=[];cal_rows=[]
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp);(root/"data").mkdir()
        protocol,heads,selection,offset=setup(root,args.offset_checkpoint)
        for key,path in (("GuitarSet_audio","GuitarSet_audio.zip"),
                         ("GuitarSet_annotations","GuitarSet_annotations.zip")):
            remote=protocol["guitar_archives"][key]
            download(remote["url"],root/"data"/path,remote["size"],remote["sha256"])
        train=[spec for spec in protocol["sources"]
               if spec["corpus"]=="GuitarSet" and spec["split"]=="train"]
        with zipfile.ZipFile(root/"data/GuitarSet_audio.zip") as wav,\
             zipfile.ZipFile(root/"data/GuitarSet_annotations.zip") as ann:
            specs=expanded_train_specs(train,wav,ann)
            fit_specs=[s for s in specs if s["composition"] in FIT_COMPS]
            cal_specs=[s for s in specs if s["composition"] in CAL_COMPS]
            assert len(fit_specs)==80 and len(cal_specs)==32
            assert not set(FIT_COMPS)&set(CAL_COMPS)
            for i,spec in enumerate(fit_specs):
                item,post,k,events,acoustic,ctx,eligible=baseline_input(
                    spec,wav,root,heads,selection,offset,network,policy,frozen["release"])
                features=causal_features(item,post,k,events)
                truth=real_notes(labels(spec,root)["notes"],item["samples"]/SR)
                y=truth_count(item["decision_ends"],truth)
                fit_rows.append(dict(X=features,gt=y,baseline=np.argmax(k,axis=1),
                                     composition=spec["composition"], name=spec["name"], texture=spec["name"].rsplit("_",1)[-1]))
                result["fit_recordings_done"]=i+1;save()
                print("S783 FIT",i+1,80,spec["name"],"k6",int((y==6).sum()),flush=True)
            comps=sorted({r["composition"] for r in fit_rows})
            assert len(comps)==5
            basem=merge_counts(fit_rows,"baseline")
            candidates=[(w,b) for w in WEIGHTS for b in K6_BIASES]
            oof={f"{w:.1f}:{b:.1f}":[] for w,b in candidates}
            foldinfo={}
            for comp in comps:
                tr=[r for r in fit_rows if r["composition"]!=comp]
                val=[r for r in fit_rows if r["composition"]==comp]
                X=np.concatenate([r["X"] for r in tr])
                y=np.concatenate([r["gt"] for r in tr])
                for weight in WEIGHTS:
                    model=fit(X,y,weight)
                    for r in val:
                        baseline_prob=np.eye(7,dtype=np.float32)[r["baseline"]]
                        for bias in K6_BIASES:
                            key=f"{weight:.1f}:{bias:.1f}"
                            oof[key].append(dict(gt=r["gt"],
                                trial=predicted(model,r["X"],baseline_prob,bias),
                                composition=comp,texture=r["texture"]))
                foldinfo[comp]=dict(training_frames=len(y),held_tracks=len(val))
                print("S783 OOF",comp,len(val),flush=True)
            summaries={key:merge_counts(oof[key],"trial") for key in oof}
            bycomp={key:{c:merge_counts([r for r in oof[key] if r["composition"]==c],"trial")
                           for c in comps} for key in oof}
            base_by_comp={c:merge_counts(
                [r for r in fit_rows if r["composition"]==c],"baseline") for c in comps}
            # Micro Exact-K FIT non-regression guard, plus per-composition guard.
            eligible=[key for key in oof
                if summaries[key]["exact_accuracy"] >= basem["exact_accuracy"]-.005
                and all(bycomp[key][c]["exact_accuracy"] >=
                        base_by_comp[c]["exact_accuracy"]-.02 for c in comps)]
            best=max(eligible,key=lambda key:robust_score(bycomp[key],summaries[key]),
                     default=None)
            weight,bias=(tuple(map(float,best.split(":"))) if best else (None,None))
            result["FIT"]=dict(baseline=basem,trial_ood=summaries,folds=foldinfo,
                               eligible=eligible,selected=best,
                               baseline_by_composition=base_by_comp,
                               oof_by_composition=bycomp,
                               selection_guard_micro_loss_max=.005,
                               selection_guard_per_composition_loss_max=.02,
                               selection_objective="maximin_K6_then_K5_composition_recall")
            result["selected_tail_weight"]=weight
            result["selected_k6_bias"]=bias
            save()
            final_model=(fit(np.concatenate([r["X"] for r in fit_rows]),
                             np.concatenate([r["gt"] for r in fit_rows]),weight)
                         if best is not None else None)
            for i,spec in enumerate(cal_specs):
                item,post,k,events,acoustic,ctx,eligible=baseline_input(
                    spec,wav,root,heads,selection,offset,network,policy,frozen["release"])
                features=causal_features(item,post,k,events)
                baseline=np.argmax(k,axis=1)
                trial=predicted(final_model,features,k,bias or 0.)
                truth=real_notes(labels(spec,root)["notes"],item["samples"]/SR)
                gt=truth_count(item["decision_ends"],truth)
                cal_rows.append(dict(gt=gt,baseline=baseline,trial=trial,
                                     composition=spec["composition"],name=spec["name"],texture=spec["name"].rsplit("_",1)[-1]))
                result["cal_recordings_done"]=i+1;save()
                print("S783 CAL",i+1,32,spec["name"],flush=True)
    result["CAL"]=dict(baseline=merge_counts(cal_rows,"baseline"),
                       selected=merge_counts(cal_rows,"trial"),
                       comp=merge_counts([r for r in cal_rows if r["texture"]=="comp"],"trial"),
                       solo=merge_counts([r for r in cal_rows if r["texture"]=="solo"],"trial"),
                       by_composition_base=by_composition(cal_rows,"baseline"),
                       by_composition_specialist=by_composition(cal_rows,"trial"),
                       tail_histogram_base=tail_histogram(cal_rows,"baseline"),
                       tail_histogram_specialist=tail_histogram(cal_rows,"trial"))
    result["complete"]=True
    save()
    print("S783_KTAIL_RESULT",json.dumps(dict(selected=best,
        fit_base=result["FIT"]["baseline"]["exact_accuracy"],
        fit_best=(result["FIT"]["trial_ood"][best]["exact_accuracy"] if best is not None else None),
        cal_base=result["CAL"]["baseline"]["exact_accuracy"],
        cal_selected=result["CAL"]["selected"]["exact_accuracy"],
        cal_K5=result["CAL"]["selected"]["per_k"][5],
        cal_K6=result["CAL"]["selected"]["per_k"][6])),flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--s708-dir",type=Path,required=True)
    parser.add_argument("--offset-checkpoint",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    evaluate(parser.parse_args())
