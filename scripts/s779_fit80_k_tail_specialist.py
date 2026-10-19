"""S779: FIT80 class-aware independent Exact-K / K5-K6 specialist.

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
SEED=779
DIM=31

def causal_features(item,post,kprob,events):
    """Current/past acoustic composition-specific cardinality evidence."""
    n=len(kprob)
    if np.asarray(post).shape!=(n,49,3) or np.asarray(item["x"]).shape!=(n,49,288):
        raise ValueError("S779 expected frozen acoustic causal ABI")
    on=np.sort(np.asarray(post[:,:,0],np.float32),axis=1)[:,-8:]
    sustain=np.sort(np.asarray(post[:,:,1],np.float32),axis=1)[:,-8:]
    counts=np.stack([(post[:,:,1]>t).sum(1)/49.
                     for t in (.1,.3,.5,.8)],axis=1).astype(np.float32)
    source_active=active_count_from_events(events,item["decision_ends"])[:,None]
    energy=item["x"][:,:,192:216].mean(axis=2)
    spectral=np.stack([np.quantile(energy,q,axis=1)
                      for q in (.5,.75,.9)],axis=1).astype(np.float32)
    result=np.concatenate((kprob,on,sustain,counts,source_active,spectral),
                          axis=1).astype(np.float32)
    if result.shape!=(n,DIM) or not np.isfinite(result).all():
        raise AssertionError("S779 class-tail feature drift")
    return result

def fit(X,y,weight):
    unique=np.unique(y)
    if len(unique)<5 or not set(unique).issubset(set(range(7))):
        raise AssertionError("S779 insufficient training K classes")
    w=np.ones(len(y),np.float32)
    w[y==5]=weight
    w[y==6]=weight*2.
    model=LogisticRegression(C=.25,max_iter=200,solver="lbfgs",
                             random_state=SEED)
    model.fit(X,y,sample_weight=w)
    return model

def predicted(model,features,baseline):
    if model is None:
        return np.argmax(baseline,axis=1).astype(np.int8)
    est=model.predict(features).astype(np.int8)
    # Restrict disruptive tail promotion: permit early-K states to
    # change only if the independent classifier is also <= K4.
    native=np.argmax(baseline,axis=1).astype(np.int8)
    return np.where((native<3)&(est>=5),native,est).astype(np.int8)

def merge_counts(records,key):
    gt=np.concatenate([r["gt"] for r in records])
    pred=np.concatenate([r[key] for r in records])
    return measurements(confusion(gt,pred),None,gt)

def score(counts):
    acc=counts["exact_accuracy"]
    cls=counts["per_k"]
    r5=cls[5]["accuracy"] or 0.
    r6=cls[6]["accuracy"] or 0.
    return (min(r5,r6),.5*(r5+r6),acc)

def write(path,data):
    path.write_text(json.dumps(data,sort_keys=True,indent=2)+"\n")

def evaluate(args):
    if args.out.exists():
        raise FileExistsError("S779 immutable output")
    args.out.mkdir(parents=True)
    signed=json.loads((args.s708_dir/"report.json").read_text())
    sha=digest((args.s708_dir/"s708_selected.pt").read_bytes())
    assert sha==signed["checkpoint_sha256"]
    frozen=signed["frozen_candidate"]["choice"]
    network=load_model(args.s708_dir/"s708_selected.pt")[0]
    policy=GuidedPolicy(**frozen["policy"])
    torch.set_num_threads(2)
    result=dict(schema="S779_CLASS_AWARE_EXACTK_TAIL_SPECIALIST_FIT80",
        complete=False,selected="S708_frozen",promoted=False,LIVE_changed=False,
        final_MIDI_changed=False,ONSET_OFFSET_changed=False,
        decision_clock_samples=1024,TEST_opened=False,HELD_opened=False,
        no_CAL_training_or_selection=True,fit_recordings_done=0,cal_recordings_done=0,
        FIT={},CAL={},selected_tail_weight=None)
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
                                     composition=spec["composition"]))
                result["fit_recordings_done"]=i+1;save()
                print("S779 FIT",i+1,80,spec["name"],"k6",int((y==6).sum()),flush=True)
            comps=sorted({r["composition"] for r in fit_rows})
            assert len(comps)==5
            basem=merge_counts(fit_rows,"baseline")
            oof={w:[] for w in WEIGHTS}
            foldinfo={}
            for comp in comps:
                tr=[r for r in fit_rows if r["composition"]!=comp]
                val=[r for r in fit_rows if r["composition"]==comp]
                X=np.concatenate([r["X"] for r in tr])
                y=np.concatenate([r["gt"] for r in tr])
                for weight in WEIGHTS:
                    model=fit(X,y,weight)
                    for r in val:
                        oof[weight].append(dict(gt=r["gt"],
                            trial=predicted(model,r["X"],np.eye(7)[r["baseline"]])))
                foldinfo[comp]=dict(training_frames=len(y),held_tracks=len(val))
                print("S779 OOF",comp,len(val),flush=True)
            summaries={str(w):merge_counts(oof[w],"trial") for w in WEIGHTS}
            # FIT controls performance: select tail specialist only when
            # preserving at least 99.5% of baseline global micro accuracy.
            viable=[w for w in WEIGHTS
                if summaries[str(w)]["exact_accuracy"]>=basem["exact_accuracy"]-.005]
            best=max(viable,key=lambda w:score(summaries[str(w)]),default=None)
            result["FIT"]=dict(baseline=basem,trial_ood=summaries,folds=foldinfo,
                               eligible=viable,selected=best)
            result["selected_tail_weight"]=best
            save()
            final_model=(fit(np.concatenate([r["X"] for r in fit_rows]),
                             np.concatenate([r["gt"] for r in fit_rows]),best)
                         if best is not None else None)
            for i,spec in enumerate(cal_specs):
                item,post,k,events,acoustic,ctx,eligible=baseline_input(
                    spec,wav,root,heads,selection,offset,network,policy,frozen["release"])
                features=causal_features(item,post,k,events)
                baseline=np.argmax(k,axis=1)
                trial=predicted(final_model,features,k)
                truth=real_notes(labels(spec,root)["notes"],item["samples"]/SR)
                gt=truth_count(item["decision_ends"],truth)
                cal_rows.append(dict(gt=gt,baseline=baseline,trial=trial,
                                     texture=spec["name"].rsplit("_",1)[-1]))
                result["cal_recordings_done"]=i+1;save()
                print("S779 CAL",i+1,32,spec["name"],flush=True)
    result["CAL"]=dict(baseline=merge_counts(cal_rows,"baseline"),
                       selected=merge_counts(cal_rows,"trial"),
                       comp=merge_counts([r for r in cal_rows if r["texture"]=="comp"],"trial"),
                       solo=merge_counts([r for r in cal_rows if r["texture"]=="solo"],"trial"))
    result["complete"]=True
    save()
    print("S779_KTAIL_RESULT",json.dumps(dict(weight=best,
        fit_base=result["FIT"]["baseline"]["exact_accuracy"],
        fit_best=(result["FIT"]["trial_ood"][str(best)]["exact_accuracy"] if best is not None else None),
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
