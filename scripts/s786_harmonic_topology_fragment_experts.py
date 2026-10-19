"""S786 causal pitch-topology expert vs source 31D S781 expert.

S781/783 discarded the spatial identity of all 49 pitches by sorting
posterior and quantile energies. S786 adds an independent 200D head that
preserves the pitch lattice and explicit harmonic interval interactions.
A per-fragment selector routes to the topology head only when its causal
harmonic-overlap evidence exceeds a FIT-only fixed gate.

FIT80 5 composition-held-out folds determine eligibility. CAL32 is only
a frozen diagnostic. No final MIDI, LIVE, offset, TEST or HELD change.
These are 1024-sample SHADOW head Exact-K metrics, NOT 256-hop MIDI scores.
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
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from causal_note.strict_live import SR
from causal_note.s708_cardinality_selection import GuidedPolicy
from scripts.s613_prepare_strict_live import download,labels,digest
from scripts.s613_train_strict_live import real_notes
from scripts.s699_player_held_full_train_K import expanded_train_specs
from scripts.s705_train_temporal_loops import setup
from scripts.s706_train_sharp_temporal_loops import FIT_COMPS,CAL_COMPS,load_model
from scripts.s733_train_onset_only import baseline_input
from scripts.s776_k5k6_count_rootcauses import truth_count
from scripts.s781_k_tail_composition_audit import causal_features,fit as fit_base,predicted as base_predicted,merge_counts,by_composition,tail_histogram

WEIGHTS=(2.,4.)
GATES=(0.,.15,.30)
INTERVALS=(12,19,24,28,31,36)
DIM=200
SEED=786

def harmonic_features(item,post,kprob,events):
    """Current/past only; (pitch position, interval topology) preserved."""
    base=causal_features(item,post,kprob,events)
    n=len(base)
    sustain=np.asarray(post[:,:,1],np.float32)
    onset=np.asarray(post[:,:,0],np.float32)
    energy=np.maximum(np.asarray(item["x"][:,:,192:216],np.float32).mean(axis=2),0.)
    energy=np.log1p(energy)
    if (sustain.shape!=(n,49) or onset.shape!=(n,49)
        or energy.shape!=(n,49)):
        raise ValueError("S786 frozen 49-pitch acoustic ABI mismatch")
    overlap=[]
    parent=np.zeros_like(sustain)
    for d in INTERVALS:
        low=sustain[:,:-d]
        high=sustain[:,d:]
        pair=low*high
        overlap.extend((pair.mean(axis=1),
                        np.max(pair,axis=1),
                        (high>low).astype(np.float32).mean(axis=1)))
        parent[:,d:]=np.maximum(parent[:,d:],low)
    conflict=sustain*parent
    mass=np.sum(sustain,axis=1)
    harmonic_density=np.sum(conflict,axis=1)/np.maximum(mass,1e-6)
    supported=np.maximum(sustain-.8*parent,0.)
    summary=np.stack((harmonic_density,
        np.mean(conflict>.1,axis=1),
        np.mean(supported>.3,axis=1),
        np.mean(supported>.6,axis=1)),axis=1).astype(np.float32)
    result=np.concatenate((base,sustain,onset,energy,
                           np.stack(overlap,axis=1),summary),axis=1).astype(np.float32)
    if result.shape!=(n,DIM) or not np.isfinite(result).all():
        raise AssertionError("S786 topology feature ABI drift")
    return result,harmonic_density.astype(np.float32)

def fit_topology(X,y,weight):
    if np.unique(y).size<5: raise AssertionError("S786 lost K classes")
    sample=np.ones(len(y),np.float32)
    sample[y==5]=weight
    sample[y==6]=weight*2.
    model=make_pipeline(StandardScaler(),
        LogisticRegression(C=.15,max_iter=200,solver="lbfgs",
                           random_state=SEED))
    model.fit(X,y,logisticregression__sample_weight=sample)
    return model

def candidate(topology_model,baseline_model,topology,base,gate,native):
    if topology_model is None:
        return native.copy()
    base_out=base_predicted(baseline_model,base,np.eye(7,dtype=np.float32)[native])
    new_out=topology_model.predict(topology).astype(np.int8)
    # Exact same tail guard as S781; low-K->K5/6 hallucinations forbidden.
    new_out=np.where((native<3)&(new_out>=5),native,new_out)
    return np.where(gate>=0.,new_out,base_out).astype(np.int8)

def gated_candidate(topology_model,baseline_model,topology,base,harmonic,gate_threshold,native):
    """Switch heads at every causal 1024-sample fragment, not by track."""
    base_out=base_predicted(baseline_model,base,np.eye(7,dtype=np.float32)[native])
    new_out=topology_model.predict(topology).astype(np.int8)
    new_out=np.where((native<3)&(new_out>=5),native,new_out)
    return np.where(harmonic>=gate_threshold,new_out,base_out).astype(np.int8)

def robust_score(comp,overall):
    def acc(m,k):
        value=m["per_k"][k]["accuracy"]
        return 0. if value is None else float(value)
    k6=[acc(m,6) for m in comp.values() if m["per_k"][6]["n"]>0]
    k5=[acc(m,5) for m in comp.values() if m["per_k"][5]["n"]>0]
    return (min(k6) if k6 else 0.,min(k5) if k5 else 0.,
            overall["exact_accuracy"])

def save(path,data):
    path.write_text(json.dumps(data,sort_keys=True,indent=2)+"\n")

def evaluate(args):
    if args.out.exists():raise FileExistsError("S786 immutable output")
    args.out.mkdir(parents=True)
    signed=json.loads((args.s708_dir/"report.json").read_text())
    sha=digest((args.s708_dir/"s708_selected.pt").read_bytes())
    if sha!=signed["checkpoint_sha256"]:
        raise AssertionError("S786 signed checkpoint mismatch")
    frozen=signed["frozen_candidate"]["choice"]
    net=load_model(args.s708_dir/"s708_selected.pt")[0]
    policy=GuidedPolicy(**frozen["policy"])
    torch.set_num_threads(2)
    result=dict(schema="S786_CAUSAL_PITCH_TOPOLOGY_FRAGMENT_EXPERTS",
        complete=False,promoted=False,LIVE_changed=False,
        final_MIDI_changed=False,ONSET_OFFSET_changed=False,
        no_CAL_training_or_selection=True,TEST_opened=False,HELD_opened=False,
        decision_clock_samples=1024,topological_dim=DIM,
        fit_recordings_done=0,cal_recordings_done=0,
        selected_policy=None,FIT={},CAL={})
    commit=lambda:save(args.out/"report.json",result)
    commit()
    fit_rows=[];cal_rows=[]
    with tempfile.TemporaryDirectory() as td:
        root=Path(td);(root/"data").mkdir()
        protocol,heads,selection,offset=setup(root,args.offset_checkpoint)
        for source,filename in (("GuitarSet_audio","GuitarSet_audio.zip"),
                                ("GuitarSet_annotations","GuitarSet_annotations.zip")):
            remote=protocol["guitar_archives"][source]
            download(remote["url"],root/"data"/filename,
                     remote["size"],remote["sha256"])
        train=[s for s in protocol["sources"]
               if s["corpus"]=="GuitarSet" and s["split"]=="train"]
        with zipfile.ZipFile(root/"data/GuitarSet_audio.zip") as wav,\
             zipfile.ZipFile(root/"data/GuitarSet_annotations.zip") as ann:
            specs=expanded_train_specs(train,wav,ann)
            fit_specs=[s for s in specs if s["composition"] in FIT_COMPS]
            cal_specs=[s for s in specs if s["composition"] in CAL_COMPS]
            if len(fit_specs)!=80 or len(cal_specs)!=32 or set(FIT_COMPS)&set(CAL_COMPS):
                raise AssertionError("S786 FIT/CAL splits changed")
            for i,spec in enumerate(fit_specs):
                item,post,k,events,acoustic,ctx,eligible=baseline_input(
                    spec,wav,root,heads,selection,offset,net,policy,frozen["release"])
                top,harm=harmonic_features(item,post,k,events)
                gt=truth_count(item["decision_ends"],
                    real_notes(labels(spec,root)["notes"],item["samples"]/SR))
                fit_rows.append(dict(base=top[:,:31],top=top,harm=harm,gt=gt,
                                     native=np.argmax(k,axis=1).astype(np.int8),
                                     composition=spec["composition"],texture=spec["name"].rsplit("_",1)[-1]))
                result["fit_recordings_done"]=i+1;commit()
                print("S786 FIT",i+1,80,spec["name"],flush=True)
            comps=sorted({r["composition"] for r in fit_rows})
            if len(comps)!=5:raise AssertionError("S786 five compositions required")
            baseline=[];oof={f"{w:g}:{g:.2f}":[] for w in WEIGHTS for g in GATES}
            for comp in comps:
                tr=[r for r in fit_rows if r["composition"]!=comp]
                val=[r for r in fit_rows if r["composition"]==comp]
                y=np.concatenate([r["gt"] for r in tr])
                base_model=fit_base(np.concatenate([r["base"] for r in tr]),y,4.)
                for r in val:
                    original=base_predicted(base_model,r["base"],
                        np.eye(7,dtype=np.float32)[r["native"]])
                    baseline.append(dict(gt=r["gt"],trial=original,
                                         composition=comp,texture=r["texture"]))
                X=np.concatenate([r["top"] for r in tr])
                for weight in WEIGHTS:
                    topology=fit_topology(X,y,weight)
                    for r in val:
                        for gate in GATES:
                            key=f"{weight:g}:{gate:.2f}"
                            trial=gated_candidate(topology,base_model,
                                r["top"],r["base"],r["harm"],gate,r["native"])
                            oof[key].append(dict(gt=r["gt"],trial=trial,
                                composition=comp,texture=r["texture"]))
                print("S786 OOF",comp,flush=True)
            base=merge_counts(baseline,"trial")
            base_group={c:merge_counts([r for r in baseline if r["composition"]==c],"trial") for c in comps}
            trial={key:merge_counts(rows,"trial") for key,rows in oof.items()}
            trial_group={key:{c:merge_counts([r for r in rows if r["composition"]==c],"trial")
                           for c in comps} for key,rows in oof.items()}
            eligible=[key for key in oof
                 if trial[key]["exact_accuracy"]>=base["exact_accuracy"]-.002
                 and all(trial_group[key][c]["exact_accuracy"]>=base_group[c]["exact_accuracy"]-.01
                         for c in comps)]
            best=max(eligible,key=lambda k:robust_score(trial_group[k],trial[k]),
                     default=None)
            result["FIT"]=dict(baseline=base,
                baseline_by_composition=base_group,
                candidates=trial,by_composition=trial_group,
                eligible=eligible,selected=best,
                selection_micro_regression_limit=.002,
                selection_comp_regression_limit=.01,
                objective="worst_composition_K6_then_K5")
            result["selected_policy"]=best
            commit()
            if best is not None:
                weight,threshold=(float(q) for q in best.split(":"))
                y=np.concatenate([r["gt"] for r in fit_rows])
                base_model=fit_base(np.concatenate([r["base"] for r in fit_rows]),y,4.)
                topology=fit_topology(np.concatenate([r["top"] for r in fit_rows]),y,weight)
            else:
                base_model=topology=None;threshold=None
            for i,spec in enumerate(cal_specs):
                item,post,k,events,acoustic,ctx,eligible=baseline_input(
                    spec,wav,root,heads,selection,offset,net,policy,frozen["release"])
                top,harm=harmonic_features(item,post,k,events)
                native=np.argmax(k,axis=1).astype(np.int8)
                output=(gated_candidate(topology,base_model,top,top[:,:31],
                                        harm,threshold,native)
                        if best is not None else native)
                gt=truth_count(item["decision_ends"],
                    real_notes(labels(spec,root)["notes"],item["samples"]/SR))
                cal_rows.append(dict(gt=gt,baseline=native,trial=output,
                                     composition=spec["composition"],
                                     texture=spec["name"].rsplit("_",1)[-1]))
                result["cal_recordings_done"]=i+1;commit()
                print("S786 CAL",i+1,32,spec["name"],flush=True)
    result["CAL"]=dict(native=merge_counts(cal_rows,"baseline"),
                       frozen=merge_counts(cal_rows,"trial"),
                       by_composition=by_composition(cal_rows,"trial"),
                       confusion_k5_k6=tail_histogram(cal_rows,"trial"))
    result["complete"]=True
    commit()
    print("S786_RESULT",json.dumps(dict(selected=best,
        fit_baseline=base["exact_accuracy"],
        fit_selected=(trial[best]["exact_accuracy"] if best else base["exact_accuracy"]),
        cal_baseline=result["CAL"]["native"]["exact_accuracy"],
        cal_selected=result["CAL"]["frozen"]["exact_accuracy"],
        cal_k5=result["CAL"]["frozen"]["per_k"][5],
        cal_k6=result["CAL"]["frozen"]["per_k"][6])),flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--s708-dir",type=Path,required=True)
    parser.add_argument("--offset-checkpoint",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    evaluate(parser.parse_args())
