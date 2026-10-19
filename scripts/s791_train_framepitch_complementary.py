"""S791 FIT80 real multi-specialist Frame Pitch OOF and frozen CAL32.

Four separately fitted domain-specialists + S708 source. FIT-only audit
removes experts that do not contribute across compositions, and a SECOND
composition-held-out router learns WHICH SUBSET to use PER FRAGMENT.
Router training sees only OOF predictions, not predictions in-sample.
No LIVE or final MIDI modifications and no use of HELD/TEST.
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
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from causal_note.strict_live import SR
from causal_note.s708_cardinality_selection import GuidedPolicy
from causal_note.s791_framepitch_expert_bank import (
    EXPERT_NAMES, ALL_MASKS,sample_expert,candidate_offers,
    frame_descriptors,oracle_mask_labels,selected_offer,
    contribution_table,retained_from_fit)
from scripts.s613_prepare_strict_live import download,labels,digest
from scripts.s613_train_strict_live import real_notes
from scripts.s699_player_held_full_train_K import expanded_train_specs
from scripts.s705_train_temporal_loops import setup
from scripts.s706_train_sharp_temporal_loops import FIT_COMPS,CAL_COMPS,load_model
from scripts.s733_train_onset_only import baseline_input
from scripts.s704_train_joint_loops import targets
from scripts.s790_independent_framepitch_expert import causal_features,counts,summarize

SCHEMA="S791_DATA_DRIVEN_FRAMEPITCH_COMPLEMENTARY_EXPERTS"
SEED=791

def save(path,r):
    path.write_text(json.dumps(r,sort_keys=True,indent=2)+"\n")

def train_bank(rows,omit=None):
    heads={}
    for h,expert in enumerate(EXPERT_NAMES):
        feature_rows=[r["sample_x"][expert]
            for r in rows if r["composition"]!=omit and len(r["sample_y"][expert])>0]
        label_rows=[r["sample_y"][expert]
            for r in rows if r["composition"]!=omit and len(r["sample_y"][expert])>0]
        if not label_rows:
            heads[expert]=None;continue
        x=np.concatenate(feature_rows,axis=0)
        y=np.concatenate(label_rows,axis=0)
        if len(np.unique(y))<2:
            heads[expert]=None
            continue
        # Independent model and training population per failure mode.
        heads[expert]=make_pipeline(StandardScaler(),
            LogisticRegression(C=.10 if expert=="general" else .18,
                 solver="liblinear",max_iter=130,
                 class_weight="balanced",random_state=SEED+h))
        heads[expert].fit(x,y)
    return heads

def head_predictions(bank,X):
    n=len(X)
    result={}
    flat=X.reshape(-1,40).astype(np.float32)
    for name in EXPERT_NAMES:
        model=bank[name]
        if model is None:
            result[name]=np.full((n,49),.5,np.float32)
        else:
            result[name]=model.predict_proba(flat)[:,1].reshape(n,49).astype(np.float32)
    return result

def head_score(offer,truth):
    return counts(offer,truth)

def totals(rows,key):
    tp=fp=fn=0
    for row in rows:
        z=row[key]
        tp+=int(z["tp"]);fp+=int(z["fp"]);fn+=int(z["fn"])
    return dict(TP=tp,FP=fp,FN=fn,
                precision=tp/max(tp+fp,1),
                recall=tp/max(tp+fn,1),
                f1=2*tp/max(2*tp+fp+fn,1))

def by_group(rows,field,key):
    return {name:totals([r for r in rows if r[field]==name],key)
            for name in sorted(set(r[field] for r in rows))}

def router_model(rows,other_comps,allowed):
    x=[];y=[]
    for r in rows:
        if r["composition"] in other_comps:
            x.append(r["descriptors"])
            y.append(oracle_mask_labels(r["offers"],r["truth"],allowed))
    xx=np.concatenate(x); yy=np.concatenate(y)
    if len(np.unique(yy))<2:return None
    # Learned on OOF offers/labels from only other compositions.
    # Finite, low-depth per-fragment architecture avoids memorizing tracks.
    model=RandomForestClassifier(n_estimators=45,max_depth=8,
        min_samples_leaf=60,class_weight="balanced_subsample",
        random_state=SEED,n_jobs=2)
    model.fit(xx,yy)
    return model

def route(model,frame_features,allowed):
    if model is None:return np.zeros(len(frame_features),np.int8)
    out=model.predict(frame_features).astype(np.int8)
    valid=np.isin(out,allowed)
    return np.where(valid,out,0).astype(np.int8)

def inspect(spec,archive,root,heads,selection,offset,network,policy,choice):
    item,post,k,emitted,*_=baseline_input(spec,archive,root,heads,selection,
                                         offset,network,policy,choice["release"])
    X,source,_,_=causal_features(item,post,k,emitted)
    # Acoustic features and emitted events computed BEFORE labels are opened.
    truth=real_notes(labels(spec,root)["notes"],item["samples"]/SR)
    expected=targets(item["decision_ends"],truth)[0][:,:,1].astype(bool)
    return X,post,source,expected

def run(args):
    if args.out.exists():raise FileExistsError("S791 output is immutable")
    args.out.mkdir(parents=True)
    signed=json.loads((args.s708/"report.json").read_text())
    sha=digest((args.s708/"s708_selected.pt").read_bytes())
    if sha!=signed["checkpoint_sha256"]:
        raise AssertionError("S791 S708 signed hash incorrect")
    network=load_model(args.s708/"s708_selected.pt")[0]
    choice=signed["frozen_candidate"]["choice"]
    policy=GuidedPolicy(**choice["policy"])
    torch.set_num_threads(2)
    report=dict(schema=SCHEMA,signed_sha256=sha,complete=False,
        promoted=False,LIVE_changed=False,final_MIDI_changed=False,
        ONSET_OFFSET_changed=False,HELD_opened=False,TEST_opened=False,
        no_CAL_training_or_selection=True,decision_clock_samples=1024,
        available_specialists=list(EXPERT_NAMES),retained_specialists=[],
        candidate_masks=len(ALL_MASKS),fit_recordings_done=0,
        cal_recordings_done=0,FIT={},CAL={})
    def commit():save(args.out/"report.json",report)
    commit()
    with tempfile.TemporaryDirectory() as td:
        root=Path(td);(root/"data").mkdir()
        protocol,heads,selection,offset=setup(root,args.offset_checkpoint)
        for a,n in (("GuitarSet_audio","GuitarSet_audio.zip"),
                    ("GuitarSet_annotations","GuitarSet_annotations.zip")):
            remote=protocol["guitar_archives"][a]
            download(remote["url"],root/"data"/n,remote["size"],remote["sha256"])
        train_specs=[s for s in protocol["sources"] if
                     s["corpus"]=="GuitarSet" and s["split"]=="train"]
        with zipfile.ZipFile(root/"data/GuitarSet_audio.zip") as audio,\
             zipfile.ZipFile(root/"data/GuitarSet_annotations.zip") as ann:
            expanded=expanded_train_specs(train_specs,audio,ann)
            fit_specs=[s for s in expanded if s["composition"] in FIT_COMPS]
            cal_specs=[s for s in expanded if s["composition"] in CAL_COMPS]
            if len(fit_specs)!=80 or len(cal_specs)!=32 or set(FIT_COMPS)&set(CAL_COMPS):
                raise AssertionError("FIT80/CAL32 sources drift")
            fits=[]
            for i,spec in enumerate(fit_specs):
                X,post,base,truth=inspect(spec,audio,root,heads,selection,
                                          offset,network,policy,choice)
                path=root/f"fit_{i:03d}.npz"
                np.savez_compressed(path,X=X.astype(np.float16),
                   post=post.astype(np.float16),source=base,truth=truth)
                samples={}
                labels={}
                for h,expert in enumerate(EXPERT_NAMES):
                    xx,yy=sample_expert(X,truth,post,expert,SEED+i*17+h)
                    samples[expert]=xx;labels[expert]=yy
                fits.append(dict(composition=spec["composition"],
                    texture=spec["name"].rsplit("_",1)[-1],
                    name=spec["name"],cache=path,
                    sample_x=samples,sample_y=labels))
                report["fit_recordings_done"]=i+1;commit()
                print("S791_FIT_RECORD",i+1,"/80",spec["name"],flush=True)
            compositions=sorted(set(row["composition"] for row in fits))
            if len(compositions)!=5:raise AssertionError("Five FIT composition folds expected")
            # First independent OOF stage trains EACH candidate with its
            # own population; held composition is invisible to every expert.
            oof=[]
            for comp in compositions:
                bank=train_bank(fits,omit=comp)
                for row in (r for r in fits if r["composition"]==comp):
                    with np.load(row["cache"]) as data:
                        X=data["X"].astype(np.float32)
                        post=data["post"].astype(np.float32)
                        base=data["source"].astype(bool)
                        truth=data["truth"].astype(bool)
                    probabilities=head_predictions(bank,X)
                    offers,domains=candidate_offers(X,post,base,probabilities)
                    descriptors=frame_descriptors(X,post,base,probabilities,domains)
                    oof.append(dict(composition=comp,texture=row["texture"],
                         name=row["name"],offers=offers,truth=truth,
                         descriptors=descriptors))
                print("S791_OOF_EXPERTS",comp,flush=True)
            table=contribution_table(oof)
            retained,allowed=retained_from_fit(table)
            report["retained_specialists"]=retained
            report["FIT"]["expert_oof"]=table
            report["FIT"]["admissible_masks"]=list(allowed)
            commit()
            # Router sees only OOF expert hypotheses from other comps.
            fit_scored=[]
            for comp in compositions:
                other=set(compositions)-{comp}
                router=router_model(oof,other,allowed)
                for r in (x for x in oof if x["composition"]==comp):
                    selected=route(router,r["descriptors"],allowed)
                    native=r["offers"][:,0,:]
                    decoded=selected_offer(r["offers"],selected)
                    fit_scored.append(dict(composition=comp,
                        texture=r["texture"],name=r["name"],
                        baseline=head_score(native,r["truth"]),
                        trial=head_score(decoded,r["truth"]),
                        mask_histogram=dict(Counter(int(v) for v in selected))))
                print("S791_OOF_ROUTER",comp,flush=True)
            base_fit=totals(fit_scored,"baseline")
            trial_fit=totals(fit_scored,"trial")
            comp_base=by_group(fit_scored,"composition","baseline")
            comp_trial=by_group(fit_scored,"composition","trial")
            selected=bool(retained) and (
                trial_fit["f1"] >= base_fit["f1"]+.001 and
                all(comp_trial[c]["f1"]>=comp_base[c]["f1"]-.01 for c in compositions))
            report["FIT"].update(dict(baseline=base_fit,router=trial_fit,
                baseline_by_composition=comp_base,
                router_by_composition=comp_trial,
                router_selected=selected,fit_compositions=compositions,
                min_global_gain=.001,max_composition_regression=.01,
                actual_retained_expert_count=len(retained),
                observed_masks=dict(sum((Counter(r["mask_histogram"])
                                        for r in fit_scored),Counter()))))
            commit()
            # After frozen FIT-only selection, CAL opens for final diagnosis.
            bank=train_bank(fits) if selected else None
            router=router_model(oof,set(compositions),allowed) if selected else None
            scored=[]
            for i,spec in enumerate(cal_specs):
                X,post,base,truth=inspect(spec,audio,root,heads,selection,
                                          offset,network,policy,choice)
                if selected:
                    probs=head_predictions(bank,X)
                    offers,domains=candidate_offers(X,post,base,probs)
                    desc=frame_descriptors(X,post,base,probs,domains)
                    choice_mask=route(router,desc,allowed)
                    predicted=selected_offer(offers,choice_mask)
                else:
                    choice_mask=np.zeros(len(base),np.int8)
                    predicted=base
                selected_mask=dict(Counter(int(v) for v in choice_mask))
                low=np.broadcast_to((np.arange(49)<8)[None,:],truth.shape)
                harmonic=np.broadcast_to(((np.arange(49)>=8)&(np.arange(49)<20))[None,:],truth.shape)
                high=np.broadcast_to((np.arange(49)>=20)[None,:],truth.shape)
                scored.append(dict(name=spec["name"],
                    composition=spec["composition"],
                    texture=spec["name"].rsplit("_",1)[-1],
                    baseline=head_score(base,truth),
                    trial=head_score(predicted,truth),
                    masks=selected_mask,
                    registers={k:dict(baseline=counts(base,truth,v),
                        trial=counts(predicted,truth,v))
                        for k,v in (("E2_B2",low),("C3_B3",harmonic),("C4_E5",high))}))
                report["cal_recordings_done"]=i+1;commit()
                print("S791_CAL_RECORD",i+1,"/32",spec["name"],flush=True)
    report["CAL"]=dict(baseline=totals(scored,"baseline"),
        frozen=totals(scored,"trial"),
        baseline_by_composition=by_group(scored,"composition","baseline"),
        frozen_by_composition=by_group(scored,"composition","trial"),
        baseline_by_texture=by_group(scored,"texture","baseline"),
        frozen_by_texture=by_group(scored,"texture","trial"),
        by_register={reg:dict(baseline=totals([{"s":v["registers"][reg]["baseline"]}
                for v in scored],"s"),frozen=totals([{"s":v["registers"][reg]["trial"]}
                for v in scored],"s")) for reg in ("E2_B2","C3_B3","C4_E5")},
        observed_masks=dict(sum((Counter(r["masks"]) for r in scored),Counter())))
    report["complete"]=True
    commit()
    print("S791_RESULT",json.dumps(dict(retained=retained,router_selected=selected,
        oof_base=base_fit["f1"],oof_router=trial_fit["f1"],
        cal_base=report["CAL"]["baseline"]["f1"],
        cal_frozen=report["CAL"]["frozen"]["f1"])),flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--s708",type=Path,required=True)
    ap.add_argument("--offset-checkpoint",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    run(ap.parse_args())
