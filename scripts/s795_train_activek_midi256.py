"""S795 data-driven heads for missed MIDI voices, FIT80 OOF + CAL32 frozen.

Five independently FIT-trained acoustic specialists. Candidate subsets are
compared on *actual emitted MIDI256 events* (all six S704 metrics). The
head-mask winner is selected exclusively from five composition-held-out FIT
folds, then frozen before CAL32. S793 forbids any source-event delay.

This remains SHADOW/RESEARCH until paired physical CPU/audio-to-MIDI
latency measurements prove no extra latency; no source LIVE deployment.
"""
from __future__ import annotations
import argparse
from collections import defaultdict,Counter
import json
from pathlib import Path
import tempfile
import zipfile

import joblib
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
from scripts.s704_joint_metrics import SCORES,metric,aggregate
from scripts.s705_train_temporal_loops import setup
from scripts.s706_train_sharp_temporal_loops import FIT_COMPS,CAL_COMPS,load_model
from scripts.s708_count_guided_joint_loops import (
    source as s708_source,evaluate_policy as s708_decode)
from causal_note.s707_causal_release import prepare_hazards,render_release
from causal_note.s793_zero_added_latency import SchedulingContract
from causal_note.s795_activek_experts import (
    HEADS,MASKS,CUTS,FEATURES,causal_features,regions,
    onset_targets,sample_head,decode
)

SCHEMA="S795_CAUSAL_MULTIHEAD_ACTIVEK_REAL_MIDI256"
SEED=795
COUNT_KEYS=("tp","fp","fn","frames","k_correct","active_frames","active_k_correct",
            "frame_pitch_tp","frame_pitch_fp","frame_pitch_fn",
            "offset_eligible","offset_correct")
CANDIDATES=tuple([(0,0.)]+[(mask,cut) for mask in MASKS if mask
                                       for cut in CUTS])

def save(path,data):
    path.write_text(json.dumps(data,sort_keys=True,indent=2)+"\n")

def feature_record(spec,archive,root,heads,selection,offset,network,policy,choice):
    """Generate full source MIDI BEFORE opening any annotation."""
    mname="s708_selected.pt"
    item=s708_source(spec,archive,heads,selection,offset,{mname:network},706)
    p,k=item["predictions"][mname]
    post=policy.blend*p+(1-policy.blend)*item["native_post"]
    original=s708_decode(item,mname,policy)
    original,_=render_release(prepare_hazards(item,original,offset),
        item["ends"],item["samples"],**choice["release"])
    X,active=causal_features(item,post,k,original)
    # LABELS ACCESSED ONLY AFTER original MIDI and causal features exist.
    truth=real_notes(labels(spec,root)["notes"],item["samples"]/SR)
    y=onset_targets(truth,item["decision_ends"],active)
    return dict(name=spec["name"],composition=spec["composition"],
        texture=spec["name"].rsplit("_",1)[-1],
        X=X,truth=truth,y=y,events=original,
        ends=item["ends"],decision_ends=item["decision_ends"],
        samples=item["samples"])

def extract_fit_training(row,seed):
    regions_map=regions(row["X"],row["X"][:,:,12]>.5)
    xs={};ys={};counts={}
    for i,name in enumerate(HEADS):
        xx,yy=sample_head(row["X"],row["y"],regions_map[name],
                          seed+37*i)
        xs[name]=xx;ys[name]=yy
        counts[name]=dict(n=int(len(yy)),positive=int(yy.sum()),
                          negative=int(len(yy)-yy.sum()))
    return xs,ys,counts

def train_bank(rows,omit=None):
    models={};audit={}
    for index,name in enumerate(HEADS):
        available=[r for r in rows if r["composition"]!=omit
                   and len(r["train_y"][name])>0]
        if not available:
            models[name]=None
            audit[name]=dict(trained=False,reason="no eligible acoustic training rows")
            continue
        X=np.concatenate([r["train_x"][name] for r in available])
        y=np.concatenate([r["train_y"][name] for r in available])
        if len(np.unique(y))!=2:
            models[name]=None
            audit[name]=dict(trained=False,reason="only one target class",
                             positive=int(y.sum()),n=len(y))
            continue
        model=make_pipeline(StandardScaler(),
            LogisticRegression(C=.12,solver="liblinear",max_iter=145,
                class_weight="balanced",random_state=SEED+index))
        model.fit(X,y)
        models[name]=model
        audit[name]=dict(trained=True,positive=int(y.sum()),negative=int(len(y)-y.sum()),
                          samples=int(len(y)),compositions=sorted({
                            r["composition"] for r in available}))
    return models,audit

def predictions(models,X):
    flat=X.reshape(-1,FEATURES).astype(np.float32)
    output={}
    for name in HEADS:
        model=models[name]
        output[name]=(model.predict_proba(flat)[:,1].reshape(len(X),49)
                      .astype(np.float32)) if model is not None else np.zeros(
                          (len(X),49),np.float32)
    return output

def score_record(record,probs,mask,cut):
    events,change=decode(record["events"],record["decision_ends"],
       record["X"].astype(np.float32),probs,mask,cut,record["samples"])
    score=metric(events,record["truth"],record["ends"])
    return score,change

def summary(rows,axis):
    return aggregate([r[axis] for r in rows])

def split_stats(rows,key,axis):
    return {value:summary([r for r in rows if r[key]==value],axis)
            for value in sorted(set(r[key] for r in rows))}

def safe_to_select(base,trial,comps_base,comps_trial):
    # Do not sacrifice ANY of the other five metrics, even for +20pt K.
    return (trial["active_exact_k"]>=base["active_exact_k"]+.001
        and all(trial[k]>=base[k]-1e-12 for k in SCORES
                if k!="active_exact_k")
        and all(comps_trial[c]["active_exact_k"]>=
                comps_base[c]["active_exact_k"]-.005 for c in comps_base))

def format_candidate(mask,cut):
    return f"mask{mask:02d}_cut{cut:.2f}" if mask else "source_identity"

def run(args):
    if args.out.exists():raise FileExistsError("S795 experiment output immutable")
    args.out.mkdir(parents=True)
    signed=json.loads((args.s708/"report.json").read_text())
    sha=digest((args.s708/"s708_selected.pt").read_bytes())
    if sha!=signed["checkpoint_sha256"] or signed["schema"]!="S708_K_GUIDED_S706_CAL_LOOPS":
        raise AssertionError("signed S708 artifact mismatch")
    choice=signed["frozen_candidate"]["choice"]
    if choice["release"]!=dict(gate=.8,persistence=1,grace=24):
        raise AssertionError("S707 NoteOff changed")
    SchedulingContract().validate()
    network=load_model(args.s708/"s708_selected.pt")[0]
    torch.set_num_threads(2)
    policy=GuidedPolicy(**choice["policy"])
    report=dict(schema=SCHEMA,complete=False,promoted=False,
       signed_source_sha256=sha,decision_clock_samples=1024,
       metric_clock_samples=256,heads=list(HEADS),
       candidate_masks=list(MASKS),candidate_policies=len(CANDIDATES),
       fitted_source_head_count=0,retained_head_count=0,
       selected_candidate="source_identity",zero_added_scheduling=True,
       source_midi_events_untouched=True,
       physical_microphone_latency_measured=False,
       production_LIVE_modified=False,HELD_opened=False,TEST_opened=False,
       cal_used_for_training=False,cal_used_for_selection=False,
       fit_done=0,cal_done=0,FIT={},CAL={})
    def checkpoint():save(args.out/"report.json",report)
    checkpoint()
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp);(root/"data").mkdir()
        protocol,heads,selection,offset=setup(root,args.offset_checkpoint)
        for key,name in (("GuitarSet_audio","GuitarSet_audio.zip"),
                         ("GuitarSet_annotations","GuitarSet_annotations.zip")):
            val=protocol["guitar_archives"][key]
            download(val["url"],root/"data"/name,val["size"],val["sha256"])
        raw=[s for s in protocol["sources"]
             if s["corpus"]=="GuitarSet" and s["split"]=="train"]
        with zipfile.ZipFile(root/"data/GuitarSet_audio.zip") as audio,\
             zipfile.ZipFile(root/"data/GuitarSet_annotations.zip") as ann:
            expanded=expanded_train_specs(raw,audio,ann)
            fit=[s for s in expanded if s["composition"] in FIT_COMPS]
            cal=[s for s in expanded if s["composition"] in CAL_COMPS]
            if len(fit)!=80 or len(cal)!=32 or set(FIT_COMPS)&set(CAL_COMPS):
                raise AssertionError("incorrect FIT80/CAL32 composition isolation")
            rows=[]
            for i,spec in enumerate(fit):
                row=feature_record(spec,audio,root,heads,selection,offset,
                                    network,policy,choice)
                xs,ys,cnt=extract_fit_training(row,SEED+i*71)
                cache=root/f"fit_{i:03d}.joblib"
                # Inference ABI is float16 cached → float32 for the same model.
                row["X"]=row["X"].astype(np.float16)
                joblib.dump(row,cache,compress=0)
                rows.append(dict(name=row["name"],composition=row["composition"],
                                 cache=cache,train_x=xs,train_y=ys,counts=cnt))
                report["fit_done"]=i+1;checkpoint()
                print("S795_FIT_SOURCE",i+1,"/80",spec["name"],flush=True)
            folds=sorted(set(row["composition"] for row in rows))
            if len(folds)!=5:raise AssertionError("five independent FIT compositions required")
            records_by_candidate={format_candidate(m,c):[] for m,c in CANDIDATES}
            trained_folds={}
            for fold in folds:
                models,training=train_bank(rows,omit=fold)
                trained_folds[fold]=training
                for row in (r for r in rows if r["composition"]==fold):
                    record=joblib.load(row["cache"])
                    P=predictions(models,record["X"])
                    for mask,cut in CANDIDATES:
                        key=format_candidate(mask,cut)
                        score,change=score_record(record,P,mask,cut)
                        records_by_candidate[key].append(dict(
                            composition=record["composition"],
                            texture=record["texture"],name=record["name"],
                            metric=score,changes=change))
                print("S795_FIT_OOF_COMPOSITION",fold,flush=True)
            control=summary(records_by_candidate["source_identity"],"metric")
            control_comps=split_stats(records_by_candidate["source_identity"],
                                      "composition","metric")
            results={}
            for mask,cut in CANDIDATES:
                key=format_candidate(mask,cut)
                entries=records_by_candidate[key]
                trial=summary(entries,"metric")
                comp=split_stats(entries,"composition","metric")
                eligible=(mask==0 or safe_to_select(control,trial,control_comps,comp))
                results[key]=dict(mask=int(mask),cut=float(cut),eligible=eligible,
                    metrics={k:trial[k] for k in SCORES},
                    six_metric_counts={k:int(trial[k]) for k in COUNT_KEYS},
                    by_composition={c:{k:v[k] for k in SCORES}
                                    for c,v in comp.items()},
                    synthetic_note_ons=sum(x["changes"]["new_onsets"] for x in entries))
            admissible=[n for n,v in results.items() if v["eligible"]]
            chosen=max(admissible,key=lambda n:(
                results[n]["metrics"]["active_exact_k"],
                -results[n]["synthetic_note_ons"]))
            selected=results[chosen]
            kept=[name for i,name in enumerate(HEADS)
                  if selected["mask"]&(1<<i)]
            report["selected_candidate"]=chosen
            report["retained_head_count"]=len(kept)
            report["FIT"]=dict(compositions=folds,training=trained_folds,
                baseline={k:control[k] for k in SCORES},
                candidate_results=results,selected=chosen,kept_experts=kept,
                source_counts={k:int(control[k]) for k in COUNT_KEYS},
                no_other_metric_regression_required=True)
            checkpoint()
            # CAL is now allowed to open ONLY after the FIT policy froze.
            models,final_training=train_bank(rows)
            report["fitted_source_head_count"]=sum(
                int(v["trained"]) for v in final_training.values())
            report["FIT"]["final_training"]=final_training
            checkpoint()
            calrows=[]
            for i,spec in enumerate(cal):
                record=feature_record(spec,audio,root,heads,selection,offset,
                                        network,policy,choice)
                P=predictions(models,record["X"]) if selected["mask"] else {
                     name:np.zeros((len(record["X"]),49),np.float32) for name in HEADS}
                original=metric(record["events"],record["truth"],record["ends"])
                scored,changes=score_record(record,P,selected["mask"],selected["cut"])
                calrows.append(dict(name=record["name"],
                    composition=record["composition"],texture=record["texture"],
                    baseline=original,trial=scored,changes=changes))
                report["cal_done"]=i+1;checkpoint()
                print("S795_CAL_REAL_MIDI256",i+1,"/32",spec["name"],flush=True)
    source=summary(calrows,"baseline");trial=summary(calrows,"trial")
    old=signed["frozen_candidate"]["CAL"]
    for k in COUNT_KEYS:
        if source[k]!=old[k]:
            raise AssertionError(f"S795 signed source count mismatch {k}: {source[k]} != {old[k]}")
    for k in SCORES:
        if not np.isclose(source[k],old[k],atol=1e-12,rtol=0):
            raise AssertionError("S795 signed source metric mismatch "+k)
    report["CAL"]=dict(
        original={k:source[k] for k in SCORES},
        candidate={k:trial[k] for k in SCORES},
        original_counts={k:int(source[k]) for k in COUNT_KEYS},
        trial_counts={k:int(trial[k]) for k in COUNT_KEYS},
        original_by_composition={c:{k:v[k] for k in SCORES}
            for c,v in split_stats(calrows,"composition","baseline").items()},
        candidate_by_composition={c:{k:v[k] for k in SCORES}
            for c,v in split_stats(calrows,"composition","trial").items()},
        by_texture_source={c:{k:v[k] for k in SCORES}
            for c,v in split_stats(calrows,"texture","baseline").items()},
        by_texture_candidate={c:{k:v[k] for k in SCORES}
            for c,v in split_stats(calrows,"texture","trial").items()},
        new_midi_note_ons=int(sum(x["changes"]["new_onsets"] for x in calrows)),
        official_source_parity=True,
        six_metric_no_regression=all(trial[k]>=source[k]-1e-12 for k in SCORES),
        active_k_over_80=trial["active_exact_k"]>.8)
    report["complete"]=True
    checkpoint()
    print("S795_RESULT",json.dumps(dict(selected=chosen,head_count=len(kept),
       trained_count=report["fitted_source_head_count"],
       source=report["CAL"]["original"],candidate=report["CAL"]["candidate"],
       six_metric_nonregression=report["CAL"]["six_metric_no_regression"],
       new_onsets=report["CAL"]["new_midi_note_ons"],
       promoted=False)),flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--s708",type=Path,required=True)
    ap.add_argument("--offset-checkpoint",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    run(ap.parse_args())
