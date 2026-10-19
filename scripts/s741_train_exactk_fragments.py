"""S741: genuine FIT80/CAL32 Exact-K fragment selector, shadow research.

Five real causal K0..K6 source distributions, original S708 H0 and K
always kept. Exact-K is trained separately, one 32-way mask per fragment.
Requires FULL MIDI causal NoteOn/NoteOff scoring and strict six-metric gate.
No claims of >80% without real completed reports, no LIVE or TEST edits.
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
from torch import nn
from torch.nn import functional as F
from threadpoolctl import threadpool_limits

from causal_note.s736_fragment_mask import (
    MASKS, FragmentMaskRouter, choose_masks)
from causal_note.s740_multitarget_dynamic import (
    target_descriptors, build_offers, apply_domain, PromotionGate)
from causal_note.s741_exactk_sources import frozen_acoustic_exactk_heads
from scripts.s737_train_fragment_onset_loops import (
    training_audit, audit_all_fit, save, STEPS, CAL_CONFIDENCE,
    MAX_FIT_BLOCKS, REFERENCE_NAME)
from scripts.s740_train_posterior_specialists import one_track
from scripts.s613_prepare_strict_live import download, labels, digest
from scripts.s613_train_strict_live import real_notes
from scripts.s699_player_held_full_train_K import expanded_train_specs
from scripts.s704_train_joint_loops import targets
from scripts.s704_joint_metrics import SCORES, metric, aggregate
from scripts.s706_train_sharp_temporal_loops import (
    FIT_COMPS, CAL_COMPS, HELD_COMPS, load_model)
from scripts.s705_train_temporal_loops import setup
from scripts.s731_auditory_scene_calibration import emit
from scripts.s735_audit_32_onset_combinations import load_s734
from causal_note.s708_cardinality_selection import GuidedPolicy
from causal_note.strict_live import SR

SEEDS=(741,742)


def one_k_track(spec,audio,root,heads,selection,offset,network,
                policy,release,h5):
    track=one_track(spec,audio,root,heads,selection,offset,network,
                    policy,release,h5,"sustain")
    k, offers=frozen_acoustic_exactk_heads(
        track["k"],track["item"]["native_k"],track["post"],
        track["baseline"],track["item"]["decision_ends"])
    avail=np.ones_like(k,dtype=bool)
    track["base_k"]=k
    track["k_heads"]=offers
    track["descriptors"]=target_descriptors(
        k,offers,avail,fusion="mean")
    return track


def k_supervision(track,truth,seed):
    _,true_k=targets(track["item"]["decision_ends"],truth)
    if len(true_k)!=len(track["k"]):
        raise AssertionError("Missing true K FIT labels")
    k=track["k"]; h=track["k_heads"]
    offers=build_offers(k,h,np.ones_like(k,bool),fusion="mean")
    offers=offers/offers.sum(-1,keepdims=True).clip(min=1e-8)
    base_class=k.argmax(-1)
    proposed=offers.argmax(-1)
    fix=(proposed==true_k[:,None])&(base_class[:,None]!=true_k[:,None])
    regression=(proposed!=true_k[:,None])&(base_class[:,None]==true_k[:,None])
    true_probability=offers[
        np.arange(len(k))[:,None],np.arange(32)[None,:],
        true_k[:,None]]
    original_probability=k[np.arange(len(k)),true_k][:,None]
    benefit=4.*fix-5.*regression+(true_probability-original_probability)
    sizes=np.array([int(z).bit_count() for z in MASKS])
    benefit-=.025*(sizes[None,:]-1)
    best=benefit.argmax(-1)
    action=np.where(
        benefit[np.arange(len(k)),best]>.01,best+1,0).astype(np.int64)
    rng=np.random.default_rng(seed)
    priority=(action>0)|(base_class!=true_k)
    selected=np.flatnonzero(priority)
    others=np.flatnonzero(~priority)
    if len(selected)>MAX_FIT_BLOCKS//2:
        selected=np.sort(rng.choice(selected,MAX_FIT_BLOCKS//2,replace=False))
    limit=MAX_FIT_BLOCKS-len(selected)
    if len(others)>limit:
        others=np.sort(rng.choice(others,limit,replace=False))
    ix=np.sort(np.r_[selected,others])
    if not len(ix):
        raise AssertionError("No FIT K fragments")
    return dict(features=track["features"][ix],
                descriptors=track["descriptors"][ix],
                k=k[ix],source=track["item"]["spec"]["composition"],
                target=action[ix],audit_true_k=true_k[ix],
                risk=np.stack((fix[ix],regression[ix]),-1).astype(np.float32),
                coverage=(np.abs(true_probability[ix]-
                                original_probability[ix])>.02).astype(np.float32),
                sampled=int(len(ix)),rescuable=int((action>0).sum()))


def tensors(sample,audit,mean,std):
    n=len(sample["k"])
    return (torch.from_numpy(((sample["features"]-mean)/std).astype(np.float32)),
            torch.from_numpy(sample["descriptors"].astype(np.float32)),
            torch.from_numpy(np.broadcast_to(audit,(n,32,7,3)).copy()),
            torch.from_numpy(sample["k"].astype(np.float32)),
            torch.ones((n,32,7),dtype=torch.bool),
            torch.from_numpy(sample["target"].copy()).long(),
            torch.from_numpy(sample["risk"].astype(np.float32)))


def train_one(seed,examples,audits,mean,std,callback):
    torch.manual_seed(seed)
    model=FragmentMaskRouter()
    opt=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=.002)
    prepared=[tensors(x,a,mean,std) for x,a in zip(examples,audits)]
    positive=sum(int(torch.count_nonzero(d[5])) for d in prepared)
    total=sum(len(d[5]) for d in prepared)
    boost=float(min(16,max(1,((total-positive)/max(positive,1))**.5)))
    for epoch in range(1,max(STEPS)+1):
        model.train()
        rng=np.random.default_rng(seed+epoch)
        costs=[]
        for s in rng.permutation(len(prepared)):
            packed=prepared[s]
            shuffled=rng.permutation(len(packed[0]))
            for start in range(0,len(shuffled),256):
                ix=shuffled[start:start+256]
                x,d,a,k,compat,target,risk=[v[ix] for v in packed]
                out=model(x,d,a,k,compat)
                loss_each=F.cross_entropy(
                    out["action_logits"],target,reduction="none")
                w=torch.where(target!=0,torch.full_like(loss_each,boost),torch.ones_like(loss_each))
                main=(loss_each*w).sum()/w.sum().clamp_min(1.)
                auxiliary=F.binary_cross_entropy(
                    torch.stack([out["p_correction"],out["p_regression"]],-1),
                    risk)
                loss=main+.05*auxiliary
                opt.zero_grad(set_to_none=True);loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),2.)
                opt.step()
                costs.append(float(loss.detach()))
        if epoch in STEPS:
            callback(model,seed,epoch,np.mean(costs))
    return model


def evaluate(tracks,model,mean,std,audit,confidence):
    old_rows=[];new_rows=[];hist=Counter()
    lost_off=0;lost_long=0
    for track in tracks:
        n=len(track["k"])
        masks,_,_=choose_masks(
            model,track["features"],track["descriptors"],
            np.broadcast_to(audit,(n,32,7,3)),track["k"],
            np.ones((n,32,7),bool),mean,std,min_confidence=confidence)
        proposed=apply_domain(
            "exact_k",track["k"],track["k_heads"],
            np.ones_like(track["k"],bool),masks)
        emitted=emit(track["item"],track["post"],proposed,
                     track["policy"],track["offset"],track["release"])
        truth=track["truth"]
        old=metric(track["baseline"],truth,track["item"]["ends"])
        new=metric(emitted,truth,track["item"]["ends"])
        old_rows.append(old);new_rows.append(new)
        lost_off+=len(set(old["correct_offset_truth_ids"])-
                      set(new["correct_offset_truth_ids"]))
        lost_long+=len(set(old["long_onset_truth_ids"])-
                       set(new["long_onset_truth_ids"]))
        hist.update(str(int(z)) for z in masks)
    original=aggregate(old_rows);actual=aggregate(new_rows)
    gate=PromotionGate(original,actual,lost_off,lost_long,
                       True,True,True)
    return dict(
        reference={m:float(original[m]) for m in SCORES},
        actual={m:float(actual[m]) for m in SCORES},
        delta={m:float(actual[m]-original[m]) for m in SCORES},
        lost_correct_offsets=lost_off,lost_long_onsets=lost_long,
        safe=gate.safe() and actual["tp"]>=original["tp"],
        all_gt80=gate.above_80(),mask_histogram=dict(hist))


def run(args):
    if args.out.exists():
        raise FileExistsError("Immutable K experiment output already exists")
    args.out.mkdir(parents=True)
    frozen=json.loads((args.s708_dir/"report.json").read_text())
    sha=digest((args.s708_dir/"s708_selected.pt").read_bytes())
    if sha!=frozen["checkpoint_sha256"]:
        raise AssertionError("S708 frozen signed K mismatch")
    h5_report=json.loads(args.s734_report.read_text())
    if h5_report["base_sha256"]!=sha or not h5_report["complete"]:
        raise AssertionError("S734 FIT H5 mismatch")
    torch.set_num_threads(2)
    network=load_model(args.s708_dir/"s708_selected.pt")[0]
    h5=load_s734(args.s734_checkpoint,sha)
    policy=GuidedPolicy(**frozen["frozen_candidate"]["choice"]["policy"])
    release=frozen["frozen_candidate"]["choice"]["release"]
    if release!=dict(gate=.8,persistence=1,grace=24):
        raise AssertionError("Original NoteOff rule changed")
    report=dict(schema="S741_REAL_AUDIO_K_32_SUBSET_FIT_CAL",
                target="exact_k",base_signed_sha256=sha,
                completed=False,selected=REFERENCE_NAME,promoted=False,
                LIVE_changed=False,TEST_opened=False,
                training="FIT80 only",CAL_rounds=[],FIT={},
                achieved_all_six_gt80=False,
                objective="all six metrics >80%; zero MIDI regression")
    save(args.out/"report.json",report)
    with tempfile.TemporaryDirectory() as td:
        root=Path(td);(root/"data").mkdir()
        proto,heads,selection,offset=setup(root,args.offset_checkpoint)
        for key,name in (("GuitarSet_audio","GuitarSet_audio.zip"),
                         ("GuitarSet_annotations","GuitarSet_annotations.zip")):
            x=proto["guitar_archives"][key]
            download(x["url"],root/"data"/name,x["size"],x["sha256"])
        specs=[s for s in proto["sources"] if
               s["corpus"]=="GuitarSet" and s["split"]=="train"]
        dev=[s for s in proto["sources"] if
             s["corpus"]=="GuitarSet" and s["split"]=="dev"]
        with zipfile.ZipFile(root/"data/GuitarSet_audio.zip") as audio, \
             zipfile.ZipFile(root/"data/GuitarSet_annotations.zip") as annotations:
            expanded=expanded_train_specs(specs,audio,annotations)
            groups={"FIT":[s for s in expanded if s["composition"] in FIT_COMPS],
                    "CAL":[s for s in expanded if s["composition"] in CAL_COMPS],
                    "HELD":[s for s in expanded if s["composition"] in HELD_COMPS],
                    "DEV":dev}
            if tuple(len(groups[z]) for z in ("FIT","CAL","HELD","DEV")) \
               != (80,32,32,12):
                raise AssertionError("Unexpected composition split")
            fits=[]
            for i,spec in enumerate(groups["FIT"]):
                row=one_k_track(spec,audio,root,heads,selection,offset,network,
                                policy,release,h5)
                truth=real_notes(labels(spec,root)["notes"],
                                 row["item"]["samples"]/SR)
                fit=k_supervision(row,truth,741+i)
                fits.append(fit)
                print("S741 FIT",i+1,80,"rescuable",fit["rescuable"],flush=True)
            both=np.concatenate([d["features"] for d in fits])
            mean=both.mean(axis=0)
            std=np.maximum(both.std(axis=0),.02)
            oof=[training_audit(fits,i) for i in range(80)]
            audit=audit_all_fit(fits)
            report["FIT"]=dict(
                sampled_blocks=sum(x["sampled"] for x in fits),
                rescuable=sum(x["rescuable"] for x in fits),
                compositions=sorted(set(x["source"] for x in fits)))
            save(args.out/"report.json",report)

            def load_group(phase):
                tracks=[]
                for i,spec in enumerate(groups[phase]):
                    x=one_k_track(spec,audio,root,heads,selection,offset,
                                  network,policy,release,h5)
                    x["truth"]=real_notes(labels(spec,root)["notes"],
                                          x["item"]["samples"]/SR)
                    x["policy"]=policy;x["offset"]=offset;x["release"]=release
                    tracks.append(x)
                    print("S741 LOAD",phase,i+1,len(groups[phase]),flush=True)
                return tracks

            cal=load_group("CAL")
            historical=aggregate([metric(x["baseline"],x["truth"],x["item"]["ends"])
                                  for x in cal])
            for metric_name in SCORES:
                if abs(historical[metric_name] -
                       frozen["frozen_candidate"]["CAL"][metric_name])>1e-8:
                    raise AssertionError("K baseline CAL replay mismatch")
            safe=[]
            for seed in SEEDS:
                def callback(model,s,epoch,fit_loss):
                    for conf in CAL_CONFIDENCE:
                        value=evaluate(cal,model,mean,std,audit,conf)
                        value.update(seed=s,epoch=epoch,confidence=conf,
                                     FIT_loss=float(fit_loss))
                        report["CAL_rounds"].append(value)
                        print("S741 REAL K CAL",json.dumps({
                            "seed":s,"epoch":epoch,"confidence":conf,
                            "actual":value["actual"],"safe":value["safe"],
                            "all_gt80":value["all_gt80"]}),flush=True)
                        if value["safe"] and any(
                                value["delta"][m]>1e-10 for m in SCORES):
                            name=f"s741_K_seed{s}_epoch{epoch}.pt"
                            torch.save(dict(model=model.state_dict(),
                                base_sha256=sha,mean=mean.tolist(),
                                std=std.tolist(),seed=s,epoch=epoch),args.out/name)
                            safe.append((value,name))
                    save(args.out/"report.json",report)
                train_one(seed,fits,oof,mean,std,callback)
            if safe:
                best,name=max(safe,key=lambda item:(
                    item[0]["all_gt80"],min(item[0]["delta"].values()),
                    sum(item[0]["actual"].values())))
                model=FragmentMaskRouter()
                state=torch.load(args.out/name,map_location="cpu",weights_only=True)
                if state["base_sha256"]!=sha:
                    raise AssertionError("K checkpoint wrong frozen source")
                model.load_state_dict(state["model"],strict=True)
                report["selected"]=name
                report["frozen_on_CAL"]=dict(
                    seed=best["seed"],epoch=best["epoch"],
                    confidence=best["confidence"])
                for phase in ("HELD","DEV"):
                    rows=load_group(phase)
                    value=evaluate(rows,model,mean,std,audit,best["confidence"])
                    expected=frozen[phase]["candidate"]
                    for m in SCORES:
                        if abs(value["reference"][m]-expected[m])>1e-8:
                            raise AssertionError("K benchmark replay mismatch "+m)
                    report[phase]=value
                    save(args.out/"report.json",report)
            else:
                report["CAL_no_safe_improvement"]=True
            report["completed"]=True
            report["achieved_all_six_gt80"]=bool(
                safe and any(row["safe"] and row["all_gt80"] for row,_ in safe)
                and all(report.get(phase,{}).get("safe",False)
                        for phase in ("HELD","DEV")))
            save(args.out/"report.json",report)
            print("S741_FINAL",json.dumps({
                "selected":report["selected"],
                "rounds":len(report["CAL_rounds"]),
                "six_gt80":report["achieved_all_six_gt80"]}),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--s708-dir",type=Path,required=True)
    p.add_argument("--s734-checkpoint",type=Path,required=True)
    p.add_argument("--s734-report",type=Path,required=True)
    p.add_argument("--offset-checkpoint",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    with threadpool_limits(limits=2):
        run(args)
