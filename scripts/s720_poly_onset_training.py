"""S720 strictly causal polyphonic Onset+Sustain *training* trial.

Train on FIT compositions ONLY. The fixed frozen S708 S706 checkpoint and
GuidedPolicy remain the reference. Two equal-budget fine-tuning conditions:
plain control and polyphonic/bass-onset positive reweighting. A second
weighted epoch is an explicit predeclared budget. All 49 pitches and all
negative frames remain present, no selective duplication or leakage.

CAL is the ONLY model choice. All six full-emission metrics plus retained
long onsets/previous correct offsets must not regress against frozen S708.
HELD/DEV are already exposed diagnostics; TEST remains unopened.
No model is promoted or installed LIVE automatically.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tempfile
import zipfile
import time

import joblib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from threadpoolctl import threadpool_limits

from causal_note.strict_live import SR
from causal_note.s708_cardinality_selection import GuidedPolicy
from causal_note.s705_temporal_heads import predict_track
from causal_note.s707_causal_release import prepare_hazards, render_release
from scripts.s613_prepare_strict_live import download, labels, digest
from scripts.s613_train_strict_live import real_notes
from scripts.s699_player_held_full_train_K import expanded_train_specs
from scripts.s704_joint_metrics import metric, aggregate, SCORES, target_rank
from scripts.s706_train_sharp_temporal_loops import record,Windows,load_model,FIT_COMPS,CAL_COMPS,HELD_COMPS
from scripts.s704_train_joint_loops import targets
from scripts.s705_train_temporal_loops import setup
from scripts.s708_count_guided_joint_loops import source,evaluate_policy

NAME="s708_selected.pt"


def onset_weights(y,k,*,poly_factor=1.,bass_factor=1.):
    """Extra loss ONLY for a true positive Onset in a polyphonic FIT frame.

    The original BCE positive weights remain [10,2,8]. Sustain positives
    belonging to the SAME Onset's active notes also receive a modest lift
    (20% of extra onset boost); not unrelated pitched negatives.
    """
    if y.ndim!=4 or y.shape[-2:]!=(49,3) or k.shape!=y.shape[:2]:
        raise ValueError("S720 must use native 49-pitch/3-head and true FIT K")
    if poly_factor<1 or bass_factor<1:
        raise ValueError("S720 weights only upweight, never downweight")
    y=y.float()
    poly=(k>=2).unsqueeze(-1)
    pitch=torch.arange(49,device=y.device)[None,None,:]
    reg=torch.where(pitch<=24,torch.as_tensor(bass_factor,device=y.device),1.)
    extra=(poly_factor-1.)*reg*poly
    onset_extra=y[...,0]*extra
    factors=torch.ones_like(y)
    factors[...,0]+=onset_extra
    # Sustain joint learning for the same true attacked pitch (not all held notes).
    factors[...,1]+=.20*onset_extra
    return factors


def training_loss(logits,k_logits,y,k,mask,*,poly_factor=1.,bass_factor=1.):
    """Same original S706 terms and normalization; controlled positive reweight."""
    y=y.float()
    k=k.long()
    positive=logits.new_tensor((10.,2.,8.))
    loss=F.binary_cross_entropy_with_logits(logits,y,pos_weight=positive,reduction="none")
    weights=onset_weights(y,k,poly_factor=poly_factor,bass_factor=bass_factor)
    pitch=(loss*weights).mean((-1,-2))
    count=F.cross_entropy(k_logits.flatten(0,1),k.flatten(),
                          reduction="none").reshape_as(k)
    consistency=(logits[:,:,:,1].sigmoid().sum(-1)-k.float()).square()
    return ((pitch+.25*count+.008*consistency)*mask).sum()/mask.sum()


def guards(baseline,trial,lost):
    return (all(trial[k]>=baseline[k]-1e-12 for k in SCORES) and
            lost["lost_correct_offsets"]==0 and lost["lost_long"]==0 and
            trial["tp"]>=baseline["tp"])


def score_phase(specs,archive,root,heads,selector,offset,models,policy,release):
    scores={name:[] for name in models}
    paired={name:dict(lost_long=0,lost_correct_offsets=0) for name in models}
    tracks=[]
    for i,spec in enumerate(specs):
        item=source(spec,archive,heads,selector,offset,models,706)
        events={}
        for name in models:
            raw=evaluate_policy(item,name,policy)
            events[name],_ = render_release(prepare_hazards(item,raw,offset),
                                            item["ends"],item["samples"],**release)
        # Source emissions are materialized before opening annotation.
        truth=real_notes(labels(spec,root)["notes"],item["samples"]/SR)
        result={name:metric(e,truth,item["ends"]) for name,e in events.items()}
        base=result["S708_frozen"]
        original_offsets=set(base["correct_offset_truth_ids"])
        original_long=set(base["long_onset_truth_ids"])
        for name,row in result.items():
            scores[name].append(row)
            paired[name]["lost_correct_offsets"]+=len(original_offsets-set(row["correct_offset_truth_ids"]))
            paired[name]["lost_long"]+=len(original_long-set(row["long_onset_truth_ids"]))
        tracks.append(dict(name=spec["name"],sha256=item["sha256"],
                           scores={name:{k:result[name][k] for k in SCORES}
                                   for name in models}))
        print("S720 EVALUATE",i+1,len(specs),spec["name"],flush=True)
    return {name:aggregate(rows) for name,rows in scores.items()},paired,tracks


def train_branch(base_checkpoint,manifest,fit_paths,out_path,
                 *,poly_factor,bass_factor,epochs):
    torch.manual_seed(720)
    model,_=load_model(base_checkpoint)
    optimizer=torch.optim.AdamW(model.parameters(),lr=0.00008,weight_decay=.001)
    trainset=Windows(fit_paths)
    assert len(trainset)>0
    history=[]
    for epoch in range(epochs):
        model.train()
        loader=DataLoader(trainset,batch_size=8,shuffle=True,num_workers=0,
                          generator=torch.Generator().manual_seed(720+epoch))
        total=0.
        for x,y,k,valid,mask in loader:
            optimizer.zero_grad(set_to_none=True)
            logits,k_logits=model(x,valid)
            loss=training_loss(logits,k_logits,y,k,mask,
                poly_factor=poly_factor,bass_factor=bass_factor)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),3.)
            optimizer.step()
            total+=float(loss.detach())
        row=dict(epoch=epoch+1,loss_sum=round(total,6),batches=len(loader),
                 poly_factor=poly_factor,bass_factor=bass_factor)
        history.append(row)
        print("S720 FIT",out_path.stem,json.dumps(row),flush=True)
    torch.save(dict(model=model.state_dict(),epoch=epochs,width=model.width,
                    s720_finetuned_from=digest(base_checkpoint.read_bytes()),
                    poly_factor=poly_factor,bass_factor=bass_factor),out_path)
    return history


def run(args):
    if args.out.exists():raise FileExistsError("S720 refuses overwriting output")
    args.out.mkdir(parents=True)
    ref=json.loads((args.s708_dir/"report.json").read_text())
    sha=digest((args.s708_dir/NAME).read_bytes())
    if sha!=ref["checkpoint_sha256"] or ref["source_family"]!=706:
        raise AssertionError("S720 baseline mismatch")
    choice=ref["frozen_candidate"]["choice"]
    frozen_policy=GuidedPolicy(**choice["policy"])
    release=choice["release"]
    assert release==dict(gate=.8,persistence=1,grace=24)
    plan=dict(S708_frozen=dict(epochs=0,poly_factor=1.,bass_factor=1.),
              plain_control_epoch1=dict(epochs=1,poly_factor=1.,bass_factor=1.),
              poly_bass_epoch1=dict(epochs=1,poly_factor=3.,bass_factor=1.5))
    report=dict(schema="S720_FIT_CAUSAL_POLY_ONSET_WEIGHTED_RETRAIN",
        promoted=False,official_live_changed=False,TEST_opened=False,
        CAL_only_selection=True,HELD_DEV_previously_exposed=True,
        base_checkpoint_sha256=sha,policy=asdict(frozen_policy),
        release=release,predeclared_training=plan,
        criterion="Six nonregressive event metrics and zero lost good offsets/long onsets",
        fits_retraining_original_model_only=True)
    def save():
        (args.out/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    save()
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory() as t:
        root=Path(t);(root/"data").mkdir();(root/"fit").mkdir()
        protocol,heads,selector,offset=setup(root,args.offset_checkpoint)
        for key,file in (("GuitarSet_audio","GuitarSet_audio.zip"),
                         ("GuitarSet_annotations","GuitarSet_annotations.zip")):
            src=protocol["guitar_archives"][key]
            download(src["url"],root/"data"/file,src["size"],src["sha256"])
        original=[s for s in protocol["sources"] if s["corpus"]=="GuitarSet" and s["split"]=="train"]
        dev=[s for s in protocol["sources"] if s["corpus"]=="GuitarSet" and s["split"]=="dev"]
        base_model,_=load_model(args.s708_dir/NAME)
        with zipfile.ZipFile(root/"data/GuitarSet_audio.zip") as archive, zipfile.ZipFile(root/"data/GuitarSet_annotations.zip") as ann:
            expanded=expanded_train_specs(original,archive,ann)
            fit=[s for s in expanded if s["composition"] in FIT_COMPS]
            cal=[s for s in expanded if s["composition"] in CAL_COMPS]
            held=[s for s in expanded if s["composition"] in HELD_COMPS]
            assert len(fit)==80 and len(cal)==32 and len(held)==32 and len(dev)==12
            manifest=dict(fit=[s["name"] for s in fit],cal=[s["name"] for s in cal],
                          held=[s["name"] for s in held],dev=[s["name"] for s in dev])
            report["splits"]=manifest
            save()
            for i,spec in enumerate(fit):
                item=record(spec,archive,heads,selector)
                # Exact FIT labels only. Other split labels are forbidden in training.
                truth=real_notes(labels(spec,root)["notes"],item["samples"]/SR)
                y,k=targets(item["decision_ends"],truth)
                path=root/"fit"/f"{i:03d}.joblib"
                joblib.dump(dict(x=item["x"].astype(np.float16),
                                 y=y,k=k,name=spec["name"]),path,compress=0)
                print("S720 PREPARE FIT",i+1,len(fit),spec["name"],flush=True)
            fit_paths=sorted((root/"fit").glob("*.joblib"))
            report["train"]=dict()
            for name,options in plan.items():
                if name=="S708_frozen":continue
                target=args.out/(name+".pt")
                report["train"][name]=train_branch(args.s708_dir/NAME,None,fit_paths,target,
                   poly_factor=options["poly_factor"],bass_factor=options["bass_factor"],
                   epochs=options["epochs"])
                report["train"][name+"_sha256"]=digest(target.read_bytes())
                save()
            models={"S708_frozen":base_model}
            for name in plan:
                if name!="S708_frozen":
                    models[name],_=load_model(args.out/(name+".pt"))
            cal_scores,cal_paired,cal_tracks=score_phase(cal,archive,root,heads,selector,
                                              offset,models,frozen_policy,release)
            old=cal_scores["S708_frozen"]
            eligible=[name for name in models if guards(old,cal_scores[name],cal_paired[name])]
            assert "S708_frozen" in eligible
            best=max(eligible,key=lambda name:target_rank(cal_scores[name]))
            report["CAL"]=dict(scores=cal_scores,paired=cal_paired,tracks=cal_tracks,
                               eligible=eligible,selected=best)
            report["selected_shadow"]=best
            save()
            print("S720 CAL FROZEN",json.dumps(dict(selected=best,eligible=eligible,
              scores={name:{key:cal_scores[name][key] for key in SCORES}
                      for name in cal_scores})),flush=True)
            for phase,specs in (("HELD",held),("DEV",dev)):
                models_chosen={"S708_frozen":models["S708_frozen"]}
                if best!="S708_frozen":models_chosen[best]=models[best]
                scores,paired,tracks=score_phase(specs,archive,root,heads,selector,offset,
                                                  models_chosen,frozen_policy,release)
                for metric_name in SCORES:
                    if abs(scores["S708_frozen"][metric_name]-ref[phase]["candidate"][metric_name])>1e-8:
                        raise AssertionError("Original S708 split changed: "+phase+" "+metric_name)
                compared=scores[best]
                ok=guards(scores["S708_frozen"],compared,paired[best])
                report[phase]=dict(tracks=tracks,
                                   baseline={k:scores["S708_frozen"][k] for k in SCORES},
                                   chosen={k:compared[k] for k in SCORES},
                                   passed=ok,paired=paired)
                save()
                print("S720",phase,"FROZEN",json.dumps(dict(selected=best,passed=ok,
                  old=report[phase]["baseline"],trial=report[phase]["chosen"])),flush=True)
    report["candidate_qualifies"]=bool(best!="S708_frozen" and
                               report["HELD"]["passed"] and report["DEV"]["passed"])
    report["promoted"]=False
    save()
    print("S720 FINAL",json.dumps(dict(selected=best,qualifies=report["candidate_qualifies"],
                                          promoted=False)),flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--s708-dir",required=True,type=Path)
    ap.add_argument("--offset-checkpoint",required=True,type=Path)
    ap.add_argument("--out",required=True,type=Path)
    with threadpool_limits(limits=2):
        run(ap.parse_args())
