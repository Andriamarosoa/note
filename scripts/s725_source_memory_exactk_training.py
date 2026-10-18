"""S725 causal multi-head NOTE ablation: new polyphonic evidence WITH Exact-K.

Unlike an independent K detector or a pure Onset threshold tweak, this trains
the original 3-head + K architecture jointly on an append-only 32D acoustic
signal, retaining the original 288D checkpoint weights exactly at epoch 0.

Compare independent-pitch harmonic novelty and K-guided weak attacks, same
FIT-trained checkpoint. CAL chooses among epoch-1/2 with K guide ON/OFF,
plus *exact frozen* S708. HELD/DEV are historically exposed diagnostics;
TEST unopened. No LIVE wiring or automatic promotion.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict,replace
import json
from pathlib import Path
import tempfile
import zipfile
import time

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader
from threadpoolctl import threadpool_limits

from causal_note.strict_live import SR
from causal_note.s725_source_memory import augmented_track,disabled_memory_track,BASE,EXTRA,TOTAL
from causal_note.s723_polyphonic_novelty import init_from_s708
from causal_note.s705_temporal_heads import TemporalHeads,predict_track
from causal_note.s708_cardinality_selection import GuidedPolicy,decode_guided
from causal_note.s707_causal_release import prepare_hazards,render_release
from scripts.s613_prepare_strict_live import download,labels,digest
from scripts.s613_train_strict_live import real_notes
from scripts.s699_player_held_full_train_K import expanded_train_specs
from scripts.s704_joint_metrics import metric,aggregate,SCORES,target_rank
from scripts.s706_train_sharp_temporal_loops import record,Windows,load_model,FIT_COMPS,CAL_COMPS,HELD_COMPS
from scripts.s705_train_temporal_loops import setup
from scripts.s704_train_joint_loops import targets
from scripts.s720_poly_onset_training import training_loss

SOURCE="s708_selected.pt"


def score_events(item,p,k,policy,offset,release):
    blended=policy.blend*p+(1-policy.blend)*item["native_post"]
    src=decode_guided(item["decision_ends"],blended,k,policy,item["samples"])
    events,_=render_release(prepare_hazards(item,src,offset),
                            item["ends"],item["samples"],**release)
    return events


def guard(old,new,paired):
    return (all(new[key]>=old[key]-1e-12 for key in SCORES)
            and paired["lost_correct_offsets"]==0
            and paired["lost_long"]==0
            and new["tp"]>=old["tp"])


def phase_scores(specs,archive,root,heads,selector,offset,
                 old_model,new_models,policies,release,phase):
    rows={name:[] for name in policies}
    paired={name:dict(lost_correct_offsets=0,lost_long=0) for name in policies}
    tracks=[]
    for i,spec in enumerate(specs):
        item=record(spec,archive,heads,selector,offset)
        original=item.pop("x")
        item.pop("cx")
        item["native_features"]=original[:,:,:126].copy()
        aug=augmented_track(original)
        pred_base=predict_track(old_model,original)
        predictions={}
        for key,model in new_models.items():
            if key.startswith("no_memory"):
                xx=disabled_memory_track(original,model.mean[BASE:].cpu().numpy())
            else:
                xx=aug
            predictions[key]=predict_track(model,xx)
        events={}
        for name,(model_name,policy) in policies.items():
            p,k=pred_base if model_name=="S708_frozen" else predictions[model_name]
            events[name]=score_events(item,p,k,policy,offset,release)
        # Ground truth is used ONLY after every candidate event stream is final.
        truth=real_notes(labels(spec,root)["notes"],item["samples"]/SR)
        measured={key:metric(e,truth,item["ends"]) for key,e in events.items()}
        base=measured["S708_frozen"]
        old_off=set(base["correct_offset_truth_ids"])
        old_long=set(base["long_onset_truth_ids"])
        for name,row in measured.items():
            rows[name].append(row)
            paired[name]["lost_correct_offsets"]+=len(old_off-set(row["correct_offset_truth_ids"]))
            paired[name]["lost_long"]+=len(old_long-set(row["long_onset_truth_ids"]))
        tracks.append(dict(name=spec["name"],sha256=item["sha256"],
             measures={key:{m:row[m] for m in SCORES} for key,row in measured.items()},
             tp_fp_fn={key:{m:row[m] for m in ("tp","fp","fn")} for key,row in measured.items()}))
        print("S725",phase,i+1,len(specs),spec["name"],flush=True)
    return {key:aggregate(scores) for key,scores in rows.items()},paired,tracks


def make_policies(base,choices):
    out={"S708_frozen":("S708_frozen",base)}
    for name in choices:
        out[f"{name}_Kguided"]=(name,base)
        # Disable K-gated weak candidate guidance, KEEP neural K
        # embeddings/learned count joint-training and predicted K.
        out[f"{name}_KweakOFF"]=(name,replace(base,guide_confidence=1.1))
    return out


class PairedWindows(Windows):
    """Same FIT frames/targets, control channels equal FIT means."""
    def __init__(self,paths,extra_mean,enable_memory):
        super().__init__(paths)
        self.extra_mean=torch.as_tensor(extra_mean,dtype=torch.float16).float()
        self.enable_memory=enable_memory

    def __getitem__(self,index):
        x,y,k,valid,mask=super().__getitem__(index)
        if not self.enable_memory:
            x[:,:,BASE:]=self.extra_mean
        return x,y,k,valid,mask


def train_one(real_original,fit_paths,mean,scale,output,*,enable_memory,epoch_budgets=(1,2)):
    torch.manual_seed(725)
    model=init_from_s708(real_original,mean,scale)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.00008,weight_decay=.001)
    dataset=PairedWindows(fit_paths,mean,enable_memory)
    name="with_memory" if enable_memory else "no_memory"
    results={}
    for epoch in range(1,max(epoch_budgets)+1):
        model.train()
        loader=DataLoader(dataset,batch_size=8,shuffle=True,num_workers=0,
             generator=torch.Generator().manual_seed(725+epoch))
        losses=[]
        for x,y,k,valid,mask in loader:
            optimizer.zero_grad(set_to_none=True)
            p,counts=model(x,valid)
            loss=training_loss(p,counts,y,k,mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),3.)
            optimizer.step()
            losses.append(float(loss.detach()))
        print("S725 TRAIN",json.dumps(dict(mode=name,epoch=epoch,batches=len(losses),
                     mean_loss=float(np.mean(losses)))),flush=True)
        if epoch in epoch_budgets:
            path=output/f"{name}_epoch{epoch}.pt"
            torch.save(dict(model=model.state_dict(),epoch=epoch,width=model.width,
                 source_checkpoint="S708_signed_original",
                 source_memory=enable_memory,feature_abi="320D_past_causal_ownership"),path)
            results[f"{name}_epoch{epoch}"]=dict(path=path.name,
                 sha256=digest(path.read_bytes()),epochs=epoch,
                 memory=enable_memory,mean_loss=float(np.mean(losses)))
    return results


def train(real_original,fit_paths,mean,scale,output):
    control=train_one(real_original,fit_paths,mean,scale,output,enable_memory=False)
    memory=train_one(real_original,fit_paths,mean,scale,output,enable_memory=True)
    return {**control,**memory}


def run(args):
    if args.out.exists():raise FileExistsError("S725 immutable experiment already exists")
    args.out.mkdir(parents=True)
    historical=json.loads((args.s708_dir/"report.json").read_text())
    original_sha=digest((args.s708_dir/SOURCE).read_bytes())
    if original_sha!=historical["checkpoint_sha256"] or historical["source_family"]!=706:
        raise ValueError("S725 requires frozen original S708")
    original_policy=GuidedPolicy(**historical["frozen_candidate"]["choice"]["policy"])
    release=historical["frozen_candidate"]["choice"]["release"]
    assert release==dict(gate=.8,persistence=1,grace=24)
    report=dict(schema="S725_CAUSAL_SOURCE_MEMORY_PITCH_EXACTK_CONTROLLED_ABLATION",
        promoted=False,official_live_changed=False,TEST_opened=False,
        HELD_DEV_previously_exposed=True,CAL_only_model_selection=True,
        s708_sha256=original_sha,new_feature_dims=EXTRA,
        old_feature_dims=BASE,total_feature_dims=TOTAL,
        policy=asdict(original_policy),release=release,
        learned_ExactK_conditioning_preserved=True,
        guidance_ablation="Kguided vs KweakOFF; K still conditions neural pitch head",
        training_ablation="matched no_memory vs with_memory, identical FIT seed and S708 weights",
        criterion="all six metrics >= frozen S708, no former correct offsets/long notes lost")
    def save():
        (args.out/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    save()
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory() as t:
        root=Path(t);(root/"data").mkdir();(root/"fit").mkdir()
        protocol,heads,selection,offset=setup(root,args.offset_checkpoint)
        for key,file in (("GuitarSet_audio","GuitarSet_audio.zip"),
                         ("GuitarSet_annotations","GuitarSet_annotations.zip")):
            spec=protocol["guitar_archives"][key]
            download(spec["url"],root/"data"/file,spec["size"],spec["sha256"])
        original=[s for s in protocol["sources"] if s["corpus"]=="GuitarSet" and s["split"]=="train"]
        dev=[s for s in protocol["sources"] if s["corpus"]=="GuitarSet" and s["split"]=="dev"]
        old_model,_=load_model(args.s708_dir/SOURCE)
        with zipfile.ZipFile(root/"data/GuitarSet_audio.zip") as audio,zipfile.ZipFile(root/"data/GuitarSet_annotations.zip") as ann:
            expanded=expanded_train_specs(original,audio,ann)
            fit=[s for s in expanded if s["composition"] in FIT_COMPS]
            cal=[s for s in expanded if s["composition"] in CAL_COMPS]
            held=[s for s in expanded if s["composition"] in HELD_COMPS]
            assert (len(fit),len(cal),len(held),len(dev))==(80,32,32,12)
            report["splits"]={key:[s["name"] for s in specs] for key,specs
                          in (("FIT",fit),("CAL",cal),("HELD",held),("DEV",dev))}
            save()
            sums=np.zeros(EXTRA,np.float64);squares=np.zeros(EXTRA,np.float64);count=0
            for i,spec in enumerate(fit):
                item=record(spec,audio,heads,selection)
                truth=real_notes(labels(spec,root)["notes"],item["samples"]/SR)
                y,k=targets(item["decision_ends"],truth)
                x=augmented_track(item["x"]).astype(np.float16)
                z=x[::8,:,:][:,:,BASE:].astype(np.float64).reshape(-1,EXTRA)
                sums+=z.sum(0);squares+=(z*z).sum(0);count+=len(z)
                joblib.dump(dict(x=x,y=y,k=k,name=spec["name"]),
                    root/"fit"/f"{i:03d}.joblib",compress=0)
                print("S725 FIT PREPARE",i+1,len(fit),spec["name"],flush=True)
            mean=sums/count
            scale=np.maximum(np.sqrt(np.maximum(0,squares/count-mean*mean)),.05)
            report["normalization"]=dict(added_mean=mean.tolist(),added_scale=scale.tolist(),
                                          derived_on_FIT_only=True)
            save()
            results=train(old_model,sorted((root/"fit").glob("*.joblib")),mean,scale,args.out)
            report["train"]=results
            save()
            models={key:load_model(args.out/value["path"])[0] for key,value in results.items()}
            policies=make_policies(original_policy,models)
            cal_scores,cal_paired,cal_tracks=phase_scores(cal,audio,root,heads,selection,offset,
                                               old_model,models,policies,release,"CAL")
            old=cal_scores["S708_frozen"]
            eligible=[name for name in policies if guard(old,cal_scores[name],cal_paired[name])]
            assert "S708_frozen" in eligible
            best=max(eligible,key=lambda name:target_rank(cal_scores[name]))
            report["CAL"]=dict(scores=cal_scores,paired=cal_paired,tracks=cal_tracks,
                               eligible=eligible,selected=best)
            report["frozen_choice"]=dict(name=best,config=dict(model=policies[best][0],
                                                               policy=asdict(policies[best][1])))
            save()
            print("S725 CAL FROZEN",json.dumps(dict(selected=best,eligible=eligible,
                 scores={name:{k:val[k] for k in SCORES} for name,val in cal_scores.items()})),flush=True)
            chosen={"S708_frozen":policies["S708_frozen"]}
            if best!="S708_frozen": chosen[best]=policies[best]
            for phase,specs in (("HELD",held),("DEV",dev)):
                scores,paired,tracks=phase_scores(specs,audio,root,heads,selection,offset,
                         old_model,models,chosen,release,phase)
                for key in SCORES:
                    if abs(scores["S708_frozen"][key]-historical[phase]["candidate"][key])>1e-8:
                        raise AssertionError("Frozen reference mismatch "+phase+" "+key)
                safe=guard(scores["S708_frozen"],scores[best],paired[best])
                report[phase]=dict(baseline={k:scores["S708_frozen"][k] for k in SCORES},
                  selected={k:scores[best][k] for k in SCORES},safe=safe,
                  paired=paired,tracks=tracks)
                save()
                print("S725",phase,"VALIDATION",json.dumps(dict(selected=best,
                    safe=safe,baseline=report[phase]["baseline"],
                    candidate=report[phase]["selected"])),flush=True)
    report["shadow_qualifies"]=bool(best!="S708_frozen" and
        report["HELD"]["safe"] and report["DEV"]["safe"])
    report["promoted"]=False
    save()
    print("S725 FINAL",json.dumps(dict(selected=best,qualifies=report["shadow_qualifies"],
                                        promoted=False)),flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--s708-dir",type=Path,required=True)
    ap.add_argument("--offset-checkpoint",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    with threadpool_limits(limits=2):
        run(ap.parse_args())
