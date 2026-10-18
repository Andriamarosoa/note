"""S737: Real-audio repeated FIT-only training for S736 fragment-by-fragment Onset.

Every training decision is a four-hop (~23.22ms) *fragment*, and one of
32 H0-inclusive combinations is chosen for ALL 49 pitches at that time.
The NOTE Exact-K/Onset base model, Sustain, Release, S707 hazard and
production microphone pathway are never re-trained or modified.
CAL32 only selects; HELD32 and previously exposed DEV12 are replayed
AFTER freeze, no new TEST. No claims of >80% without actual metrics.
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
    MASKS, FEATURE_DIM, FragmentMaskRouter, block_features,
    all_mask_offers, subset_descriptors, subset_class_compatibility,
    apply_fragment_masks, choose_masks, validate_decision_grid)
from scripts.s733_train_onset_only import baseline_input, REFERENCE
from scripts.s735_audit_32_onset_combinations import load_s734
from scripts.s613_prepare_strict_live import download, labels, digest
from scripts.s613_train_strict_live import real_notes
from scripts.s699_player_held_full_train_K import expanded_train_specs
from scripts.s704_train_joint_loops import targets
from scripts.s704_joint_metrics import SCORES, metric, aggregate
from scripts.s706_train_sharp_temporal_loops import (
    FIT_COMPS, CAL_COMPS, HELD_COMPS, load_model)
from scripts.s705_train_temporal_loops import setup
from scripts.s731_auditory_scene_calibration import emit
from causal_note.s708_cardinality_selection import GuidedPolicy
from causal_note.strict_live import SR

REFERENCE_NAME = "S708_frozen"
STEPS = (1, 3, 6)
SEEDS = (737, 738)
CAL_CONFIDENCE = (0.0, .20, .40)
MAX_FIT_BLOCKS = 512
ONSET_THRESHOLD = .45


def save(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def frozen_H5(model, candidates, ctx, eligible, k):
    """Use only the trained S734 native H5 Onset subnetwork, not an invented H5.

    The H5 network takes [context16, frozen K7, baseline1], nothing else.
    """
    t,p = eligible.shape
    full = np.zeros((t,p,6), np.float32)
    full[:,:,:5] = candidates
    ids = np.flatnonzero(eligible.ravel())
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(ids), 1024):
            ix = ids[start:start+1024]
            ti,pi = np.divmod(ix,p)
            packed = np.concatenate((ctx[ti,pi],k[ti],
                                      candidates[ti,pi,0,None]),axis=1)
            pred = torch.sigmoid(model.k_linked_onset(
                torch.from_numpy(packed.astype(np.float32)))).numpy().ravel()
            full[ti,pi,5] = pred
    return full


def one_track(spec, audio, root, heads, selection, offset, network,
              policy, release, h5):
    item,post,k,baseline,candidates,context,eligible = baseline_input(
        spec,audio,root,heads,selection,offset,network,policy,release)
    validate_decision_grid(item["decision_ends"])
    six = frozen_H5(h5,candidates,context,eligible,k)
    feat = block_features(six,context,eligible,k)
    desc = subset_descriptors(post,six,eligible,context)
    # The exact 32 proposals are generated BEFORE opening any annotations.
    return dict(item=item,post=post,k=k,baseline=baseline,head=six,
                eligible=eligible,features=feat,descriptors=desc)


def supervision(track, truth, seed):
    item=track["item"]
    y,ktrue = targets(item["decision_ends"],truth)
    if len(y)!=len(track["k"]):
        raise AssertionError("S736 target time grid mismatch")
    offers=all_mask_offers(track["post"],track["head"],track["eligible"])
    base = track["post"][:,:,0]>=ONSET_THRESHOLD
    new = (offers>=ONSET_THRESHOLD)&~base[:,None,:]
    real = y[:,:,0].astype(bool)
    rescued=(new & real[:,None,:]).sum(axis=-1)
    regressed=(new & ~real[:,None,:]).sum(axis=-1)
    activation=new.sum(-1)
    bits=np.asarray([int(mask).bit_count() for mask in MASKS])
    gain=2.0*rescued-2.5*regressed-.025*(bits[None,:]-1)
    chosen=gain.argmax(-1)
    best=gain[np.arange(len(gain)),chosen]
    # 0 = KEEP, others = H0-inclusive mask. Ground truth is never used in inference.
    target=np.where(best>.0,chosen+1,0).astype(np.int64)
    # Balance selection chances: all valuable frames, all onset-bearing frames,
    # plus a random sample of quiet and difficult negatives per recording.
    rng=np.random.default_rng(seed)
    priority=(target>0) | real.any(-1)
    essential=np.flatnonzero(priority)
    ordinary=np.flatnonzero(~priority)
    if len(essential)>MAX_FIT_BLOCKS//2:
        essential=np.sort(rng.choice(essential,MAX_FIT_BLOCKS//2,replace=False))
    available=MAX_FIT_BLOCKS-len(essential)
    if len(ordinary)>available:
        ordinary=np.sort(rng.choice(ordinary,available,replace=False))
    ix=np.sort(np.concatenate((essential,ordinary)))
    if not len(ix):
        raise AssertionError("Empty FIT block selection")
    risk=np.stack((rescued[ix]>0,regressed[ix]>0),axis=-1).astype(np.float32)
    coverage=(activation[ix]>0).astype(np.float32)
    return dict(features=track["features"][ix],
                descriptors=track["descriptors"][ix],
                k=track["k"][ix],
                source=track["item"]["spec"]["composition"],
                target=target[ix],audit_true_k=ktrue[ix],
                risk=risk,coverage=coverage,
                total=int(len(y)),rescuable=int(np.count_nonzero(target>0)),
                sampled=int(len(ix)))


def training_audit(examples, index):
    """OOF FIT-composition audit for each mask and each true-K class.

    FIX, REGRESSION, COVERAGE use the other FIT compositions ONLY.
    The current fragment's true K is never supplied to its input.
    """
    source=examples[index]["source"]
    other=[ex for ex in examples if ex["source"]!=source]
    if not other:
        raise ValueError("Risk audit requires disjoint FIT composition")
    numerator=np.zeros((32,7,3), np.float64)
    denominator=np.zeros(7, np.float64)
    for sample in other:
        y=sample["audit_true_k"]
        labels=np.concatenate((sample["risk"],sample["coverage"][:,:,None]),axis=-1)
        for ki in range(7):
            chosen=(y==ki)
            denominator[ki]+=chosen.sum()
            numerator[:,ki,:]+=labels[chosen].sum(axis=0)
    return (numerator/(denominator[None,:,None]+16.)).astype(np.float32)


def audit_all_fit(examples):
    """Single global FIT-only audit for CAL/HELD/DEV."""
    numerator=np.zeros((32,7,3), np.float64)
    denom=np.zeros(7, np.float64)
    for sample in examples:
        risk=np.concatenate((sample["risk"],sample["coverage"][:,:,None]),axis=-1)
        for k in range(7):
            ix=sample["audit_true_k"]==k
            denom[k]+=ix.sum()
            numerator[:,k,:]+=risk[ix].sum(axis=0)
    return (numerator/(denom[None,:,None]+16.)).astype(np.float32)


def structural_mask(b):
    # H0 is unconditional for K0..K6. Optional Onset heads are allowed
    # for K1..K6. Never lock a mask to observed argmax-K or true K.
    h=torch.ones((b,6),dtype=torch.bool)
    km=torch.ones((b,6,7),dtype=torch.bool)
    km[:,1:,0]=False
    return subset_class_compatibility(h,km)


def model_data(ex, audit, mean, std):
    n=len(ex["features"])
    return (torch.from_numpy(((ex["features"]-mean)/std).astype(np.float32)),
            torch.from_numpy(ex["descriptors"].astype(np.float32)),
            torch.from_numpy(np.broadcast_to(audit,(n,32,7,3)).copy()),
            torch.from_numpy(ex["k"].astype(np.float32)),
            structural_mask(n),
            torch.from_numpy(ex["target"].copy()).long(),
            torch.from_numpy(ex["risk"].astype(np.float32)))


def train_one(seed, examples, audits, mean, std, epochs, cal_callback):
    torch.manual_seed(seed)
    model=FragmentMaskRouter()
    opt=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=.002)
    prepared=[model_data(ex,audit,mean,std)
              for ex,audit in zip(examples,audits)]
    # FIT-only imbalance correction: roughly 1-2% of observed blocks
    # admit a useful correction. A plain CE objective otherwise learns KEEP.
    # Cap weights so rare positives cannot overwhelm regression penalties.
    positives=sum(int(torch.count_nonzero(item[5])) for item in prepared)
    rows=sum(len(item[5]) for item in prepared)
    neg=max(rows-positives, 1)
    boost=float(min(16., max(1., (neg/max(positives,1))**.5)))
    results=[]
    for epoch in range(1,epochs+1):
        model.train()
        rng=np.random.default_rng(seed+epoch)
        ordering=rng.permutation(len(prepared))
        cost=0.;batches=0
        for index in ordering:
            tensors=prepared[index]
            perm=rng.permutation(len(tensors[0]))
            for start in range(0,len(perm),256):
                ix=perm[start:start+256]
                x,d,a,k,mask,target,risk=[
                    tensor[ix] for tensor in tensors]
                out=model(x,d,a,k,mask)
                # Learn one action per fragment, not one selector per song/pitch.
                per_example=F.cross_entropy(
                    out["action_logits"],target,reduction="none")
                weights=torch.where(
                    target!=0,torch.full_like(per_example,boost),
                    torch.ones_like(per_example))
                ce=(per_example*weights).sum()/weights.sum().clamp_min(1.)
                auxiliary=F.binary_cross_entropy(
                    torch.stack([out["p_correction"],out["p_regression"]],-1),
                    risk)
                loss=ce + .05*auxiliary
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),2.)
                opt.step()
                cost+=float(loss.detach());batches+=1
        if epoch in STEPS:
            score=cal_callback(model,seed,epoch,cost/max(batches,1))
            results.append(score)
    return model,results


def eval_tracks(tracks,model,mean,std,audit,confidence):
    perf={REFERENCE_NAME:[], "dynamic":[]}
    lost_offset=0;lost_long=0
    mask_hist=Counter()
    for track in tracks:
        n=len(track["k"])
        mask,_,_=choose_masks(model,track["features"],
            track["descriptors"],np.broadcast_to(audit,(n,32,7,3)),
            track["k"],structural_mask(n).numpy(),mean,std,
            min_confidence=confidence)
        new_post=apply_fragment_masks(
            track["post"],track["head"],track["eligible"],mask)
        emitted=emit(track["item"],new_post,track["k"],
                     track["policy"],track["offset"],track["release"])
        baseline=track["baseline"]
        true=track["truth"]
        old=metric(baseline,true,track["item"]["ends"])
        current=metric(emitted,true,track["item"]["ends"])
        lost_offset+=len(set(old["correct_offset_truth_ids"]) -
                         set(current["correct_offset_truth_ids"]))
        lost_long+=len(set(old["long_onset_truth_ids"]) -
                       set(current["long_onset_truth_ids"]))
        perf[REFERENCE_NAME].append(old)
        perf["dynamic"].append(current)
        mask_hist.update(str(int(i)) for i in mask)
    original=aggregate(perf[REFERENCE_NAME])
    actual=aggregate(perf["dynamic"])
    changed={k:float(actual[k]-original[k]) for k in SCORES}
    # Strict no-regression gate; deliberately hard to pass.
    passes=(all(actual[k]>=original[k]-1e-12 for k in SCORES) and
            lost_offset==0 and lost_long==0 and actual["tp"]>=original["tp"])
    return dict(
        reference={k:float(original[k]) for k in SCORES},
        actual={k:float(actual[k]) for k in SCORES},delta=changed,
        tp=int(actual["tp"]),fp=int(actual["fp"]),fn=int(actual["fn"]),
        lost_correct_offset=lost_offset,lost_long_onset=lost_long,
        safe=passes,all_gt80=all(actual[k]>.80 for k in SCORES),
        mask_histogram=dict(mask_hist))


def main(args):
    if args.out.exists():
        raise FileExistsError("S737 output is immutable")
    args.out.mkdir(parents=True)
    frozen=json.loads((args.s708_dir/"report.json").read_text())
    sha=digest((args.s708_dir/"s708_selected.pt").read_bytes())
    if sha!=frozen["checkpoint_sha256"]:
        raise AssertionError("S708 official signed model changed")
    h5report=json.loads(args.s734_report.read_text())
    if h5report["base_sha256"]!=sha or not h5report["complete"]:
        raise AssertionError("S734 source mismatch")
    torch.set_num_threads(2)
    h5=load_s734(args.s734_checkpoint,sha)
    network=load_model(args.s708_dir/"s708_selected.pt")[0]
    policy=GuidedPolicy(**frozen["frozen_candidate"]["choice"]["policy"])
    release=frozen["frozen_candidate"]["choice"]["release"]
    if release!=dict(gate=.8,persistence=1,grace=24):
        raise AssertionError("S707 release policy changed")
    report=dict(schema="S737_DYNAMIC_EXACTK_STYLE_32_MASK_FRAGMENT_ONSET",
        status="FIT_CAL_LOOP",selected=REFERENCE_NAME,
        S708_signed_sha256=sha,training="FIT80 only",
        objective_all_6_metrics_strictly_above_80=True,
        condition="all six > .80 and no regressions, preserved correct Offsets/long onsets",
        target="Onset alone; Sustain/Release/Offset/Exact-K/pitch FROZEN",
        selection_unit="4-hop 23.22ms block, one combination shared across 49 pitches",
        target_granularity="each individual note remains separately predicted",
        architecture="PyTorch port of original Exact-K ClassConditionalAuditSelector with learned class decoder and separate KEEP",
        selection_imbalance_correction="FIT-only sqrt(negative/positive) positive CE boost capped at 16",
        seeds=list(SEEDS),epochs=list(STEPS),confidences=list(CAL_CONFIDENCE),
        post_s708_held_dev_previously_exposed=True,
        promoted=False,LIVE_changed=False,TEST_opened=False,
        completed=False,FIT={},CAL_rounds=[])
    save(args.out/"report.json",report)
    with tempfile.TemporaryDirectory() as td:
        root=Path(td);(root/"data").mkdir()
        proto,heads,selection,offset=setup(root,args.offset_checkpoint)
        for key,name in (("GuitarSet_audio","GuitarSet_audio.zip"),
                         ("GuitarSet_annotations","GuitarSet_annotations.zip")):
            asset=proto["guitar_archives"][key]
            download(asset["url"],root/"data"/name,asset["size"],asset["sha256"])
        originals=[s for s in proto["sources"]
                   if s["corpus"]=="GuitarSet" and s["split"]=="train"]
        dev=[s for s in proto["sources"]
             if s["corpus"]=="GuitarSet" and s["split"]=="dev"]
        with zipfile.ZipFile(root/"data/GuitarSet_audio.zip") as audio, \
             zipfile.ZipFile(root/"data/GuitarSet_annotations.zip") as annotations:
            expanded=expanded_train_specs(originals,audio,annotations)
            groups={"FIT":[s for s in expanded if s["composition"] in FIT_COMPS],
                    "CAL":[s for s in expanded if s["composition"] in CAL_COMPS],
                    "HELD":[s for s in expanded if s["composition"] in HELD_COMPS],
                    "DEV":dev}
            if (len(groups["FIT"]),len(groups["CAL"]),len(groups["HELD"]),len(dev))!=(80,32,32,12):
                raise AssertionError("Incompatible GuitarSet split")

            training=[]
            for i,spec in enumerate(groups["FIT"]):
                x=one_track(spec,audio,root,heads,selection,offset,
                            network,policy,release,h5)
                true=real_notes(labels(spec,root)["notes"],x["item"]["samples"]/SR)
                sample=supervision(x,true,seed=737+i)
                training.append(sample)
                print("S737 PREPARE FIT",i+1,80,spec["name"],
                      "blocks",sample["sampled"],
                      "rescuable",sample["rescuable"],flush=True)
            mean=np.mean(np.concatenate(
                [d["features"] for d in training],axis=0),axis=0)
            std=np.maximum(np.std(np.concatenate(
                [d["features"] for d in training],axis=0),axis=0),.02)
            audits=[training_audit(training,j) for j in range(80)]
            overall_audit=audit_all_fit(training)
            report["FIT"]=dict(
                total_blocks=sum(d["total"] for d in training),
                sampled_blocks=sum(d["sampled"] for d in training),
                candidate_positive=sum(d["rescuable"] for d in training),
                feature_mean=mean.tolist(),feature_scale=std.tolist(),
                fit_compositions=sorted(set(d["source"] for d in training)),
                source_leak_check="Each audited FIT sample excludes its entire composition")
            save(args.out/"report.json",report)

            def load_group(phase):
                ret=[]
                for i,spec in enumerate(groups[phase]):
                    x=one_track(spec,audio,root,heads,selection,offset,
                                network,policy,release,h5)
                    # All candidate probabilities and features already frozen.
                    x["truth"]=real_notes(
                        labels(spec,root)["notes"],x["item"]["samples"]/SR)
                    x["policy"]=policy;x["offset"]=offset;x["release"]=release
                    ret.append(x)
                    print("S737 PREPARE",phase,i+1,len(groups[phase]),flush=True)
                return ret

            cal=load_group("CAL")
            def check_signed_cal():
                totals=aggregate([metric(x["baseline"],x["truth"],
                                x["item"]["ends"]) for x in cal])
                expected=frozen["frozen_candidate"]["CAL"]
                for name in SCORES:
                    if abs(totals[name]-expected[name])>1e-8:
                        raise AssertionError("S708 official CAL replay mismatch "+name)
            check_signed_cal()
            candidate_choices=[]
            for seed in SEEDS:
                def cal_round(model,run_seed,epoch,fit_loss):
                    records=[]
                    for confidence in CAL_CONFIDENCE:
                        result=eval_tracks(cal,model,mean,std,overall_audit,confidence)
                        records.append(dict(seed=run_seed,epoch=epoch,
                                           confidence=confidence,
                                           FIT_loss=fit_loss,**result))
                    for record in records:
                        report["CAL_rounds"].append(record)
                        print("S737 CAL ROUND",json.dumps({
                            k:record[k] for k in
                            ("seed","epoch","confidence","FIT_loss",
                             "actual","safe","all_gt80","mask_histogram")}),flush=True)
                        # Don't evaluate held until a gate passing checkpoint is frozen.
                        if record["safe"] and any(
                                record["delta"][k]>1e-10 for k in SCORES):
                            candidate_choices.append((record,seed,epoch,confidence))
                            name=f"s737_seed{seed}_epoch{epoch}.pt"
                            torch.save(dict(model=model.state_dict(),seed=seed,
                                epoch=epoch,source_sha256=sha,
                                mean=mean.tolist(),std=std.tolist()),
                                args.out/name)
                    save(args.out/"report.json",report)
                    return records
                train_one(seed,training,audits,mean,std,max(STEPS),cal_round)
            if candidate_choices:
                selected=max(candidate_choices,key=lambda a:(
                    a[0]["all_gt80"],
                    min(a[0]["actual"][k]-a[0]["reference"][k] for k in SCORES),
                    sum(a[0]["actual"].values())))
                outcome,seed,epoch,conf=selected
                report["selected"]=f"seed{seed}_epoch{epoch}_confidence{conf}"
                report["frozen_on_CAL"]=dict(seed=seed,epoch=epoch,confidence=conf)
                save(args.out/"report.json",report)
                model=FragmentMaskRouter()
                ckpt=torch.load(args.out/f"s737_seed{seed}_epoch{epoch}.pt",
                                map_location="cpu",weights_only=True)
                if ckpt["source_sha256"]!=sha:
                    raise AssertionError("S737 frozen provenance mismatch")
                model.load_state_dict(ckpt["model"],strict=True)
                for phase in ("HELD","DEV"):
                    hold=load_group(phase)
                    result=eval_tracks(hold,model,mean,std,overall_audit,conf)
                    expected=frozen[phase]["candidate"]
                    for name in SCORES:
                        if abs(result["reference"][name]-expected[name])>1e-8:
                            raise AssertionError("S708 holdout replay mismatch "+name)
                    report[phase]=result
                    save(args.out/"report.json",report)
                    del hold
            else:
                # No good new candidate: retain the exact signed baseline and
                # refuse to waste previously exposed held-out data on tuning.
                report["CAL_no_safe_improvement"]=True
                report["selected"]=REFERENCE_NAME
            report["completed"]=True
            report["achieved_all_six_gt80"]=bool(
                candidate_choices and report["CAL_rounds"] and
                any(x["all_gt80"] and x["safe"]
                    for x in report["CAL_rounds"]))
            report["promoted"]=False
            report["eligible_for_LIVE"]=False
            save(args.out/"report.json",report)
            print("S737_FINAL",json.dumps(dict(
                selected=report["selected"],
                completed=True,
                achieved_all_six_gt80=report["achieved_all_six_gt80"],
                rounds=len(report["CAL_rounds"]),
                eligible_for_LIVE=False)),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s708-dir",required=True,type=Path)
    parser.add_argument("--s734-checkpoint",required=True,type=Path)
    parser.add_argument("--s734-report",required=True,type=Path)
    parser.add_argument("--offset-checkpoint",required=True,type=Path)
    parser.add_argument("--out",required=True,type=Path)
    args=parser.parse_args()
    with threadpool_limits(limits=2):
        main(args)
