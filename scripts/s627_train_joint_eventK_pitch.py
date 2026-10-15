"""S627: mixed-source train-only joint onset + EVENT cardinality network.

This is not original S80 Exact-K, which remains frozen. Also not a count
of sounding/active notes. The shared encoder predicts BOTH new pitch onsets
and how many unique new pitches start in a short causal frame.

Experiment / artifact only. Never read S614 test labels or use its 22
published recordings to choose the model; GuitarSet and URMP dev groups
were used by historical S614 and are NOT newly independent.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import tempfile
import time
import zipfile

import numpy as np
from scipy.io import wavfile
from causal_note.strict_live import SR,Selection,feature_track,load_heads,CausalSelection
from causal_note.s627_joint_heads import (
    INPUT_DIM,LATENT,SCHEMA,frames,targets,JointNPZ,variants,fuse)
from scripts.s623_mixed_source_event_head import pick_original_guitar_training
from scripts.s626_train_sharp_native_onset import get_source
from scripts.s616_live_dev_ablation import ROOT,MODEL
from scripts.s613_prepare_strict_live import download,labels,digest
from scripts.s613_train_strict_live import decode,real_notes,event_score,summary,offset_audit

CONTROL={"GuitarSet":dict(TP=998,FP=585,FN=582),
         "URMP":dict(TP=243,FP=149,FN=256)}

def choose(row,rng):
    f=frames(row["x"],row["p"])
    y,k=targets(row["ends"],row["truth"])
    active=(k>0)
    pos=np.flatnonzero(active)
    # Over-sample frames around actual new onsets; include quieter negatives,
    # especially native onset false positives near events of OTHER pitches.
    if len(pos)>420:pos=rng.choice(pos,420,replace=False)
    negative=np.flatnonzero(~active)
    likely=row["p"][:,:,0].max(1)>.38
    hard=negative[likely[negative]]
    if len(hard)>160:hard=rng.choice(hard,160,replace=False)
    background=negative[~likely[negative]]
    if len(background)>160:background=rng.choice(background,160,replace=False)
    ids=np.unique(np.concatenate((pos,hard,background)))
    rng.shuffle(ids)
    return (f[ids].astype(np.float16),y[ids].astype(np.float16),
            k[ids].astype(np.int64),dict(
                positive_frames=int(active.sum()),selected_positive=int(sum(active[ids])),
                selected_hard_neg=int(sum(likely[ids]&(~active[ids]))),
                selected_background=int(sum((~likely[ids])&(~active[ids]))),
                selected_frames=len(ids)))

def train_network(train_rows,out,epochs=10):
    import torch
    from torch import nn
    torch.set_num_threads(3)
    torch.manual_seed(627)
    class Joint(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder=nn.Sequential(nn.Linear(INPUT_DIM,64),nn.ReLU(),
                    nn.Linear(64,LATENT),nn.ReLU())
            self.k=nn.Linear(LATENT*2,7)
            self.pitch=nn.Linear(LATENT+7,1)
        def forward(self,f):
            h=self.encoder(f)
            pooled=torch.cat((h.mean(1),h.amax(1)),dim=1)
            logits=self.k(pooled)
            count=torch.softmax(logits,dim=-1)
            joint=torch.cat((h,count[:,None,:].expand(-1,49,-1)),dim=-1)
            return self.pitch(joint).squeeze(-1),logits

    a=np.concatenate([r[0] for r in train_rows])
    y=np.concatenate([r[1] for r in train_rows])
    k=np.concatenate([r[2] for r in train_rows])
    assert a.shape[1:]==(49,INPUT_DIM) and len(a)>2500 and y.sum()>1000
    means=a.astype(np.float32).mean(axis=(0,1))
    var=a.astype(np.float32).var(axis=(0,1))
    scales=np.maximum(np.sqrt(var),.05).astype(np.float32)
    a=((a.astype(np.float32)-means)/scales).astype(np.float32)
    a=np.clip(a,-16,16)
    x=torch.from_numpy(a);yl=torch.from_numpy(y.astype(np.float32))
    kl=torch.from_numpy(k)
    network=Joint()
    optimizer=torch.optim.AdamW(network.parameters(),lr=.002,weight_decay=.015)
    torch_gen=torch.Generator().manual_seed(627)
    history=[]
    for epoch in range(epochs):
        order=torch.randperm(len(a),generator=torch_gen)
        total=0.;seen=0
        for ids in order.split(72):
            logits,ck=network(x[ids])
            bce=nn.functional.binary_cross_entropy_with_logits(
                logits,yl[ids],pos_weight=torch.tensor(8.),reduction="mean")
            ce=nn.functional.cross_entropy(
                ck,kl[ids],weight=torch.tensor([1.,2.,2.,2.,2.,2.,2.]),
                reduction="mean")
            loss=bce+.35*ce
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(network.parameters(),4.)
            optimizer.step()
            total+=float(loss.detach())*len(ids);seen+=len(ids)
        history.append(dict(epoch=epoch+1,loss=total/seen))
        print("S627 JOINT EPOCH",epoch+1,"loss",round(total/seen,6),flush=True)
    arrays=dict(schema=np.asarray(SCHEMA),mean=means,scale=scales,
        metadata=np.asarray(json.dumps(dict(epochs=epochs,architecture="joint_new_pitches_Kevent",
        training_sources=len(train_rows),note="Kevent not S80 and not S614 active-K"))))
    for prefix,layer in (
       ("encoder_w0",network.encoder[0]),("encoder_w1",network.encoder[2]),
       ("k",network.k),("pitch",network.pitch)):
        if prefix.startswith("encoder"):
            arrays[prefix]=layer.weight.detach().numpy().T.astype(np.float32)
            arrays[prefix.replace("_w","_b")]=layer.bias.detach().numpy().astype(np.float32)
        else:
            arrays[prefix+"_w"]=layer.weight.detach().numpy().T.astype(np.float32)
            arrays[prefix+"_b"]=layer.bias.detach().numpy().astype(np.float32)
    path=out/"S627_joint_new_pitch_eventK_train_only.npz"
    np.savez_compressed(path,**arrays)
    # Test Torch-vs-Numpy ABI on the initial 8 samples (all future frames differ).
    original=train_rows[0][0][:8].astype(np.float32)
    # x data comes from arbitrary frames; their p delta already part of row,
    # so compare exported forward math exactly, not stateful .frame.
    with torch.no_grad():
        torch_q,torch_k=network(torch.from_numpy(np.clip((original-means)/scales,-16,16)))
        torch_q=torch.sigmoid(torch_q).numpy()
        torch_k=torch.softmax(torch_k,-1).numpy()
    expected=export_forward(arrays,original)
    np.testing.assert_allclose(expected[0],torch_q,rtol=1e-4,atol=1e-4)
    np.testing.assert_allclose(expected[1],torch_k,rtol=1e-4,atol=1e-4)
    return path,history,dict(total_train_frames=len(a),
                            positive_pitch_rows=int(y.sum()),
                            positive_event_frames=int((k>0).sum()),
                            eventK_hist=np.bincount(k,minlength=7).tolist())


def export_forward(a,feature):
    z=np.clip((np.asarray(feature,np.float32)-a["mean"])/a["scale"],-16,16)
    h=np.maximum(z@a["encoder_w0"]+a["encoder_b0"],0)
    h=np.maximum(h@a["encoder_w1"]+a["encoder_b1"],0)
    pool=np.concatenate((h.mean(1),h.max(1)),axis=-1)
    kl=pool@a["k_w"]+a["k_b"]
    ek=np.exp(kl-kl.max(-1,keepdims=True));ek/=ek.sum(-1,keepdims=True)
    joint=np.concatenate((h,np.broadcast_to(ek[:,None,:],(len(ek),49,7))),axis=-1)
    logits=joint@a["pitch_w"]+a["pitch_b"]
    q=1/(1+np.exp(-np.clip(logits[...,0],-30,30)))
    return q,ek


def scores_for(row,checkpoint,cfg):
    ends,x,p,k=row["ends"],row["x"],row["p"],row["k"]
    inp=frames(x,p)
    with np.load(checkpoint,allow_pickle=False) as z:
        arr={name:z[name] for name in z.files if name not in ("schema","metadata")}
    q,eventK=export_forward(arr,inp)
    events_out={}
    for name,policy in variants():
        if name=="S614_original":
            events=decode(ends,p,k,cfg,row["n"])
        else:
            state=CausalSelection(cfg);events=[]
            for t,pitch,oldK,newq,newk in zip(ends,p,k,q,eventK):
                corrected=fuse(pitch,newq,newk,policy)
                events.extend(state.step(corrected,oldK,int(t),int(t)))
            events.extend(state.panic(row["n"]))
        score,pairs=event_score(events,row["truth"])
        offset=offset_audit(events,row["truth"],pairs,row["n"]/SR)
        events_out[name]=dict(score=score,
            conditional_offset=offset["conditional_offset_accuracy"],
            total_emissions=score["TP"]+score["FP"])
    return events_out


def run(out,epochs):
    if out.exists():raise FileExistsError("Evidence immutable")
    out.mkdir(parents=True)
    protocol=json.loads((ROOT/"protocol.json").read_text())
    frozen=json.loads((ROOT/"selection_frozen.json").read_text())
    basehash=digest(MODEL.read_bytes())
    assert basehash==frozen["model_sha256"]
    assert digest((ROOT/"protocol.json").read_bytes())==frozen["protocol_sha256"]
    heads=load_heads(MODEL,expected_sha256=basehash)
    cfg=Selection(**heads.metadata["selection"])
    trainU=[s for s in protocol["sources"] if s["corpus"]=="URMP" and s["split"]=="train"]
    devU=[s for s in protocol["sources"] if s["corpus"]=="URMP" and s["split"]=="dev"]
    trainG=pick_original_guitar_training(protocol)
    devG=[s for s in protocol["sources"] if s["corpus"]=="GuitarSet" and s["split"]=="dev"]
    assert (len(trainU),len(devU),len(trainG),len(devG))==(17,5,18,12)
    key=lambda s:(s["corpus"],s["composition"])
    assert not set(map(key,trainU+trainG))&set(map(key,devU+devG))
    start=time.perf_counter()
    training=[];proof=[];dev_rows=[]
    rng=np.random.default_rng(627)
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);(root/"data").mkdir()
        for spec in trainU:
            row=get_source(spec,root,heads)
            sample=choose(row,rng)
            training.append(sample)
            proof.append(dict(source=spec["name"],corpus=spec["corpus"],**sample[3]))
            print("S627 TRAIN URMP",spec["name"],sample[3],flush=True)
        a=protocol["guitar_archives"]["GuitarSet_audio"]
        b=protocol["guitar_archives"]["GuitarSet_annotations"]
        download(a["url"],root/"data/GuitarSet_audio.zip",a["size"],a["sha256"])
        download(b["url"],root/"data/GuitarSet_annotations.zip",b["size"],b["sha256"])
        with zipfile.ZipFile(root/"data/GuitarSet_audio.zip") as arc:
            for spec in trainG:
                row=get_source(spec,root,heads,arc)
                sample=choose(row,rng)
                training.append(sample)
                proof.append(dict(source=spec["name"],corpus=spec["corpus"],**sample[3]))
                print("S627 TRAIN GUITAR",spec["name"],sample[3],flush=True)
            checkpoint,history,counts=train_network(training,out,epochs=epochs)
            del training
            print("S627 CHECKPOINT",digest(checkpoint.read_bytes()),counts,flush=True)
            for spec in devU:
                row=get_source(spec,root,heads)
                tested=scores_for(row,checkpoint,cfg)
                dev_rows.append(dict(source=spec["name"],corpus=spec["corpus"],policies=tested))
                print("S627 DEV URMP",spec["name"],flush=True)
            for spec in devG:
                row=get_source(spec,root,heads,arc)
                tested=scores_for(row,checkpoint,cfg)
                dev_rows.append(dict(source=spec["name"],corpus=spec["corpus"],policies=tested))
                print("S627 DEV GUITAR",spec["name"],flush=True)
    totals=[]
    for name,policy in variants():
        corps={}
        for corpus,control in CONTROL.items():
            data=[row for row in dev_rows if row["corpus"]==corpus]
            coll=summary([r["policies"][name]["score"] for r in data])
            if name=="S614_original":
                assert all(coll[k]==v for k,v in control.items()),(corpus,coll)
            f1_orig=2*control["TP"]/(2*control["TP"]+control["FP"]+control["FN"])
            corps[corpus]=dict(TP=coll["TP"],FP=coll["FP"],FN=coll["FN"],F1=coll["F1"],
                F1_delta_points=100*(coll["F1"]-f1_orig),
                wins=sum(r["policies"][name]["score"]["F1"]>r["policies"]["S614_original"]["score"]["F1"]
                         for r in data),
                losses=sum(r["policies"][name]["score"]["F1"]<r["policies"]["S614_original"]["score"]["F1"]
                           for r in data),
                offset_conditional_by_source={r["source"]:r["policies"][name]["conditional_offset"] for r in data})
        totals.append(dict(name=name,params=asdict(policy),by_corpus=corps))
        print("S627 METRIC",name,
              "G",round(corps["GuitarSet"]["F1"],5),
              "U",round(corps["URMP"]["F1"],5),flush=True)
    viable=[r for r in totals if all(r["by_corpus"][c]["F1_delta_points"]>0
                                   for c in ("GuitarSet","URMP"))]
    report=dict(status="S627_JOINT_NATIVE_PITCH_ATTACK_EVENTK_TRAIN_DEV_ONLY",
        original_checkpoint_SHA256=basehash,new_joint_checkpoint_SHA256=digest(checkpoint.read_bytes()),
        old_S80_untouched=True,old_S614_untouched=True,source_count_train=35,source_count_dev=17,
        training_counts=counts,loss_history=history,training_proof=proof,
        eventK_is_NEW_start_count_not_activeK=True,test_source_and_labels_not_read=True,
        all_dev_preexposed_historically=True,
        threshold_variants_tuned_on_dev=True,
        results=totals,
        both_corpus_positive=[r["name"] for r in viable],no_official_promotion=True,
        elapsed_seconds=round(time.perf_counter()-start,2))
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("S627 SUMMARY",json.dumps({k:report[k] for k in (
        "original_checkpoint_SHA256","new_joint_checkpoint_SHA256",
        "training_counts","both_corpus_positive","elapsed_seconds")},indent=2),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--epochs",type=int,default=10)
    args=parser.parse_args()
    run(args.out,args.epochs)
