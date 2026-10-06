"""Symmetric pitch test-time augmentation on frozen B_low K2/K3 cases."""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np

from causal_note.guitarset import index_guitarset
from scripts.train_boundaries import decode_pcm16_mono_wav
from scripts.v273_window_experiment import load_bundle, require
from scripts.audit_v273_semitone_disappearance_fold import (
    robust_model, shifted_map, model_probability, FOLDS
)

NEG_STEPS=(-1,-2,-3,-4,-5,-6)


def load_positive(path):
    with np.load(path,allow_pickle=False) as z:
        return {k:z[k] for k in z.files}


def main():
    p=argparse.ArgumentParser()
    for n in ("source","dataset","bundle","config","fold-root","output"):
        p.add_argument("--"+n,type=Path,required=True)
    p.add_argument("--val-fold",type=int,required=True)
    a=p.parse_args()
    require(a.val_fold in FOLDS and a.val_fold!=3,"bad fold")
    require(not a.output.exists(),"refusing overwrite")
    src=load_positive(a.source)
    ids=np.asarray(src["row_id"],np.int64)
    y=np.asarray(src["true_k"],np.int32)
    pos_pred=np.asarray(src["predicted"],np.int32)
    pos_prob=np.asarray(src["probability"],np.float32)
    require(np.array_equal(src["steps"],np.arange(25)),"source steps drift")

    cache,_,_=load_bundle(a.bundle,a.config)
    model=robust_model(a.fold_root,a.val_fold)
    wanted=set(np.asarray(cache["members"][ids]).astype(str))
    tracks={t.annotation_member:t for t in index_guitarset(a.dataset) if t.annotation_member in wanted}
    require(set(tracks)==wanted,"missing tracks")
    audio={}
    for m,t in tracks.items():
        wav=decode_pcm16_mono_wav(t.audio_zip,t.audio_member)
        audio[m]=np.asarray(wav.samples,np.float32)/32768.0

    neg_prob=np.zeros((len(ids),len(NEG_STEPS),7),np.float32)
    neg_pred=np.zeros((len(ids),len(NEG_STEPS)),np.int8)
    for si,step in enumerate(NEG_STEPS):
        maps=np.empty((len(ids),31,64,3),np.float32)
        for j,rid in enumerate(ids):
            member=str(cache["members"][rid]); start=int(cache["cluster_start_samples"][rid])
            maps[j]=shifted_map(audio[member],start,step)
        pr=model_probability(model,cache,ids,maps)
        neg_prob[:,si]=pr
        neg_pred[:,si]=pr.argmax(1).astype(np.int8)
        print(json.dumps({"fold":a.val_fold,"step":step,"k2":int(np.sum(neg_pred[:,si]==2)),
                          "k3":int(np.sum(neg_pred[:,si]==3))}),flush=True)

    # Assemble ordered -6..+6 views.
    steps=np.arange(-6,7,dtype=np.int8)
    probs=np.empty((len(ids),13,7),np.float32)
    preds=np.empty((len(ids),13),np.int8)
    # NEG_STEPS stored -1,-2...; map them.
    byneg={s:i for i,s in enumerate(NEG_STEPS)}
    for j,s in enumerate(steps):
        if s<0:
            i=byneg[int(s)]; probs[:,j]=neg_prob[:,i]; preds[:,j]=neg_pred[:,i]
        else:
            probs[:,j]=pos_prob[:,int(s)]; preds[:,j]=pos_pred[:,int(s)]

    a.output.mkdir(parents=True)
    np.savez_compressed(a.output/"symmetric.npz",row_id=ids,true_k=y,steps=steps,
                        probability=probs,predicted=preds)
    report={"status":"completed","experiment":"v273_symmetric_pitch_tta_fold",
            "validation_fold":a.val_fold,"rows":len(ids),"K2":int(np.sum(y==2)),
            "K3":int(np.sum(y==3)),"outer_fold_3_used":False,
            "steps":steps.astype(int).tolist()}
    (a.output/"report.json").write_text(json.dumps(report,indent=2)+"\n")
if __name__=="__main__": main()
