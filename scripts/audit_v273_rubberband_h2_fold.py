"""Evaluate fixed symmetric H2 with Rubber Band on one internal fold."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

from causal_note.guitarset import index_guitarset
from scripts.train_boundaries import decode_pcm16_mono_wav
from scripts.v273_window_experiment import load_bundle, require
from scripts.audit_v273_semitone_disappearance_fold import robust_model, model_probability, FOLDS
from scripts.v273_rubberband_pitch import rubberband_shifted_map

STEPS=(-2,-1,0,1,2)

def load_npz(path):
    with np.load(path,allow_pickle=False) as z:
        return {k:np.asarray(z[k]) for k in z.files}

def main():
    p=argparse.ArgumentParser()
    for n in ("source","dataset","bundle","config","fold-root","output"):
        p.add_argument("--"+n,type=Path,required=True)
    p.add_argument("--val-fold",type=int,required=True)
    a=p.parse_args()
    require(a.val_fold in FOLDS and a.val_fold!=3,"bad fold")
    require(not a.output.exists(),"refusing overwrite")

    src=load_npz(a.source)
    require(np.array_equal(src["steps"],np.arange(25)),"source steps drift")
    ids=np.asarray(src["row_id"],np.int64)
    y=np.asarray(src["true_k"],np.int32)
    require(np.all(np.isin(y,(2,3))),"cohort must be K2/K3 only")

    cache,_,_=load_bundle(a.bundle,a.config)
    model=robust_model(a.fold_root,a.val_fold)

    wanted=set(np.asarray(cache["members"][ids]).astype(str))
    tracks={t.annotation_member:t for t in index_guitarset(a.dataset) if t.annotation_member in wanted}
    require(set(tracks)==wanted,"missing tracks")
    audio={}
    for m,t in tracks.items():
        wav=decode_pcm16_mono_wav(t.audio_zip,t.audio_member)
        audio[m]=np.asarray(wav.samples,np.float32)/32768.0

    probs=np.empty((len(ids),5,7),np.float64)
    probs[:,2]=np.asarray(src["probability"][:,0],np.float64)

    for si,step in enumerate(STEPS):
        if step==0: continue
        maps=np.empty((len(ids),31,64,3),np.float32)
        for j,rid in enumerate(ids):
            member=str(cache["members"][rid])
            start=int(cache["cluster_start_samples"][rid])
            maps[j]=rubberband_shifted_map(audio[member],start,step)
        probs[:,si]=model_probability(model,cache,ids,maps)
        print(json.dumps({"fold":a.val_fold,"step":step,
                          "k2":int(np.sum(probs[:,si].argmax(1)==2)),
                          "k3":int(np.sum(probs[:,si].argmax(1)==3))}),flush=True)

    margin=(probs[:,:,2]-probs[:,:,3]).mean(axis=1)
    action=margin>0
    corr=int(np.sum(action&(y==2)))
    reg=int(np.sum(action&(y==3)))
    report={
      "status":"completed","experiment":"v273_rubberband_h2_fold",
      "validation_fold":a.val_fold,"rows":int(len(ids)),
      "K2":int(np.sum(y==2)),"K3":int(np.sum(y==3)),
      "actions":int(action.sum()),"corrections":corr,"regressions":reg,"net":corr-reg,
      "steps":list(STEPS),"rule":"mean(P2-P3)>0","outer_fold_3_used":False
    }
    a.output.mkdir(parents=True)
    (a.output/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    np.savez_compressed(a.output/"rubberband_h2.npz",row_id=ids,true_k=y,
                        probability=probs.astype(np.float32),
                        mean_margin=margin.astype(np.float32),
                        action=action.astype(np.uint8))
    print(json.dumps(report),flush=True)

if __name__=="__main__": main()
