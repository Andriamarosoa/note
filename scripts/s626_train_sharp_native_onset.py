"""S626 — direct native causal onset-head retraining after S625 missed-note audit.

S625 found 702/838 DEV missing notes lacked a sufficient attack-head
response. This experimental head therefore learns sharper 0–40 ms acoustic
novelty, with same-time *other-pitch* hard negatives, on original S614 PCM
features. Train=17 URMP plus 18 GuitarSet (two predetermined recordings
per train composition). DEV=5 URMP +12 GuitarSet composition-disjoint.
S80, S614 and prior NOTE checkpoints never change. No held TEST used.
"""
from __future__ import annotations

from collections import defaultdict
import argparse
import hashlib
import io
import json
from pathlib import Path
import tempfile
import time
import zipfile

import joblib
import numpy as np
from scipy.io import wavfile
from sklearn.ensemble import HistGradientBoostingClassifier

from causal_note.strict_live import SR,Selection,feature_track,load_heads
from causal_note.s618_attack_head import input_features,candidate_mask,DIM
from scripts.s620_frozen_novelty_audit import decode_novelty
from scripts.s623_mixed_source_event_head import pick_original_guitar_training
from scripts.s616_live_dev_ablation import ROOT,MODEL,prepare_dev_source
from scripts.s613_prepare_strict_live import download,labels,digest
from scripts.s613_train_strict_live import decode,real_notes,event_score,summary


CONFIGS=(
    ("S614_original",2.,2.,2.),
    ("attack626_075_015_008",.75,.15,.08),
    ("attack626_075_025_008",.75,.25,.08),
    ("attack626_085_015_008",.85,.15,.08),
    ("attack626_085_025_008",.85,.25,.08),
    ("attack626_090_015_008",.90,.15,.08),
    ("attack626_090_025_008",.90,.25,.08),
    ("attack626_095_015_008",.95,.15,.08),
    ("attack626_095_025_008",.95,.25,.08)
)
CONTROL={"GuitarSet":dict(TP=998,FP=585,FN=582),
         "URMP":dict(TP=243,FP=149,FN=256)}


def get_source(spec,root,heads,audio_zip=None):
    if spec["corpus"]=="URMP":
        pack=prepare_dev_source(spec,root)
        ends,x=pack["ends"],pack["features"]
        truth=pack["truth"]
        n=pack["samples"]
        stamp=pack["source_sha256"]
    else:
        assert audio_zip is not None
        blob=audio_zip.read(spec["audio"])
        rate,audio=wavfile.read(io.BytesIO(blob))
        assert rate==SR and audio.dtype in (np.int16,np.int32)
        audio=audio.astype(np.float32)/(32768 if audio.dtype==np.int16 else 2147483648)
        if audio.ndim==2:audio=audio.mean(1)
        audio=audio[:int(spec["seconds"]*SR)].copy()
        n=len(audio)
        ends,x=feature_track(audio)
        truth=real_notes(labels(spec,root)["notes"],n/SR)
        stamp=digest(blob)
    p,k=heads.predict(x)
    return dict(name=spec["name"],corpus=spec["corpus"],
                composition=spec["composition"],ends=ends,x=x,p=p,k=k,
                truth=truth,n=n,source_sha256=stamp)


def sharper_positive(ends,truth):
    y=np.zeros((len(ends),49),np.bool_)
    for note in truth:
        idx=note["pitch"]-40
        if 0<=idx<49:
            begin=np.searchsorted(ends,round(note["time"]*SR),side="left")
            stop=np.searchsorted(ends,round((note["time"]+.040)*SR),side="right")
            y[begin:stop,idx]=True
    return y


def choose_examples(row,rng):
    y=sharper_positive(row["ends"],row["truth"])
    x=row["x"];p=row["p"]
    f=input_features(x,p)
    available=candidate_mask(x,p)
    # Hard negatives: another pitch truly attacks within +/- 6 hops, while
    # this pitch does not. Label guidance is used at training only.
    any_onset=y.any(axis=1)
    vicinity=np.zeros(len(any_onset),np.bool_)
    for delta in range(-6,7):
        if delta>=0:vicinity[delta:] |= any_onset[:len(any_onset)-delta]
        else:vicinity[:delta] |= any_onset[-delta:]
    hard=np.flatnonzero((available & ~y & vicinity[:,None]).ravel())
    general=np.flatnonzero((available & ~y & ~vicinity[:,None]).ravel())
    positive=np.flatnonzero(y.ravel())
    if len(positive)>2500:positive=rng.choice(positive,2500,replace=False)
    if len(hard)>5000:hard=rng.choice(hard,5000,replace=False)
    if len(general)>3500:general=rng.choice(general,3500,replace=False)
    selected=np.concatenate((positive,hard,general))
    rng.shuffle(selected)
    a=f.reshape(-1,DIM)[selected]
    b=y.ravel()[selected].astype(np.int8)
    return a,b,dict(positive=int(len(positive)),hard_neg=int(len(hard)),
                    other_neg=int(len(general)),
                    annotated_onsets=int(len(row["truth"])),
                    candidate_positive_coverage=int((available&y).sum()),
                    candidate_positive_total=int(y.sum()))


def fit(a,b):
    assert a.shape[1]==DIM and len(np.unique(b))==2
    model=HistGradientBoostingClassifier(
        max_iter=115,max_leaf_nodes=18,max_depth=5,
        learning_rate=.07,min_samples_leaf=35,
        l2_regularization=4.,early_stopping=False,random_state=626)
    model.fit(a,b,sample_weight=np.where(b==1,4.,1.))
    return model


def evaluate(row,trained,cfg):
    ends,x,p,k=row["ends"],row["x"],row["p"],row["k"]
    score=trained.predict_proba(input_features(x,p).reshape(-1,DIM))[:,1].reshape(len(ends),49)
    results={}
    original=decode(ends,p,k,cfg,row["n"])
    for name,c,j,physical in CONFIGS:
        if name=="S614_original":
            emitted=original; opportunities=0
        else:
            emitted,opportunities=decode_novelty(ends,x,p,k,cfg,row["n"],score,c,j,physical)
        metric,pairs=event_score(emitted,row["truth"])
        results[name]=dict(TP=metric["TP"],FP=metric["FP"],
            FN=metric["FN"],F1=metric["F1"],activated=int(opportunities))
    return results


def run(out):
    if out.exists():raise FileExistsError("S626 evidence must not be overwritten")
    out.mkdir(parents=True)
    protocol=json.loads((ROOT/"protocol.json").read_text())
    frozen=json.loads((ROOT/"selection_frozen.json").read_text())
    sha=digest(MODEL.read_bytes())
    assert sha==frozen["model_sha256"]
    assert digest((ROOT/"protocol.json").read_bytes())==frozen["protocol_sha256"]
    head=load_heads(MODEL,expected_sha256=sha)
    config=Selection(**head.metadata["selection"])
    trainU=[s for s in protocol["sources"] if s["corpus"]=="URMP" and s["split"]=="train"]
    devU=[s for s in protocol["sources"] if s["corpus"]=="URMP" and s["split"]=="dev"]
    trainG=pick_original_guitar_training(protocol)
    devG=[s for s in protocol["sources"] if s["corpus"]=="GuitarSet" and s["split"]=="dev"]
    assert (len(trainU),len(devU),len(trainG),len(devG))==(17,5,18,12)
    assert not ({(s["corpus"],s["composition"]) for s in trainU+trainG}&
                {(s["corpus"],s["composition"]) for s in devU+devG})
    t0=time.perf_counter()
    arrays=[];targets=[];training_info=[]
    rng=np.random.default_rng(626)
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp);(root/"data").mkdir()
        for spec in trainU:
            data=get_source(spec,root,head)
            x,y,info=choose_examples(data,rng)
            arrays.append(x);targets.append(y)
            training_info.append(dict(source=data["name"],corpus=data["corpus"],**info))
            print("S626 URMP TRAIN",data["name"],info,flush=True)
        a=protocol["guitar_archives"]["GuitarSet_audio"]
        b=protocol["guitar_archives"]["GuitarSet_annotations"]
        download(a["url"],root/"data/GuitarSet_audio.zip",a["size"],a["sha256"])
        download(b["url"],root/"data/GuitarSet_annotations.zip",b["size"],b["sha256"])
        with zipfile.ZipFile(root/"data/GuitarSet_audio.zip") as archive:
            for spec in trainG:
                data=get_source(spec,root,head,audio_zip=archive)
                x,y,info=choose_examples(data,rng)
                arrays.append(x);targets.append(y)
                training_info.append(dict(source=data["name"],corpus=data["corpus"],**info))
                print("S626 GUITAR TRAIN",data["name"],info,flush=True)
            X=np.concatenate(arrays);Y=np.concatenate(targets)
            del arrays,targets
            model=fit(X,Y)
            checkpoint=out/"S626_native_sharp_onset_joint_train.joblib"
            joblib.dump(model,checkpoint,compress=3)
            head_sha=digest(checkpoint.read_bytes())
            print("S626 FIT",len(Y),"positive",int(Y.sum()),"SHA",head_sha,flush=True)
            test_records=[]
            for spec in devU:
                data=get_source(spec,root,head)
                test_records.append(dict(source=spec["name"],corpus="URMP",
                                         results=evaluate(data,model,config)))
                print("S626 DEV URMP",spec["name"],flush=True)
            for spec in devG:
                data=get_source(spec,root,head,audio_zip=archive)
                test_records.append(dict(source=spec["name"],corpus="GuitarSet",
                                         results=evaluate(data,model,config)))
                print("S626 DEV GUITAR",spec["name"],flush=True)
    measured=[]
    for name,*_ in CONFIGS:
        per_corpus={}
        for corpus in ("GuitarSet","URMP"):
            selected=[r for r in test_records if r["corpus"]==corpus]
            tp=sum(r["results"][name]["TP"] for r in selected)
            fp=sum(r["results"][name]["FP"] for r in selected)
            fn=sum(r["results"][name]["FN"] for r in selected)
            f1=2*tp/max(1,2*tp+fp+fn)
            per_corpus[corpus]=dict(TP=tp,FP=fp,FN=fn,F1=f1,
                extra_accepted=sum(r["results"][name]["activated"] for r in selected),
                improved_sources=sum(r["results"][name]["F1"]>r["results"]["S614_original"]["F1"] for r in selected),
                regressed_sources=sum(r["results"][name]["F1"]<r["results"]["S614_original"]["F1"] for r in selected))
            if name=="S614_original":
                assert all(per_corpus[corpus][key]==v for key,v in CONTROL[corpus].items()),per_corpus
            per_corpus[corpus]["F1_delta_points"]=100*(f1-2*CONTROL[corpus]["TP"]/(
                2*CONTROL[corpus]["TP"]+CONTROL[corpus]["FP"]+CONTROL[corpus]["FN"]))
        measured.append(dict(name=name,by_corpus=per_corpus))
        print("S626 VARIANT",name,
              "G",round(per_corpus["GuitarSet"]["F1"],4),
              "U",round(per_corpus["URMP"]["F1"],4),flush=True)
    both=[v for v in measured if all(v["by_corpus"][c]["F1_delta_points"]>=0
                                   for c in ("GuitarSet","URMP"))]
    result=dict(status="S626_RETRAIN_NATIVE_CAUSAL_ONSET_HEAD_MIXED_TRAIN_DEV_ONLY",
        original_S80_S614_unchanged=True, original_checkpoint_sha256=sha,
        trained_sharp_attack_head_sha256=head_sha,
        train_sources=training_info, training_positive_rows=int(Y.sum()),training_total_rows=len(Y),
        dev_source_count=len(test_records),dev_results=measured,
        no_regression_both_corpora=[v["name"] for v in both],
        tested_new_sealed_sources=False,original_S614_test_not_read=True,
        all_dev_previously_used_by_original_S614=True,
        no_new_heldout_claim=True,no_promotion=True,
        elapsed_seconds=round(time.perf_counter()-t0,2))
    (out/"report.json").write_text(json.dumps(result,indent=2)+"\n")
    print("S626 FINAL",json.dumps({"train_rows":len(Y),"positive":int(Y.sum()),
       "head_sha":head_sha,"nonregressing_both":result["no_regression_both_corpora"],
       "variants":[dict(name=r["name"],G=r["by_corpus"]["GuitarSet"]["F1"],
                        U=r["by_corpus"]["URMP"]["F1"]) for r in measured]},indent=2),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,required=True)
    run(parser.parse_args().out)
