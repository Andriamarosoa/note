"""S654: learn multi-view Tone K head, composition-held 5-fold, real GuitarSet.

DIAGNOSTIC ONLY: K estimates are sampled at annotated onset +30ms,
as in S652. Using annotation timestamps in evaluation does NOT imply
a fully autonomous microphone detector. The classifier features are
strictly original and causally filtered NOTE K posteriors at that time.
The labels are the unique simultaneous active pitches (clipped at 6).
All 36 TRAIN composition groups provide OOF model selection; 12 DEV
used only for a frozen final evaluation. NO GuitarSet TEST opened.

Goal: check whether original+2800Hz+1100Hz count posteriors provide
independent information allowing K1/K2/K3+ to improve together.
No NOTE pitch, original Exact-K, S614, S641 or microphone changes.
"""
from __future__ import annotations
from collections import Counter
import argparse
import json
from pathlib import Path
import tempfile
import time
import zipfile

import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier,RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from causal_note.strict_live import SR,Selection,load_heads
from scripts.s652_tone_filter_multiview import causal_lowpass,predict_causal_view,FILTERS
from scripts.s637_guitarset_three_view_groundtruth import ROOT,MODEL,raw_guitar_audio
from scripts.s613_prepare_strict_live import download,digest,labels
from scripts.s613_train_strict_live import real_notes

MODEL_KINDS=("linear","linear_simple","trees","forest")
TARGET_K=range(1,7)
LAG_AFTER_ONSET_MS=30
THRESHOLD_CV_GAIN=.03  # precommitted min +3 points overall K accuracy
MIN_IMPROVED_TRAIN_FOLDS=3


def physical_rows(ends,score_orig,score_mild,score_dark,truth):
    """S652 exact count label at 30ms following annotated onset."""
    result=[]
    base_orig=[];base_mild=[];base_dark=[];count_by=Counter()
    for note in truth:
        t=float(note["time"])+.030
        if t>=ends[-1]/SR:continue
        K=min(6,len({int(other["pitch"]) for other in truth
           if float(other["time"])<=t and
              float(other["time"])+float(other["duration"])>t}))
        if K<1:continue
        i=min(int(np.searchsorted(ends,round(t*SR),side="left")),len(ends)-1)
        a=score_orig[i];b=score_mild[i];c=score_dark[i]
        feat=np.concatenate((
            a,b,c,
            (a+b+c)/3.,
            b-a,c-a,
            [a.max(),b.max(),c.max()],
            [a.argmax()/6.,b.argmax()/6.,c.argmax()/6.],
            [float(np.sum(a[2:])),float(np.sum(b[2:])),float(np.sum(c[2:]))],
            [-float(np.sum(z*np.log(np.clip(z,1e-8,1.)))) for z in (a,b,c)],
        )).astype(np.float32)
        if feat.shape!=(46,):
            raise AssertionError(f"S654 multi Tone K features ABI {feat.shape}")
        result.append((feat,K))
        base_orig.append(int(a.argmax()))
        base_mild.append(int(b.argmax()))
        base_dark.append(int(c.argmax()))
        count_by[K]+=1
    return dict(
        features=np.asarray([v[0] for v in result],np.float32).reshape(-1,46),
        truth=np.asarray([v[1] for v in result],np.int8),
        original_pred=np.asarray(base_orig,np.int8),
        mild_pred=np.asarray(base_mild,np.int8),
        dark_pred=np.asarray(base_dark,np.int8),
        counts=dict(count_by))


def one(spec,root,archive,heads):
    sound,sha=raw_guitar_audio(spec,archive)
    truth=real_notes(labels(spec,root)["notes"],len(sound)/SR)
    ends,p,k=predict_causal_view(sound,heads)
    views={}
    for name,fc in FILTERS.items():
        e,q,ks=predict_causal_view(causal_lowpass(sound,fc),heads)
        if not np.array_equal(ends,e):raise AssertionError("Tone altered frame clock")
        views[name]=ks
    rows=physical_rows(ends,k,views["tone_mild_2800"],views["tone_dark_1100"],truth)
    return dict(name=spec["name"],composition=spec["composition"],sha256=sha,**rows)


def project(x,mode):
    if mode=="linear_simple":
        return x[:,0:21] # K native, mild, dark only
    return x


def models():
    return dict(
        linear=make_pipeline(StandardScaler(),LogisticRegression(
            C=.15,max_iter=1400,random_state=654)),
        linear_simple=make_pipeline(StandardScaler(),LogisticRegression(
            C=.50,max_iter=1400,random_state=654)),
        trees=HistGradientBoostingClassifier(max_iter=90,max_depth=4,
            max_leaf_nodes=12,min_samples_leaf=20,learning_rate=.06,
            l2_regularization=8.,random_state=654),
        forest=RandomForestClassifier(n_estimators=150,max_depth=12,
            min_samples_leaf=8,max_features=".sqrt" if False else "sqrt",
            n_jobs=1,random_state=654)
    )


def metrics(y,guess):
    y=np.asarray(y,dtype=int);guess=np.asarray(guess,dtype=int)
    return dict(n=len(y),correct=int(np.sum(y==guess)),
                accuracy=float(np.mean(y==guess)) if len(y) else 0.,
                per_true_K={str(k):dict(n=int(np.sum(y==k)),
                    correct=int(np.sum((y==k)&(guess==k))),
                    recall=float(np.mean(guess[y==k]==k)) if np.any(y==k) else None)
                    for k in TARGET_K})


def train_cv(records):
    x=np.concatenate([s["features"] for s in records])
    y=np.concatenate([s["truth"] for s in records])
    groups=np.concatenate([np.repeat(s["composition"],len(s["truth"])) for s in records])
    if len(set(groups))!=9 or len(y)<600 or len(np.unique(y))<4:
        raise ValueError("Insufficient independent GuitarSet K1-K6 training samples")
    splits=list(GroupKFold(n_splits=5).split(x,y,groups))
    baseline=np.concatenate([s["original_pred"] for s in records])
    scores={}
    trained={}
    candidate={}
    for name,model in models().items():
        OOF=np.full(len(y),-1,np.int32)
        compare=[]
        for tr,va in splits:
            if set(groups[tr])&set(groups[va]):raise AssertionError("Group K leakage")
            head=clone(model).fit(project(x[tr],name),y[tr])
            OOF[va]=head.predict(project(x[va],name))
            compare.append(dict(
                held_groups=sorted(set(groups[va])),
                baseline=metrics(y[va],baseline[va])["accuracy"],
                new=metrics(y[va],OOF[va])["accuracy"]))
        report=metrics(y,OOF)
        report["folds"]=compare
        report["improved_folds"]=sum(r["new"]>r["baseline"] for r in compare)
        scores[name]=report
        trained[name]=clone(model).fit(project(x,name),y)
        print("S654 TRAIN OOF",name,"n",len(y),"acc",report["accuracy"],
              "improved_folds",report["improved_folds"],flush=True)
    original=metrics(y,baseline)
    candidate_name=max(MODEL_KINDS,key=lambda n:(
        scores[n]["accuracy"],scores[n]["improved_folds"],n))
    choice=(candidate_name
            if scores[candidate_name]["accuracy"]>=original["accuracy"]+THRESHOLD_CV_GAIN
            and scores[candidate_name]["improved_folds"]>=MIN_IMPROVED_TRAIN_FOLDS
            else "S614_original_K")
    return trained,choice,candidate_name,dict(
        train_original=original,train_models=scores,
        predeclared_f1_gain=THRESHOLD_CV_GAIN,
        predeclared_improved_folds=MIN_IMPROVED_TRAIN_FOLDS)


def report_on_set(records,heads_fitted):
    x=np.concatenate([s["features"] for s in records])
    y=np.concatenate([s["truth"] for s in records])
    predictions={
        "S614_original_K":np.concatenate([s["original_pred"] for s in records]),
        "tone_mild_diagnostic":np.concatenate([s["mild_pred"] for s in records]),
        "tone_dark_diagnostic":np.concatenate([s["dark_pred"] for s in records]),
    }
    for name,model in heads_fitted.items():
        predictions[name]=model.predict(project(x,name))
    return {name:metrics(y,p) for name,p in predictions.items()}


def run(out):
    if out.exists():raise FileExistsError("S654 evidence immutable")
    frozen=json.loads((ROOT/"selection_frozen.json").read_text())
    assert digest(MODEL.read_bytes())==frozen["model_sha256"]
    assert digest((ROOT/"protocol.json").read_bytes())==frozen["protocol_sha256"]
    proto=json.loads((ROOT/"protocol.json").read_text())
    train=[s for s in proto["sources"] if s["corpus"]=="GuitarSet" and s["split"]=="train"]
    dev=[s for s in proto["sources"] if s["corpus"]=="GuitarSet" and s["split"]=="dev"]
    tests=[s for s in proto["sources"] if s["split"]=="test"]
    assert len(train)==36 and len(dev)==12 and len({s["composition"] for s in train})==9
    assert not {(s["corpus"],s["composition"]) for s in train}&{
        (s["corpus"],s["composition"]) for s in dev+tests}
    heads=load_heads(MODEL,expected_sha256=frozen["model_sha256"])
    started=time.perf_counter()
    with tempfile.TemporaryDirectory() as td:
        root=Path(td);(root/"data").mkdir()
        for key,filename in (("GuitarSet_audio","GuitarSet_audio.zip"),
                             ("GuitarSet_annotations","GuitarSet_annotations.zip")):
            d=proto["guitar_archives"][key]
            download(d["url"],root/"data"/filename,d["size"],d["sha256"])
        with zipfile.ZipFile(root/"data/GuitarSet_audio.zip") as arc:
            training=[]
            for idx,spec in enumerate(train,1):
                item=one(spec,root,arc,heads)
                training.append(item)
                print("S654 TRAIN K",idx,"/36",item["name"],len(item["truth"]),flush=True)
            fitted,choice,best,cv=train_cv(training)
            print("S654 TRAIN CHOICE",json.dumps(dict(choice=choice,best=best,
                train_base_accuracy=cv["train_original"]["accuracy"])),flush=True)
            evaluated=[]
            for idx,spec in enumerate(dev,1):
                item=one(spec,root,arc,heads)
                evaluated.append(item)
                print("S654 REAL DEV K",idx,"/12",item["name"],len(item["truth"]),flush=True)
    outcome=report_on_set(evaluated,fitted)
    print("S654 SCORES",json.dumps({n:dict(
        correct=v["correct"],n=v["n"],accuracy=v["accuracy"])
        for n,v in outcome.items()}),flush=True)
    report=dict(
        status="S654_TONE_3_VIEW_GROUPED_TRAIN_EXACTK_DIAGNOSTIC",
        K_Diagnostic_at_ANNOTATED_ONSET_PLUS_30ms=True,
        this_is_not_autonomous_LIVE_K=True,
        model_original_sha=frozen["model_sha256"],
        native_S614_pitch_count_heads_unchanged=True,
        original_NOTE_selection_untouched=True,
        conditions="causal_original_2800_1100_Tone",
        training_original_GuitarSet=36,
        validation_GuitarSet_DEV=12,
        TEST_unopened=True,
        DEV_historically_exposed=True,
        selected_by_TRAIN_group_CV=choice,
        best_cv_alternative=best,
        train_cv=cv,
        dev_result=outcome,
        dev_per_source=[dict(name=s["name"],composition=s["composition"],
              reference=metrics(s["truth"],s["original_pred"]),
              chosen=(metrics(s["truth"],fitted[choice].predict(
                     project(s["features"],choice))) if choice in fitted
                     else metrics(s["truth"],s["original_pred"])))
              for s in evaluated],
        elapsed_seconds=round(time.perf_counter()-started,2),
        promoted=False)
    out.mkdir(parents=True)
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("S654 FINAL",json.dumps(dict(choice=choice,
        baseline=outcome["S614_original_K"]["accuracy"],
        best_observed=sorted(((n,m["accuracy"]) for n,m in outcome.items()),
                             key=lambda z:-z[1])[:3])),flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",required=True,type=Path)
    args=ap.parse_args()
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        run(args.out)
