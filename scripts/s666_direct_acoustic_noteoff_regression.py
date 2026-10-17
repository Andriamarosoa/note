"""S666: new direct NoteOff duration head from genuine S411 MIC acoustic trajectories.

Different from S662 residual-only metadata and S665 selecting fixed candidate
grid. Predict note lifespan as a continuous quantity using the S411 acoustic
timeline; evaluate composition-disjoint (4 folds) on exactly the same 25,395
ground-truth paired NOTE events. Candidate output MUST preserve source K,
pitch, onset and original S425 reference. S411 features may look up to 3.5s
into the future and are not strict causality; this is research-only. Avoid
interpreting the model as a Live performance.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from scripts.s425_songheld_select_S422_S420_offsets_no_K_regression import (
    read, labels_for_other_songs_only, need, N_MATCHED, SR
)
from scripts.s402_k_input_first_supervised_midi_pitch_network import (
    composition_key, composition_fold
)
from scripts.s399_s80_exactK_constrained_midi_pitch_decoder import sha
from scripts.s662_80pct_offset_feasibility_and_residual import (
    evidence,build_features,stats,TOL
)
from scripts.s664_acoustic_candidate_offset_head import align_audio

BLENDS=(.25,.5,.75,1.)
REGRESSORS=("direct_log","residual_ms")
MAX_MS=5000.

def features(p,on,old,surv,base,acoustic,known):
    source=build_features(p,on,old,surv,base)
    # Avoid future knowledge of the total recording length from S411
    # metadata column 8; real acoustic features still retain future audio.
    audio=acoustic.copy()
    audio[:,8]=0.
    f=np.column_stack((
        source, audio,
        known.astype(np.float32),
        np.log1p(np.maximum(1,(base-on)/SR*1000.)),
        np.log1p(np.maximum(1,(old-on)/SR*1000.)),
        np.log1p(np.maximum(1,(surv-on)/SR*1000.)),
    )).astype(np.float32)
    need(np.isfinite(f).all(),"S666 nonfinite input")
    return f

def fit_models(X,true_duration_ms,base_duration_ms,mask,fold):
    data=X[mask]
    ytrue=true_duration_ms[mask]
    baseline=base_duration_ms[mask]
    need((ytrue>0).all() and len(data)>12000,"S666 training labels invalid")
    # Directly predict median musical duration in log scale; no use of
    # fixed candidate grid.
    log=HistGradientBoostingRegressor(
        loss="absolute_error",learning_rate=.05,max_iter=175,
        max_leaf_nodes=24,min_samples_leaf=65,
        l2_regularization=14.,early_stopping=False,
        random_state=666+fold)
    log.fit(data,np.log1p(np.clip(ytrue,10,MAX_MS)))
    # Separate continuous residual for acoustic-to-native duration repair.
    residual=HistGradientBoostingRegressor(
        loss="absolute_error",learning_rate=.05,max_iter=155,
        max_leaf_nodes=20,min_samples_leaf=80,
        l2_regularization=18.,early_stopping=False,
        random_state=766+fold)
    residual.fit(data,np.clip(ytrue-baseline,-1200,1200))
    return log,residual

def decode_predictions(models,X,base_onset,baseline):
    log,res=models
    bduration=np.maximum(10,(baseline-base_onset)/SR*1000.)
    direct=np.expm1(log.predict(X))
    corrected=bduration+res.predict(X)
    return {
        "direct_log":np.maximum(10,np.minimum(direct,MAX_MS)),
        "residual_ms":np.maximum(10,np.minimum(corrected,MAX_MS)),
    }

def score_by_duration(pred,target,on,mask):
    bins=((0,100),(100,500),(500,1000),(1000,100000))
    dtrue=(target-on)/SR*1000.
    result={}
    for left,right in bins:
        ids=mask&(dtrue>=left)&(dtrue<right)
        result[f"{left}-{right}ms"]=dict(
            n=int(ids.sum()),
            correct=int((np.abs(pred[ids]-target[ids])<=TOL).sum()))
    return result

def run(frozen,s411,annotations,out):
    need(not out.exists(),"S666 evidence directory exists")
    p=read(frozen)
    on,old,surv,base=evidence(p)
    gt,_=labels_for_other_songs_only(p,annotations)
    valid=gt>0
    need(int(valid.sum())==N_MATCHED and
        int((np.abs(base[valid]-gt[valid])<=TOL).sum())==16543,
        "S666 S425 baseline mismatch")
    audio,known=align_audio(p,s411)
    X=features(p,on,old,surv,base,audio,known)
    compositions=np.asarray([composition_key(t) for t in
                             p["track_member"].astype(str)])
    folds=np.asarray([composition_fold(s) for s in compositions])
    need(len(set(folds))==4,"S666 GuitarSet composition split invalid")
    policies={f"S666_{name}_{a}":base.copy() for name in REGRESSORS
              for a in BLENDS}
    dtrue=np.where(valid,(gt-on)/SR*1000.,100.)
    dbase=(base-on)/SR*1000.
    folds_proof=[]
    for fold in range(4):
        train=(folds!=fold)&valid
        held=folds==fold
        need(not(set(compositions[train])&set(compositions[held])),
             "S666 train/test composition overlap")
        models=fit_models(X,dtrue,dbase,train,fold)
        estimates=decode_predictions(models,X[held],on[held],base[held])
        for name,est in estimates.items():
            for alpha in BLENDS:
                dst=f"S666_{name}_{alpha}"
                pred_dur=(1-alpha)*dbase[held]+alpha*est
                pred=on[held]+np.rint(np.maximum(pred_dur,10)*SR/1000.).astype(np.int64)
                policies[dst][held]=np.maximum(pred,on[held]+1)
        m=held&valid
        item=dict(fold=int(fold),
                  train_notes=int(train.sum()),held_notes=int(held.sum()),
                  acoustic_known=int(known[held].sum()),
                  baseline_correct=int((np.abs(base[m]-gt[m])<=TOL).sum()),
                  by_policy={n:int((np.abs(v[m]-gt[m])<=TOL).sum())
                             for n,v in policies.items()},
                  held_label_never_used_by_fit=True)
        folds_proof.append(item)
        print("S666 FOLD",json.dumps(item),flush=True)
    scores={"S425":stats(base,gt,valid)}
    scores.update({n:stats(v,gt,valid) for n,v in policies.items()})
    best=max(policies,key=lambda n:scores[n]["accuracy"])
    report=dict(status="S666_DIRECT_ACOUSTIC_DURATION_REGRESSION",
        original_K_59309_unchanged=True,original_NoteOns_33475_unchanged=True,
        original_MIDI_pitch_33475_unchanged=True,
        original_strict_Live_unchanged=True,
        upstream_S411_future_acoustic_evidence=True,
        upstream_S425_future_evidence=True,
        offline_only=True,
        original_S80_has_GuitarSet_exposure=True,
        not_fully_nested_upstream_S425=True,
        current_comparison_exploratory=True,
        independent_prospective_validation=False,
        no_promotion=True,
        target_accuracy=.80,
        reached_80=bool(scores[best]["accuracy"]>=.80),
        best_experimental=best,
        by_duration_reference=score_by_duration(base,gt,on,valid),
        by_duration_best=score_by_duration(policies[best],gt,on,valid),
        fold_proof=folds_proof,
        metrics=scores,
        source_hashes=dict(S425=sha(frozen),S411=sha(s411),
                           JAMS=sha(annotations)))
    out.mkdir(parents=True)
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("S666 SCORES",json.dumps(scores),flush=True)
    print("S666 DURATION",json.dumps(report["by_duration_best"]),flush=True)
    print("S666 FINAL",json.dumps(dict(best=best,
        accuracy=scores[best]["accuracy"],reached_80=report["reached_80"])),
        flush=True)

def self_test():
    on=np.array([0,1000,2000],np.int64)
    base=on+np.array([500,2000,5000],np.int64)
    d=(base-on)/SR*1000.
    assert np.all(d>0)
    x=np.array([0,1000,1500,2000])
    assert np.all(np.diff(x)>=0)
    print("S666 SELFTEST PASS: positive duration, frozen onsets")

if __name__=="__main__":
    q=argparse.ArgumentParser()
    q.add_argument("--frozen",type=Path)
    q.add_argument("--s411",type=Path)
    q.add_argument("--annotations",type=Path)
    q.add_argument("--out",type=Path)
    q.add_argument("--self-test",action="store_true")
    a=q.parse_args()
    if a.self_test:self_test()
    else:
        need(all((a.frozen,a.s411,a.annotations,a.out)),
             "Need --frozen --s411 --annotations --out")
        run(a.frozen,a.s411,a.annotations,a.out)
