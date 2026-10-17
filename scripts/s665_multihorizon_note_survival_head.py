"""S665 -- multi-horizon note survival evidence for candidate NoteOff selection.

Research-only. Frozen GuitarSet S425 predictions and original K/Onset/MIDI
unchanged. Train four composition-disjoint folds. A separate model can focus
on short, medium, long NOTE durations; never use actual held duration at
routing time, only proposed offset age. Do not deploy into causal live:
S411 historical centered STFT and original S425 future WAV use are noncausal.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from scripts.s425_songheld_select_S422_S420_offsets_no_K_regression import (
    read, labels_for_other_songs_only, need, N_MATCHED, SR
)
from scripts.s402_k_input_first_supervised_midi_pitch_network import (
    composition_key, composition_fold
)
from scripts.s399_s80_exactK_constrained_midi_pitch_decoder import sha
from scripts.s662_80pct_offset_feasibility_and_residual import (
    evidence, candidates, build_features, stats, TOL
)
from scripts.s663_candidate_offset_ranker import get_matrix, pick_from_prob
from scripts.s664_acoustic_candidate_offset_head import (
    AGES, align_audio, joined_features
)

PENALTIES=(.025,.06)
EXPERT_EDGES=(300.,1000.)
MODEL_KW=dict(loss="log_loss",learning_rate=.05,max_iter=155,
              max_leaf_nodes=24,min_samples_leaf=75,
              l2_regularization=15.,early_stopping=False)
MAX_NOTE_AGE_H=3500

def survival_history_features(acoustic,known,C,on,indices):
    """Additional history summary indexed only from acoustic anchors at
    times <= proposed NoteOff age. Never use future anchors to decide
    a candidate's score. Previous S411 STFT itself is centered and
    original native offsets remain noncausal; research only.
    """
    n=len(indices);m=C.shape[1]
    signal=acoustic[indices,9:].reshape(n,10,5)
    duration=np.maximum(1,(C[indices]-on[indices,None])/SR*1000)
    last=np.searchsorted(AGES,duration,side="right")-1
    valid=(last>=0)&known[indices,None]
    li=np.clip(last,0,9)
    row=np.arange(n)[:,None]
    summary=[]
    for step in (0,1,2,3):
        past=np.maximum(li-step,0)
        z=signal[row,past]
        exist=valid & (li>=step) & (z[:,:,0]>.5)
        summary.extend([
            np.where(exist,z[:,:,j],0) for j in (1,2,3,4)])
        summary.append(exist.astype("float32"))
    # Full trajectory summary only over anchors <= candidate age:
    horizon=np.arange(10)[None,None,:]
    mask=(horizon<=last[:,:,None]) & known[indices,None,None]
    powers=np.broadcast_to(signal[:,None,:,1],(n,m,10))
    rises=np.broadcast_to(signal[:,None,:,2],(n,m,10))
    falls=np.broadcast_to(signal[:,None,:,3],(n,m,10))
    energies=np.broadcast_to(signal[:,None,:,4],(n,m,10))
    count=np.maximum(mask.sum(axis=-1),1)
    for arr in (powers,rises,falls,energies):
        val=np.where(mask,arr,0.)
        summary.extend([
            val.sum(axis=-1)/count,
            np.max(val,axis=-1),
            np.where(mask.any(axis=-1),
                     np.min(np.where(mask,arr,np.inf),axis=-1),0.)
        ])
    # Dynamic relative decline and short-vs-history disagreement.
    current=np.where(valid,signal[row,li,1],0.)
    power_at_on=acoustic[indices,2][:,None]
    summary.extend([
        np.where(valid,current-power_at_on,0.),
        np.where(valid,np.exp(np.clip(current-power_at_on,-12,8)),0.),
        np.log1p(duration)/10.,
        valid.astype("float32")
    ])
    z=np.stack(summary,axis=-1).reshape(n*m,-1)
    need(z.shape==(n*m,36) and np.isfinite(z).all(),
         "S665 causal-horizon history malformed")
    return z.astype(np.float32)

def full_features(X,C,on,old,surv,base,acoustic,known,indices):
    static=joined_features(X,C,on,old,surv,base,acoustic,known,indices)
    extra=survival_history_features(acoustic,known,C,on,indices)
    z=np.column_stack((static,extra)).astype(np.float32)
    need(z.shape[0]==len(indices)*C.shape[1] and np.isfinite(z).all(),
         "S665 fused multihorizon features invalid")
    return z

def train_general(features,correct,seed):
    y=correct.astype(np.uint8)
    need(int(y.sum())>500 and int((1-y).sum())>500,
         "S665 class balance not sufficient")
    model=HistGradientBoostingClassifier(random_state=seed,**MODEL_KW)
    model.fit(features,y)
    return model

def train_experts(features,correct,duration,seed):
    experts=[]
    for k,(left,right) in enumerate(((0.,300.),(300.,1000.),(1000.,float("inf")))):
        take=(duration>=left)&(duration<right)
        if take.sum()<1200 or correct[take].sum()<50 or (~correct[take]).sum()<50:
            experts.append(None)
            continue
        config=dict(MODEL_KW)
        config.update(max_iter=125,max_leaf_nodes=19,min_samples_leaf=40)
        model=HistGradientBoostingClassifier(random_state=seed+k+20,**config)
        model.fit(features[take],correct[take])
        experts.append(model)
    return experts

def probabilities(experts, general, feats, ages, use_expert):
    scores=general.predict_proba(feats)[:,1]
    if not use_expert:
        return scores
    for i,(left,right) in enumerate(((0.,300.),(300.,1000.),(1000.,float("inf")))):
        here=(ages>=left)&(ages<right)
        if here.any() and experts[i] is not None:
            scores[here]=experts[i].predict_proba(feats[here])[:,1]
    return scores

def run(frozen,s411,annotations,out):
    need(not out.exists(),"S665 immutable report already exists")
    p=read(frozen)
    on,old,surv,base=evidence(p)
    target,_=labels_for_other_songs_only(p,annotations)
    matched=target>0
    need(int(matched.sum())==N_MATCHED and
         int((np.abs(base[matched]-target[matched])<=TOL).sum())==16543,
         "S665 S425 protected baseline not recovered")
    acoustics,known=align_audio(p,s411)
    X=build_features(p,on,old,surv,base)
    names,C=get_matrix(candidates(on,old,surv,base))
    composers=np.asarray([composition_key(t) for t in
                          p["track_member"].astype(str)])
    folds=np.asarray([composition_fold(c) for c in composers])
    need(len(np.unique(folds))==4,"Expected 4 held-fold splits")
    configs=tuple((mode,penalty) for mode in ("history","age_experts")
                  for penalty in PENALTIES)
    result={f"S665_{mode}_{penalty}":base.copy() for mode,penalty in configs}
    fold_stats=[]
    for fold in range(4):
        held=folds==fold
        train=(folds!=fold)&matched
        need(not(set(composers[held])&set(composers[train])),
             "S665 leakage across music compositions")
        tr=np.flatnonzero(train);va=np.flatnonzero(held)
        ftr=full_features(X,C,on,old,surv,base,acoustics,known,tr)
        corr=(np.abs(C[tr]-target[tr,None])<=TOL).reshape(-1)
        trainage=((C[tr]-on[tr,None])/SR*1000).reshape(-1)
        general=train_general(ftr,corr,665+fold)
        experts=train_experts(ftr,corr,trainage,765+fold)
        fv=full_features(X,C,on,old,surv,base,acoustics,known,va)
        vaage=((C[va]-on[va,None])/SR*1000).reshape(-1)
        for mode in ("history","age_experts"):
            proba=probabilities(experts,general,fv,vaage,
                                use_expert=(mode=="age_experts"))
            proba=proba.reshape(len(va),len(names))
            for penalty in PENALTIES:
                result[f"S665_{mode}_{penalty}"][va]=pick_from_prob(
                    proba,C[va],names,penalty)
        query=held&matched
        row=dict(fold=int(fold),train=int(train.sum()),held=int(held.sum()),
            held_correct_onset_notes=int(query.sum()),
            s411_matched=int(known[va].sum()),
            age_experts_trained=sum(q is not None for q in experts),
            baseline=int((np.abs(base[query]-target[query])<=TOL).sum()),
            correct_by_policy={
                name:int((np.abs(offset[query]-target[query])<=TOL).sum())
                for name,offset in result.items()
            },train_labels_used_to_score_held=False)
        fold_stats.append(row)
        print("S665 FOLD",json.dumps(row),flush=True)
    scores={"S425":stats(base,target,matched)}
    scores.update({k:stats(v,target,matched) for k,v in result.items()})
    report=dict(status="S665_MULTIHORIZON_NOTE_SURVIVAL_HEAD",
        source_S425_immutable=True,source_S80_original_K_immutable=True,
        original_59309_K_events_unchanged=True,
        original_33475_note_pitches_and_attacks_unchanged=True,
        native_strict_Live_unchanged=True,
        upstream_S425_and_S411_can_use_future=True,
        S411_centered_STFT_noncausal=True,
        historically_exposed_GuitarSet=True,
        evaluation_by_held_composition=True,
        S425_non_nested_training=True,
        frozen_promotion=False,
        no_independent_test_passed=True,
        target_offset_50ms=.80,
        goal_reached=any(r["accuracy"]>=.80 for r in scores.values()),
        s411_coverage=int(known.sum()),folds=fold_stats,metrics=scores,
        input_hashes=dict(S425=sha(frozen),S411=sha(s411),
                          JAMS=sha(annotations)))
    out.mkdir(parents=True)
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("S665 RESULTS",json.dumps(scores),flush=True)
    print("S665 FINAL",json.dumps(dict(
        best=max(x["accuracy"] for k,x in scores.items() if k!="S425"),
        reference=scores["S425"]["accuracy"],
        goal_reached=report["goal_reached"])),flush=True)

def self_test():
    acoustics=np.zeros((2,59),dtype=np.float32)
    acoustics[:,2]=5
    acoustics[:,9::5]=1
    acoustics[:,10::5]=2
    a=np.asarray([[1200,5000],[5000,30000]],np.int64)
    old=survival_history_features(acoustics,np.array([True,True]),
                                  a,np.zeros(2,dtype=np.int64),np.array([0,1]))
    assert old.shape==(4,36)
    # Alter only 1s+ features; early 27/113ms candidates cannot change.
    changed=acoustics.copy()
    changed[:,9+5*6:]=100
    new=survival_history_features(changed,np.array([True,True]),
                                  a,np.zeros(2,dtype=np.int64),np.array([0,1]))
    assert np.array_equal(old[0],new[0])
    assert np.array_equal(old[1],new[1])
    print("S665 SELFTEST PASS: history size and future anchor invisibility")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--frozen",type=Path)
    ap.add_argument("--s411",type=Path)
    ap.add_argument("--annotations",type=Path)
    ap.add_argument("--out",type=Path)
    ap.add_argument("--self-test",action="store_true")
    q=ap.parse_args()
    if q.self_test:self_test()
    else:
        need(all((q.frozen,q.s411,q.annotations,q.out)),
             "need --frozen --s411 --annotations --out")
        run(q.frozen,q.s411,q.annotations,q.out)
