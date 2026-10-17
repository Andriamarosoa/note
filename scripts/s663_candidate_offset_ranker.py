"""S663 — select NoteOff using a trained per-candidate probability-of-success head.

A new selection head scores whether each physically feasible proposed
duration (S420/S422/S425 and a preregistered interpolation/stretch grid)
will fall within 50 ms of the true offset. The label is read for training
songs only; all held-song selections are label-free. 4 song-composition
folds. S425 baseline, MIDI pitch/onset and all 59,309 original S80 K
predictions frozen. Research-only: upstream meta heads not fully nested,
GuitarSet historically exposed, and input offsets can incorporate up to
5 seconds of future audio. DO NOT integrate this into strict-Live.
"""
from __future__ import annotations
import argparse,collections,json
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from scripts.s425_songheld_select_S422_S420_offsets_no_K_regression import (
    read,labels_for_other_songs_only,need,SR,N_MATCHED,
)
from scripts.s402_k_input_first_supervised_midi_pitch_network import (
    composition_key,composition_fold,
)
from scripts.s399_s80_exactK_constrained_midi_pitch_decoder import sha
from scripts.s662_80pct_offset_feasibility_and_residual import (
    evidence,candidates,build_features,stats,compute_oracles,TOL,
)

PENALTIES=(0.,.025,.06,.12)
WINNER_THRESHOLD=.80


def get_matrix(cands):
    names=tuple(cands)
    return names,np.column_stack([cands[n] for n in names]).astype(np.int64)


def candidate_matrix_features(X,C,on,old,surv,base,indices):
    """Input: candidate-independent native predictions/features, no GT.
    One row per candidate. Can score new variants without changing source K.
    """
    n=len(indices);m=C.shape[1]
    baseX=np.repeat(X[indices],m,axis=0)
    durations=np.maximum(1,(C[indices]-on[indices,None])/SR*1000.)
    duration425=np.maximum(1,(base[indices]-on[indices])/SR*1000.)
    duration420=np.maximum(1,(old[indices]-on[indices])/SR*1000.)
    duration422=np.maximum(1,(surv[indices]-on[indices])/SR*1000.)
    db=duration425[:,None]
    d0=duration420[:,None]
    d1=duration422[:,None]
    family=np.broadcast_to(np.arange(m,dtype=np.float32)[None,:],(n,m))
    extra=np.column_stack((
        family.ravel(),
        np.log1p(durations).ravel(),
        np.log1p(np.abs(durations-db)).ravel(),
        np.log1p(np.abs(durations-d0)).ravel(),
        np.log1p(np.abs(durations-d1)).ravel(),
        ((durations-db)/1000.).ravel(),
        ((durations-d0)/1000.).ravel(),
        ((durations-d1)/1000.).ravel(),
        (durations/db).ravel(),
    )).astype(np.float32)
    out=np.column_stack((baseX,extra)).astype(np.float32)
    need(out.shape==(n*m,X.shape[1]+9) and np.isfinite(out).all(),
         "S663 candidate ranking features wrong dimension or nonfinite")
    return out


def train_ranking(X,C,on,old,surv,base,target,idx,seed):
    features=candidate_matrix_features(X,C,on,old,surv,base,idx)
    truth=target[idx,None]
    positive=(np.abs(C[idx]-truth)<=TOL).reshape(-1)
    need(positive.sum()>=3000 and (~positive).sum()>=4000,
         "S663 insufficient real corrected/incorrect candidates")
    model=HistGradientBoostingClassifier(
        loss="log_loss",learning_rate=.065,max_iter=130,
        max_leaf_nodes=15,min_samples_leaf=90,
        l2_regularization=20.,early_stopping=False,
        random_state=seed,
    )
    model.fit(features,positive)
    return model,dict(train_candidate_rows=int(len(positive)),
                      positives=int(positive.sum()),
                      negative=int((~positive).sum()))


def pick_from_prob(prob,C,names,penalty):
    """Selection uses scores ONLY. No true note end, no oracle labels."""
    s=prob.copy()
    j=names.index("S425")
    prior=np.full(len(names),float(penalty),dtype=np.float32)
    prior[j]=0.
    s-=prior[None,:]
    choice=np.argmax(s,axis=1)
    return C[np.arange(len(C)),choice]


def run(frozen,annotations,out):
    need(not out.exists(),"S663 immutable evidence directory already exists")
    p=read(frozen)
    on,old,surv,base=evidence(p)
    target,gt=labels_for_other_songs_only(p,annotations)
    matched=target>0
    need(int(matched.sum())==N_MATCHED and
         int((np.abs(base[matched]-target[matched])<=TOL).sum())==16543,
         "Frozen S425 16543/25395 baseline changed")
    cands=candidates(on,old,surv,base)
    names,C=get_matrix(cands)
    X=build_features(p,on,old,surv,base)
    comps=np.asarray([composition_key(q) for q in
                      p["track_member"].astype(str)])
    folds=np.asarray([composition_fold(c) for c in comps])
    need(len(set(folds))==4,"S663 expected four held-composition folds")
    predictions={f"S663_rank_penalty_{x}":base.copy() for x in PENALTIES}
    fold_ledger=[]
    for fold in range(4):
        hold=(folds==fold)
        train=(folds!=fold)&matched
        testids=np.flatnonzero(hold)
        trainids=np.flatnonzero(train)
        need(not(set(comps[train])&set(comps[hold])) and
             len(trainids)>12000 and len(testids)>4000,
             "S663 composition intersection or invalid fold")
        model,training=train_ranking(X,C,on,old,surv,base,
                                     target,trainids,663+fold)
        score=model.predict_proba(candidate_matrix_features(
             X,C,on,old,surv,base,testids))[:,1].reshape(len(testids),len(names))
        for penalty in PENALTIES:
            col=f"S663_rank_penalty_{penalty}"
            predictions[col][testids]=pick_from_prob(
                score,C[testids],names,penalty)
        fold_ledger.append(dict(
             fold=int(fold),
             train_songs=len(set(comps[train])),
             held_songs=len(set(comps[hold])),
             overlap=0,
             training=training,
             correct_425=int(
                 (np.abs(base[hold&matched]-target[hold&matched])<=TOL).sum()),
             corrected={
                name:int((np.abs(value[hold&matched]-
                                   target[hold&matched])<=TOL).sum())
                for name,value in predictions.items()
             },
             held_ground_truth_not_training_input=True,
        ))
        print("S663 FOLD",json.dumps(fold_ledger[-1]),flush=True)
    metrics={"S425":stats(base,target,matched)}
    for key,v in predictions.items():
        metrics[key]=stats(v,target,matched)
    audit=compute_oracles(cands,target,matched)
    selected=None # no promotion on historically exposed GuitarSet
    result=dict(
       status="S663_HELD_COMPOSITION_CANDIDATE_RANKER_OFFSET_AUDIT",
       original_S80_exact_K_unchanged=True,
       MIDI_note_pitches_and_onsets_unchanged=True,
       historical_GuitarSet_upstream_overlap=True,
       old_S425_not_fully_nested=True,
       strict_live_not_modified=True,
       future_lookahead_for_upstream_offsets_up_to_5s=True,
       scored_only_held_composition_predictions=True,
       candidate_count=len(names),
       candidate_names=list(names),
       frozen_selection=None,
       target_80_percent_within_50ms=True,
       reached_80_percent=bool(any(
            r["accuracy"]>=WINNER_THRESHOLD
            for n,r in metrics.items() if n!="S425")),
       oracle_is_diagnostic_only=True,
       theoretical_oracle=audit,
       fold_proofs=fold_ledger,
       metrics=metrics,
       no_automatic_promotion=True,
       source_hashes=dict(S425=sha(frozen),JAMS=sha(annotations)),
    )
    out.mkdir(parents=True)
    (out/"report.json").write_text(json.dumps(result,indent=2)+"\n")
    print("S663 SCORES",json.dumps(metrics),flush=True)
    print("S663 FINAL",json.dumps(dict(
       baseline=metrics["S425"]["accuracy"],
       best_experimental=max((r["accuracy"] for name,r in metrics.items()
                               if name!="S425")),
       oracle=audit["diagnostic_interpolation_grid"]["accuracy"],
       achieved_80=result["reached_80_percent"])),flush=True)


def self_test():
    base=np.array([[10,20,30],[40,50,60]],np.int64)
    names=("S420","S425","S422")
    proba=np.array([[.6,.65,.85],[.60,.65,.64]],np.float32)
    normal=pick_from_prob(proba,base,names,0.)
    guarded=pick_from_prob(proba,base,names,.25)
    assert np.array_equal(normal,[30,50])
    assert np.array_equal(guarded,[20,50])
    print("S663 SELFTEST PASS: candidate probabilities and conservative baseline selection")


if __name__=="__main__":
    q=argparse.ArgumentParser()
    q.add_argument("--frozen",type=Path)
    q.add_argument("--annotations",type=Path)
    q.add_argument("--out",type=Path)
    q.add_argument("--self-test",action="store_true")
    a=q.parse_args()
    if a.self_test:self_test()
    else:
        need(a.frozen and a.annotations and a.out,
             "Supply --frozen --annotations --out")
        run(a.frozen,a.annotations,a.out)
