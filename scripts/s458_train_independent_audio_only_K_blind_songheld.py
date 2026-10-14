"""S458 -- new 1090D acoustic K0..K6 candidate, no original S80 modification.

Train four GuitarSet composition-held acoustic models on original real V88
S258 events only. Never use S80 predicted K as a classifier input
(not true K) and real S97 attack/flow/physics/morphology features. For
each held composition fold, an inner DIFFERENT composition subset selects
risk-gated thresholds individually by *predicted K*, never by actual K
at inference. Final fold model is fitted on all OTHER compositions.
All 59309 S80 official K remain separate and untouched. This is a
GuitarSet DEV model: original S80/S397 previously exposed to GuitarSet.
No YourMT teacher or external IDMT labels used.
"""
from __future__ import annotations
import argparse,hashlib,json,collections
from pathlib import Path
import joblib,numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from scripts.s402_k_input_first_supervised_midi_pitch_network import (
    load,need,composition_key,composition_fold)
from scripts.s444_balanced_joint_note_swap_songheld_Klocked import balanced_inner_partition
from scripts.s399_s80_exactK_constrained_midi_pitch_decoder import sha

N=59309
FOLDS=4
DIM=1090
K_MIN=0
K_MAX=6
GATES=(0,1,2,3,4)
THRESHOLDS=(.15,.25,.35,.45,.55,.65,.75,.85)
SEED=458
REQUIRE_CORRECTIONS=5
MIN_NET=5

def inputs(data,K):
    n=len(K)
    need(np.isin(K,range(7)).all()
         and all(data[key].shape==(n,width) for key,width in
          (("summary58",58),("flow245",245),("morphology490",490),
           ("physics294",294),("geometry3",3))),
         "S458 original acoustic geometry or K0..6 invalid")
    # Deliberately NEVER concatenate the original K or source logits.
    # This is the S457 motivated ablation: acoustic-only K prediction.
    x=np.column_stack((data["summary58"],data["flow245"],
        data["morphology490"],data["physics294"],data["geometry3"])).astype(np.float32)
    need(x.shape==(n,DIM) and np.isfinite(x).all(),
         "S458 1090 real acoustic head feature schema invalid")
    return x

def model(seed):
    return ExtraTreesClassifier(
        n_estimators=85,min_samples_leaf=9,max_depth=15,
        max_features=.5,class_weight="balanced_subsample",
        n_jobs=2,random_state=seed)

def proba_model(est,x):
    scores=est.predict_proba(x)
    need(np.array_equal(est.classes_,np.arange(7))
         and scores.shape==(len(x),7)
         and np.isfinite(scores).all(),
         "S458 trained original K0..6 model missing classes")
    return scores

def candidate(source,scores,threshold_by_original_K):
    """Non-oracular optional K edits: no threshold for K => abstain."""
    old=np.asarray(source,np.int8)
    res=old.copy()
    p=np.argmax(scores,axis=1)
    certainty=np.max(scores,axis=1)
    for k,value in threshold_by_original_K.items():
        use=(old==int(k))&(p!=old)&(certainty>=float(value))
        res[use]=p[use]
    need(res.shape==old.shape and np.isin(res,range(7)).all(),
         "S458 invalid categorical K intervention")
    return res

def ledger(y,old,new,indices):
    ii=np.asarray(indices,int)
    if not len(ii):return dict(n=0,fixes=0,regressions=0,net=0)
    a=old[ii]==y[ii];b=new[ii]==y[ii]
    return dict(n=int(len(ii)),fixes=int(np.sum(~a&b)),
        regressions=int(np.sum(a&~b)),net=int(b.sum()-a.sum()),
        correct_base=int(a.sum()),correct_new=int(b.sum()))

def run(audio,s80,truth,out):
    need(not out.exists(),"S458 cannot overwrite prior source experiment")
    a=load(audio);s=load(s80);t=load(truth)
    ids=a["row_id"].astype(str)
    K=s["s80_K"].astype(np.int8)
    y=t["true_K"].astype(np.int8)
    need(len(ids)==N and len(set(ids))==N
         and np.array_equal(ids,s["row_id"].astype(str))
         and np.array_equal(ids,t["row_id"].astype(str))
         and np.array_equal(a["incumbent_freeze_K"],s["incumbent_freeze_K"])
         and np.array_equal(a["incumbent_freeze_K"],t["original_freeze_K"])
         and int(np.sum(K==y))==53529
         and int(np.sum((K==y)&(y>=2)))==6054,
         "S458 protected original S80 genuine N=59309 exact-K reference drift")
    x=inputs(a,K)
    comps=np.asarray([composition_key(v) for v in a["member"].astype(str)])
    outer=np.asarray([composition_fold(v) for v in comps],np.int8)
    chosen=K.copy()
    folds=[]
    out.mkdir(parents=True)
    for fold in range(4):
        outertrain=np.flatnonzero(outer!=fold)
        held=np.flatnonzero(outer==fold)
        inner_train,inner_val,partition=balanced_inner_partition(
            outertrain,comps)
        need(len(inner_val)>1000 and len(inner_train)>10000
             and len(held)>1000
             and not(set(comps[held])&set(comps[outertrain]))
             and not(set(comps[inner_val])&set(comps[inner_train])),
             "S458 leakage or insufficient all-K composition holdout")
        inner=model(SEED+fold)
        inner.fit(x[inner_train],y[inner_train])
        scores=proba_model(inner,x[inner_val])
        gate={}
        calibration={}
        for kk in GATES:
            k_indices=inner_val[K[inner_val]==kk]
            choices=[]
            for cut in THRESHOLDS:
                trial=candidate(K[inner_val],scores,{kk:cut})
                truth_relative=y[inner_val]
                pids=np.flatnonzero(K[inner_val]==kk)
                overall=ledger(truth_relative,K[inner_val],trial,pids)
                # All sparse source classes K2/K3/K4 are guarded: a
                # catastrophic rare-class regression can block even a
                # positive mean on the dominant K0/K1 population.
                by_true={
                   str(k):ledger(truth_relative,K[inner_val],trial,
                     np.flatnonzero((K[inner_val]==kk)&(truth_relative==k)))
                    for k in range(7)}
                choices.append(dict(threshold=float(cut),
                    overall=overall,by_true_K=by_true))
            safe=[entry for entry in choices if
                entry["overall"]["fixes"]>=REQUIRE_CORRECTIONS and
                entry["overall"]["net"]>=MIN_NET and
                entry["overall"]["fixes"]>=2*entry["overall"]["regressions"]
                and all(entry["by_true_K"][str(t)]["net"]>=0
                        for t in (2,3,4))]
            best=max(safe,key=lambda q:(q["overall"]["net"],
                -q["overall"]["regressions"],q["threshold"])) if safe else None
            if best:gate[kk]=best["threshold"]
            calibration[str(kk)]=dict(n=int(len(k_indices)),
                candidate_thresholds=choices,
                safe_gate_selected=best)
        del inner
        final=model(SEED+100+fold)
        final.fit(x[outertrain],y[outertrain])
        conf=proba_model(final,x[held])
        chosen[held]=candidate(K[held],conf,gate)
        path=out/f"S458_acoustic_K_head_fold{fold}.joblib"
        joblib.dump(dict(model=final,original_S80_K_required=True,
            feature_width=DIM,held_fold=fold,
            threshold_by_predicted_source_K=gate,
            no_external_IDMT_training_or_labels=True),path,compress=3)
        report=dict(fold=fold,inner_count_balance=partition,
           composition_heldout_song_count=len(set(comps[held])),
           training_songs=len(set(comps[outertrain])),
           train_held_song_overlap=0,
           final_training_examples=len(outertrain),
           held_events=len(held),inner_training_events=len(inner_train),
           inner_validation_events=len(inner_val),
           gates=gate,inner_by_predicted_K=calibration,
           model_SHA256=sha(path),held_labels_not_used_for_model_or_gate=True)
        folds.append(report)
        print("S458 FOLD",json.dumps(dict(fold=fold,
            n_inner_val=len(inner_val),gates=gate,held_events=len(held),
            changes=int(np.count_nonzero(chosen[held]!=K[held])))),flush=True)
    need(len(chosen)==N and np.isin(chosen,range(7)).all(),
         "S458 invalid held-composition prediction count")
    predfile=out/"S458_K_CANDIDATE_GUITARSET_OOF_ONLY.npz"
    np.savez_compressed(predfile,row_id=ids,original_source_S80_K=K,
                        S458_candidate_K=chosen,outer_composition_fold=outer)
    all_ix=np.arange(N)
    poly_ix=np.flatnonzero(y>=2)
    stats=dict(all=ledger(y,K,chosen,all_ix),
        true_poly=ledger(y,K,chosen,poly_ix),
        by_true_K={str(k):ledger(y,K,chosen,
            np.flatnonzero(y==k)) for k in range(7)},
        by_original_predicted_K={str(k):ledger(y,K,chosen,
            np.flatnonzero(K==k)) for k in range(7)})
    result=dict(status="S458_K_BLIND_1090D_ACOUSTIC_ABLATION_COMP_DISJOINT_CANDIDATE",
        original_K_reference_guitarset_global=53529,
        original_K_reference_guitarset_poly=6054,
        original_S80_changed_or_promoted=False,
        threshold_gate_by_original_predicted_K=True,
        labels_used_only_other_compositions_for_training_and_inner_gate=True,
        no_external_IDMT_labels_or_weights=True,
        GuitarSet_historically_exposed_original_S80_models=True,
        not_independent_external_K_benchmark=True,
        no_YourMT_teacher=True,
        source_K_not_a_network_feature=True,
        source_K_used_only_for_post_network_gate=True,
        held_original_S80_K_from_S397=True,
        result_description="GuitarSet retrospective development; originals immutable",
        dimensions=DIM,real_source_N=N,
        original_K_total=int(K.sum()),
        candidate_K_total=int(chosen.sum()),
        candidate_changed_event_count=int(np.count_nonzero(chosen!=K)),
        all_model_folds=folds,
        paired_validation=stats,
        source_SHA256=dict(S258=sha(audio),S397=sha(s80),truth=sha(truth)),
        candidate_prediction_SHA256=sha(predfile),
        no_promotion=True)
    (out/"report.json").write_text(json.dumps(result,indent=2)+"\n")
    print("S458 DEVELOPMENT",json.dumps(dict(
        global_dev=stats["all"],true_poly_dev=stats["true_poly"],
        changed=int(np.count_nonzero(chosen!=K)),old_S80_untouched=True)),
        flush=True)

def selftest():
    rng=np.random.default_rng(SEED)
    mock={}
    for p,width in (("summary58",58),("flow245",245),
           ("morphology490",490),("physics294",294),("geometry3",3)):
        mock[p]=rng.normal(size=(4,width)).astype(np.float32)
    source=np.array([0,2,3,4],np.int8)
    features=inputs(mock,source)
    need(features.shape==(4,DIM),"S458 morphology feature bridge selftest")
    q=np.zeros((4,7),float)
    q[0,3]=.94;q[1,3]=.86;q[2,2]=.91;q[3,3]=.70
    baseline=candidate(source,q,{2:.80,3:.95})
    need(baseline.tolist()==[0,3,3,4],
         "S458 must never override ungated K0 K4 and conservative K3")
    alternate=np.array([4,4,4,4],np.int8)
    need(np.array_equal(inputs(mock,alternate),features),
         "S458 AUDIO-ONLY network feature must not depend on source S80 K")
    print("S458 SELFTEST PASS true K-blind acoustic network, optional source-K gate")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--self-test",action="store_true")
    for v in ("audio","s80","truth","out"):p.add_argument("--"+v,type=Path)
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        for attr in ("audio","s80","truth","out"):
            need(getattr(a,attr)is not None,"S458 missing --"+attr)
        run(a.audio,a.s80,a.truth,a.out)
