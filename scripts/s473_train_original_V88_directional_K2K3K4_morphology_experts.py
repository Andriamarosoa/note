"""S473 - train safe directional K2<->K3<->K4 specialists on original V88, NOT S80.

Why: S471 third corpus piano proves original V88 overpredicts K0
at real MIDI attacks; S472 shows binary S466 birth confidence does not
separate K2/K3/K4. Build separate learned morphology/spectral experts
per *original V88 predicted* K2,K3,K4, not one global K classifier.

Nested whole-composition validation selects source-only-safe proposals:
a proposal must have positive net gain, 2x more fixes than regressions,
and NO worse on any true poly K2,K3,K4 stratum. Original K0,K1,K5,K6
and all S80/live/note identities stay untouched. No external IDMT or
MAESTRO annotations opened, fitted, or used for thresholds.

Original GuitarSet V88 model has historical dataset exposure: this is
development, not an independently blind test.
"""
from __future__ import annotations
import argparse,hashlib,json,collections
from pathlib import Path
import joblib,numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from scripts.s402_k_input_first_supervised_midi_pitch_network import (
    need,load,composition_key,composition_fold)
from scripts.s444_balanced_joint_note_swap_songheld_Klocked import balanced_inner_partition
from scripts.s458_train_independent_audio_only_K_blind_songheld import inputs
from scripts.s399_s80_exactK_constrained_midi_pitch_decoder import sha

SOURCE_KS=(2,3,4)
CUTS=(.30,.40,.50,.60,.70,.80,.90)
SEED=473
N=59309
def model(seed):
    return ExtraTreesClassifier(
        n_estimators=40,min_samples_leaf=12,max_depth=13,
        max_features="sqrt",class_weight="balanced_subsample",
        random_state=seed,n_jobs=2)
def scores(est,X):
    a=est.predict_proba(X)
    r=np.zeros((len(X),7),np.float32)
    r[:,est.classes_.astype(int)]=a
    need(r.shape==(len(X),7) and np.isfinite(r).all(),
         "S473 unchanged predicted V88 K seven-class probability unsupported")
    return r
def proposal(old,probs,k,cut):
    res=np.asarray(old,np.int8).copy()
    available=np.flatnonzero(old==k)
    if len(available):
        chosen=np.argmax(probs[available],axis=1)
        maxprob=probs[available,chosen]
        permitted=(np.abs(chosen-k)==1)&(maxprob>=cut)
        res[available[permitted]]=chosen[permitted]
    need(np.array_equal(res[old!=k],old[old!=k]),
         "S473 source specialist altered protected original V88 classes")
    return res
def ledger(true,old,candidate,ix):
    if not len(ix):return dict(n=0,fix=0,reg=0,net=0)
    a=(old[ix]==true[ix]);b=(candidate[ix]==true[ix])
    fix=int(np.sum(~a&b));reg=int(np.sum(a&~b))
    return dict(n=int(len(ix)),fix=fix,reg=reg,net=fix-reg,
        before=int(a.sum()),after=int(b.sum()))
def gate(p,y,K,k):
    indices=np.flatnonzero(K==k)
    options=[]
    for cut in CUTS:
        cand=proposal(K,p,k,cut)
        allstat=ledger(y,K,cand,indices)
        pertrue={str(yy):ledger(y,K,cand,
             np.flatnonzero((K==k)&(y==yy)))
                 for yy in range(7)}
        valid=(allstat["fix"]>=5 and allstat["net"]>=3
               and allstat["fix"]>=2*allstat["reg"]
               and all(pertrue[str(t)]["net"]>=0 for t in (1,2,3,4)))
        options.append(dict(cut=float(cut),net=allstat["net"],
            fixes=allstat["fix"],regressions=allstat["reg"],
            safe=valid,each_trueK=pertrue))
    candidates=[x for x in options if x["safe"]]
    best=max(candidates,key=lambda a:(a["net"],-a["regressions"],a["cut"])) if candidates else None
    return (best["cut"] if best else None),options,best
def run(audio,truth,out):
    need(not out.exists(),"S473 immutable original V88 specialist experiment exists")
    a=load(audio);t=load(truth)
    ids=a["row_id"].astype(str)
    K=a["incumbent_freeze_K"].astype(np.int8)
    y=t["true_K"].astype(np.int8)
    need(len(ids)==N and len(set(ids))==N
         and np.array_equal(ids,t["row_id"].astype(str))
         and np.array_equal(K,t["original_freeze_K"].astype(np.int8))
         and np.isin(K,range(7)).all() and np.isin(y,range(7)).all()
         and int(np.sum(y>=2))==7385,
         "S473 actual original V88 native GuitarSet or true chord labels drifted")
    features=inputs(a,K)
    members=np.asarray([composition_key(v) for v in a["member"].astype(str)])
    folds=np.asarray([composition_fold(v) for v in members],np.int8)
    selected=K.copy()
    summaries=[]
    out.mkdir(parents=True)
    for k in SOURCE_KS:
        preds_all=np.full((N,7),np.nan,np.float32)
        calibrated=[]
        for outer in range(4):
            tr=np.flatnonzero((folds!=outer)&(K==k))
            held=np.flatnonzero((folds==outer)&(K==k))
            need(len(tr)>60 and len(held)>10 and
                 len(set(y[tr]))>=2 and len(set(members[tr]))>=5,
                 "S473 insufficient actual source V88 K2-K4 sample for one fold")
            sub_train,sub_val,partition=balanced_inner_partition(tr,members)
            need(not(set(members[sub_train])&set(members[sub_val]))
                 and not(set(members[tr])&set(members[held])),
                 "S473 incomplete genuine nested composition independence")
            calibrator=model(SEED+outer*7+k)
            calibrator.fit(features[sub_train],y[sub_train])
            pp=scores(calibrator,features[sub_val])
            c,choices,best=gate(pp,y[sub_val],
                np.full(len(sub_val),k,dtype=np.int8),k)
            del calibrator
            specialist=model(SEED+200+outer*7+k)
            specialist.fit(features[tr],y[tr])
            scores_test=scores(specialist,features[held])
            preds_all[held]=scores_test
            if c is not None:
                selected[held]=proposal(K[held],scores_test,k,c)
            location=out/f"S473_actual_V88_K{k}_outerfold{outer}.joblib"
            joblib.dump(dict(model=specialist,source_predicted_V88_K=k,
                four_fold_outer_id=outer,feature_width=1090,
                original_S80_not_input=True,
                calibration_cut=c,
                no_external_MAESTRO_or_IDMT_training=True),location,compress=3)
            calibrated.append(dict(outer_fold=outer,source_K=k,
                train_count=len(tr),held_count=len(held),
                inner_training_count=len(sub_train),
                inner_validation_count=len(sub_val),
                calibration_protocol=partition,
                selected_cut=c,selected_inner_result=best,
                all_threshold_evidence=choices,
                original_model_SHA256=sha(location),
                heldout_labels_used_to_choose_threshold=False))
            print("S473 SPECIALIST",json.dumps(dict(
                original_predicted_K=k,fold=outer,
                train=len(tr),held=len(held),cut=c,
                altered_proposals=int(np.sum(selected[held]!=K[held])))),flush=True)
        relevant=np.flatnonzero(K==k)
        summaries.append(dict(original_predicted_K=k,
            all_source_events=int(len(relevant)),
            fold_specs=calibrated,
            paired_GuitarSet_development_result=ledger(y,K,selected,relevant),
            per_true_K={str(t):ledger(y,K,selected,
               np.flatnonzero((K==k)&(y==t))) for t in range(7)}))
    other=~np.isin(K,SOURCE_KS)
    need(np.array_equal(selected[other],K[other]),
         "S473 change to protected original predicted K0 K1 K5 K6")
    resultall=ledger(y,K,selected,np.arange(N))
    resultpoly=ledger(y,K,selected,np.flatnonzero(y>=2))
    frozen=out/"S473_SOURCE_V88_GUITARSET_OOF_RESIDUAL_COUNTS.npz"
    np.savez_compressed(frozen,row_id=ids,original_V88_K=K,
                        new_candidate_K=selected,outer_fold=folds)
    rep=dict(status="S473_ACTUAL_V88_SONGHELD_DIRECTIONAL_K2_K3_K4_SPECIALISTS",
        original_source_events=N,
        original_source_V88_K_correct=int(np.sum(K==y)),
        original_source_V88_poly_K_correct=int(np.sum((K==y)&(y>=2))),
        true_poly_events=7385,
        original_S80_training_unchanged=True,
        source_non_poly_classes_K0K1_K5K6_untouched=True,
        source_original_K2K3K4_permitted_adjacent_only=True,
        original_GuitarSet_previous_train_exposure=True,
        nested_composition_calibration_only=True,
        no_external_truth_training_or_threshold_tuning=True,
        nonofficial_GuitarSet_pair_all=resultall,
        nonofficial_GuitarSet_pair_poly=resultpoly,
        per_source_predicted_K=summaries,
        count_K_changes=int(np.sum(selected!=K)),
        source_input_SHA256=dict(original_V88_S258=sha(audio),
            actual_JAMS_truth=sha(truth),candidate_prediction=sha(frozen)),
        no_promotion=True)
    (out/"report.json").write_text(json.dumps(rep,indent=2)+"\n")
    print("S473 FINAL",json.dumps(dict(
        original_V88_score=rep["original_source_V88_K_correct"],
        poly_correct=rep["original_source_V88_poly_K_correct"],
        global_pair=resultall,poly_pair=resultpoly,
        proposed_changes=rep["count_K_changes"])),flush=True)

def selftest():
    y=np.asarray([3,2,3,4],int)
    old=np.asarray([2,2,3,4],np.int8)
    p=np.zeros((4,7));p[0,3]=.8;p[1,2]=.7;p[2,2]=.7;p[3,3]=.8
    q=proposal(old,p,2,.6)
    need(q.tolist()==[3,2,3,4],"S473 adjacent only specialist touched outside source K")
    z=ledger(y,old,q,np.arange(4))
    need(z["fix"]==1 and z["reg"]==0,"S473 paired source event audit wrong")
    print("S473 SELFTEST PASS source-K-conditional K2+1, no K0/K1/K5K6 modifications")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--self-test",action="store_true")
    for name in ("audio","truth","out"):p.add_argument("--"+name,type=Path)
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        for n in ("audio","truth","out"):need(getattr(a,n)is not None,"S473 missing "+n)
        run(a.audio,a.truth,a.out)
