"""S481: LightGBM boosted trees on acoustic real features, true K0->K1/K2+ recovery from acoustic features, NOT generic Exact-K.

Train a residual classifier ONLY on candidates predicted K0 by original V88.
Labels: true newly assigned K0, K1, K>=2 (the last becomes proposed K2,
never claim exact K3+ from 2+). Nested whole-composition validation chooses
a risk-averse threshold using corrections vs regressions (and abstains if
none). Four held-composition folds produce genuine OOF source predictions.
Other source predicted Ks never change. No IDMT/MAESTRO truth used in
training or gate calibration. No deployment or official S80 modification.
"""
from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path
import numpy as np,joblib
from lightgbm import LGBMClassifier
from scripts.s402_k_input_first_supervised_midi_pitch_network import (
    need,load,composition_key,composition_fold)
from scripts.s444_balanced_joint_note_swap_songheld_Klocked import balanced_inner_partition
from scripts.s458_train_independent_audio_only_K_blind_songheld import inputs
from scripts.s399_s80_exactK_constrained_midi_pitch_decoder import sha
CUTS=(.35,.40,.45,.50,.55,.60,.65,.70,.75,.80,.85,.90,.925,.95,.975,.99)
N=59309
def classifier(seed):
    return LGBMClassifier(
        objective="multiclass",num_class=3,
        n_estimators=140,learning_rate=.045,
        num_leaves=24,max_depth=-1,
        min_child_samples=45,colsample_bytree=.66,
        subsample=.85,reg_lambda=5.,reg_alpha=.25,
        n_jobs=2,random_state=seed,verbosity=-1)

def train_balanced(est,X,labels):
    z=np.asarray(labels,np.int8)
    counts=np.bincount(z,minlength=3)
    need(np.all(counts>0),"S481 source K0 training population missing one class")
    # Square root class balancing dampens rare-poly false positives.
    w=np.sqrt(np.max(counts)/counts)[z]
    est.fit(X,z,sample_weight=w.astype(np.float32))
    return est
def targets(trueK):
    return np.minimum(np.asarray(trueK,int),2).astype(np.int8)
def distribution(model,X):
    p=model.predict_proba(X)
    a=np.zeros((len(X),3),np.float32)
    a[:,np.asarray(model.classes_,int)]=p
    need(a.shape==(len(X),3) and np.isfinite(a).all()
         and np.allclose(a.sum(1),1,atol=1e-5),
         "S481 K0 specialist lacks valid probability contract")
    return a
def proposal(old,prob,threshold):
    p=np.asarray(old,np.int8).copy()
    if threshold is None:return p
    need(prob.shape==(len(p),3),"S481 candidate source K0 probabilities missing")
    chosen=np.argmax(prob,axis=1)
    strength=np.max(prob,axis=1)
    edit=(old==0)&(chosen>=1)&(strength>=threshold)
    p[edit]=chosen[edit]
    need(np.array_equal(p[old!=0],old[old!=0]),
         "S481 accidental original V88 K1+ mutation")
    return p
def score(trueK,old,new,ix):
    if len(ix)==0:return dict(n=0,fixes=0,regressions=0,net=0)
    before=(old[ix]==trueK[ix]);after=(new[ix]==trueK[ix])
    fixes=int(np.sum(~before&after));regs=int(np.sum(before&~after))
    return dict(n=len(ix),fixes=fixes,regressions=regs,net=fixes-regs,
        source_K0_wrong_rescued_K1=int(np.sum(
            (old[ix]==0)&(trueK[ix]==1)&(new[ix]==1))),
        source_K0_wrong_rescued_K2=int(np.sum(
            (old[ix]==0)&(trueK[ix]==2)&(new[ix]==2))),
        true_K0_lost=int(np.sum((old[ix]==0)&(trueK[ix]==0)&(new[ix]>0))),
        real_poly_K0_episodes_unchanged=int(np.sum(
            (old[ix]==0)&(trueK[ix]>=2)&(new[ix]==0))))
def calibrate(p,true):
    old=np.zeros(len(true),np.int8)
    choices=[]
    for cut in CUTS:
        res=proposal(old,p,cut)
        k=score(true,old,res,np.arange(len(true)))
        okay=(k["fixes"]>=5 and k["net"]>=5
              and k["fixes"]>=3*max(1,k["regressions"])
              and k["regressions"]<=2
              and k["true_K0_lost"]<=2)
        choices.append(dict(cut=cut,valid=bool(okay),**k))
    safe=[a for a in choices if a["valid"]]
    best=max(safe,key=lambda a:(a["net"],-a["regressions"],a["cut"])) if safe else None
    return (float(best["cut"]) if best else None),choices,best
def run(audio,truth,out):
    need(not out.exists(),"S481 immutable model history source exists")
    a=load(audio);t=load(truth)
    ids=a["row_id"].astype(str)
    original=a["incumbent_freeze_K"].astype(np.int8)
    y=t["true_K"].astype(np.int8)
    need(len(ids)==N and len(set(ids))==N
         and np.array_equal(ids,t["row_id"].astype(str))
         and np.array_equal(original,t["original_freeze_K"])
         and np.isin(original,range(7)).all()
         and np.isin(y,range(7)).all()
         and int(np.sum(y>=2))==7385,
         "S481 original source GuitarSet V88 true K contract broken")
    X=inputs(a,original)
    composition=np.asarray([composition_key(v) for v in a["member"].astype(str)])
    fold=np.asarray([composition_fold(v) for v in composition],np.int8)
    corrected=original.copy()
    predicted=np.full((N,3),np.nan,np.float32)
    model_meta=[]
    out.mkdir(parents=True)
    for f in range(4):
        outertrain=np.flatnonzero((fold!=f)&(original==0))
        held=np.flatnonzero((fold==f)&(original==0))
        need(len(held)>100 and len(outertrain)>1000
             and len(set(y[outertrain]))>=3,
             "S481 source real K0 three-class outertrain insufficient")
        innertrain,innerval,partition=balanced_inner_partition(
             outertrain,composition)
        need(len(innertrain)>1000 and len(innerval)>100
             and not(set(composition[innertrain])&set(composition[innerval]))
             and not(set(composition[outertrain])&set(composition[held])),
             "S481 composition leakage in label-blind holdout")
        inner=classifier(4810+f)
        train_balanced(inner,X[innertrain],targets(y[innertrain]))
        pp=distribution(inner,X[innerval])
        cut,choices,best=calibrate(pp,y[innerval])
        del inner
        net=classifier(4814+f)
        train_balanced(net,X[outertrain],targets(y[outertrain]))
        scores=distribution(net,X[held])
        predicted[held]=scores
        old=np.zeros(len(held),np.int8)
        edit=proposal(old,scores,cut)
        corrected[held]=edit
        path=out/f"S481_source_V88_K0_rescue_fold{f}.joblib"
        joblib.dump(dict(
            status="S481_GUITARSET_K0_SPECIALIST",
            model=net,outer_fold=f,feature_count=1090,
            original_source_class=0,
            label_classes="original_trueK_0_1_2plus",
            decoded_trueK2plus_proposal=2,
            chosen_guitarset_only_calibration_threshold=cut,
            no_IDMT_or_MAESTRO_training=True),path,compress=3)
        all_f=score(y,original,corrected,held)
        meta=dict(fold=f,original_source_K0_train=len(outertrain),
            original_source_K0_held=len(held),
            separate_inner_partition=partition,original_K0_inner=len(innerval),
            selected_cut=cut,inner_selected_evidence=best,
            inner_all_thresholds=choices,held_pair_report=all_f,
            original_model_SHA256=sha(path))
        model_meta.append(meta)
        print("S481 FOLD",json.dumps(dict(fold=f,cut=cut,
            train=len(outertrain),held=len(held),pair=all_f)),flush=True)
    masked=(original==0)
    need(np.isfinite(predicted[masked]).all() and
         np.array_equal(corrected[~masked],original[~masked]),
         "S481 immutable original V88 non-K0 source rows not preserved")
    paired=score(y,original,corrected,np.arange(N))
    true_poly=score(y,original,corrected,np.flatnonzero(y>=2))
    preds=out/"S481_original_V88_GuitarSet_K0_rescue_OOF_candidate.npz"
    np.savez_compressed(preds,row_id=ids,original_V88_K=original,
        proposed_rescue_V88_K=corrected,fold=fold,
        source_K0_multiclass_prob=np.nan_to_num(predicted,nan=0.),
        original_trueK_ONLY_FOR_AUDIT=y)
    result=dict(status="S481_GUITARSET_ONLY_V88_K0_SOURCE_RESIDUAL_4FOLD",
        original_GuitarSet_native_events=N,
        original_V88_global_correct=int(np.sum(original==y)),
        original_V88_true_poly_correct=int(np.sum((original==y)&(y>=2))),
        original_V88_source_K0_count=int(masked.sum()),
        total_directional_K0_proposals=int(np.sum(corrected!=original)),
        global_paired_GuitarSet_only=paired,
        poly_paired_GuitarSet_only=true_poly,
        four_GuitarSet_train_only_outerfolds=model_meta,
        original_K_ge1_never_edited=True,
        original_S80_and_notes_never_edited=True,
        no_external_IDMT_or_MAESTRO_truth_used=True,
        source_GuitarSet_historical_model_training_exposure=True,
        no_promotion=True,
        SHA256=dict(GuitarSet_audio=sha(audio),GuitarSet_labels=sha(truth),
            OOF_candidate=sha(preds)))
    (out/"report.json").write_text(json.dumps(result,indent=2)+"\n")
    print("S481 PAIR FINAL",json.dumps(dict(
        source_V88=int(np.sum(original==y)),
        source_poly=int(np.sum((original==y)&(y>=2))),
        rescue_global=paired,rescue_poly=true_poly,
        changed=result["total_directional_K0_proposals"])),flush=True)
def selftest():
    p=np.array([[.01,.95,.04],[.01,.04,.95],[.9,.08,.02]],float)
    old=np.array([0,0,0],np.int8)
    n=proposal(old,p,.9)
    need(n.tolist()==[1,2,0],"S481 source K0 can rescue K1 and K2+ while abstaining")
    q=score(np.array([1,2,0]),old,n,np.arange(3))
    need(q["fixes"]==2 and q["regressions"]==0,
         "S481 individual K1 K2+ false positive accounting wrong")
    print("S481 SELFTEST PASS true K0 rescue vs false K0 losses")
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--self-test",action="store_true")
    for k in ("audio","truth","out"):p.add_argument("--"+k,type=Path)
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        for k in ("audio","truth","out"):need(getattr(a,k)is not None,"S481 missing "+k)
        run(a.audio,a.truth,a.out)
