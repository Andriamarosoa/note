"""S133: LightGBM S5 teacher-only student; never ORIGINAL S5 or S80.

Two predetermined feature sets: 111D authentic prefix and 111D+245D
original audio temporal flow. Original pinned historical S5 posterior
argmax is the ONLY training/evaluation target. Held-out fold0 fidelity
measures teacher imitation, NOT real K accuracy.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import joblib
import lightgbm as lgb
import numpy as np
from scripts.s129_s5_111d_student import archived_fold,require,FOLDS
from scripts.loop_v273_birth_flow import flow_profile
from scripts.s130_s5_student_rare_k import measure

ARMS=("control111","control111_flow245")
PARAMS=dict(n_estimators=165,learning_rate=.055,num_leaves=31,
            max_bin=127,min_child_samples=40,colsample_bytree=.86,
            reg_lambda=2.5,random_state=133,n_jobs=2,verbosity=-1,
            objective="multiclass")
FORBIDDEN={"true_K","truth","target","label","labels","midi","y","k"}

def estimator():
    return lgb.LGBMClassifier(**PARAMS)

def original(root,teacher_root,fold):
    paths=list((root/f"fold{fold}").rglob(f"features-fold-{fold}.npz"))
    require(len(paths)==1,f"original fold {fold} missing")
    ids,prefix,teacher=archived_fold(paths[0],teacher_root,fold)
    with np.load(paths[0],allow_pickle=False) as z:
        require(np.array_equal(ids,z["global_index"]),"historical row order")
        seq=z["sequence"].astype(np.float32)
    require(seq.shape==(len(ids),42,49),"original 42x49 source")
    flow=flow_profile(seq).astype(np.float32)
    require(flow.shape==(len(ids),245) and np.isfinite(flow).all(),"real 245D flow")
    return ids,teacher,{"control111":prefix,
                        "control111_flow245":np.column_stack((prefix,flow)).astype(np.float32)}

def goat(s116,s97):
    with np.load(s116,allow_pickle=False) as z:
        require(set(z.files)=={"row_id","control111","freeze_K","geometry46"},
                "S116 exact input contract")
        require(not FORBIDDEN.intersection(z.files),"truth prohibited S116")
        ids=z["row_id"].copy()
        prefix=z["control111"].astype(np.float32)
    with np.load(s97,allow_pickle=False) as z:
        require({"row_id","flow245"}.issubset(z.files),"S97 flow missing")
        require(not FORBIDDEN.intersection(z.files),"truth prohibited S97")
        require(np.array_equal(ids,z["row_id"]),"GOAT row alignment")
        flow=z["flow245"].astype(np.float32)
    require(prefix.shape==(57,111) and flow.shape==(57,245),"GOAT 57 rows")
    require(len(set(map(str,ids)))==57,"duplicate GOAT ids")
    views={"control111":prefix,"control111_flow245":
           np.column_stack((prefix,flow)).astype(np.float32)}
    require(all(np.isfinite(v).all() for v in views.values()),"nonfinite GOAT features")
    return ids,views

def seven_prob(model,x):
    y=np.zeros((len(x),7),np.float64)
    y[:,np.asarray(model.classes_,int)]=np.asarray(model.predict_proba(x))
    require(np.isfinite(y).all() and np.allclose(y.sum(1),1,atol=1e-5),
            "seven class normalized posterior")
    return y

def run(features_root,teacher_root,s116,s97,s129,out):
    require(not out.exists(),"immutable S133 output")
    hist={f:original(features_root,teacher_root,f) for f in FOLDS}
    ids_all=np.concatenate([hist[f][0] for f in FOLDS])
    require(len(ids_all)==59309 and len(set(ids_all.tolist()))==59309,
            "historic 59309 native rows")
    ids,y,held=hist[0]
    require(len(ids)==15952,"fold0 heldout row count")
    with np.load(s129,allow_pickle=False) as z:
        require(set(z.files)=={"global_index","student_predicted_K",
                               "student_probabilities"},"S129 cached baseline")
        require(np.array_equal(ids,z["global_index"]),"S129 fold0 native order")
        reference=z["student_predicted_K"].astype(np.int8)
        require(np.array_equal(reference,z["student_probabilities"].argmax(1)),
                "S129 predicted K integrity")
    require(int((reference==y).sum())==14557,"S129 original teacher fidelity changed")
    goat_ids,goat_views=goat(s116,s97)
    baseline=measure(y,reference,reference)
    out.mkdir(parents=True)
    results={}
    for name in ARMS:
        train=np.concatenate([hist[f][2][name] for f in (1,2,4)])
        teacher=np.concatenate([hist[f][1] for f in (1,2,4)])
        require(len(train)==len(teacher) and train.shape[1] in (111,356),
                "training array shape")
        model=estimator()
        model.fit(train,teacher)
        hold_prob=seven_prob(model,held[name])
        pred=hold_prob.argmax(1).astype(np.int8)
        metrics=measure(y,reference,pred)
        require(metrics["heldout_teacher_agreement"]==
                14557+metrics["paired_net_teacher_fidelity"],
                "paired conservation check")
        np.savez_compressed(out/(name+"_fold0.npz"),global_index=ids,
                            predicted_K=pred,probabilities=hold_prob.astype(np.float32))
        all_train=np.concatenate([hist[f][2][name] for f in FOLDS])
        all_teacher=np.concatenate([hist[f][1] for f in FOLDS])
        deployed=estimator()
        deployed.fit(all_train,all_teacher)
        external=seven_prob(deployed,goat_views[name])
        np.savez_compressed(out/(name+"_GOAT.npz"),row_id=goat_ids,
                            predicted_K=external.argmax(1).astype(np.int8),
                            probabilities=external.astype(np.float32))
        weights=out/(name+"_all_teacher.joblib")
        joblib.dump(deployed,weights,compress=3)
        metrics.update({
             "feature_dimensions":int(train.shape[1]),
             "teacher_train_folds":[1,2,4],
             "heldout_fold0_used_for_fit":False,
             "new_audio_groups":57,
             "GOAT_K_histogram":np.bincount(external.argmax(1),minlength=7).tolist(),
             "deploy_model_sha256":hashlib.sha256(weights.read_bytes()).hexdigest()})
        results[name]=metrics
        print(json.dumps({"variant":name,
                "teacher_agreement":metrics["heldout_teacher_agreement"],
                "teacher_agreement_rate":metrics["heldout_teacher_agreement_rate"],
                "teacher_macro_K2_K4":metrics["macro_K2_K3_K4_teacher_recall"],
                "fixes_vs_S129":metrics["paired_fixes_vs_s129"],
                "breaks_vs_S129":metrics["paired_breaks_vs_s129"],
                "GOAT_K_histogram":metrics["GOAT_K_histogram"]}),flush=True)
    report={
        "status":"S133_TWO_FIXED_LIGHTGBM_TEACHER_DISTILLATIONS",
        "original_teacher_source":"SHA-pinned S5 OOF argmax predictions",
        "arms_fixed_before_evaluation":list(ARMS),
        "parameters":PARAMS,"heldout_rows":len(y),
        "heldout_teacher_baseline_S129":baseline,"arms":results,
        "GOAT_new_wave_events":len(goat_ids),
        "historical_true_K_read":False,"external_MIDI_labels_read":False,
        "teacher_agreement_not_Exact_K_accuracy":True,
        "original_S5_527D_reproduced":False,"S80_promoted":False}
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    return report

def selftest():
    rng=np.random.default_rng(133)
    x=rng.normal(size=(273,7)).astype(np.float32)
    y=np.tile(np.arange(7),39)
    m=lgb.LGBMClassifier(n_estimators=6,random_state=133,n_jobs=2,
                         min_child_samples=5,verbosity=-1)
    m.fit(x,y)
    p=seven_prob(m,x[:8])
    require(p.shape==(8,7) and np.allclose(p.sum(1),1),"seven-class test")
    print("S133 PASS: teacher-only student; 2 fixed arms")
if __name__=="__main__":
    ap=argparse.ArgumentParser()
    for a in ("features-root","teacher-root","s116","s97","s129","out"):
        ap.add_argument("--"+a,type=Path)
    ap.add_argument("--self-test",action="store_true")
    a=ap.parse_args()
    if a.self_test:selftest()
    else:
        require(all(getattr(a,p.replace("-","_")) is not None for p in
                    ("features-root","teacher-root","s116","s97","s129","out")),
                "six sources required")
        run(a.features_root,a.teacher_root,a.s116,a.s97,a.s129,a.out)
