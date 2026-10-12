"""S123: independent code replay of *historical* S5 fold models and held folds.

This compares a FIXED log/activation-first 527D feature-construction
hypothesis to archived Series5 development-fold raw/tilt1/freeze/mix1
confusion. The actual GuitarSet truth 'k' is loaded ONLY AFTER forward
predictions for audit; not used in features, model tuning or selection.

Even a perfect confusion match is weaker than per-event posterior agreement
and is NOT an independent S80 validation or grounds for promotion.
"""
from __future__ import annotations

import argparse,hashlib,json,time
from pathlib import Path
import numpy as np,joblib
from scripts.s118_exploratory_s4_owned527 import extra416

POLICY = "open_k0k6_series5__raw__tilt1__freeze__mix1"
FOLDS=(0,1,2,4)

def require(value,reason):
    if not value: raise ValueError(reason)

def make_design(source):
    with np.load(source,allow_pickle=False) as z:
        needed={"global_index","summary","geometry","sequence","baseline","k","fold"}
        require(needed.issubset(z.files),"original 2026 feature archive incomplete")
        features=np.asarray(z["summary"],np.float32)
        geometry=np.asarray(z["geometry"],np.float32)
        sequence=np.asarray(z["sequence"],np.float32)
        reference=np.asarray(z["baseline"],np.int8)
        rowid=z["global_index"].copy()
        fold=np.asarray(z["fold"],np.int8)
    n=len(rowid)
    require(n>0 and features.shape==(n,58) and geometry.shape==(n,46)
            and sequence.shape==(n,42,49),"original GuitarSet input layout mismatch")
    require(np.isfinite(features).all() and np.isfinite(geometry).all()
            and np.isfinite(sequence).all(),"nonfinite original input")
    require(np.isin(reference,np.arange(7)).all(),"frozen K must be 0..6")
    require(np.all((geometry[:,:42]==0)|(geometry[:,:42]==1)),"ownership mask drift")
    check=np.eye(7,dtype=np.float32)[reference]
    # No labels are read or referenced in the actual input reconstruction.
    extension=np.empty((n,416),np.float32)
    started=time.monotonic()
    for idx in range(n):
        extension[idx]=extra416(sequence[idx],geometry[idx,:42])
        if (idx+1)%2000==0:
            print(json.dumps({"stage":"fixed-feature-replay","processed":idx+1,
                              "total":n,"elapsed":round(time.monotonic()-started,1)}),
                  flush=True)
    X=np.column_stack((features,geometry,check,extension)).astype(np.float32)
    require(X.shape==(n,527) and np.isfinite(X).all(),"527D original fold input mismatch")
    return rowid,fold,X

def run(fold,feature_file,models,original_report,out):
    require(fold in FOLDS,"non-held fold prohibited")
    require(not out.exists(),"immutable replay output")
    original=json.loads(original_report.read_text())
    reference=original["policies"][POLICY]["folds"][str(fold)]["metrics"]
    modeldir=models/f"fold{fold}"
    state_json=json.loads((modeldir/"report.json").read_text())
    weight=modeldir/"model.joblib"
    require(hashlib.sha256(weight.read_bytes()).hexdigest()==state_json["model_sha256"],
            "original S5 checkpoint checksum drift")
    state=joblib.load(weight)
    require(state["held_fold"]==fold and
            state["designs_sha256"]==state_json["designs_sha256"],
            "S5 input training provenance mismatch")
    model=state["model"]
    require(model.n_features_in_==527 and np.array_equal(model.classes_,np.arange(7)),
            "S5 original class/input contract")
    rowid,folds,X=make_design(feature_file)
    require((folds==fold).all(),"source contains other folds")
    require(len(rowid)==reference["rows"],"original source held-fold count mismatch")
    # THIS is the frozen model forward. k/y/truth not part of any input.
    prediction_prob=np.asarray(model.predict_proba(X),np.float64)
    require(prediction_prob.shape==(len(rowid),7) and np.isfinite(prediction_prob).all()
            and np.allclose(prediction_prob.sum(axis=1),1,atol=1e-5),
            "failed historical S5 original forward")
    prediction=prediction_prob.argmax(1).astype(np.int8)
    # Only now read truth in order to compare prediction confusion with the
    # already-reported 2026 development-fold metric (not independent).
    with np.load(feature_file,allow_pickle=False) as z:
        truth=np.asarray(z["k"],np.int8)
    require(len(truth)==len(prediction) and np.isin(truth,np.arange(7)).all(),
            "holdout annotations invalid")
    confusion=np.zeros((7,7),np.int64)
    np.add.at(confusion,(truth,prediction),1)
    old=np.array(reference["confusion_true_by_predicted"],np.int64)
    require(old.shape==(7,7),"archived Series5 confusion not 7x7")
    delta=confusion-old
    by_k={str(k):{"original_correct":int(old[k,k]),
                 "replay_correct":int(confusion[k,k]),
                 "original_rows":int(old[k].sum()),
                 "replay_rows":int(confusion[k].sum())} for k in range(7)}
    result={"status":"S123_FIXED_LOG527_S5_HISTORICAL_FOLD_REPLAY",
       "fold":fold,"source_rows":len(rowid),"model_sha256":state_json["model_sha256"],
       "training_designs_sha256":state_json["designs_sha256"],
       "policy_compared":POLICY,
       "original_confusion_match":bool(np.array_equal(confusion,old)),
       "original_exact_correct":int(np.trace(old)),
       "candidate_replay_exact_correct":int(np.trace(confusion)),
       "absolute_confusion_difference":int(np.abs(delta).sum()),
       "max_abs_confusion_cell_difference":int(np.abs(delta).max()),
       "original_confusion":old.tolist(),"replay_confusion":confusion.tolist(),
       "per_K":by_k,"truth_used_for_features":False,
       "labels_used_only_for_posthoc_historical_confusion":True,
       "feature_sha_match_proven":False,
       "historical_per_event_posterior_identity_proven":False,
       "external_GOAT_accuracy":None,
       "independent_S80_validation":False,"S80_promoted":False}
    out.mkdir(parents=True)
    np.savez_compressed(out/"replay_predictions.npz",global_index=rowid,fold=folds,
                        predicted_K=prediction,probabilities=prediction_prob.astype(np.float32))
    (out/"report.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"status":result["status"],"fold":fold,
        "source_rows":len(rowid),"original_confusion_match":result["original_confusion_match"],
        "original_exact_correct":result["original_exact_correct"],
        "candidate_replay_exact_correct":result["candidate_replay_exact_correct"],
        "absolute_confusion_difference":result["absolute_confusion_difference"],
        "feature_sha_match_proven":False,"S80_promoted":False},indent=2))
    return result

def selftest():
    z=extra416(np.zeros((42,49),np.float32),np.r_[np.zeros(8),np.ones(5),np.zeros(29)])
    require(z.shape==(416,) and np.isfinite(z).all(),"S123 candidate unit failure")
    c=np.zeros((7,7),np.int64)
    y=np.array([0,1,1,2]); pred=np.array([0,1,0,2])
    np.add.at(c,(y,pred),1)
    require(c[0,0]==1 and c[1,0]==1 and c[1,1]==1 and c[2,2]==1,
           "posthoc confusion contract")
    print("S123 PASS: candidate 416D and posthoc-only historical scoring")
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--fold",type=int)
    p.add_argument("--features",type=Path)
    p.add_argument("--models",type=Path)
    p.add_argument("--original-report",type=Path)
    p.add_argument("--out",type=Path)
    p.add_argument("--self-test",action="store_true")
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        require(all(v is not None for v in (a.fold,a.features,a.models,a.original_report,a.out)),
                "missing original source or replay contract")
        run(a.fold,a.features,a.models,a.original_report,a.out)
