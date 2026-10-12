"""S125 historical S5 fold0 forensic replay of TWO extra 416D column hypotheses.

Predictions generated from original GuitarSet 58+42x49+46D features and
unchanged SHA-pinned S5 checkpoint. Cached S123 predictions serve ONLY as a
third fixed diagnostic, never as inference inputs or labels.
Truth is loaded after all forwards and used solely to compare against
the archived historical S5 confusion, not to train or promote models.
"""
from __future__ import annotations
import argparse,hashlib,json,time
from pathlib import Path
import joblib,numpy as np
from scripts.s118_s4_owned527_candidate import owned416
from scripts.s120_s4_s5_416_layout_audit import reorder_amplitude

FOLD=0
POLICY="open_k0k6_series5__raw__tilt1__freeze__mix1"

def require(ok,reason):
    if not ok:raise ValueError(reason)

def load_features(path):
    with np.load(path,allow_pickle=False) as z:
        need={"global_index","summary","sequence","geometry","baseline","k","fold"}
        require(need.issubset(z.files),"historical archive fields missing")
        ids=z["global_index"].copy()
        summaries=z["summary"].astype(np.float32)
        geom=z["geometry"].astype(np.float32)
        temporal=z["sequence"].astype(np.float32)
        b=z["baseline"].astype(np.int8)
        folds=z["fold"].astype(np.int8)
    require(len(ids)==15952 and np.all(folds==FOLD),"wrong original fold0")
    require(summaries.shape==(len(ids),58) and geom.shape==(len(ids),46) and
            temporal.shape==(len(ids),42,49),"historical feature shapes incorrect")
    require(np.isfinite(summaries).all() and np.isfinite(geom).all() and
            np.isfinite(temporal).all(),"nonfinite original acoustics")
    require(np.isin(b,range(7)).all(),"bad original frozen parent")
    require(np.all((geom[:,:42]==0)|(geom[:,:42]==1)),"native mask mismatch")
    prefix=np.column_stack([summaries,geom,np.eye(7,dtype=np.float32)[b]])
    require(prefix.shape==(len(ids),111),"111D historical control prefix mismatch")
    return ids,prefix,geom[:,:42],temporal

def confusion(truth,pred):
    matrix=np.zeros((7,7),np.int64)
    np.add.at(matrix,(truth,pred),1)
    return matrix

def run(features,archive,modeldir,original,out):
    require(not out.exists(),"immutable output required")
    original_report=json.loads(original.read_text())
    fixed=original_report["policies"][POLICY]["folds"][str(FOLD)]["metrics"]
    archive_confusion=np.asarray(fixed["confusion_true_by_predicted"],np.int64)
    checkpoint=modeldir/"fold0"/"model.joblib"
    model_report=json.loads((modeldir/"fold0"/"report.json").read_text())
    require(hashlib.sha256(checkpoint.read_bytes()).hexdigest()==
            model_report["model_sha256"],"S5 original fold0 SHA drift")
    model=joblib.load(checkpoint)["model"]
    require(model.n_features_in_==527 and np.array_equal(model.classes_,np.arange(7)),
            "S5 original 527D K0..6 model mismatch")
    ids,prefix,native,seq=load_features(features)
    with np.load(archive,allow_pickle=False) as z:
        require(set(z.files)=={"global_index","fold","predicted_K","probabilities"},
                "original S123 cached forward contract mismatch")
        require(np.array_equal(ids,z["global_index"]),"S123 cached events mismatch")
        fixed_log=z["predicted_K"].astype(np.int8).copy()
    predict={"log_activation_first":fixed_log}
    suffixes={}
    started=time.monotonic()
    for name in ("amplitude_lag_first","amplitude_activation_first"):
        suffix=np.empty((len(ids),416),np.float32)
        for i in range(len(ids)):
            v=owned416(seq[i],native[i])
            if name=="amplitude_activation_first":
                v=reorder_amplitude(v)
            suffix[i]=v
            if (i+1)%3000==0:
                print(json.dumps({"variant":name,"rows":i+1,
                    "elapsed":round(time.monotonic()-started,1)}),flush=True)
        X=np.column_stack([prefix,suffix]).astype(np.float32)
        require(X.shape==(len(ids),527) and np.isfinite(X).all(),
                "bad candidate layout")
        scores=model.predict_proba(X)
        require(scores.shape==(len(ids),7) and np.isfinite(scores).all() and
                np.allclose(scores.sum(1),1,atol=1e-5),"S5 posterior invalid")
        predict[name]=scores.argmax(1).astype(np.int8)
        suffixes[name]={"all_zero_columns":int(np.all(suffix==0,axis=0).sum()),
                        "input_min":float(X[:,111:].min()),
                        "input_max":float(X[:,111:].max())}
    # Historical truth is accessed only after all predictions have been frozen.
    with np.load(features,allow_pickle=False) as z:
        truth=z["k"].astype(np.int8)
    require(truth.shape==(len(ids),) and np.isin(truth,range(7)).all(),
            "historical development truth invalid")
    results={}
    for name,pred in predict.items():
        mat=confusion(truth,pred)
        diff=np.abs(mat-archive_confusion)
        results[name]={"true_correct":int(np.trace(mat)),
            "original_correct":int(np.trace(archive_confusion)),
            "correct_difference":int(np.trace(mat)-np.trace(archive_confusion)),
            "confusion_L1_error":int(diff.sum()),
            "confusion_exact_match":bool(np.array_equal(mat,archive_confusion)),
            "predicted_K_histogram":np.bincount(pred,minlength=7).tolist(),
            "original_confusion":archive_confusion.tolist(),
            "candidate_confusion":mat.tolist(),
            "suffix":suffixes.get(name)}
    pairwise={}
    names=list(predict)
    for i,a in enumerate(names):
        for b in names[i+1:]:
            pairwise[a+"__"+b]=int((predict[a]!=predict[b]).sum())
    report={"status":"S125_THREE_UNVERIFIED_527D_HISTORICAL_MODEL_REPLAYS",
        "fold":0,"events":len(ids),"results":results,
        "pairwise_prediction_disagreement":pairwise,
        "fixed_checkpoint_SHA256":model_report["model_sha256"],
        "historical_development_truth_only_posthoc":True,
        "features_used_truth":False,
        "historical_fold_used_for_model_selection":False,
        "feature_equivalence_proven":False,
        "S80_independent_validation":False,"S80_promoted":False}
    out.mkdir(parents=True)
    np.savez_compressed(out/"candidate_predictions.npz",global_index=ids,
                        **{"K_"+key:pred for key,pred in predict.items()})
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"status":report["status"],"rows":len(ids),
        "diagnostics":{k:{"correct":v["true_correct"],
           "vs_historical":v["correct_difference"],
           "confusion_L1":v["confusion_L1_error"]} for k,v in results.items()},
        "pairwise":pairwise,"S80_promoted":False},indent=2))
def selftest():
    a=np.arange(416,dtype=np.float32)
    reordered=reorder_amplitude(a)
    require(np.array_equal(np.sort(reordered),a),"variant permutation drift")
    aa=np.zeros((7,7),np.int64)
    y=np.array([0,1,1,2]);p=np.array([0,1,0,2])
    np.add.at(aa,(y,p))
    require(np.array_equal(confusion(y,p),aa),"posthoc confusion mismatch")
    print("S125 PASS: label-isolated replay, row/order and permutation invariants")
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for name in ("features","s123","models","original-report","out"):
        p.add_argument("--"+name,type=Path)
    p.add_argument("--self-test",action="store_true")
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        require(all(x is not None for x in (a.features,a.s123,a.models,a.original_report,a.out)),
                "missing immutable original evidence")
        run(a.features,a.s123,a.models,a.original_report,a.out)
