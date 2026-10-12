"""S130: teacher-only rare-K weighted S5-student111 ablations, never S80 promotion.

Use original SHA-verified S5 OOF predictions as *pseudo-labels*, never true K.
Train only on original folds 1/2/4, compare to fixed S129 held-out-fold0
predictions and the frozen original S5 posterior on fold0. GOAT labels never
enter the runtime. Count all regressions against the teacher for every class.
"""
from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path
import joblib
import numpy as np
from scripts.s129_s5_111d_student import archived_fold,classifier,probabilities,require,FOLDS

ALPHAS=(0.25,0.5,0.75)

def weights_from_teacher(y,alpha):
    counts=np.bincount(y,minlength=7)
    require(len(y)>0 and counts[0]>0 and np.all(np.isin(y,range(7))),
            "teacher predictions invalid")
    scale=(counts.max()/np.maximum(counts,1))**alpha
    scale=np.minimum(scale,12.0)
    weights=scale[y].astype(np.float64)
    weights/=weights.mean()
    require(np.isfinite(weights).all() and (weights>0).all(),"teacher weights invalid")
    return weights

def measure(y,prev,new):
    baseline=(prev==y)
    current=(new==y)
    gains=int(np.sum(~baseline&current))
    losses=int(np.sum(baseline&~current))
    require(int(current.sum())==int(baseline.sum())+gains-losses,
            "paired gain/loss conservation")
    detail={}
    for k in range(7):
        rows=(y==k)
        p=int(rows.sum())
        new_pred=int(np.sum(new==k))
        hits=int(np.sum(rows&current))
        detail[str(k)]={
            "original_teacher_K_rows":p,
            "student_predictions_K":new_pred,
            "student_teacher_agreement_K":hits,
            "teacher_recall":float(hits/p) if p else None,
            "teacher_precision":float(hits/new_pred) if new_pred else None,
            "fixes_vs_unweighted":int(np.sum(rows&~baseline&current)),
            "breaks_vs_unweighted":int(np.sum(rows&baseline&~current))}
    recall=[detail[str(k)]["teacher_recall"] for k in (2,3,4)]
    return {
        "heldout_teacher_agreement":int(current.sum()),
        "heldout_teacher_agreement_rate":float(current.mean()),
        "paired_fixes_vs_s129":gains,
        "paired_breaks_vs_s129":losses,
        "paired_net_teacher_fidelity":gains-losses,
        "macro_K2_K3_K4_teacher_recall":float(np.mean(recall)),
        "per_teacher_K":detail}

def run(features_root,original_s5,s116,s129,out):
    require(not out.exists(),"do not overwrite S130 evidence")
    folds={}
    for f in FOLDS:
        originals=list((features_root/f"fold{f}").rglob(f"features-fold-{f}.npz"))
        require(len(originals)==1,"missing/ambiguous fold acoustic source")
        folds[f]=archived_fold(originals[0],original_s5,f)
    train_x=np.concatenate([folds[f][1] for f in (1,2,4)])
    train_y=np.concatenate([folds[f][2] for f in (1,2,4)])
    held_ids,held_x,y=folds[0]
    with np.load(s129,allow_pickle=False) as z:
        require(set(z.files)=={"global_index","student_predicted_K","student_probabilities"},
                "S129 cached baseline schema")
        require(np.array_equal(z["global_index"],held_ids),
                "held fold row order drift")
        baseline=z["student_predicted_K"].copy()
        require(np.array_equal(z["student_probabilities"].argmax(1),baseline),
                "cached S129 prediction drift")
    baseline_metrics=measure(y,baseline,baseline)
    require(int((baseline==y).sum())==14557,"S129 historical baseline changed")
    with np.load(s116,allow_pickle=False) as z:
        require(set(z.files)=={"row_id","control111","freeze_K","geometry46"},
                "S116 external label-free schema")
        ids=z["row_id"].copy()
        external_x=z["control111"].astype(np.float32)
    require(external_x.shape==(57,111) and np.isfinite(external_x).all(),
            "external 57x111 native GOAT controls")
    out.mkdir(parents=True)
    results={}
    for alpha in ALPHAS:
        w=weights_from_teacher(train_y,alpha)
        model=classifier()
        model.fit(train_x,train_y,sample_weight=w)
        pred=probabilities(model,held_x).argmax(1).astype(np.int8)
        ext=probabilities(model,external_x)
        metric=measure(y,baseline,pred)
        metric["teacher_weight_exponent"]=alpha
        metric["max_weight_over_min_weight"]=float(w.max()/w.min())
        metric["external_K0_K6_histogram"]=np.bincount(ext.argmax(1),minlength=7).tolist()
        tag=str(alpha).replace(".","p")
        joblib.dump(model,out/f"student_K_weight_{tag}.joblib",compress=3)
        np.savez_compressed(out/f"teacher_weight_{tag}_GOAT.npz",row_id=ids,
                            predicted_K=ext.argmax(1).astype(np.int8),
                            probabilities=ext.astype(np.float32))
        metric["model_sha256"]=hashlib.sha256(
            (out/f"student_K_weight_{tag}.joblib").read_bytes()).hexdigest()
        results[tag]=metric
    report={
        "status":"S130_S5_STUDENT_RARE_CLASS_ABLATIONS_HISTORICAL_TEACHER_ONLY",
        "fold_heldout":0,
        "heldout_teacher_events":len(held_ids),
        "unweighted_S129":baseline_metrics,
        "fixed_predefined_weight_exponents":list(ALPHAS),
        "ablations":results,
        "selection_applied_to_GOAT":False,
        "original_S5_source_model_equivalence_proven":False,
        "S80_pipeline_complete":False,
        "historical_true_K_read":False,
        "GOAT_annotation_read":False,
        "true_accuracy_scored":False,
        "external_S80_accuracy":None,"S80_promoted":False}
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({
        "status":report["status"],
        "unweighted":{k:baseline_metrics[k] for k in
                       ("heldout_teacher_agreement","macro_K2_K3_K4_teacher_recall")},
        "ablations":{k:{p:v[p] for p in
            ("heldout_teacher_agreement","macro_K2_K3_K4_teacher_recall",
             "paired_fixes_vs_s129","paired_breaks_vs_s129",
             "paired_net_teacher_fidelity","external_K0_K6_histogram")}
                    for k,v in results.items()},
        "GOAT_annotation_read":False,"S80_promoted":False},indent=2))
    return report

def selftest():
    y=np.array([0,1,2,3,4,0,1,2])
    p=np.array([0,1,1,2,4,0,1,2])
    n=np.array([0,1,2,3,4,1,1,2])
    d=measure(y,p,n)
    assert d["paired_fixes_vs_s129"]==2
    assert d["paired_breaks_vs_s129"]==1
    assert d["paired_net_teacher_fidelity"]==1
    assert np.isclose(weights_from_teacher(y,.5).mean(),1)
    print("S130 PASS: paired gains/regressions conserved, teacher-only weights")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    for f in ("features-root","original-s5","s116","s129","out"):
        ap.add_argument("--"+f,type=Path)
    ap.add_argument("--self-test",action="store_true")
    args=ap.parse_args()
    if args.self_test:selftest()
    else:
        require(all(v is not None for v in
              (args.features_root,args.original_s5,args.s116,args.s129,args.out)),
              "source teacher/GuitarSet/GOAT inputs required")
        run(args.features_root,args.original_s5,args.s116,args.s129,args.out)
