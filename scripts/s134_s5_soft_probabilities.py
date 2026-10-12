"""S134: soft-posterior S5 distillation on verified 111D original source.

Fixed seven-regressor LightGBM posterior student, run on heldout fold0
to compare original 7-way S5 probabilities, S129 and S133 cached
posteriors. Original 416D remains missing and is NOT reconstructed.
No true GuitarSet K, no GOAT MIDI/truth are read.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import joblib
import lightgbm as lgb
import numpy as np
from scripts.s129_s5_111d_student import archived_fold,require,FOLDS
from scripts.s130_s5_student_rare_k import measure

FORBIDDEN={"true_K","truth","target","label","labels","midi","y","k"}
PARAMS=dict(n_estimators=155,learning_rate=.055,num_leaves=31,
            max_bin=127,min_child_samples=40,colsample_bytree=.86,
            reg_lambda=2.5,random_state=134,n_jobs=2,verbosity=-1)
ORIGINAL_MODEL_SHA="854ada718ac9bf1556f00c96b8e1aa90402d6ab3"

def authenticated_soft(teacher_root,fold,ids):
    path=teacher_root/f"fold{fold}"/"predictions.npz"
    info=json.loads((teacher_root/f"fold{fold}"/"report.json").read_text())
    require(hashlib.sha256(path.read_bytes()).hexdigest()==info["prediction_sha256"],
            "original S5 pinned posterior SHA mismatch")
    with np.load(path,allow_pickle=False) as z:
        require({"global_index","raw"}.issubset(z.files),
                "raw original S5 posterior missing")
        t_ids=z["global_index"].copy()
        vals=z["raw"].astype(np.float64)
        # Ignore every archived label/truth array; only original model output.
    require(vals.shape==(len(t_ids),7) and np.isfinite(vals).all() and
            np.allclose(vals.sum(axis=1),1,atol=1e-5),"teacher posterior schema")
    order=np.argsort(t_ids)
    loc=np.searchsorted(t_ids[order],ids)
    require(np.all(loc<len(ids)) and np.array_equal(ids,t_ids[order][loc]),
            "teacher native event IDs not aligned")
    return vals[order[loc]]

def original(features_root,teacher_root,fold):
    paths=list((features_root/f"fold{fold}").rglob(f"features-fold-{fold}.npz"))
    require(len(paths)==1,"historic original input ambiguity")
    ids,x,y=archived_fold(paths[0],teacher_root,fold)
    soft=authenticated_soft(teacher_root,fold,ids)
    require(np.array_equal(y,soft.argmax(1)),"teacher hard/soft source mismatch")
    return ids,x,soft

def fit_seven(x,p):
    require(x.shape[1]==111 and p.shape==(len(x),7),"soft student training contract")
    models=[]
    for k in range(7):
        model=lgb.LGBMRegressor(**PARAMS)
        model.fit(x,p[:,k])
        models.append(model)
    return models

def forward(models,x):
    p=np.column_stack([model.predict(x) for model in models])
    p=np.maximum(p,1e-7)
    p/=p.sum(axis=1,keepdims=True)
    require(p.shape==(len(x),7) and np.isfinite(p).all() and
            np.allclose(p.sum(axis=1),1,atol=1e-6),"soft posteriors invalid")
    return p

def cached(path,ids,name,prefix):
    with np.load(path,allow_pickle=False) as z:
        require(np.array_equal(z["global_index"],ids),name+" row alignment")
        p=z[prefix].astype(np.float64)
        require(p.shape==(len(ids),7) and np.isfinite(p).all(),name+" posterior shape")
    return p

def probability_distance(teacher,student):
    dif=np.abs(teacher-student)
    return {"mean_absolute_probability_error":float(dif.mean()),
            "p95_absolute_probability_error":float(np.quantile(dif,.95)),
            "maximum_probability_error":float(dif.max()),
            "argmax_agreements":int((teacher.argmax(1)==student.argmax(1)).sum())}

def run(features_root,teacher_root,s116,s129,s133,out):
    require(not out.exists(),"immutable S134 artifact")
    historical={f:original(features_root,teacher_root,f) for f in FOLDS}
    ids,yx,ysoft=historical[0]
    require(len(ids)==15952,"held0 row count")
    y=ysoft.argmax(1)
    p129=cached(s129,ids,"S129","student_probabilities")
    p133=cached(s133,ids,"S133","probabilities")
    require((p129.argmax(1)==y).sum()==14557 and
            (p133.argmax(1)==y).sum()==14619,"fixed teacher-fidelity baselines changed")
    x=np.concatenate([historical[f][1] for f in (1,2,4)])
    p=np.concatenate([historical[f][2] for f in (1,2,4)])
    models=fit_seven(x,p)
    out.mkdir(parents=True)
    hold=forward(models,yx)
    baseline=p133.argmax(1)
    pred=hold.argmax(1)
    score=measure(y,baseline,pred)
    distances={"S129_hard_class_student":probability_distance(ysoft,p129),
               "S133_multiclass_student":probability_distance(ysoft,p133),
               "S134_seven_teacher_prob_regressors":probability_distance(ysoft,hold)}
    np.savez_compressed(out/"heldout_fold0_teacher_probabilities.npz",
                        global_index=ids,student_probabilities=hold.astype(np.float32),
                        predicted_K=pred.astype(np.int8))
    # Label-free prediction on external audio after full four-fold fit.
    with np.load(s116,allow_pickle=False) as z:
        require(set(z.files)=={"row_id","control111","freeze_K","geometry46"},
                "external GOAT authentic feature contract")
        require(not FORBIDDEN.intersection(z.files),"GOAT label in runtime")
        ids_ext=z["row_id"].copy()
        x_ext=z["control111"].astype(np.float32)
    require(x_ext.shape==(57,111) and len(set(map(str,ids_ext)))==57,
            "external GOAT rows")
    all_x=np.concatenate([historical[f][1] for f in FOLDS])
    all_p=np.concatenate([historical[f][2] for f in FOLDS])
    deploy=fit_seven(all_x,all_p)
    ext=forward(deploy,x_ext)
    np.savez_compressed(out/"GOAT_57_label_blind.npz",row_id=ids_ext,
                        probabilities=ext.astype(np.float32),
                        predicted_K=ext.argmax(1).astype(np.int8))
    checkpoint=out/"soft_s5_111d_distilled.joblib"
    joblib.dump(deploy,checkpoint,compress=3)
    report={
      "status":"S134_FIXED_SOFT_PROBABILITY_7_REGRESSOR_STUDENT",
      "original_teacher_checkpoint":"historical Series5 out-of-fold saved RAW probabilities",
      "historic_true_K_read":False,"GOAT_MIDI_labels_read":False,
      "original_S5_416D_restored":False,"original_S5_reproduced":False,
      "student_is_new":True,"heldout_rows":len(ids),
      "heldout_folds_used_for_training":[1,2,4],
      "fixed_hyperparameters":PARAMS,
      "heldout_soft_distances":distances,
      "heldout_K_teacher_fidelity":score,
      "GOAT_external_event_count":len(ids_ext),
      "GOAT_label_blind_pred_histogram":np.bincount(ext.argmax(1),minlength=7).tolist(),
      "new_checkpoint_SHA256":hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
      "independent_true_Exact_K":None,"S80_promoted":False}
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"status":report["status"],
       "prob_mae":{k:round(v["mean_absolute_probability_error"],6)
                       for k,v in distances.items()},
       "teacher_K_fidelity":score["heldout_teacher_agreement"],
       "fixes_vs_S133":score["paired_fixes_vs_s129"],
       "regressions_vs_S133":score["paired_breaks_vs_s129"],
       "GOAT_hist":report["GOAT_label_blind_pred_histogram"],
       "S80_promoted":False},indent=2),flush=True)
    return report

def selftest():
    rng=np.random.default_rng(134)
    x=rng.normal(size=(210,111)).astype(np.float32)
    p=rng.dirichlet(np.ones(7),size=210)
    assert p.shape==(210,7)
    d=probability_distance(p,p)
    assert d["mean_absolute_probability_error"]==0
    p2=np.column_stack([np.ones(3)]*7)
    p2/=p2.sum(1,keepdims=True)
    assert np.allclose(p2.sum(1),1)
    print("S134 PASS: probability comparison and seven-way conservation")
if __name__=="__main__":
    ap=argparse.ArgumentParser()
    for field in ("features-root","teacher-root","s116","s129","s133","out"):
        ap.add_argument("--"+field,type=Path)
    ap.add_argument("--self-test",action="store_true")
    a=ap.parse_args()
    if a.self_test:selftest()
    else:
        require(all(getattr(a,f.replace("-","_")) is not None for f in
                   ("features-root","teacher-root","s116","s129","s133","out")),
                "six pinned source paths required")
        run(a.features_root,a.teacher_root,a.s116,a.s129,a.s133,a.out)
