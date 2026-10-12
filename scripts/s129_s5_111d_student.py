"""S129: experimental S5 111D distillation to bypass missing ORIGINAL owned527 code.

The S5 ORIGINAL checkpoint is NOT reimplemented by this student.
Train only on pinned historical S5 *predicted K* as teacher labels.
Held-out original fold0 teacher predictions test teacher fidelity, NOT accuracy.
A final model trained on all historic teacher predictions produces deployable
new-GOAT audio proposals WITHOUT MIDI/truth, but must be versioned S5-student.
No original S80 scores or model promotion may be claimed from this experiment.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

FOLDS = (0, 1, 2, 4)
FORBIDDEN = {"true_K", "truth", "target", "label", "labels", "midi", "y", "k"}

def require(v,why):
    if not v:
        raise ValueError(why)

def archived_fold(feature_path,teacher_dir,fold):
    report_path=teacher_dir/f"fold{fold}"/"report.json"
    teacher_path=teacher_dir/f"fold{fold}"/"predictions.npz"
    report=json.loads(report_path.read_text())
    digest=hashlib.sha256(teacher_path.read_bytes()).hexdigest()
    require(digest == report["prediction_sha256"], f"fold{fold}: teacher SHA mismatch")
    with np.load(feature_path,allow_pickle=False) as z:
        required={"global_index","summary","geometry","baseline","fold"}
        require(required.issubset(z.files),f"fold{fold}: source acoustic fields absent")
        # Never read z["k"], despite this historical archive containing it.
        ids=z["global_index"].copy()
        summary=z["summary"].astype(np.float32)
        geometry=z["geometry"].astype(np.float32)
        parent=z["baseline"].astype(np.int8)
        f=z["fold"].copy()
    require(np.all(f==fold) and len(ids)==len(set(ids.tolist())),
            f"fold{fold}: fold/ID errors")
    require(summary.shape==(len(ids),58) and geometry.shape==(len(ids),46)
            and np.isin(parent,range(7)).all(), "111D features invalid")
    require(np.isin(geometry[:,:42],(0,1)).all(), "ownership mask invalid")
    features=np.column_stack((summary,geometry,
                              np.eye(7,dtype=np.float32)[parent])).astype(np.float32)
    require(features.shape==(len(ids),111) and np.isfinite(features).all(),
            "111D original S4 control prefix invalid")
    with np.load(teacher_path,allow_pickle=False) as z:
        # Teacher archive may contain "truth", but never load it.
        t_ids=z["global_index"].copy()
        post=z["raw"].astype(np.float64)
    require(len(t_ids)==len(set(t_ids.tolist())) and post.shape==(len(t_ids),7),
            f"fold{fold}: teacher posterior schema")
    order=np.argsort(t_ids)
    loc=np.searchsorted(t_ids[order],ids)
    require(np.all(loc<len(t_ids)) and np.array_equal(t_ids[order][loc],ids),
            f"fold{fold}: teacher/event native IDs not paired")
    p=post[order[loc]]
    require(np.isfinite(p).all() and np.allclose(p.sum(1),1,atol=1e-5),
            f"fold{fold}: bad teacher posterior")
    return ids,features,p.argmax(1).astype(np.int8)

def classifier():
    return HistGradientBoostingClassifier(
        max_iter=75,learning_rate=0.09,max_leaf_nodes=15,
        min_samples_leaf=25,l2_regularization=0.5,early_stopping=False,
        random_state=129)

def probabilities(model,features):
    raw=np.asarray(model.predict_proba(features),np.float64)
    out=np.zeros((len(features),7),np.float64)
    out[:,model.classes_.astype(int)]=raw
    require(np.isfinite(out).all() and np.allclose(out.sum(1),1,atol=1e-5),
            "student seven-class probability shape")
    return out

def run(features_root,original_s5,s116,out):
    require(not out.exists(),"immutable S129 output")
    hist={}
    for fold in FOLDS:
        features=list((features_root/f"fold{fold}").rglob(f"features-fold-{fold}.npz"))
        require(len(features)==1,f"fold{fold}: original acoustic artifact missing/ambiguous")
        hist[fold]=archived_fold(features[0],original_s5,fold)
    all_ids=np.concatenate([hist[f][0] for f in FOLDS])
    require(len(set(all_ids.tolist()))==len(all_ids),"folds overlap native event IDs")
    train=np.concatenate([hist[f][1] for f in (1,2,4)])
    teacher=np.concatenate([hist[f][2] for f in (1,2,4)])
    require(len(train)>40000 and len(np.unique(teacher))>=3,"teacher training populations")
    student=classifier()
    student.fit(train,teacher)
    ids_test,x_test,y_test=hist[0]
    p_test=probabilities(student,x_test)
    pred=p_test.argmax(1).astype(np.int8)
    agree=(pred==y_test)
    detail={}
    for k in range(7):
        mask=y_test==k
        detail[str(k)]={
            "original_teacher_predicted_rows":int(mask.sum()),
            "student_teacher_agreements":int(np.sum(agree&mask)),
            "teacher_fidelity_recall":(float(np.mean(agree[mask])) if mask.any() else None)}
    with np.load(s116,allow_pickle=False) as z:
        require(set(z.files)=={"row_id","control111","freeze_K","geometry46"},
                "external GOAT S116 strict input schema")
        require(not FORBIDDEN.intersection(z.files),"GOAT truth in inputs")
        external_ids=z["row_id"].copy()
        external=z["control111"].astype(np.float32)
    require(external.shape==(57,111) and len(set(map(str,external_ids)))==57,
            "GOAT 57 authentic native groups")
    require(np.isfinite(external).all(),"GOAT audio nonfinite")
    all_x=np.concatenate([hist[f][1] for f in FOLDS])
    all_y=np.concatenate([hist[f][2] for f in FOLDS])
    deployed=classifier()
    deployed.fit(all_x,all_y)
    goat_probs=probabilities(deployed,external)
    out.mkdir(parents=True)
    joblib.dump(deployed,out/"s5_student111_all_historic_teacher.joblib",compress=3)
    np.savez_compressed(out/"s5_student111_GOAT_label_free.npz",
                        row_id=external_ids,probabilities=goat_probs.astype(np.float32),
                        predicted_K=goat_probs.argmax(1).astype(np.int8))
    np.savez_compressed(out/"held_fold0_teacher_fidelity.npz",
                        global_index=ids_test,student_predicted_K=pred,
                        student_probabilities=p_test.astype(np.float32))
    report={
        "status":"S129_DISTILLED_S5_111D_ON_EXTERNAL_AUDIO_RESEARCH_ONLY",
        "training_teacher_source":"original pinned Series5 out-of-fold saved raw predictions",
        "original_527D_recovered":False,
        "student_is_original_S5":False,
        "held_out_fold":0,
        "held_out_teacher_rows":len(ids_test),
        "held_out_teacher_K_agreement":int(agree.sum()),
        "held_out_teacher_K_disagreement":int((~agree).sum()),
        "held_out_teacher_K_agreement_rate":float(agree.mean()),
        "held_out_per_teacher_K":detail,
        "train_fold_ids":[1,2,4],
        "deploy_student_train_fold_ids":list(FOLDS),
        "GOAT_external_audio_groups":len(external_ids),
        "GOAT_student_predicted_histogram":np.bincount(goat_probs.argmax(1),minlength=7).tolist(),
        "deploy_checkpoint_sha256":hashlib.sha256(
            (out/"s5_student111_all_historic_teacher.joblib").read_bytes()).hexdigest(),
        "historic_true_K_read":False,"external_labels_read":False,
        "label_accuracy_evaluated":False,"new_model_name":"S5-student111",
        "original_S80_pipeline_equivalence":False,
        "external_S80_accuracy":None,"promotion":False}
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
    return report

def selftest():
    rng=np.random.default_rng(129)
    a=rng.normal(size=(120,111)).astype(np.float32)
    labels=(a[:,0]>0).astype(np.int8)
    m=classifier();m.fit(a,labels)
    p=probabilities(m,a[:7])
    require(p.shape==(7,7) and np.allclose(p.sum(1),1),"7-column expansion")
    require(np.all(p[:,2:]==0),"missing-class zero prob")
    print("S129 PASS: teacher-only fit, 111D, seven-class output")
if __name__=="__main__":
    ap=argparse.ArgumentParser()
    for f in ("features-root","original-s5","s116","out"):
        ap.add_argument("--"+f,type=Path)
    ap.add_argument("--self-test",action="store_true")
    args=ap.parse_args()
    if args.self_test:
        selftest()
    else:
        require(all(v is not None for v in (args.features_root,args.original_s5,args.s116,args.out)),
                "historical teacher artifacts and GOAT 111D required")
        run(args.features_root,args.original_s5,args.s116,args.out)
