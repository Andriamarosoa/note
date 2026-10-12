"""S135: deterministic Pareto stitching of two NEW teacher students.

S133 supplies hard K. S134 supplies original S5 posterior approximation.
To respect S133 K, only when the S134 top class disagrees, increase
the S133-designated probability to the previous maximum + fixed epsilon,
then renormalize. No evaluation labels nor tuning. This is not original S5.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np

EPS=1e-6
REQUIRED_133={"global_index","predicted_K","probabilities"}
REQUIRED_134={"global_index","predicted_K","student_probabilities"}

def check(a,msg):
    if not a:raise ValueError(msg)

def anchor_posterior(prob,hard):
    p=np.asarray(prob,np.float64)
    h=np.asarray(hard,np.int8)
    check(p.ndim==2 and p.shape[1]==7 and h.shape==(len(p),),"K posterior shape")
    check(np.isfinite(p).all() and (p>=0).all() and
          np.allclose(p.sum(1),1,atol=1e-5),"original soft posterior normalization")
    check(np.isin(h,np.arange(7)).all(),"hard K outside 0..6")
    q=p.copy()
    original=q.argmax(1)
    flip=original!=h
    ix=np.flatnonzero(flip)
    if len(ix):
        # All non-K entries retain relative ratios, while designated K
        # receives enough probability to become the unique maximum.
        old_max=q[ix].max(axis=1)
        q[ix,h[ix]]=old_max+EPS
        q[ix]/=q[ix].sum(axis=1,keepdims=True)
    check(np.all(q>=0) and np.allclose(q.sum(1),1,atol=1e-6),
          "modified probability simplex")
    check(np.array_equal(q.argmax(1),h),"hard-K alignment not preserved")
    # When teachers agree on K, all values are bit-identical.
    check(np.array_equal(q[~flip],p[~flip]),"compatible probabilities changed")
    return q,int(flip.sum())

def load(path,ids_key,p_key):
    with np.load(path,allow_pickle=False) as z:
        check({ids_key,"predicted_K",p_key}.issubset(z.files),"source prediction schema")
        ids=z[ids_key].copy()
        predicted=z["predicted_K"].astype(np.int8)
        posterior=z[p_key].astype(np.float64)
    check(len(set(map(str,ids)))==len(ids),"duplicate native predictions")
    check(posterior.shape==(len(ids),7) and
          np.array_equal(posterior.argmax(1),predicted),"prediction/probability integrity")
    return ids,predicted,posterior

def teacher(archive,info,ids):
    r=json.loads(info.read_text())
    check(hashlib.sha256(archive.read_bytes()).hexdigest()==r["prediction_sha256"],
          "historic original S5 archive fingerprint")
    with np.load(archive,allow_pickle=False) as z:
        # DO NOT read archived truth, even though it may exist.
        keys=z["global_index"].copy()
        p=z["raw"].astype(np.float64)
    check(p.shape==(len(ids),7) and np.isfinite(p).all(),"teacher source shape")
    order=np.argsort(keys)
    loc=np.searchsorted(keys[order],ids)
    check(np.all(loc<len(ids)) and np.array_equal(keys[order][loc],ids),
          "original S5 native order mismatch")
    return p[order[loc]]

def mae(original,estimated):
    err=np.abs(original-estimated)
    return {"MAE":float(np.mean(err)), "P95":float(np.quantile(err,.95)),
            "MAX":float(np.max(err)),
            "original_teacher_topK_agreement":int(np.sum(original.argmax(1)==estimated.argmax(1)))}

def run(s133_fold0,s134_fold0,s133_goat,s134_goat,archived_teacher,teacher_info,out):
    check(not out.exists(),"immutable S135 output")
    ids,h133,p133=load(s133_fold0,"global_index","probabilities")
    ids2,h134,p134=load(s134_fold0,"global_index","student_probabilities")
    check(np.array_equal(ids,ids2) and len(ids)==15952,"fold0 row alignment")
    original=teacher(archived_teacher,teacher_info,ids)
    check((h133==original.argmax(1)).sum()==14619,"heldout S133 K provenance drift")
    check((h134==original.argmax(1)).sum()==14550,"heldout S134 K provenance drift")
    projected,changed=anchor_posterior(p134,h133)
    stats={"S133_best_K":mae(original,p133),
           "S134_best_soft":mae(original,p134),
           "S135_anchored_soft":mae(original,projected)}
    check(stats["S135_anchored_soft"]["original_teacher_topK_agreement"]==14619,
          "S133 strongest heldout K lost in fusion")
    g133_id,g133_K,g133_p=load(s133_goat,"row_id","probabilities")
    g134_id,g134_K,g134_p=load(s134_goat,"row_id","probabilities")
    check(np.array_equal(g133_id,g134_id) and len(g133_id)==57,
          "GOAT native 57 row join error")
    qgoat,nchanged=anchor_posterior(g134_p,g133_K)
    check(np.array_equal(qgoat.argmax(1),g133_K),"GOAT fixed S133 K lost")
    out.mkdir(parents=True)
    np.savez_compressed(out/"fold0_soft_Kanchored.npz",global_index=ids,
                        predicted_K=h133,probabilities=projected.astype(np.float32))
    np.savez_compressed(out/"GOAT_57_label_blind_soft_Kanchored.npz",
                        row_id=g133_id,predicted_K=g133_K,
                        probabilities=qgoat.astype(np.float32))
    report={
       "status":"S135_DETERMINISTIC_NEW_STUDENT_PARETO_STITCH",
       "original_S5_recovered":False,
       "original_416D_recovered":False,
       "new_model_not_original_S5":True,
       "historic_fold0_rows":int(len(ids)),
       "GOAT_real_wave_events":int(len(g133_id)),
       "predefined_hard_anchor":"S133 control111 LightGBM",
       "predefined_soft_posterior":"S134 control111 seven LightGBM regressors",
       "probability_rule":"increase chosen S133 class to old maximum + 1e-6, renormalize, only on disagreement",
       "fold0_events_modified_for_anchor":changed,
       "GOAT_events_modified_for_anchor":nchanged,
       "fold0_teacher_probabilities":stats,
       "GOAT_K_histogram":np.bincount(g133_K,minlength=7).tolist(),
       "GuitarSet_true_K_read":False,"GOAT_MIDI_truth_read":False,
       "true_Exact_K_accuracy_claimed":False,"S80_promoted":False}
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"status":report["status"],
      "heldout_anchor_changes":changed,"GOAT_anchor_changes":nchanged,
      "S133_MAE":stats["S133_best_K"]["MAE"],
      "S134_MAE":stats["S134_best_soft"]["MAE"],
      "S135_MAE":stats["S135_anchored_soft"]["MAE"],
      "S135_teacher_K_fidelity":stats["S135_anchored_soft"]["original_teacher_topK_agreement"],
      "GOAT_K_histogram":report["GOAT_K_histogram"],
      "S80_promoted":False},indent=2),flush=True)
    return report

def selftest():
    p=np.asarray([[.4,.3,.2,.1,0,0,0],
                  [.1,.6,.1,.1,.05,.03,.02]],float)
    hard=np.asarray([2,1],np.int8)
    q,n=anchor_posterior(p,hard)
    assert n==1 and np.array_equal(q.argmax(1),hard)
    assert np.array_equal(q[1],p[1]) and np.allclose(q.sum(1),1)
    assert np.all(q>=0)
    print("S135 PASS: anchor only changed K, preserved exact compatible rows")
if __name__=="__main__":
    ap=argparse.ArgumentParser()
    for x in ("s133-fold0","s134-fold0","s133-goat","s134-goat",
              "archived-teacher","teacher-info","out"):
        ap.add_argument("--"+x,type=Path)
    ap.add_argument("--self-test",action="store_true")
    a=ap.parse_args()
    if a.self_test:selftest()
    else:
        check(all(getattr(a,x.replace("-","_")) is not None for x in
                  ("s133-fold0","s134-fold0","s133-goat","s134-goat",
                   "archived-teacher","teacher-info","out")),"inputs required")
        run(a.s133_fold0,a.s134_fold0,a.s133_goat,a.s134_goat,
            a.archived_teacher,a.teacher_info,a.out)
