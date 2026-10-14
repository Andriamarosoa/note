"""S472 — can S466's *independently OOF* onset confidence explain Exact-K errors?

Read only: source GuitarSet 59,309 originals, 4-fold OOF S466 birth scores,
original S397 S80, and JAMS true-K. No training, threshold selection or
new reference. Stratify by predicted K2,K3,K4 and actual delta -1/0/+1;
measure whether birth confidence discriminates wrong-K at all.
This avoids inventing a new K correction based on a signal unrelated to
note cardinality. Historical GuitarSet model exposure remains caveat.
"""
from __future__ import annotations
import argparse,hashlib,json,collections
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
from scripts.s402_k_input_first_supervised_midi_pitch_network import need,load,composition_key,composition_fold
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def group_stats(v):
    if len(v)==0:return dict(n=0,median=None,p10=None,p90=None)
    return dict(n=int(len(v)),median=float(np.median(v)),
        p10=float(np.percentile(v,10)),p90=float(np.percentile(v,90)))
def auc(scores,mask):
    if len(np.unique(mask))!=2:return None
    return float(roc_auc_score(mask,scores))
def run(birth,birthreport,s80,truth,out):
    need(not out.exists(),"S472 immutable research signal audit exists")
    spec=json.loads(birthreport.read_text())
    need(spec["status"]=="S466_GUITARSET_TRAIN_ONLY_COMPOSITION_DISJOINT_BINARY_BIRTH_HEAD"
         and spec["source_GuitarSet_original_events"]==59309
         and spec["IDMT_not_read_during_training_or_threshold_selection"]
         and spec["SHA256"]["OOF"]==sha(birth)
         and spec["no_promotion"],
         "S472 requires only original per-composition held S466 onset models")
    a=load(birth);o=load(s80);t=load(truth)
    ids=a["row_id"].astype(str)
    K=a["original_S80_K"].astype(np.int8)
    y=t["true_K"].astype(np.int8)
    scores=a["birth_prob_OOF"].astype(float)
    folds=a["fold"].astype(int)
    need(len(ids)==59309
         and np.array_equal(ids,o["row_id"].astype(str))
         and np.array_equal(ids,t["row_id"].astype(str))
         and np.array_equal(K,o["s80_K"])
         and np.array_equal(t["true_K"],a["true_K_ONLY_FOR_AUDIT"])
         and np.isfinite(scores).all()
         and np.all((scores>=0)&(scores<=1))
         and int(np.sum(K==y))==53529
         and int(np.sum((K==y)&(y>=2)))==6054,
         "S472 source original GuitarSet protected Exact-K 59k baseline drift")
    tables={}
    for k in range(5):
        valid=(K==k)
        cases={}
        for tag,offset in (("under_by_one",1),("correct",0),("over_by_one",-1)):
            ix=np.flatnonzero(valid&(y==k+offset))
            cases[tag]=group_stats(scores[ix])
        val=np.flatnonzero(valid&np.isin(y,[k,k+1,k-1]))
        yy=(y[val]!=k).astype(int)
        cases["AUC_detect_nonzero_plusminus_one_vs_correct"]=auc(scores[val],yy)
        valunder=np.flatnonzero(valid&np.isin(y,[k,k+1]))
        cases["AUC_detect_trueK_plus1_vs_trueK_same"]=auc(
            scores[valunder],(y[valunder]==k+1).astype(int))
        valover=np.flatnonzero(valid&np.isin(y,[k,k-1]))
        cases["AUC_detect_trueK_minus1_vs_trueK_same"]=auc(
            scores[valover],(y[valover]==k-1).astype(int))
        cases["n_predicted_original_K"]=int(valid.sum())
        # Interpretation of AUC: score predicts TRUE birth, not new
        # fundamental count, so do not treat as proof of corrected K.
        tables[str(k)]=cases
    # Cross-fold descriptive tests only: NEVER decide a gate from them.
    byfold={}
    for f in range(4):
        ff=folds==f
        byfold[str(f)]=dict(n=int(ff.sum()),
            original_K_correct=int(np.sum(ff&(K==y))),
            original_true_poly=int(np.sum(ff&(y>=2))),
            birth_score_true_poly_median=(
                float(np.median(scores[ff&(y>=2)]))
                if np.sum(ff&(y>=2)) else None))
    out.mkdir(parents=True)
    result=dict(status="S472_BIRTH_CONFIDENCE_OOF_VS_SOURCE_EXACTK_ERROR_SIGNAL_AUDIT",
        genuine_source_events=59309,
        original_trueK_correct=53529,
        original_true_poly_correct=6054,
        original_onset_birth_head_trained_on_GuitarSet_only=True,
        source_K_under_vs_over_not_auto_repaired=True,
        trained_birth_OOF_not_groundtruth_cardinality=True,
        original_audio_source_predictions_unchanged=True,
        no_regression_or_promotion_claim=True,
        previously_exposed_original_GuitarSet=True,
        OOF_birth_parent_models_can_crossfit_other_folds_caution=True,
        no_training_no_threshold_tuning_no_reference_change=True,
        per_predicted_K_with_direction=tables,
        by_outer_composition_fold=byfold,
        SHA256=dict(birth=sha(birth),S397_original_S80=sha(s80),original_JAMS=sha(truth)),
        no_promotion=True)
    (out/"report.json").write_text(json.dumps(result,indent=2)+"\n")
    print("S472 BIRTH-vs-K ERROR SIGNAL",json.dumps(tables),flush=True)
def selftest():
    x=np.asarray([.1,.2,.3,.4])
    a=auc(x,np.asarray([0,0,1,1]))
    need(a==1.,"S472 sign of AUC score wrong")
    print("S472 SELFTEST PASS AUC direction and audit no new K prediction")
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--self-test",action="store_true")
    for x in ("birth","birthreport","s80","truth","out"):
        p.add_argument("--"+x,type=Path)
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        for x in ("birth","birthreport","s80","truth","out"):
            need(getattr(a,x)is not None,"S472 missing "+x)
        run(a.birth,a.birthreport,a.s80,a.truth,a.out)
