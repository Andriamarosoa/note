"""S463: training-free, matched-K GuitarSet→IDMT acoustic shift inspection.

Join original real V88 S258/S283 views with exact original IDMT S453 truth
from prior freeze. Compare pitch-class-controlled acoustic-feature quantile
coverage, without model fitting, tuning or alteration of S80. Report which
original feature blocks shift on IDMT and which direction they move,
but *not* as a proven causal explanation for errors.
"""
from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path
import numpy as np

BLOCKS=(("summary58",58),("flow245",245),("morphology490",490),
        ("physics294",294),("geometry3",3))
def need(c,msg):
    if not c:raise ValueError(msg)
def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
    with np.load(p,allow_pickle=False) as z:
        return {k:z[k].copy() for k in z.files}
def quantile_ood(reference,queries):
    """Only same-K GuitarSet distribution; descriptive, no trained classifier."""
    q01=np.quantile(reference,.01,axis=0)
    q99=np.quantile(reference,.99,axis=0)
    q25=np.quantile(reference,.25,axis=0)
    q50=np.quantile(reference,.50,axis=0)
    q75=np.quantile(reference,.75,axis=0)
    qmin=np.maximum(q75-q25,1e-5)
    outside=(queries<q01)|(queries>q99)
    typical=(reference<q01)|(reference>q99)
    shifts=np.median((queries-q50)/qmin,axis=0)
    median_centers=np.median(queries,axis=0)
    return dict(reference_rows=int(len(reference)),
        query_rows=int(len(queries)),
        query_fraction_below_reference_p01=float(np.mean(queries<q01)),
        query_fraction_above_reference_p99=float(np.mean(queries>q99)),
        query_fraction_outside_reference_p01_p99=float(outside.mean()),
        training_reference_fraction_outside_own_p01_p99=float(typical.mean()),
        median_abs_standardized_feature_shift_in_IQR=float(np.median(np.abs(shifts))),
        maximum_abs_standardized_feature_shift_in_IQR=float(np.max(np.abs(shifts))),
        strongest_shifted_dimensions=[
           dict(feature_number=int(i),signed_median_IQR_shift=float(shifts[i]),
                reference_median=float(q50[i]),
                external_median=float(median_centers[i]))
           for i in np.argsort(np.abs(shifts))[-5:][::-1]],
        source_percentiles_calculated_from_GuitarSet_only=True)

def run(s258,s258_truth,s283,s297,s453,s453_report,out):
    need(not out.exists(),"S463 immutable source shift audit may not be overwritten")
    earlier=json.loads(s453_report.read_text())
    need(earlier["status"]=="S453_ORIGINAL_S5_S8_S18_S80_EXTERNAL_IDMT_WRONGK_STAGE_AUDIT"
         and earlier["valid_pitch_supported_events"]==45
         and earlier["matched_true_poly_events"]==21
         and earlier["annotated_rows_SHA256"]==sha(s453),
         "S463 original S453 annotation-matched row ledger must be unaltered")
    a=load(s258)
    yt=load(s258_truth)
    b=load(s283)
    pred=load(s297)
    trainids=a["row_id"].astype(str)
    extids=b["row_id"].astype(str)
    need(len(trainids)==59309 and len(set(trainids))==59309
         and len(extids)==506 and len(set(extids))==506
         and np.array_equal(trainids,yt["row_id"].astype(str))
         and np.array_equal(extids,pred["row_id"].astype(str))
         and all(a[field].shape==(59309,width) and
                 b[field].shape==(506,width)
                 for field,width in BLOCKS),
         "S463 original identical acoustic feature contract 58/245/490/294/3")
    original=pred["s80_K"].astype(np.int8)
    ytrain=yt["true_K"].astype(np.int8)
    matched=[json.loads(line) for line in s453.read_text().splitlines()]
    idmap={x:i for i,x in enumerate(extids)}
    need(len(matched)==47 and
         len({m["source_row_id"] for m in matched})==47 and
         all(m["source_row_id"] in idmap for m in matched),
         "S463 original IDMT 47 annotation matched events changed")
    subset=[m for m in matched if m["within_supported_notes"]]
    ix=np.asarray([idmap[x["source_row_id"]] for x in subset],int)
    ye=np.asarray([int(x["actual_K"]) for x in subset],np.int8)
    need(len(ix)==45 and np.array_equal(
       original[ix],np.array([int(x["original_S80_K"]) for x in subset],int))
       and int(np.sum(ye>=2))==21
       and int(np.sum(original[ix]==ye))==25,
       "S463 external S80 true K source ids changed; impossible original benchmark")
    same=np.asarray(original[ix]==ye)
    per_block={}
    per_K_block={}
    for field,width in BLOCKS:
        # No source labels are considered for model selection; this is
        # retrospective feature-distribution analysis only.
        all_refs=a[field][np.isin(ytrain,ye)].astype(float)
        ext=b[field][ix].astype(float)
        need(np.isfinite(all_refs).all() and np.isfinite(ext).all(),
             "S463 original source features corrupted")
        whole=quantile_ood(all_refs,ext)
        buckets={}
        all_outlier_correct=[];all_outlier_incorrect=[]
        for k in sorted(set(ye)):
            tr=a[field][ytrain==k].astype(float)
            external=ext[ye==k]
            need(len(tr)>200 and len(external)>0,
                 "S463 cannot match original known true K GuitarSet acoustic population")
            sample=quantile_ood(tr,external)
            buckets[str(k)]=sample
            low=np.quantile(tr,.01,axis=0)
            high=np.quantile(tr,.99,axis=0)
            diagnostic_frac=np.mean((external<low)|(external>high),axis=1)
            all_outlier_correct.extend(map(float,diagnostic_frac[same[ye==k]]))
            all_outlier_incorrect.extend(map(float,diagnostic_frac[~same[ye==k]]))
        per_block[field]=dict(
            all_45_IDMT_vs_all_GuitarSet_same_true_K1to4_unstratified=whole,
            class_K_stratified_external_outside_training_percentiles_01_99={
                str(k):buckets[str(k)]["query_fraction_outside_reference_p01_p99"]
                for k in sorted(set(ye))},
            K_stratified_mean_outlier_fraction_on_IDMT_correct_K=(
                float(np.mean(all_outlier_correct)) if all_outlier_correct else None),
            K_stratified_mean_outlier_fraction_on_IDMT_wrong_K=(
                float(np.mean(all_outlier_incorrect)) if all_outlier_incorrect else None),
            K_stratified_n_correct=len(all_outlier_correct),
            K_stratified_n_wrong=len(all_outlier_incorrect))
        per_K_block[field]=buckets
        print("S463 BLOCK",json.dumps(dict(feature_block=field,
             overall_IDMT_ood=whole["query_fraction_outside_reference_p01_p99"],
             by_true_K=per_block[field]["class_K_stratified_external_outside_training_percentiles_01_99"],
             S80_correct_ood=per_block[field]["K_stratified_mean_outlier_fraction_on_IDMT_correct_K"],
             S80_wrong_ood=per_block[field]["K_stratified_mean_outlier_fraction_on_IDMT_wrong_K"])),
             flush=True)
    result=dict(status="S463_ORIGINAL_IDMT_GUITARSET_MATCHED_TRUE_K_ACOUSTIC_DOMAIN_SHIFT_READONLY",
        original_GuitarSet_native_rows=59309,
        original_IDMT_native_rows=506,
        actual_known_IDMT_true_K_rows=45,
        actual_known_IDMT_true_poly_rows=21,
        original_S80_exactK_correct_on_matched=25,
        original_S80_exactK_wrong_on_matched=20,
        original_GuitarSet_original_V88_incumbent_histogram=np.bincount(
            a["incumbent_freeze_K"],minlength=7).tolist(),
        original_IDMT_S80_histogram=np.bincount(original,minlength=7).tolist(),
        original_GuitarSet_trueK_K1to4_histogram=np.bincount(ytrain,minlength=7).tolist(),
        original_IDMT_matched_trueK_K1to4_histogram=np.bincount(ye,minlength=7).tolist(),
        note_source_recording_count=5,
        caution_45_matched_only_not_representative_of_all_506=True,
        original_GuitarSet_populations_may_have_training_overlap=True,
        no_train_refit_reweighting_or_model_promotion=True,
        not_a_causal_audio_shift_proof=True,
        per_feature_block=per_block,
        details_per_block_and_true_K=per_K_block,
        SHA256=dict(GuitarSet_S258=sha(s258),GuitarSet_Ktruth=sha(s258_truth),
                    IDMT_S283=sha(s283),IDMT_S297=sha(s297),
                    IDMT_S453_matched_truth=sha(s453)),
        no_promotion=True)
    out.mkdir(parents=True)
    (out/"report.json").write_text(json.dumps(result,indent=2)+"\n")
    print("S463 DOMAIN_SUMMARY",json.dumps(dict(
        external_45=45,original_K_right=25,original_K_wrong=20,
        acoustic_ood_by_group={
           name:val["all_45_IDMT_vs_all_GuitarSet_same_true_K1to4_unstratified"][
            "query_fraction_outside_reference_p01_p99"]
           for name,val in per_block.items()})),flush=True)

def selftest():
    x=np.arange(1000,dtype=float).reshape(100,10)
    y=x[:3]+2000
    q=quantile_ood(x,y)
    need(q["query_fraction_outside_reference_p01_p99"]==1
         and q["reference_rows"]==100 and q["query_rows"]==3,
         "S463 original training quantile outlier scoring broken")
    print("S463 SELFTEST PASS external OOD quantile rank no train fit")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--self-test",action="store_true")
    for k in ("s258","s258-truth","s283","s297","s453","s453-report","out"):
        p.add_argument("--"+k,type=Path)
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        for k in ("s258","s258_truth","s283","s297","s453","s453_report","out"):
            need(getattr(a,k)is not None,"S463 missing --"+k)
        run(a.s258,a.s258_truth,a.s283,a.s297,a.s453,a.s453_report,a.out)
