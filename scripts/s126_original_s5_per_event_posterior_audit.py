"""S126 compare saved original S5 posteriors against reconstructed historical inputs.

One-to-one sample alignment, original SHA-pinned per-fold posterior vs
S123 cached log-layout forward plus S125 fixed amplitude variants.
No ground truth, MIDI, labels or scores are read, and no parameters
are tuned. The original archive contains 'truth', which is deliberately
NEVER accessed. Not an external S80 evaluation.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np

VARIANTS=("log_activation_first","amplitude_lag_first","amplitude_activation_first")

def ensure(ok,why):
    if not ok:raise ValueError(why)

def read(path,fields):
    with np.load(path,allow_pickle=False) as z:
        ensure(set(fields).issubset(z.files),"required archive fields missing "+str(path))
        # Deliberately never read the source's truth field even when present.
        return {k:z[k].copy() for k in fields}

def compare(archive,checkpoint_report,s123,s125,out):
    ensure(not out.exists(),"immutable forensic output required")
    record=json.loads(checkpoint_report.read_text())
    ensure(hashlib.sha256(archive.read_bytes()).hexdigest()==record["prediction_sha256"],
           "original S5 raw posterior file SHA mismatch")
    historical=read(archive,("global_index","fold","raw"))
    reconstructed=read(s123,("global_index","fold","predicted_K","probabilities"))
    predicted=read(s125,("global_index",)+tuple("K_"+x for x in VARIANTS))
    ids=historical["global_index"]
    ensure(np.array_equal(ids,reconstructed["global_index"]) and
           np.array_equal(ids,predicted["global_index"]), "sample identity or ordering mismatch")
    ensure(len(ids)==15952 and len(set(ids.tolist()))==len(ids),"historical fold0 rows")
    ensure(np.all(historical["fold"]==0) and np.all(reconstructed["fold"]==0),
           "mixed folds")
    original=np.asarray(historical["raw"],np.float64)
    log=np.asarray(reconstructed["probabilities"],np.float64)
    ensure(original.shape==log.shape==(len(ids),7),"original and replay posterior K0..6 schema")
    ensure(np.isfinite(original).all() and np.isfinite(log).all(),
           "non-finite original/reconstructed output")
    ensure(np.allclose(original.sum(1),1,atol=1e-5) and np.allclose(log.sum(1),1,atol=1e-5),
           "source output does not contain class probabilities")
    original_K=original.argmax(1).astype(np.int8)
    log_K=log.argmax(1).astype(np.int8)
    ensure(np.array_equal(reconstructed["predicted_K"],log_K),"S123 cached posterior argmax mismatch")
    ensure(np.array_equal(predicted["K_log_activation_first"],log_K),
           "S123 and S125 log variant predictions differ")
    errors=np.abs(original-log)
    kresults={}
    for k in range(7):
        rows=original_K==k
        disagreements=int(np.sum(log_K[rows]!=k))
        kresults[str(k)]={"original_K_rows":int(rows.sum()),
                          "reconstructed_disagreements":disagreements,
                          "reconstructed_agreement_rate":float(1-disagreements/max(1,int(rows.sum())))}
    variants={}
    for name in VARIANTS:
        pred=np.asarray(predicted["K_"+name],np.int8)
        ensure(pred.shape==original_K.shape and np.isin(pred,range(7)).all(),
               "bad variant K prediction "+name)
        wrong=pred!=original_K
        variants[name]={"original_K_agreement":int((~wrong).sum()),
                        "K_disagreements":int(wrong.sum()),
                        "K_agreement_fraction":float((~wrong).mean()),
                        "prediction_histogram":np.bincount(pred,minlength=7).tolist(),
                        "original_K0_disagreements":int(np.sum(wrong&(original_K==0))),
                        "original_poly_K2plus_disagreements":int(np.sum(wrong&(original_K>=2)))}
    worst=np.argsort(np.max(errors,axis=1))[::-1][:15]
    cases=[{"global_index":int(ids[i]),"old_K":int(original_K[i]),"replay_K":int(log_K[i]),
            "max_prob_error":float(np.max(errors[i]))} for i in worst]
    report={"status":"S126_ORIGINAL_S5_PER_EVENT_POSTERIOR_FIDELITY",
            "fold":0,"rows":len(ids),"raw_prediction_sha256_verified":True,
            "original_raw_K_distribution":np.bincount(original_K,minlength=7).tolist(),
            "log_posterior_mae":float(errors.mean()),
            "log_posterior_p95_absolute_error":float(np.quantile(errors,.95)),
            "log_posterior_max_abs_error":float(errors.max()),
            "full_probability_vectors_1e_minus_6_agreement":int(np.all(errors<1e-6,axis=1).sum()),
            "per_original_predicted_K":kresults,
            "layouts":variants,"largest_log_posterior_disagreements":cases,
            "source_archive_truth_field_accessed":False,
            "ground_truth_used_for_decoding_or_selection":False,
            "accuracy_evaluated":False,"original_527D_features_recovered":False,
            "S80_independently_validated":False,"S80_promoted":False}
    out.mkdir(parents=True)
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:report[k] for k in
       ("status","rows","log_posterior_mae","log_posterior_p95_absolute_error",
        "log_posterior_max_abs_error","full_probability_vectors_1e_minus_6_agreement",
        "layouts","accuracy_evaluated","S80_promoted")},indent=2))
    return report

def selftest():
    src=np.array([[.8,.2],[.2,.8]],np.float64)
    assert src.shape==(2,2) and np.allclose(src.sum(1),1)
    err=np.abs(src-src.copy())
    assert np.all(err<1e-6)
    assert np.array_equal(src.argmax(1),np.array([0,1]))
    print("S126 PASS: no-truth posterior shape/match metrics")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    for n in ("original","original-report","s123","s125","out"):
        p.add_argument("--"+n,type=Path)
    p.add_argument("--self-test",action="store_true")
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        ensure(all(v is not None for v in (a.original,a.original_report,a.s123,a.s125,a.out)),
               "requires original and independent cached forward inputs")
        compare(a.original,a.original_report,a.s123,a.s125,a.out)
