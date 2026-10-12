"""S127 original-vs-replayed S5 cardinality transition forensic audit.

Original archived S5 raw posteriors and S123 fixed log 527D replay only.
No ground truth/MIDI used, no parameter fitting, no accuracy claims.
Characterizes K under/over count and original model confidence at mismatches.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np

def require(ok,msg):
    if not ok:raise ValueError(msg)

def audit(reference,manifest,replica,out):
    require(not out.exists(),"immutable output")
    rep=json.loads(manifest.read_text())
    require(hashlib.sha256(reference.read_bytes()).hexdigest()==rep["prediction_sha256"],
            "source S5 archive SHA mismatch")
    with np.load(reference,allow_pickle=False) as z:
        require(set(["global_index","raw"]).issubset(z.files),"source posteriors missing")
        ids=z["global_index"].copy()
        p=z["raw"].copy()
        # The truth array is present in this source, but NEVER accessed here.
    with np.load(replica,allow_pickle=False) as z:
        require(set(["global_index","probabilities","predicted_K"]).issubset(z.files),
                "replay posterior fields missing")
        require(np.array_equal(ids,z["global_index"]),"source/replay event misalignment")
        r=z["probabilities"].copy()
        outk=z["predicted_K"].copy()
    require(p.shape==r.shape==(15952,7),"expected original class probabilities")
    require(np.allclose(p.sum(1),1,atol=1e-5) and np.allclose(r.sum(1),1,atol=1e-5),
            "not normalized K0..6 posteriors")
    original=p.argmax(1).astype(np.int8)
    replay=r.argmax(1).astype(np.int8)
    require(np.array_equal(replay,outk),"S123 saved decisions and probabilities disagree")
    flips=(replay!=original)
    transition=np.zeros((7,7),np.int64)
    np.add.at(transition,(original,replay),1)
    transitions=[]
    for i in range(7):
        for j in range(7):
            if i!=j and transition[i,j]:
                transitions.append({"original_K":i,"replay_K":j,"count":int(transition[i,j]),
                                    "direction":"under" if j<i else "over"})
    transitions.sort(key=lambda x:x["count"],reverse=True)
    margins=np.sort(p,axis=1)[:,-1]-np.sort(p,axis=1)[:,-2]
    conf=p.max(1)
    result={
       "status":"S127_ORIGINAL_S5_TO_LOG_REPLAY_CARDINALITY_TRANSITIONS",
       "events":len(original),"mismatches":int(flips.sum()),
       "under_count":int((replay<original).sum()),
       "over_count":int((replay>original).sum()),
       "original_poly_K2plus_mismatches":int(np.sum(flips&(original>=2))),
       "original_K0_mismatches":int(np.sum(flips&(original==0))),
       "original_K_histogram":np.bincount(original,minlength=7).tolist(),
       "replayed_K_histogram":np.bincount(replay,minlength=7).tolist(),
       "transitions_original_K_by_replay_K":transition.tolist(),
       "top15_mismatch_transitions":transitions[:15],
       "original_model_mean_confidence_on_mismatches":float(conf[flips].mean()),
       "original_model_mean_confidence_on_agreements":float(conf[~flips].mean()),
       "original_model_mismatch_confidence_ge_0_8":int(np.sum(flips&(conf>=.8))),
       "original_model_mismatch_confidence_ge_0_9":int(np.sum(flips&(conf>=.9))),
       "original_model_mean_margin_on_mismatches":float(margins[flips].mean()),
       "original_model_mean_margin_on_agreements":float(margins[~flips].mean()),
       "all_audio_source_labels_unused":True,
       "no_accuracy_or_fitting":True,
       "original_527D_generator_verified":False,
       "S80_external_accuracy":None,"S80_promoted":False}
    require(result["under_count"]+result["over_count"]==result["mismatches"],
            "under/over mismatch conservation")
    out.mkdir(parents=True)
    (out/"report.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:result[k] for k in
      ("status","events","mismatches","under_count","over_count",
       "original_poly_K2plus_mismatches","original_K0_mismatches",
       "top15_mismatch_transitions",
       "original_model_mean_confidence_on_mismatches",
       "original_model_mismatch_confidence_ge_0_8",
       "original_model_mismatch_confidence_ge_0_9",
       "no_accuracy_or_fitting","S80_promoted")},indent=2))
def selftest():
    a=np.array([0,2,3,1]);b=np.array([0,1,4,2])
    assert int((a!=b).sum())==3
    assert int((b<a).sum())==1 and int((b>a).sum())==2
    print("S127 PASS: transitions conserved; no truth needed")
if __name__=="__main__":
    ap=argparse.ArgumentParser()
    for name in ("original","original-report","s123","out"):
        ap.add_argument("--"+name,type=Path)
    ap.add_argument("--self-test",action="store_true")
    a=ap.parse_args()
    if a.self_test:selftest()
    else:
        require(all(x is not None for x in (a.original,a.original_report,a.s123,a.out)),
                "missing original or replay evidence")
        audit(a.original,a.original_report,a.s123,a.out)
