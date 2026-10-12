"""S119: numerical split-threshold coverage audit for unverified S4/S5 527D input.

Tests whether the candidate GOAT values ever straddle the actual historical
tree thresholds. This is a diagnostic ONLY, not proof of original feature order.
Do not read labels or alter the canonical freeze reference.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import joblib,numpy as np
from scripts.s105_s5_frozen_forward import load as load_s5

def require(ok,why):
    if not ok:raise ValueError(why)

def model_audit(model, X):
    require(model.n_features_in_==527 and np.array_equal(model.classes_,np.arange(7)),
            "original checkpoint mismatch")
    lo=X.min(axis=0);hi=X.max(axis=0)
    crossings=np.zeros(527,dtype=np.int64)
    seen=np.zeros(527,dtype=np.int64)
    for stage in model._predictors:
        for tree in stage:
            nodes=tree.nodes
            active=nodes["is_leaf"]==0
            idx=nodes["feature_idx"][active].astype(int)
            threshold=nodes["num_threshold"][active].astype(float)
            require((idx>=0).all() and (idx<527).all() and np.isfinite(threshold).all(),
                    "bad tree node")
            np.add.at(seen,idx,1)
            within=(lo[idx] <= threshold) & (threshold <= hi[idx])
            np.add.at(crossings,idx,within.astype(int))
    summaries={}
    for key,section in (("control111",slice(0,111)),("extension416",slice(111,527)),
                        ("all527",slice(0,527))):
        nodes=int(seen[section].sum())
        crossing=int(crossings[section].sum())
        active=int((seen[section]>0).sum())
        active_crossing=int(((seen[section]>0)&(crossings[section]>0)).sum())
        constant=int((np.ptp(X[:,section],axis=0)<1e-9).sum())
        summaries[key]={"used_tree_splits":nodes,
                        "crossing_splits":crossing,
                        "crossing_fraction":crossing/nodes if nodes else None,
                        "active_feature_columns":active,
                        "active_columns_with_any_crossing":active_crossing,
                        "constant_candidate_columns":constant}
    return summaries

def run(candidate,s4dir,s5dir,out):
    require(not out.exists(),"immutable output required")
    with np.load(candidate,allow_pickle=False) as z:
        require(set(z.files)=={"row_id","control111","extension416","candidate527"},
                "S118 candidate-only contract violated")
        ids=z["row_id"].astype(str);X=z["candidate527"].copy()
        control=z["control111"].copy();added=z["extension416"].copy()
    require(X.shape==(57,527) and len(set(ids))==57 and
            np.array_equal(X[:,:111],control) and np.array_equal(X[:,111:],added)
            and np.isfinite(X).all(),"invalid S118 audit input")
    results={}
    for fold in (0,1,2):
        for poly in (1,2):
            path=s4dir/f"fold{fold}__owned__poly{poly}.joblib"
            state=joblib.load(path)
            require(state["held_fold"]==fold and state["family"]==f"owned__poly{poly}",
                    "S4 fold/provenance mismatch")
            results[f"S4_f{fold}_p{poly}"]=model_audit(state["model"],X)
    for fold,model in zip((0,1,2,4),load_s5(s5dir)):
        results[f"S5_f{fold}"]=model_audit(model,X)
    summary={}
    for family in ("S4","S5"):
        group=[v for k,v in results.items() if k.startswith(family+"_")]
        summary[family]={}
        for section in ("control111","extension416","all527"):
            crossing=sum(x[section]["crossing_splits"] for x in group)
            total=sum(x[section]["used_tree_splits"] for x in group)
            summary[family][section]={"crossing_splits":crossing,
               "used_tree_splits":total,"crossing_fraction":crossing/total if total else None}
    out.mkdir(parents=True)
    report={"status":"S119_UNVERIFIED_FEATURE_THRESHOLD_COVERAGE",
            "source":"S118 labelled-free 527D candidate, 57 GOAT events",
            "audited_genuine_original_models":10,
            "group_threshold_coverage":summary,"model_details":results,
            "feature_equivalence_proven":False,
            "accuracy_or_generalisation_claim":False,"S80_promoted":False,
            "interpretation":"Lower crossing fraction could indicate domain shift, wrong preprocessing or column order; it cannot distinguish these causes without original training feature arrays."}
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k!="model_details"},indent=2))
    return report

if __name__=="__main__":
    p=argparse.ArgumentParser()
    for name in ("candidate","s4-models","s5-models","out"):
        p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args();run(a.candidate,a.s4_models,a.s5_models,a.out)
