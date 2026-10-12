"""S120: frozen, label-blind comparison of three distinct S4-owned 416D reconstructions.

This tests feature-family/column-order hypotheses against unmodified archived
S4 and S5 split thresholds on the SAME 57 GOAT audio events. Threshold
coverage is only a range diagnostic, not a score or a way to infer the
original feature order. Never declare a winner based on these tests.
"""
from __future__ import annotations

import argparse, json
from pathlib import Path
import joblib
import numpy as np
from scripts.s118_s4_owned527_candidate import owned416
from scripts.s118_exploratory_s4_owned527 import extra416
from scripts.s119_s4_s5_tree_threshold_coverage import model_audit
from scripts.s105_s5_frozen_forward import load as load_s5

BAD = {"target", "truth", "labels", "label", "true_K", "midi", "y"}
NAMES = ("amplitude_lag_first", "amplitude_activation_first", "log_activation_first")
def check(b, why):
    if not b: raise ValueError(why)

def reorder_amplitude(v):
    check(v.shape == (416,), "expected four 104D zone blocks")
    blocks = []
    for zone in range(4):
        a = v[zone*104:(zone+1)*104]
        blocks.append(np.r_[a[84:104], a[:84]])
    return np.concatenate(blocks).astype(np.float32)

def variant_sets(s97, s116):
    with np.load(s97, allow_pickle=False) as z:
        check(not BAD.intersection(z.files), "truth in S97 input")
        keys = z["row_id"].astype(str).copy()
        seq = z["sequence42x49"].copy()
    with np.load(s116, allow_pickle=False) as z:
        check(not BAD.intersection(z.files), "truth in S116 input")
        check(set(z.files) == {"row_id","control111","freeze_K","geometry46"},
              "bad S116 feature contract")
        check(np.array_equal(keys,z["row_id"].astype(str)), "event identities differ")
        control = z["control111"].copy()
        geom = z["geometry46"].copy()
    check(seq.shape == (57,42,49) and control.shape == (57,111)
          and geom.shape == (57,46), "GOAT shapes changed")
    check(len(set(keys))==57, "duplicate events")
    check(np.all(np.isfinite(seq)) and np.all(np.isfinite(control)), "nonfinite source")
    check(np.all(np.isin(geom[:,:42], [0,1])), "nonbinary ownership")
    rows = {k: [] for k in NAMES}
    for signal, owned in zip(seq, geom[:,:42]):
        a = owned416(signal,owned)
        rows["amplitude_lag_first"].append(a)
        rows["amplitude_activation_first"].append(reorder_amplitude(a))
        rows["log_activation_first"].append(extra416(signal,owned))
    Xs = {}
    for variant,blocks in rows.items():
        suffix = np.stack(blocks).astype(np.float32)
        X = np.column_stack((control,suffix)).astype(np.float32)
        check(X.shape==(57,527) and np.isfinite(X).all(), "invalid "+variant)
        check(np.array_equal(X[:,:111],control), "original control prefix changed")
        Xs[variant]=X
    return keys, Xs

def evaluate(s97,s116,s4dir,s5dir,out):
    check(not out.exists(), "immutable output required")
    keys, variants=variant_sets(s97,s116)
    frozen={}
    for f in (0,1,2):
        for poly in (1,2):
            src=s4dir / f"fold{f}__owned__poly{poly}.joblib"
            state=joblib.load(src)  # trusted historical models only
            check(state["held_fold"]==f and state["family"]==f"owned__poly{poly}",
                  "original S4 lineage mismatch")
            frozen[f"S4_f{f}_p{poly}"]=state["model"]
    for f,m in zip((0,1,2,4),load_s5(s5dir)):
        frozen[f"S5_f{f}"]=m
    result={"status":"S120_UNVERIFIED_416D_LAYOUT_COMPARISON",
        "events":57,"original_models":len(frozen),
        "variants":{},"no_ground_truth_access":True,
        "numeric_feature_equivalence_proven":False,
        "accuracy_measured":False,"S80_promoted":False}
    for name,X in variants.items():
        report = {"by_model":{},"summary":{}}
        for k,m in frozen.items():
            audit=model_audit(m,X)
            probs=m.predict_proba(X)
            check(probs.shape==(57,7) and np.isfinite(probs).all() and
                  np.allclose(probs.sum(axis=1),1,atol=1e-5),
                  "original model prediction contract changed")
            report["by_model"][k]={"threshold_coverage":audit,
               "K_histogram":np.bincount(probs.argmax(1),minlength=7).tolist(),
               "mean_max_probability":float(probs.max(axis=1).mean())}
        for family in ("S4","S5"):
            models=[v["threshold_coverage"] for key,v in report["by_model"].items()
                    if key.startswith(family+"_")]
            groups={}
            for block in ("control111","extension416","all527"):
                split_count=sum(r[block]["used_tree_splits"] for r in models)
                crosses=sum(r[block]["crossing_splits"] for r in models)
                groups[block]={"crossing_fraction":crosses/split_count,
                   "split_count":split_count,
                   "crossing_count":crosses,
                   "feature_columns_with_any_crossing_sum":sum(
                     r[block]["active_columns_with_any_crossing"] for r in models),
                   "constant_columns_sum":sum(r[block]["constant_candidate_columns"]
                                               for r in models)}
            report["summary"][family]=groups
        # Per-zone value statistics are useful to understand why the suffix
        # has a different split occupancy; do not fit/calibrate to labels.
        report["suffix_zones"]={}
        for zi,zone in enumerate(("native","dilated2","before","after")):
            sub=X[:,111+zi*104:111+(zi+1)*104]
            report["suffix_zones"][zone]={
                "median":float(np.median(sub)),
                "p95":float(np.quantile(sub,0.95)),
                "max":float(sub.max()),
                "constant_columns":int((np.ptp(sub,axis=0)<1e-9).sum())}
        result["variants"][name]=report
    out.mkdir(parents=True)
    (out/"report.json").write_text(json.dumps(result,indent=2)+"\n")
    compact={n:{"S4":d["summary"]["S4"],"S5":d["summary"]["S5"],
                "zones":d["suffix_zones"]} for n,d in result["variants"].items()}
    print(json.dumps({"status":result["status"],"comparisons":compact,
       "accuracy_measured":False,"S80_promoted":False},indent=2))
    return result

def selftest():
    z=np.zeros(416,np.float32)
    assert np.array_equal(reorder_amplitude(z),z)
    v=np.arange(416,dtype=np.float32)
    a=reorder_amplitude(v)
    assert np.array_equal(a[:20],v[84:104])
    assert np.array_equal(a[20:104],v[:84])
    assert np.array_equal(a[104:124],v[188:208])
    assert np.array_equal(a[-84:],v[-104:-20])
    assert np.array_equal(np.sort(a),v)
    print("S120 PASS: per-zone 104D permutation is exact, reversible by inspection")
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for n in ("s97","s116","s4-models","s5-models","out"):
        p.add_argument("--"+n,type=Path)
    p.add_argument("--self-test",action="store_true")
    a=p.parse_args()
    if a.self_test: selftest()
    else:
        check(all(v is not None for v in (a.s97,a.s116,a.s4_models,a.s5_models,a.out)),
              "all evidence paths required")
        evaluate(a.s97,a.s116,a.s4_models,a.s5_models,a.out)
