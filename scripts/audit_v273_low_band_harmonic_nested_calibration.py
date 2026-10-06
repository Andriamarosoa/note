"""Nested calibration of the harmonic ambiguity score inside the low p=.60-.65 band.

Outer validation folds: 0,1,2,4.

For each held-out outer fold:
1. Fit the harmonic ambiguity logistic on low-register K2/K3 rows with
   0.50 <= frozen residual p(K2) < 0.70 from the other folds.
2. Choose the p(K3) abstention threshold ONLY from inner-OOF predictions inside
   those outer-fit folds, restricted to the target action band 0.60<=p(K2)<0.65.
3. A threshold is feasible only if abstention gain (K3 caught - K2 sacrificed)
   is non-negative on every inner fold. Choose maximum total inner gain, then
   higher K3 precision, then the higher threshold. If no positive-gain feasible
   threshold exists, disable abstention for that outer fold.
4. Apply the fitted model + selected threshold to ALL outer-fold actions in the
   low-register .60-.65 band, including other-K rows. Labels are evaluation-only.

No outer fold 3. No automatic promotion.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np

from scripts.audit_v273_low_confidence_harmonic_signature_lofo import (
    FOLDS,TRAIN_P0,TRAIN_P1,ACTION_P0,ACTION_P1,NAMES,
    model,load_rows,add_features,counts
)
from scripts.v273_residual_audit import require

GRID=tuple(round(x,2) for x in np.arange(.30,.901,.05))

def xy(rows):
    X=np.asarray([[r["features"][n] for n in NAMES] for r in rows],float)
    y=np.asarray([r["true_k"]==3 for r in rows],int)
    return X,y

def fit_model(rows):
    X,y=xy(rows)
    require(len(rows)>=20 and len(np.unique(y))==2,"bad harmonic fit")
    m=model();m.fit(X,y);return m

def predict_one(m,r):
    return float(m.predict_proba(np.asarray([[r["features"][n] for n in NAMES]],float))[0,1])

def inner_oof_predictions(outer_fit_k23):
    folds=sorted({r["fold"] for r in outer_fit_k23})
    out=[]
    for inner_val in folds:
        tr=[r for r in outer_fit_k23 if r["fold"]!=inner_val]
        va=[r for r in outer_fit_k23 if r["fold"]==inner_val]
        m=fit_model(tr)
        for r in va:
            if not (ACTION_P0<=r["probability"]<ACTION_P1):
                continue
            out.append({"fold":int(inner_val),"true_k":int(r["true_k"]),
                        "p_k3":predict_one(m,r),"row_id":int(r["row_id"])})
    return out

def choose_threshold(preds):
    inner_folds=sorted({q["fold"] for q in preds})
    require(inner_folds,"no inner band rows")
    candidates=[]
    for th in GRID:
        per=[];k3=k2=0
        for f in inner_folds:
            hit=[q for q in preds if q["fold"]==f and q["p_k3"]>=th]
            r=sum(q["true_k"]==3 for q in hit)
            c=sum(q["true_k"]==2 for q in hit)
            per.append({"fold":int(f),"K3_caught":r,"K2_sacrificed":c,"gain":r-c,"n":len(hit)})
            k3+=r;k2+=c
        feasible=all(q["gain"]>=0 for q in per)
        precision=None if k3+k2==0 else k3/(k3+k2)
        candidates.append({"threshold":float(th),"K3_caught":k3,"K2_sacrificed":k2,
                           "gain":k3-k2,"precision":precision,"per_fold":per,
                           "feasible":feasible})
    feasible=[q for q in candidates if q["feasible"] and q["gain"]>0]
    if not feasible:
        return 1.01,candidates,None
    best=max(feasible,key=lambda q:(q["gain"],
                                    -1 if q["precision"] is None else q["precision"],
                                    q["threshold"]))
    return float(best["threshold"]),candidates,best

def main():
    ap=argparse.ArgumentParser()
    for n in ("internal-exports","bundle","config","dataset","output"):
        ap.add_argument("--"+n,type=Path,required=True)
    a=ap.parse_args();require(not a.output.exists(),"overwrite")

    rows=load_rows(a.internal_exports)
    feat_rows=add_features(rows,a.bundle,a.config,a.dataset)
    train_rows=[r for r in feat_rows if r["true_k"] in (2,3)]
    require(len(train_rows)>=30,"small harmonic cohort")

    baseline=np.asarray([r["probability"]>=ACTION_P0 for r in rows],bool)
    abstain=np.zeros(len(rows),bool)
    outer_details=[]

    for f in FOLDS:
        fit_k23=[r for r in train_rows if r["fold"]!=f]
        val_feature_rows=[r for r in feat_rows if r["fold"]==f]
        inner=inner_oof_predictions(fit_k23)
        th,cands,best=choose_threshold(inner)
        final_model=fit_model(fit_k23)

        scored=0;band_scored=0
        for i,r in enumerate(rows):
            if r["fold"]!=f: continue
            if not baseline[i]: continue
            if r["register"]!="low" or not (ACTION_P0<=r["probability"]<ACTION_P1): continue
            require("features" in r,"missing inference feature in target band")
            p_k3=predict_one(final_model,r)
            scored+=1;band_scored+=1
            abstain[i]=p_k3>=th

        outer_details.append({
          "fold":int(f),"selected_threshold":float(th),
          "inner_band_rows":len(inner),"inner_best":best,
          "inner_candidates":cands,
          "fit_rows":len(fit_k23),"outer_target_band_rows_scored":band_scored
        })

    final=baseline&~abstain
    base_all=counts(rows,baseline);ab_all=counts(rows,abstain);final_all=counts(rows,final)

    per=[]
    for f in FOLDS:
        fm=np.asarray([r["fold"]==f for r in rows],bool)
        b=counts(rows,baseline&fm);ab=counts(rows,abstain&fm);fin=counts(rows,final&fm)
        per.append({"fold":int(f),"baseline":b,"abstained":ab,
                    "abstention_gain":ab["regressions"]-ab["corrections"],
                    "final":fin,
                    "selected_threshold":next(q["selected_threshold"] for q in outer_details if q["fold"]==f)})

    strict={
      "global_not_worse":final_all["net"]>=base_all["net"],
      "global_positive":final_all["net"]>0,
      "all_folds_nonnegative":all(q["final"]["net"]>=0 for q in per),
      "fold2_nonnegative":next(q for q in per if q["fold"]==2)["final"]["net"]>=0,
      "fold4_not_worse":next(q for q in per if q["fold"]==4)["final"]["net"]>=next(q for q in per if q["fold"]==4)["baseline"]["net"],
    }

    rep={"status":"completed","experiment":"v273_low_band_harmonic_nested_calibration",
         "features":list(NAMES),"classifier":"StandardScaler + balanced LogisticRegression(C=0.1)",
         "training_cohort":"register=low, 0.50<=raw residual p(K2)<0.70, K2/K3 only",
         "target_band":"register=low, 0.60<=raw residual p(K2)<0.65",
         "threshold_grid":list(GRID),
         "selection_rule":"inner-OOF only; each inner fold gain>=0; maximize total gain, then precision, then higher threshold; disable if no positive feasible threshold",
         "baseline":base_all,"abstained":ab_all,"final":final_all,
         "per_fold":per,"outer_calibration":outer_details,"strict_pass":strict,
         "outer_fold_3_used":False,"threshold_uses_held_fold":False,
         "prediction_changes":False,"automatic_promotion":False}

    a.output.mkdir(parents=True)
    (a.output/"report.json").write_text(json.dumps(rep,indent=2,sort_keys=True)+"\n")
    lines=["# Low-band harmonic nested calibration","",
           "Held outer fold is excluded from both harmonic-model fitting and score-threshold calibration.","",
           f"Baseline raw residual p>=.60: **{base_all['net']:+d}**. Nested policy: **{final_all['net']:+d}**.","",
           "| fold | selected harmonic p(K3) | baseline net | abstained K2 | abstained K3 | abstention gain | final net |",
           "|---:|---:|---:|---:|---:|---:|---:|"]
    for q in per:
        ab=q["abstained"]
        th=q["selected_threshold"]
        ths="disabled" if th>1 else f"{th:.2f}"
        lines.append(f"| {q['fold']} | {ths} | {q['baseline']['net']:+d} | {ab['corrections']} | {ab['regressions']} | {q['abstention_gain']:+d} | {q['final']['net']:+d} |")
    lines+=["",f"Strict pass: **{strict}**.","",
            "No outer fold 3; no automatic promotion."]
    (a.output/"report.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))

if __name__=="__main__":main()
