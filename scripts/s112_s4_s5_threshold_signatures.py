"""S112 compare numeric split thresholds on original S4/S5 527D features.

High feature-index overlap alone does not establish identical semantics.
For every shared split coordinate, compare median numeric thresholds of
the original gradient-boosted models. Never read ground truth or test audio.
"""
from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path
import numpy as np,joblib
from scipy.stats import spearmanr
FOLDS=(0,1,2)
def thresholds(model):
    if not hasattr(model,"_predictors"):raise ValueError("only original HistGradientBoostingClassifier supported")
    out={}
    for iteration in model._predictors:
        for tree in iteration:
            nodes=tree.nodes
            cut=nodes[nodes["is_leaf"]==0]
            for f,t in zip(cut["feature_idx"],cut["num_threshold"]):
                if np.isfinite(t):out.setdefault(int(f),[]).append(float(t))
    return {k:np.asarray(v) for k,v in out.items()}
def one(a,b):
    common=sorted(set(a)&set(b))
    med_a=np.array([np.median(a[i]) for i in common])
    med_b=np.array([np.median(b[i]) for i in common])
    sd=np.array([max(np.std(np.r_[a[i],b[i]]),1e-6) for i in common])
    d=np.abs(med_a-med_b)/sd
    rho,p=spearmanr(med_a,med_b)
    return {"overlap":len(common),"spearman_median_split_threshold":round(float(rho),5),
        "spearman_pvalue":float(p),"median_normalized_difference":round(float(np.median(d)),6),
        "p90_normalized_difference":round(float(np.quantile(d,.9)),6),
        "fraction_median_disagreement_under_1_pooled_std":round(float(np.mean(d<1)),5),
        "threshold_coordinate_equivalence_proven":False}
def run(root,out):
    allrows=[]
    for f in FOLDS:
        report=json.loads((root/"series5"/"models"/f"fold{f}"/"report.json").read_text())
        ck=root/"series5"/"models"/f"fold{f}"/"model.joblib"
        if hashlib.sha256(ck.read_bytes()).hexdigest()!=report["model_sha256"]:raise ValueError("S5 SHA drift")
        original=joblib.load(ck)["model"]
        base=thresholds(original)
        for poly in (1,2):
            ck4=root/"series4"/"models"/f"fold{f}__owned__poly{poly}.joblib"
            if not ck4.exists():raise ValueError("S4 model missing "+str(ck4))
            state=joblib.load(ck4)
            rows={"fold":f,"S4_variant":f"owned__poly{poly}",
                **one(thresholds(state["model"]),base)}
            allrows.append(rows)
    res={"status":"S112_S4_S5_SPLIT_THRESHOLD_COMPARISON",
      "comparisons":allrows,
      "S4_S5_identical_feature_transform_verified":False,
      "original_527D_constructor_recovered":False,"S80_promoted":False}
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(res,indent=2)+"\n")
    print(json.dumps(res,indent=2))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args();run(a.root,a.out)
