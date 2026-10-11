"""S113 test S4 control 111D coordinate prefix against owned 527D.

Numeric tree thresholds are fingerprints, not proof of exact feature
identity. No labels, no inference on new audio, no promotion.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np, joblib
from scipy.stats import spearmanr
def thresholds(model):
    c={}
    for step in model._predictors:
        for tree in step:
            z=tree.nodes
            for i,v in zip(z["feature_idx"][z["is_leaf"]==0],z["num_threshold"][z["is_leaf"]==0]):
                if np.isfinite(v): c.setdefault(int(i),[]).append(float(v))
    return {k:np.median(v) for k,v in c.items()}
def run(root,out):
    records=[]
    for fold in (0,1,2):
        for poly in (1,2):
            prefix=root/f"fold{fold}__control__poly{poly}.joblib"
            full=root/f"fold{fold}__owned__poly{poly}.joblib"
            u=joblib.load(prefix);v=joblib.load(full)
            a,b=thresholds(u["model"]),thresholds(v["model"])
            shared=sorted(set(a)&set(b))
            if len(shared)<30: raise ValueError("too few shared split coordinates")
            corr,p=spearmanr([a[i] for i in shared],[b[i] for i in shared])
            records.append({"fold":fold,"poly_weight":poly,"control_input":u["model"].n_features_in_,
                            "owned_input":v["model"].n_features_in_,
                            "common_prefix_features":len(shared),
                            "spearman_median_numeric_threshold":float(corr),
                            "spearman_pvalue":float(p)})
    report={"status":"S113_111_TO_527_S4_PREFIX_CHECK",
            "comparisons":records,"prefix_order_equivalence_proven":False,
            "missing_416D_feature_builder_recovered":False,
            "independent_S80_score":None,"promotion":False}
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True);a=p.parse_args()
    run(a.root,a.out)
