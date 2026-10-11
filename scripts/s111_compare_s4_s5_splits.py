"""S111 compare feature-index signatures of original S4 and S5 estimators.

Models with 527 inputs need not have the same semantic feature order.
We measure split-coordinate overlap, not equivalence or prediction accuracy.
"""
from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path
from collections import Counter
import numpy as np,joblib
FOLDS=(0,1,2,4)
def splits(estimator):
    if not hasattr(estimator,"_predictors"):return None
    c=Counter()
    for iteration in estimator._predictors:
        for tree in iteration:
            nodes=tree.nodes
            for x in nodes["feature_idx"][nodes["is_leaf"]==0]:
                c[int(x)]+=1
    return c
def analyze(root,out):
    models={}
    for fold in FOLDS:
        p=root/"series5"/"models"/f"fold{fold}"
        rep=json.loads((p/"report.json").read_text())
        src=p/"model.joblib"
        if hashlib.sha256(src.read_bytes()).hexdigest()!=rep["model_sha256"]:raise ValueError("S5 SHA mismatch")
        ob=joblib.load(src)
        c=splits(ob["model"])
        if not c or ob["model"].n_features_in_!=527:raise ValueError("S5 invalid")
        models[f"S5_fold{fold}"]={"size":len(c),"splits":sum(c.values()),"features":sorted(c),"counts":c}
    found=[]
    for p in sorted((root/"series4"/"models").glob("*.joblib")):
        ob=joblib.load(p);clf=ob["model"] if isinstance(ob,dict) else ob
        n=getattr(clf,"n_features_in_",-1)
        c=splits(clf)
        line={"model":p.name,"input_dimensions":n,"estimator":type(clf).__name__,
              "active_feature_count":len(c) if c is not None else None}
        if n==527 and c:
            a=set(c)
            line["overlap_with_S5"]={}
            for key,v in models.items():
                b=set(v["features"])
                line["overlap_with_S5"][key]={"shared_count":len(a&b),
                      "jaccard":round(len(a&b)/len(a|b),4),
                      "S4_exclusive":len(a-b),"S5_exclusive":len(b-a)}
        found.append(line)
    if not found:raise ValueError("S4 checkpoints missing")
    result={"status":"S111_S4_S5_FEATURE_COORDINATE_COMPARISON",
        "S4_models":found,"S5_count":len(models),
        "same_527_dimension_does_not_prove_same_feature_semantics":True,
        "original_feature_generator_recovered":False,
        "GOAT_S80_score":None,"promote":False}
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"status":result["status"],
        "by_model":[{"name":x["model"],"dimensions":x["input_dimensions"],"active":x["active_feature_count"],
            "overlap_fold0":x.get("overlap_with_S5",{}).get("S5_fold0")} for x in found],
        "S5_active_counts":{k:v["size"] for k,v in models.items()}},indent=2))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args();analyze(a.root,a.out)
