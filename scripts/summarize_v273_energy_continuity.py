"""Cross-fold evaluation of energy-conditioned flow / continuity residuals."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FOLDS=(0,1,2,4)
EXPERIMENT="v273_energy_continuity_extract"


def require(c,m):
    if not c: raise RuntimeError(m)


def discover(root):
    rows=[]; reports={}
    for rp in root.rglob("report.json"):
        try: rep=json.loads(rp.read_text())
        except Exception: continue
        if rep.get("experiment")!=EXPERIMENT: continue
        f=int(rep["fold"]); require(f in FOLDS and f not in reports,"duplicate/bad fold")
        rr=[json.loads(x) for x in (rp.parent/"rows.jsonl").read_text().splitlines() if x.strip()]
        require(len(rr)==int(rep["rows"]),"row drift")
        rows+=rr; reports[f]=rep
    require(set(reports)==set(FOLDS),f"missing {set(FOLDS)-set(reports)}")
    return rows,reports


def clf():
    return make_pipeline(StandardScaler(),LogisticRegression(
        C=.1,max_iter=5000,class_weight="balanced",solver="lbfgs",random_state=27401))


def auc(y,s):
    return float(roc_auc_score(y,s)) if len(np.unique(y))==2 else None


def thresholds(s):
    v=np.unique(np.asarray(s,float)); out=[float(np.nextafter(v.max(),np.inf)),*map(float,v)]
    if len(v)>1: out += [float((a+b)/2) for a,b in zip(v[:-1],v[1:])]
    return sorted(set(out))


def account(y,s,t):
    take=np.asarray(s)>=t; y=np.asarray(y)
    c=int(np.sum(take&(y==1))); r=int(np.sum(take&(y==0)))
    return {"actions":int(take.sum()),"corrections":c,"regressions":r,"net":c-r}


def choose(y,s):
    c=[{"threshold":t,**account(y,s,t)} for t in thresholds(s)]
    b=min(c,key=lambda q:(-q["net"],q["regressions"],q["actions"],-q["threshold"]))
    if b["net"]<=0:
        t=float(np.nextafter(np.max(s),np.inf)); return {"threshold":t,**account(y,s,t)}
    return b


def inner_oof(X,y,folds,fit):
    out=np.full(len(y),np.nan)
    for vf in sorted(set(int(v) for v in folds[fit])):
        tr=fit&(folds!=vf); va=fit&(folds==vf)
        require(tr.any() and va.any() and len(np.unique(y[tr]))==2,"inner split")
        m=clf(); m.fit(X[tr],y[tr]); out[va]=m.predict_proba(X[va])[:,1]
    require(np.isfinite(out[fit]).all(),"inner missing"); return out


def evaluate(X,y,folds):
    oof=np.full(len(y),np.nan); rots=[]
    for vf in FOLDS:
        fit=folds!=vf; val=folds==vf
        inn=inner_oof(X,y,folds,fit); sel=choose(y[fit],inn[fit])
        m=clf(); m.fit(X[fit],y[fit]); p=m.predict_proba(X[val])[:,1]; oof[val]=p
        rots.append({
            "val_fold":vf,"val_auc":auc(y[val],p),"selected_threshold":float(sel["threshold"]),
            "val_policy":account(y[val],p,sel["threshold"])
        })
    total={k:int(sum(r["val_policy"][k] for r in rots)) for k in ("actions","corrections","regressions","net")}
    return {"oof_auc":auc(y,oof),"per_fold_auc":{str(r["val_fold"]):r["val_auc"] for r in rots},
            "selected_policy_total":total,"rotations":rots}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-root",type=Path,required=True); ap.add_argument("--output",type=Path,required=True)
    a=ap.parse_args(); require(not a.output.exists(),"refusing overwrite")
    rows,reps=discover(a.input_root); require(len(rows)==488 and len({r["row_id"] for r in rows})==488,"cohort drift")
    rows=sorted(rows,key=lambda r:(int(r["fold"]),int(r["row_id"])))
    y=np.asarray([1 if int(r["true_k"])==2 else 0 for r in rows],int)
    folds=np.asarray([int(r["fold"]) for r in rows],int)
    require(int(np.sum(y==1))==216 and int(np.sum(y==0))==272,"truth drift")
    names=sorted(rows[0]["features"]); require(all(sorted(r["features"])==names for r in rows),"schema")

    groups={
      "absolute_control":("absolute__",),
      "relative_flow":("relative__",),
      "log_energy_flow":("logflow__",),
      "weighted_relative_flow":("weighted_relative__",),
      "passive_state":("passive__",),
      "continuity_residual":("residual__",),
      "relative_continuity_residual":("relative_residual__",),
      "weighted_continuity_residual":("weighted_residual__",),
      "coupled_relative_plus_residual":("relative__","relative_residual__"),
      "coupled_all_continuity":("relative__","logflow__","weighted_relative__",
                                "residual__","relative_residual__","weighted_residual__","passive__"),
    }
    fam={}; fmap={}
    for k,prefs in groups.items():
        fn=[n for n in names if any(n.startswith(p) for p in prefs)]
        X=np.asarray([[float(r["features"][n]) for n in fn] for r in rows],float)
        require(fn and np.isfinite(X).all(),f"bad {k}")
        fmap[k]=fn; fam[k]={"dimensions":len(fn),**evaluate(X,y,folds)}

    best=max(fam,key=lambda k:(fam[k]["oof_auc"],fam[k]["selected_policy_total"]["net"]))
    report={"status":"completed","experiment":"v273_energy_continuity_crossfold","rows":488,
            "K2":216,"K3":272,"folds":list(FOLDS),"fold3_used":False,
            "families":fam,"family_features":fmap,"best_family_by_oof_auc_then_net":best,
            "interpretation_guard":"Negative result falsifies this continuity proxy, not the general physical hypothesis.",
            "automatic_promotion":False,"source_reports":{str(k):v for k,v in reps.items()}}
    a.output.mkdir(parents=True)
    (a.output/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    lines=["# V27.3 energy-continuity audit","",
      "Coupled observables: relative flow and PRE-estimated passive-continuation residual.","",
      "| family | dims | OOF AUC | actions | corrections | regressions | net |",
      "|---|---:|---:|---:|---:|---:|---:|"]
    for k,q in fam.items():
        z=q["selected_policy_total"]
        lines.append(f"| {k} | {q['dimensions']} | {q['oof_auc']:.4f} | {z['actions']} | {z['corrections']} | {z['regressions']} | {z['net']:+d} |")
    lines += ["",f"Best: **{best}**.","",
      "A weak/negative result rejects only this mathematical proxy, not the broader energy-flow hypothesis."]
    (a.output/"report.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines),flush=True)


if __name__=="__main__": main()
