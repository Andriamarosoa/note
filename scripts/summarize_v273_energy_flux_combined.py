"""Nested OOF evaluation of combined energy + flux + birth/death features."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
FOLDS=(0,1,2,4)
def require(c,m):
    if not c: raise RuntimeError(m)
def clf():
    return make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=5000,class_weight="balanced",solver="lbfgs",random_state=27402))
def auc(y,s): return float(roc_auc_score(y,s)) if len(np.unique(y))==2 else None
def thresholds(s):
    v=np.unique(np.asarray(s,float)); out=[float(np.nextafter(v.max(),np.inf)),*map(float,v)]
    if len(v)>1: out += [float((a+b)/2) for a,b in zip(v[:-1],v[1:])]
    return sorted(set(out))
def account(y,s,t):
    take=np.asarray(s)>=t; y=np.asarray(y); return {"actions":int(take.sum()),"corrections":int(np.sum(take&(y==1))),"regressions":int(np.sum(take&(y==0))),"net":int(np.sum(take&(y==1))-np.sum(take&(y==0)))}
def choose(y,s):
    c=[{"threshold":t,**account(y,s,t)} for t in thresholds(s)]
    b=min(c,key=lambda q:(-q["net"],q["regressions"],q["actions"],-q["threshold"]))
    return b if b["net"]>0 else {"threshold":float(np.nextafter(np.max(s),np.inf)),**account(y,s,float(np.nextafter(np.max(s),np.inf)))}
def inner(X,y,folds,fit):
    out=np.full(len(y),np.nan)
    for vf in FOLDS:
        tr=fit&(folds!=vf); va=fit&(folds==vf)
        if not va.any(): continue
        require(tr.any() and len(np.unique(y[tr]))==2,"inner split")
        m=clf(); m.fit(X[tr],y[tr]); out[va]=m.predict_proba(X[va])[:,1]
    require(np.isfinite(out[fit]).all(),"inner missing"); return out
def evaluate(X,y,folds):
    oof=np.full(len(y),np.nan); rots=[]
    for vf in FOLDS:
        fit=folds!=vf; val=folds==vf
        inn=inner(X,y,folds,fit); sel=choose(y[fit],inn[fit])
        m=clf(); m.fit(X[fit],y[fit]); p=m.predict_proba(X[val])[:,1]; oof[val]=p
        rots.append({"val_fold":vf,"val_auc":auc(y[val],p),"selected_threshold":sel["threshold"],"val_policy":account(y[val],p,sel["threshold"])})
    total={k:int(sum(r["val_policy"][k] for r in rots)) for k in ("actions","corrections","regressions","net")}
    return {"oof_auc":auc(y,oof),"selected_policy_total":total,"rotations":rots}
def main():
    p=argparse.ArgumentParser(); p.add_argument("--input-root",type=Path,required=True); p.add_argument("--output",type=Path,required=True); a=p.parse_args()
    require(not a.output.exists(),"refusing overwrite"); rows=[]; reps={}
    for rp in a.input_root.rglob("report.json"):
        rep=json.loads(rp.read_text())
        if rep.get("experiment")!="v273_energy_flux_combined_extract": continue
        f=int(rep["fold"]); require(f in FOLDS and f not in reps,"duplicate fold")
        rr=[json.loads(x) for x in (rp.parent/"rows.jsonl").read_text().splitlines() if x.strip()]
        rows+=rr; reps[f]=rep
    require(set(reps)==set(FOLDS),"missing folds"); require(len(rows)==488,"cohort drift")
    rows=sorted(rows,key=lambda r:(int(r["fold"]),int(r["row_id"])))
    y=np.asarray([1 if r["true_k"]==2 else 0 for r in rows]); folds=np.asarray([r["fold"] for r in rows])
    require(int(y.sum())==216,"K2 drift")
    names=sorted(rows[0]["features"])
    groups={
      "energy_state":("energy__","passive__"),
      "signed_flux":("flux__","absolute__"),
      "birth_death_flow":("logflow__","relative__","weighted_relative__"),
      "continuity_residual":("residual__","relative_residual__","weighted_residual__"),
      "flux_plus_energy":("energy__","flux__","geometry__","interaction__"),
      "energy_flux_birth_death":("energy__","flux__","geometry__","interaction__","logflow__","relative__","weighted_relative__","residual__","relative_residual__","weighted_residual__","passive__"),
    }
    fam={}
    for k,prefs in groups.items():
        fn=[n for n in names if any(n.startswith(p) for p in prefs)]
        X=np.asarray([[float(r["features"][n]) for n in fn] for r in rows],float)
        require(fn and np.isfinite(X).all(),f"bad {k}")
        fam[k]={"dimensions":len(fn),**evaluate(X,y,folds)}
    best=max(fam,key=lambda k:(fam[k]["oof_auc"],fam[k]["selected_policy_total"]["net"]))
    report={"status":"completed","experiment":"v273_energy_flux_combined_crossfold","rows":488,"K2":216,"K3":272,"folds":list(FOLDS),"fold3_used":False,"families":fam,"best_family_by_oof_auc_then_net":best,"automatic_promotion":False}
    a.output.mkdir(parents=True); (a.output/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    lines=["# V27.3 combined energy + flux + birth/death audit","","| family | dims | OOF AUC | actions | corrections | regressions | net |","|---|---:|---:|---:|---:|---:|---:|"]
    for k,q in fam.items():
        z=q["selected_policy_total"]; lines.append(f"| {k} | {q['dimensions']} | {q['oof_auc']:.4f} | {z['actions']} | {z['corrections']} | {z['regressions']} | {z['net']:+d} |")
    lines+=["",f"Best by OOF AUC/net: **{best}**.","","No automatic promotion."]
    (a.output/"report.md").write_text("\n".join(lines)+"\n")
if __name__=="__main__": main()
