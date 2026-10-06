"""Reliability gate for fixed symmetric pitch TTA H=2."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scripts.v273_residual_audit import FOLDS,require,write_json

FEATURES=(
 "mean_margin","median_margin","std_margin","min_margin","max_margin",
 "positive_margin_fraction","k2_vote_fraction","k3_vote_fraction",
 "pair_asymmetry_1","pair_asymmetry_2","mean_pair_asymmetry",
 "center_margin","outer_mean_margin","inner_mean_margin","margin_range"
)

def discover(root):
    out={}
    for p in root.rglob("report.json"):
        try:r=json.loads(p.read_text())
        except Exception:continue
        if r.get("experiment")!="v273_symmetric_pitch_tta_fold":continue
        f=int(r["validation_fold"]); npz=p.parent/"symmetric.npz"
        require(f in FOLDS and f not in out and npz.exists(),"bad symmetric fold")
        out[f]=npz
    require(set(out)==set(FOLDS),"missing folds")
    return out

def feats(steps,p,pr):
    idx=np.flatnonzero(np.abs(steps)<=2)
    pp=p[idx]
    kk=pr[idx]
    m=pp[:,2]-pp[:,3]
    by={int(s):float((p[np.where(steps==s)[0][0],2]-p[np.where(steps==s)[0][0],3])) for s in (-2,-1,0,1,2)}
    d={
      "mean_margin":float(np.mean(m)),
      "median_margin":float(np.median(m)),
      "std_margin":float(np.std(m)),
      "min_margin":float(np.min(m)),
      "max_margin":float(np.max(m)),
      "positive_margin_fraction":float(np.mean(m>0)),
      "k2_vote_fraction":float(np.mean(kk==2)),
      "k3_vote_fraction":float(np.mean(kk==3)),
      "pair_asymmetry_1":abs(by[1]-by[-1]),
      "pair_asymmetry_2":abs(by[2]-by[-2]),
      "mean_pair_asymmetry":0.5*(abs(by[1]-by[-1])+abs(by[2]-by[-2])),
      "center_margin":by[0],
      "outer_mean_margin":0.5*(by[-2]+by[2]),
      "inner_mean_margin":0.5*(by[-1]+by[1]),
      "margin_range":float(np.max(m)-np.min(m)),
    }
    return np.asarray([d[k] for k in FEATURES],np.float64),d

def load(root):
    rows=[]
    for f,p in discover(root).items():
        with np.load(p,allow_pickle=False) as z:
            st=np.asarray(z["steps"],np.int32)
            for i,rid in enumerate(z["row_id"]):
                x,named=feats(st,np.asarray(z["probability"][i],np.float64),
                              np.asarray(z["predicted"][i],np.int32))
                rows.append({"row_id":int(rid),"fold":f,"true_k":int(z["true_k"][i]),"x":x,"named":named})
    require(len(rows)==488 and len({r["row_id"] for r in rows})==488,"cohort drift")
    return rows

def raw_action(rows):
    return np.asarray([r["named"]["mean_margin"]>0 for r in rows],bool)

def account(rows,action):
    y=np.asarray([r["true_k"] for r in rows],np.int32)
    a=np.asarray(action,bool)
    c=int(np.sum(a&(y==2)));g=int(np.sum(a&(y==3)))
    return {"actions":int(a.sum()),"corrections":c,"regressions":g,"net":c-g}

def gate():
    return Pipeline([("scale",StandardScaler()),
      ("lr",LogisticRegression(C=1.0,class_weight="balanced",solver="lbfgs",
                               max_iter=3000,random_state=39631))])

def run(a):
    require(not a.output.exists(),"refusing overwrite")
    rows=load(a.input_root)
    folds=[]
    for vf in FOLDS:
        fit=[r for r in rows if r["fold"]!=vf]; val=[r for r in rows if r["fold"]==vf]
        af=raw_action(fit); av=raw_action(val)
        raw=account(val,av)
        fit_actions=[r for r,t in zip(fit,af) if t]
        entry={"fold":vf,"raw":raw,"fit_action_rows":len(fit_actions)}
        if len(fit_actions)<16 or len({r["true_k"] for r in fit_actions})<2:
            gated=np.zeros(len(val),bool); entry["status"]="abstain_insufficient_fit"
            entry["val_reliability_auc"]=None
        else:
            X=np.stack([r["x"] for r in fit_actions]); y=np.asarray([r["true_k"]==2 for r in fit_actions],np.int32)
            m=gate();m.fit(X,y)
            Xv=np.stack([r["x"] for r in val]); rel=m.predict_proba(Xv)[:,1]
            gated=av&(rel>=.5)
            yy=np.asarray([r["true_k"]==2 for r in val],np.int32)
            take=av
            entry["val_reliability_auc"]=float(roc_auc_score(yy[take],rel[take])) if take.sum() and len(np.unique(yy[take]))==2 else None
            entry["status"]="completed"
            entry["coefficients_standardized"]={k:float(v) for k,v in zip(FEATURES,m.named_steps["lr"].coef_[0])}
        entry["gated"]=account(val,gated)
        entry["blocked_actions"]=int(np.sum(av&~gated))
        folds.append(entry)
    rawt={k:int(sum(f["raw"][k] for f in folds)) for k in ("actions","corrections","regressions","net")}
    gt={k:int(sum(f["gated"][k] for f in folds)) for k in ("actions","corrections","regressions","net")}
    nonneg=sum(f["gated"]["net"]>=0 for f in folds); worst=min(f["gated"]["net"] for f in folds)
    keep=gt["corrections"]/max(1,rawt["corrections"])
    crit={"net_gt_10":gt["net"]>10,"at_least_3_nonnegative_folds":nonneg>=3,
          "worst_fold_gte_minus3":worst>=-3,"keeps_at_least_half_corrections":keep>=.5}
    crit["all_met"]=all(crit.values())
    report={"status":"completed","experiment":"v273_symmetric_h2_reliability_gate",
            "features":list(FEATURES),"folds":folds,"raw_total":rawt,"gated_total":gt,
            "criteria":crit,"outer_fold_3_used":False,"automatic_promotion":False}
    a.output.mkdir(parents=True);write_json(a.output/"report.json",report)
    lines=["# Symmetric H2 reliability gate","",
      "| fold | raw net | gated corr | gated reg | gated net | rel AUC |",
      "|---:|---:|---:|---:|---:|---:|"]
    for f in folds:
        au=f["val_reliability_auc"]
        lines.append(f"| {f['fold']} | {f['raw']['net']:+d} | {f['gated']['corrections']} | {f['gated']['regressions']} | {f['gated']['net']:+d} | {'n/a' if au is None else f'{au:.3f}'} |")
    lines+=["",f"Raw H2: **{rawt['corrections']} / {rawt['regressions']} = {rawt['net']:+d}**.",
            f"Gated H2: **{gt['corrections']} / {gt['regressions']} = {gt['net']:+d}**.",
            f"Criteria met: **{crit['all_met']}**.","","Fold 3 excluded."]
    (a.output/"report.md").write_text("\n".join(lines)+"\n");print("\n".join(lines))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--input-root",type=Path,required=True);p.add_argument("--output",type=Path,required=True);run(p.parse_args())
