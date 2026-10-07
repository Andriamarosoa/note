"""Class-conditional FIT-vs-VAL shift audit on archived V27.3 residual features.

Scope: players 00-04, folds 0/1/2/4 only.
No fold3, no player05, no audio, no refit, no threshold tuning.

For every feature:
- freeze K2-vs-K3 orientation from FIT only;
- compare oriented K2-vs-K3 AUC on FIT vs VAL;
- compare FIT-vs-VAL distribution shift separately inside true K2 and true K3.

Diagnostic only.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

from scripts.audit_v273_feature_family_nested import arrays,split,validate_split,ALL_FEATURES,FOLDS

ALLOWED={"00","01","02","03","04"}

def require(c,m):
    if not c: raise RuntimeError(m)

def domain_auc(fit_vals,val_vals):
    x=np.concatenate([fit_vals,val_vals]).astype(float)
    y=np.concatenate([np.zeros(len(fit_vals),int),np.ones(len(val_vals),int)])
    if len(np.unique(x))<2:return .5
    a=float(roc_auc_score(y,x))
    return max(a,1.0-a)

def robust_shift(fit_vals,val_vals):
    f=np.asarray(fit_vals,float);v=np.asarray(val_vals,float)
    q1,q3=np.quantile(f,[.25,.75]);scale=max(float(q3-q1),1e-12)
    return float((np.median(v)-np.median(f))/scale)

def oriented_class_auc(y,x,sign):
    y=np.asarray(y,int);x=np.asarray(x,float)
    return float(roc_auc_score((y==2).astype(int),sign*x))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--inputs",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    a=ap.parse_args();require(not a.output.exists(),"overwrite")
    data=arrays(a.inputs)

    results=[]
    prevalence=[]
    for f in FOLDS:
        fit,val=split(data,f,"fit"),split(data,f,"val")
        validate_split(fit,val,f)
        for d,label in ((fit,"fit"),(val,"val")):
            players={str(r)[:2] for r in d["recording"]}
            require(players<=ALLOWED and "05" not in players,f"excluded player in {label} fold {f}: {players}")
            require(set(map(int,d["fold"]))<=set(FOLDS),"excluded fold present")
        fk=np.isin(fit["y"],(2,3));vk=np.isin(val["y"],(2,3))
        prevalence.append({
          "fold":int(f),
          "fit_K2":int(np.sum(fit["y"][fk]==2)),
          "fit_K3":int(np.sum(fit["y"][fk]==3)),
          "val_K2":int(np.sum(val["y"][vk]==2)),
          "val_K3":int(np.sum(val["y"][vk]==3)),
        })
        for j,name in enumerate(ALL_FEATURES):
            xf=fit["X"][:,j].astype(float);xv=val["X"][:,j].astype(float)
            yf=fit["y"].astype(int);yv=val["y"].astype(int)
            # FIT-only class orientation.
            raw_auc=float(roc_auc_score((yf[fk]==2).astype(int),xf[fk]))
            sign=1.0 if raw_auc>=.5 else -1.0
            fit_auc=max(raw_auc,1.0-raw_auc)
            val_auc=oriented_class_auc(yv[vk],xv[vk],sign)
            class_flip=val_auc<.5
            cls={}
            for k in (2,3):
                af=xf[yf==k];av=xv[yv==k]
                require(len(af)>0 and len(av)>0,f"missing K{k} fold {f}")
                cls[str(k)]={
                  "fit_n":int(len(af)),"val_n":int(len(av)),
                  "fit_median":float(np.median(af)),
                  "val_median":float(np.median(av)),
                  "domain_auc":domain_auc(af,av),
                  "robust_median_shift_iqr_fit":robust_shift(af,av),
                }
            results.append({
              "fold":int(f),"feature":name,
              "fit_class_auc_oriented_K2":fit_auc,
              "val_class_auc_with_fit_orientation_K2":val_auc,
              "class_auc_delta_val_minus_fit":val_auc-fit_auc,
              "class_direction_flip_on_val":bool(class_flip),
              "class_shift":cls,
            })

    summary=[]
    for name in ALL_FEATURES:
        rr=[q for q in results if q["feature"]==name]
        dom=[q["class_shift"][k]["domain_auc"] for q in rr for k in ("2","3")]
        medshift=[abs(q["class_shift"][k]["robust_median_shift_iqr_fit"]) for q in rr for k in ("2","3")]
        deltas=[q["class_auc_delta_val_minus_fit"] for q in rr]
        summary.append({
          "feature":name,
          "direction_flip_folds":sum(q["class_direction_flip_on_val"] for q in rr),
          "mean_domain_auc_at_fixed_K":float(np.mean(dom)),
          "max_domain_auc_at_fixed_K":float(np.max(dom)),
          "mean_abs_robust_median_shift":float(np.mean(medshift)),
          "mean_class_auc_delta_val_minus_fit":float(np.mean(deltas)),
          "min_val_class_auc_with_fit_orientation":float(min(q["val_class_auc_with_fit_orientation_K2"] for q in rr)),
          "mean_fit_class_auc":float(np.mean([q["fit_class_auc_oriented_K2"] for q in rr])),
          "mean_val_class_auc":float(np.mean([q["val_class_auc_with_fit_orientation_K2"] for q in rr])),
        })
    summary.sort(key=lambda q:(q["direction_flip_folds"],
                               q["mean_domain_auc_at_fixed_K"],
                               -q["mean_class_auc_delta_val_minus_fit"]),reverse=True)

    rep={
      "status":"completed",
      "experiment":"v273_fit_val_class_conditional_shift",
      "folds":list(FOLDS),
      "players":["00","01","02","03","04"],
      "player05_used":False,
      "outer_fold_3_used":False,
      "audio_loaded":False,
      "model_refitted":False,
      "threshold_search":False,
      "prevalence":prevalence,
      "per_fold_feature":results,
      "feature_summary":summary,
      "diagnostic_only":True,
    }
    a.output.mkdir(parents=True)
    (a.output/"report.json").write_text(json.dumps(rep,indent=2,sort_keys=True)+"\n")
    lines=["# FIT↔VAL shift conditionné par vrai K","",
           "Players 00–04, folds 0/1/2/4 uniquement. Aucun player 05, aucun fold 3.","",
           "| feature | flips K2/K3 | domain AUC moyen à K fixé | max domain AUC | Δ AUC classe VAL-FIT | AUC FIT | AUC VAL |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    for q in summary:
        lines.append(f"| {q['feature']} | {q['direction_flip_folds']} | {q['mean_domain_auc_at_fixed_K']:.3f} | {q['max_domain_auc_at_fixed_K']:.3f} | {q['mean_class_auc_delta_val_minus_fit']:+.3f} | {q['mean_fit_class_auc']:.3f} | {q['mean_val_class_auc']:.3f} |")
    lines+=["","Diagnostic only; aucune règle ou feature n'est promue depuis ce run."]
    (a.output/"report.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))

if __name__=="__main__":main()
