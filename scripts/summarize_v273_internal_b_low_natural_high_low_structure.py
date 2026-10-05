"""Aggregate natural high-vs-low structural audit across internal folds."""
from __future__ import annotations
import argparse,json,statistics
from pathlib import Path
FOLDS=(0,1,2,4)

def discover(root):
    out={}
    for p in root.rglob("report.json"):
        try:r=json.loads(p.read_text())
        except Exception:continue
        if r.get("protocol",{}).get("experiment")!="v273_internal_B_low_natural_high_vs_low_structure":
            continue
        f=r["protocol"].get("validation_fold")
        if f in FOLDS:
            if f in out:raise RuntimeError(f"duplicate fold {f}")
            out[f]=r
    if set(out)!=set(FOLDS):raise RuntimeError(f"missing folds {set(FOLDS)-set(out)}")
    return out

def avg(vals):
    vals=[x for x in vals if x is not None]
    return statistics.mean(vals) if vals else None

def amin(vals):
    vals=[x for x in vals if x is not None]
    return min(vals) if vals else None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input-root",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    a=ap.parse_args()
    if a.output.exists():raise RuntimeError("refusing overwrite")
    a.output.mkdir(parents=True)
    r=discover(a.input_root)

    features=r[FOLDS[0]]["protocol"]["relative_features"]
    rows=[]
    for feat in features:
        reg=[];low=[];high=[];same=[]
        dirs=[]
        for f in FOLDS:
            rr=next(x for x in r[f]["register_separation"] if x["feature"]==feat)
            cc=next(x for x in r[f]["direction_consistency"] if x["feature"]==feat)
            reg.append(rr.get("auc"));low.append(cc.get("low_val_auc"));high.append(cc.get("high_val_auc"))
            same.append(bool(cc.get("same_direction")))
            dirs.append((cc.get("fit_low_direction"),cc.get("fit_high_direction")))
        good_both=[min(l,h) for l,h in zip(low,high) if l is not None and h is not None]
        rows.append({
          "feature":feat,
          "register_separation_mean_auc":avg(reg),
          "low_k2k3_mean_auc":avg(low),
          "high_k2k3_mean_auc":avg(high),
          "min_register_k2k3_mean":avg(good_both),
          "same_direction_folds":sum(same),
          "low_auc_by_fold":{str(f):low[i] for i,f in enumerate(FOLDS)},
          "high_auc_by_fold":{str(f):high[i] for i,f in enumerate(FOLDS)},
          "register_auc_by_fold":{str(f):reg[i] for i,f in enumerate(FOLDS)},
          "fit_directions_by_fold":{str(f):dirs[i] for i,f in enumerate(FOLDS)}
        })
    rows.sort(key=lambda x:(x["same_direction_folds"],
                            x["min_register_k2k3_mean"] if x["min_register_k2k3_mean"] is not None else -1,
                            x["high_k2k3_mean_auc"] if x["high_k2k3_mean_auc"] is not None else -1),reverse=True)

    low_multi=[r[f]["multivariate"]["low"]["auc"] for f in FOLDS]
    high_multi=[r[f]["multivariate"]["high"]["auc"] for f in FOLDS]
    result={
      "status":"completed","outer_fold_3_used":False,"folds":list(FOLDS),
      "multivariate":{
        "low_by_fold":{str(f):low_multi[i] for i,f in enumerate(FOLDS)},
        "high_by_fold":{str(f):high_multi[i] for i,f in enumerate(FOLDS)},
        "low_mean_auc":avg(low_multi),"low_min_auc":amin(low_multi),
        "high_mean_auc":avg(high_multi),"high_min_auc":amin(high_multi)
      },
      "features":rows
    }
    (a.output/"report.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")

    lines=["# Natural high vs low structure — multi-fold summary","",
      f"Relative-feature model, LOW: mean AUC **{result['multivariate']['low_mean_auc']:.3f}**, min **{result['multivariate']['low_min_auc']:.3f}**.",
      f"Relative-feature model, HIGH: mean AUC **{result['multivariate']['high_mean_auc']:.3f}**, min **{result['multivariate']['high_min_auc']:.3f}**.","",
      "| feature | high-vs-low AUC | K2/K3 low AUC | K2/K3 high AUC | same K2/K3 direction folds |",
      "|---|---:|---:|---:|---:|"]
    for x in rows:
        def ff(v):return "n/a" if v is None else f"{v:.3f}"
        lines.append(f"| {x['feature']} | {ff(x['register_separation_mean_auc'])} | "
                     f"{ff(x['low_k2k3_mean_auc'])} | {ff(x['high_k2k3_mean_auc'])} | "
                     f"{x['same_direction_folds']}/4 |")
    lines+=["","Interpretation:",
      "- Features with high high-vs-low AUC explain why naturally high and low examples differ structurally.",
      "- Features with the same K2/K3 direction across both registers and multiple folds are candidates for a register-invariant discriminator.",
      "- No outer evaluation and no promotion."]
    (a.output/"report.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))

if __name__=="__main__":
    main()
