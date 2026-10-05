"""Aggregate temporal-onset feature audit across internal folds."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
FOLDS=(0,1,2,4)

def discover(root):
    out={}
    for p in root.rglob("report.json"):
        try:r=json.loads(p.read_text())
        except Exception:continue
        if r.get("protocol",{}).get("experiment")!="v273_internal_B_low_temporal_onset_feature_audit":
            continue
        f=r["protocol"].get("validation_fold")
        if f in FOLDS:
            if f in out: raise RuntimeError(f"duplicate fold {f}")
            out[f]=r
    if set(out)!=set(FOLDS): raise RuntimeError(f"missing folds {set(FOLDS)-set(out)}")
    return out

def avg(xs):
    xs=[x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input-root",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    a=ap.parse_args()
    if a.output.exists(): raise RuntimeError("refusing overwrite")
    a.output.mkdir(parents=True)

    r=discover(a.input_root)
    variants={}
    for name in ("current","onset","current_plus_onset"):
        k23=[r[f]["variants"][name]["k2_vs_k3"]["auc"] for f in FOLDS]
        rest=[r[f]["variants"][name]["k2_vs_rest"]["auc"] for f in FOLDS]
        variants[name]={
          "k2_vs_k3_by_fold":{str(f):r[f]["variants"][name]["k2_vs_k3"]["auc"] for f in FOLDS},
          "k2_vs_rest_by_fold":{str(f):r[f]["variants"][name]["k2_vs_rest"]["auc"] for f in FOLDS},
          "mean_k2_vs_k3_auc":avg(k23),"min_k2_vs_k3_auc":min(k23),
          "mean_k2_vs_rest_auc":avg(rest),"min_k2_vs_rest_auc":min(rest)
        }

    # Aggregate univariate AUCs by feature using same FIT-learned orientation per fold.
    names=r[FOLDS[0]]["feature_names"]
    univ=[]
    for feature in names:
        vals=[]
        for f in FOLDS:
            row=next(x for x in r[f]["univariate"] if x["feature"]==feature and x["task"]=="k2_vs_k3")
            vals.append(row["auc"])
        univ.append({
          "feature":feature,
          "by_fold":{str(f):vals[i] for i,f in enumerate(FOLDS)},
          "mean_auc":avg(vals),"min_auc":min(vals),"max_auc":max(vals)
        })
    univ.sort(key=lambda x:(x["mean_auc"],x["min_auc"]),reverse=True)

    base=variants["current"]["mean_k2_vs_k3_auc"]
    onset=variants["onset"]["mean_k2_vs_k3_auc"]
    both=variants["current_plus_onset"]["mean_k2_vs_k3_auc"]
    result={
      "status":"completed","outer_fold_3_used":False,"folds":list(FOLDS),
      "variants":variants,"univariate_k2_vs_k3":univ,
      "mean_auc_delta_onset_vs_current":onset-base,
      "mean_auc_delta_combined_vs_current":both-base,
      "verdict":{
        "temporal_onset_adds_signal":bool(both>=base+0.02 or onset>=base+0.02),
        "target_auc_065_reached_all_folds":bool(variants["current_plus_onset"]["min_k2_vs_k3_auc"]>=0.65),
        "target_auc_070_mean_reached":bool(both>=0.70)
      }
    }
    (a.output/"report.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    lines=[
      "# B_low temporal-onset features — multi-fold summary","",
      "| variant | mean AUC K2/K3 | min fold | mean AUC K2/rest | min fold |",
      "|---|---:|---:|---:|---:|"
    ]
    for name in ("current","onset","current_plus_onset"):
        x=variants[name]
        lines.append(f"| {name} | {x['mean_k2_vs_k3_auc']:.3f} | {x['min_k2_vs_k3_auc']:.3f} | {x['mean_k2_vs_rest_auc']:.3f} | {x['min_k2_vs_rest_auc']:.3f} |")
    lines+=["",
      f"Combined-current mean K2/K3 AUC delta: **{both-base:+.3f}**.",
      f"Onset-only vs current delta: **{onset-base:+.3f}**.","",
      "Top univariate temporal features:","",
      "| feature | mean AUC | min fold | max fold |",
      "|---|---:|---:|---:|"
    ]
    for x in univ[:10]:
        lines.append(f"| {x['feature']} | {x['mean_auc']:.3f} | {x['min_auc']:.3f} | {x['max_auc']:.3f} |")
    lines+=["",
      f"Temporal onset adds >=0.02 mean AUC: **{result['verdict']['temporal_onset_adds_signal']}**.",
      f"Combined reaches AUC >=0.65 on every fold: **{result['verdict']['target_auc_065_reached_all_folds']}**.",
      f"Combined mean reaches 0.70: **{result['verdict']['target_auc_070_mean_reached']}**.",
      "",
      "No outer evaluation and no promotion."
    ]
    (a.output/"report.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))

if __name__=="__main__":
    main()
