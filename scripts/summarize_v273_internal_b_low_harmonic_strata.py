"""Aggregate stratified harmonic audit across internal folds."""
from __future__ import annotations
import argparse,json,statistics
from pathlib import Path
FOLDS=(0,1,2,4)

def discover(root):
    out={}
    for p in root.rglob("report.json"):
        try:r=json.loads(p.read_text())
        except Exception:continue
        if r.get("protocol",{}).get("experiment")!="v273_internal_B_low_harmonic_stratified_audit":
            continue
        f=r["protocol"].get("validation_fold")
        if f in FOLDS:
            if f in out:raise RuntimeError(f"duplicate fold {f}")
            out[f]=r
    if set(out)!=set(FOLDS):raise RuntimeError(f"missing folds {set(FOLDS)-set(out)}")
    return out

def mean(vals):
    vals=[x for x in vals if x is not None]
    return statistics.mean(vals) if vals else None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input-root",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    a=ap.parse_args()
    if a.output.exists():raise RuntimeError("refusing overwrite")
    a.output.mkdir(parents=True)
    r=discover(a.input_root)

    global_auc={str(f):r[f]["global"]["combined_auc"] for f in FOLDS}
    strata={}
    for sname in r[FOLDS[0]]["strata"]:
        labels=[x["label"] for x in r[FOLDS[0]]["strata"][sname]]
        rows=[]
        for label in labels:
            vals=[];feat={}
            for f in FOLDS:
                hit=next(x for x in r[f]["strata"][sname] if x["label"]==label)
                vals.append(hit["combined_auc"])
                for k,v in hit["features"].items():
                    feat.setdefault(k,[]).append(v)
            rows.append({
              "label":label,
              "combined_auc_by_fold":{str(f):vals[i] for i,f in enumerate(FOLDS)},
              "mean_combined_auc":mean(vals),
              "min_combined_auc":min([x for x in vals if x is not None],default=None),
              "positive_fold_count":sum(x is not None and x>=0.55 for x in vals),
              "feature_mean_auc":{k:mean(v) for k,v in feat.items()}
            })
        strata[sname]=rows

    candidates=[]
    for sname,rows in strata.items():
        for row in rows:
            if row["mean_combined_auc"] is None:continue
            candidates.append({
              "stratum_family":sname,
              **row
            })
    candidates.sort(key=lambda x:(x["positive_fold_count"],x["mean_combined_auc"],x["min_combined_auc"] if x["min_combined_auc"] is not None else -1),reverse=True)

    result={
      "status":"completed","outer_fold_3_used":False,"folds":list(FOLDS),
      "global_combined_auc_by_fold":global_auc,
      "global_combined_mean_auc":mean(list(global_auc.values())),
      "strata":strata,
      "ranked_strata":candidates
    }
    (a.output/"report.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")

    lines=["# Stratified harmonic evidence — multi-fold summary","",
      f"Global 4-feature mean AUC: **{result['global_combined_mean_auc']:.3f}**.","",
      "| stratum | label | mean combined AUC | min fold | folds >=0.55 | third coeff | pair residual | triplet residual |",
      "|---|---|---:|---:|---:|---:|---:|---:|"]
    for x in candidates[:18]:
        def ff(v):return "n/a" if v is None else f"{v:.3f}"
        fm=x["feature_mean_auc"]
        lines.append(
          f"| {x['stratum_family']} | {x['label']} | {ff(x['mean_combined_auc'])} | "
          f"{ff(x['min_combined_auc'])} | {x['positive_fold_count']}/4 | "
          f"{ff(fm.get('third_coefficient_fraction'))} | "
          f"{ff(fm.get('best_pair_residual_ratio'))} | "
          f"{ff(fm.get('best_triplet_residual_ratio'))} |"
        )
    lines+=["","No outer evaluation and no promotion."]
    (a.output/"report.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))

if __name__=="__main__":
    main()
