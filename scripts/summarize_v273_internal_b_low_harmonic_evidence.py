"""Aggregate fine-STFT harmonic evidence across internal folds."""
from __future__ import annotations
import argparse,json,statistics
from pathlib import Path
FOLDS=(0,1,2,4)

def discover(root):
    out={}
    for p in root.rglob("report.json"):
        try:r=json.loads(p.read_text())
        except Exception:continue
        if r.get("protocol",{}).get("experiment")!="v273_internal_B_low_fine_STFT_harmonic_evidence":
            continue
        f=r["protocol"].get("validation_fold")
        if f in FOLDS:
            if f in out:raise RuntimeError(f"duplicate fold {f}")
            out[f]=r
    if set(out)!=set(FOLDS):raise RuntimeError(f"missing folds {set(FOLDS)-set(out)}")
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input-root",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    a=ap.parse_args()
    if a.output.exists():raise RuntimeError("refusing overwrite")
    a.output.mkdir(parents=True)
    r=discover(a.input_root)
    variants={}
    for name in ("current","harmonic","current_plus_harmonic"):
        vals=[r[f]["variants"][name]["auc"] for f in FOLDS]
        variants[name]={
          "by_fold":{str(f):vals[i] for i,f in enumerate(FOLDS)},
          "mean_auc":statistics.mean(vals),"min_auc":min(vals),"max_auc":max(vals)
        }
    feats=r[FOLDS[0]]["feature_names"];rows=[]
    for feat in feats:
        vals=[]
        for f in FOLDS:
            hit=next((x for x in r[f]["univariate"] if x["feature"]==feat),None)
            vals.append(hit["auc"] if hit else None)
        good=[x for x in vals if x is not None]
        rows.append({"feature":feat,"by_fold":{str(f):vals[i] for i,f in enumerate(FOLDS)},
                     "mean_auc":statistics.mean(good) if good else None,
                     "min_auc":min(good) if good else None,
                     "max_auc":max(good) if good else None})
    rows.sort(key=lambda x:(x["mean_auc"] if x["mean_auc"] is not None else -1,
                            x["min_auc"] if x["min_auc"] is not None else -1),reverse=True)
    base=variants["current"]["mean_auc"];harm=variants["harmonic"]["mean_auc"];both=variants["current_plus_harmonic"]["mean_auc"]
    result={
      "status":"completed","outer_fold_3_used":False,"folds":list(FOLDS),
      "variants":variants,"univariate":rows,
      "delta_harmonic_vs_current":harm-base,
      "delta_combined_vs_current":both-base,
      "verdict":{
        "harmonic_adds_mean_auc_002":bool(both>=base+0.02 or harm>=base+0.02),
        "combined_min_auc_ge_065":bool(variants["current_plus_harmonic"]["min_auc"]>=0.65),
        "combined_mean_auc_ge_070":bool(both>=0.70)
      }
    }
    (a.output/"report.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    lines=["# Fine-STFT harmonic evidence — multi-fold summary","",
      "| variant | mean AUC K2/K3 | min fold | max fold |",
      "|---|---:|---:|---:|"]
    for n in ("current","harmonic","current_plus_harmonic"):
        x=variants[n];lines.append(f"| {n} | {x['mean_auc']:.3f} | {x['min_auc']:.3f} | {x['max_auc']:.3f} |")
    lines+=["",
      f"Harmonic-only delta vs current: **{harm-base:+.3f}**.",
      f"Combined delta vs current: **{both-base:+.3f}**.","",
      "Top harmonic features:","","| feature | mean AUC | min fold | max fold |","|---|---:|---:|---:|"]
    for x in rows[:10]:
        lines.append(f"| {x['feature']} | {x['mean_auc']:.3f} | {x['min_auc']:.3f} | {x['max_auc']:.3f} |")
    lines+=["",
      f"Adds >=0.02 mean AUC: **{result['verdict']['harmonic_adds_mean_auc_002']}**.",
      f"Combined >=0.65 every fold: **{result['verdict']['combined_min_auc_ge_065']}**.",
      f"Combined mean >=0.70: **{result['verdict']['combined_mean_auc_ge_070']}**.",
      "",
      "No outer evaluation and no promotion."]
    (a.output/"report.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))

if __name__=="__main__":
    main()
