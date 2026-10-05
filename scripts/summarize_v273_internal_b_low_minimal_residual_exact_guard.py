"""Aggregate minimal pair+triplet residual exact-K guard across folds."""
from __future__ import annotations
import argparse,json,statistics
from pathlib import Path
FOLDS=(0,1,2,4)

def discover(root):
    out={}
    for p in root.rglob("report.json"):
        try:r=json.loads(p.read_text())
        except Exception:continue
        if r.get("protocol",{}).get("experiment")!="v273_internal_B_low_minimal_pair_triplet_residual_exact_guard":
            continue
        f=r["protocol"].get("validation_fold")
        if f in FOLDS:
            if f in out:raise RuntimeError(f"duplicate fold {f}")
            out[f]=r
    if set(out)!=set(FOLDS):raise RuntimeError(f"missing folds {set(FOLDS)-set(out)}")
    return out

def avg(xs): return statistics.mean(xs) if xs else None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input-root",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    a=ap.parse_args()
    if a.output.exists():raise RuntimeError("refusing overwrite")
    a.output.mkdir(parents=True)
    r=discover(a.input_root)

    aucs=[r[f]["diagnostic"]["k2_vs_k3_auc"] for f in FOLDS]
    gs=[r[f]["action"]["global_net"] for f in FOLDS]
    ps=[r[f]["action"]["poly_net"] for f in FOLDS]
    ls=[r[f]["action"]["low_net"] for f in FOLDS]
    k23=[r[f]["action"]["k23_exact_net"] for f in FOLDS]
    corrections=[r[f]["action"]["corrections"] for f in FOLDS]
    regressions=[r[f]["action"]["regressions"] for f in FOLDS]
    applied=[r[f]["action"]["applied_rows"] for f in FOLDS]

    strict=(all(x>=0 for x in ps) and all(x>=0 for x in ls)
            and sum(x>0 for x in gs)>=3 and sum(gs)>0)

    by_k={str(k):sum(r[f]["action"]["by_k_net"][str(k)] for f in FOLDS) for k in range(7)}
    result={
      "status":"completed","outer_fold_3_used":False,"folds":list(FOLDS),
      "k2_vs_k3_auc_by_fold":{str(f):aucs[i] for i,f in enumerate(FOLDS)},
      "mean_k2_vs_k3_auc":avg(aucs),
      "min_k2_vs_k3_auc":min(aucs),
      "global_net_by_fold":{str(f):gs[i] for i,f in enumerate(FOLDS)},
      "k23_exact_net_by_fold":{str(f):k23[i] for i,f in enumerate(FOLDS)},
      "positive_global_folds":sum(x>0 for x in gs),
      "total_global_net":sum(gs),"min_global_net":min(gs),
      "total_poly_net":sum(ps),"min_poly_net":min(ps),
      "total_low_net":sum(ls),"min_low_net":min(ls),
      "total_k23_exact_net":sum(k23),
      "total_applied":sum(applied),
      "total_corrections":sum(corrections),
      "total_regressions":sum(regressions),
      "by_k_net":by_k,
      "strict_robust":strict
    }
    (a.output/"report.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    lines=["# Minimal pair+triplet residual exact-K guard — multi-fold summary","",
      f"K2/K3 mean AUC: **{result['mean_k2_vs_k3_auc']:.3f}** (min **{result['min_k2_vs_k3_auc']:.3f}**).",
      f"Global Exact-K: **{result['total_global_net']:+d}** total, min fold **{result['min_global_net']:+d}**, positive folds **{result['positive_global_folds']}/4**.",
      f"K2/K3-only Exact-K: **{result['total_k23_exact_net']:+d}**.",
      f"Corrections/regressions: **{result['total_corrections']}/{result['total_regressions']}** over **{result['total_applied']}** actions.",
      f"Strict robust: **{result['strict_robust']}**.","",
      "| fold | AUC K2/K3 | global net | K2/K3 net |",
      "|---|---:|---:|---:|"]
    for i,f in enumerate(FOLDS):
        lines.append(f"| {f} | {aucs[i]:.3f} | {gs[i]:+d} | {k23[i]:+d} |")
    lines+=["","By-K net: "+json.dumps(by_k,sort_keys=True),"",
            "No outer evaluation and no promotion."]
    (a.output/"report.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))

if __name__=="__main__":
    main()
