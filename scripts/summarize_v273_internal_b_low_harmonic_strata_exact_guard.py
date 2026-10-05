"""Aggregate exact-K utility of harmonic strata gates across internal folds."""
from __future__ import annotations
import argparse,json
from pathlib import Path
FOLDS=(0,1,2,4)

def discover(root):
    out={}
    for p in root.rglob("report.json"):
        try:r=json.loads(p.read_text())
        except Exception:continue
        if r.get("protocol",{}).get("experiment")!="v273_internal_B_low_harmonic_strata_exact_guard":
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
    gates=set.intersection(*[{x["gate"] for x in r[f]["rows"]} for f in FOLDS])
    rows=[]
    for gate in sorted(gates):
        per={f:next(x for x in r[f]["rows"] if x["gate"]==gate) for f in FOLDS}
        gs=[per[f]["global_net"] for f in FOLDS]
        ps=[per[f]["poly_net"] for f in FOLDS]
        ls=[per[f]["low_net"] for f in FOLDS]
        strict=(all(x>=0 for x in ps) and all(x>=0 for x in ls)
                and sum(x>0 for x in gs)>=3 and sum(gs)>0)
        rows.append({
          "gate":gate,
          "per_fold":{str(f):{
            "global_net":per[f]["global_net"],
            "poly_net":per[f]["poly_net"],
            "low_net":per[f]["low_net"],
            "applied_rows":per[f]["applied_rows"],
            "corrections":per[f]["corrections"],
            "regressions":per[f]["regressions"],
            "by_k_net":per[f]["by_k_net"],
          } for f in FOLDS},
          "positive_global_folds":sum(x>0 for x in gs),
          "total_global_net":sum(gs),
          "min_global_net":min(gs),
          "total_poly_net":sum(ps),
          "min_poly_net":min(ps),
          "total_low_net":sum(ls),
          "min_low_net":min(ls),
          "total_corrections":sum(per[f]["corrections"] for f in FOLDS),
          "total_regressions":sum(per[f]["regressions"] for f in FOLDS),
          "strict_robust":strict
        })
    ranked=sorted(rows,key=lambda x:(x["strict_robust"],x["positive_global_folds"],x["total_global_net"],x["min_global_net"]),reverse=True)
    strict=[x for x in ranked if x["strict_robust"]]
    result={
      "status":"completed","outer_fold_3_used":False,
      "fresh_validation":False,
      "reason_not_fresh":"gate families were discovered on the same internal folds in run 37330190680",
      "folds":list(FOLDS),"gates_tested":len(rows),
      "strict_candidate_count":len(strict),
      "descriptive_ranking":ranked,
      "best_descriptive_gate":ranked[0] if ranked else None
    }
    (a.output/"report.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    lines=["# Harmonic strata exact-K guards — multi-fold summary","",
      "**Exploratory only:** the strata were discovered on these same internal folds.","",
      "| gate | positive folds | total global | min global | total poly | min poly | corrections | regressions | strict |",
      "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for x in ranked:
        lines.append(f"| {x['gate']} | {x['positive_global_folds']}/4 | {x['total_global_net']:+d} | "
                     f"{x['min_global_net']:+d} | {x['total_poly_net']:+d} | {x['min_poly_net']:+d} | "
                     f"{x['total_corrections']} | {x['total_regressions']} | {x['strict_robust']} |")
    if ranked:
        b=ranked[0]
        lines+=["",f"Best descriptive gate: **{b['gate']}** ({b['total_global_net']:+d} exacts total, min fold {b['min_global_net']:+d})."]
    lines+=["","No outer evaluation and no promotion."]
    (a.output/"report.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))

if __name__=="__main__":
    main()
